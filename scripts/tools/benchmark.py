import json
from pathlib import Path
from typing import Optional, Union

import click
import torch

from astrai import setup_logging
from astrai.config import BaseModelConfig, ConfigFactory
from astrai.extension import ATTN_BACKEND, AttentionBackendFactory, attn_backend
from astrai.inference.core.cache import PagePool
from astrai.inference.core.graph import CudaGraphContext
from astrai.inference.core.workspace import InferenceWorkspace
from astrai.model import AutoModel, AutoRegressiveLM

_DTYPES = ["bfloat16", "float16", "float32"]
_CACHES = ["contiguous", "paged"]
_BACKENDS = AttentionBackendFactory.list_registered()

# Default 1B GQA preset matching the project checkpoint architecture.
_DEFAULT_CONFIG = {
    "vocab_size": 100000,
    "hidden_size": 1536,
    "num_hidden_layers": 24,
    "intermediate_size": 6912,
    "num_attention_heads": 24,
    "num_key_value_heads": 4,
    "max_position_embeddings": 32768,
    "rms_norm_eps": 1e-05,
    "tie_word_embeddings": False,
}


class BenchmarkResult:
    def __init__(
        self,
        name: str,
        batch_size: int,
        seq_len: int,
        tokens_per_second: float,
        latency_ms: float,
        metadata: Optional[dict] = None,
    ):
        self.name = name
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tokens_per_second = tokens_per_second
        self.latency_ms = latency_ms
        self.metadata = metadata or {}


class GenerationBenchmark:
    def __init__(
        self,
        model: AutoModel,
        config: BaseModelConfig,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_type: str = "contiguous",
        backend: Union[str, ATTN_BACKEND] = ATTN_BACKEND.CUDA,
        cuda_graph: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.cache_type = cache_type
        self.model = model
        self.config = config
        self.backend = backend
        self.cuda_graph = cuda_graph

    def _make_pool(self, batch_size: int, max_seq_len: int) -> PagePool:
        return PagePool(
            n_layers=self.config.num_hidden_layers,
            n_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.hidden_size // self.config.num_attention_heads,
            max_batch_size=batch_size,
            max_seq_len=max_seq_len,
            device=self.device,
            dtype=self.dtype,
            page_size=1,
            n_tokens=None,
        )

    @staticmethod
    def _make_workspace(pool: PagePool, config: BaseModelConfig) -> InferenceWorkspace:
        return InferenceWorkspace(
            pool.max_batch_size,
            pool.max_seq_len,
            max_q_heads=config.num_attention_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            device=pool.device,
            dtype=pool.dtype,
        )

    def _run_prefill(
        self,
        pool: PagePool,
        batch_size: int,
        prompt_len: int,
        workspace: InferenceWorkspace,
    ) -> list:
        input_ids = torch.randint(
            0, self.config.vocab_size, (batch_size, prompt_len), device=self.device
        )
        position_ids = (
            torch.arange(0, prompt_len, dtype=torch.long, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        input_mask = position_ids.unsqueeze(-1) >= torch.arange(
            prompt_len, device=self.device
        )

        task_ids = [f"bench_{i}" for i in range(batch_size)]
        for tid in task_ids:
            pool.task_alloc(tid, list(range(prompt_len)))

        kv_cache = pool.bind_tasks(task_ids, workspace, self.device, start_pos=0)
        with torch.inference_mode(), attn_backend(self.backend):
            self.model(
                input_ids,
                input_mask=input_mask,
                kv_cache=kv_cache,
                position_ids=position_ids,
            )
        torch.cuda.synchronize()
        return task_ids

    def _run_decode_step(
        self,
        pool: PagePool,
        task_ids: list,
        seq_len: int,
        workspace: InferenceWorkspace,
    ):
        batch_size = len(task_ids)
        input_ids = torch.randint(
            0, self.config.vocab_size, (batch_size, 1), device=self.device
        )
        position_ids = torch.tensor(
            [[seq_len] for _ in range(batch_size)], dtype=torch.long, device=self.device
        )
        total_len = seq_len + 1
        for tid in task_ids:
            pool.task_extend(tid, seq_len)
        input_mask = position_ids[:, :, None] >= torch.arange(
            total_len, device=self.device
        )
        kv_cache = pool.bind_tasks(task_ids, workspace, self.device)
        with torch.inference_mode(), attn_backend(self.backend):
            self.model(
                input_ids,
                input_mask=input_mask,
                kv_cache=kv_cache,
                position_ids=position_ids,
            )

    def run_prefill_benchmark(
        self,
        batch_size: int = 4,
        prompt_length: int = 512,
        num_trials: int = 5,
    ) -> BenchmarkResult:
        import time

        pool = self._make_pool(batch_size, prompt_length)
        workspace = self._make_workspace(pool, self.config)
        task_ids = [f"bench_prefill_{i}" for i in range(batch_size)]
        for tid in task_ids:
            pool.task_alloc(tid, list(range(prompt_length)))

        input_ids = torch.randint(
            0, self.config.vocab_size, (batch_size, prompt_length), device=self.device
        )
        position_ids = (
            torch.arange(0, prompt_length, dtype=torch.long, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        input_mask = position_ids.unsqueeze(-1) >= torch.arange(
            prompt_length, device=self.device
        )
        kv_cache = pool.bind_tasks(task_ids, workspace, self.device, start_pos=0)

        for _ in range(3):
            with torch.inference_mode(), attn_backend(self.backend):
                self.model(
                    input_ids,
                    input_mask=input_mask,
                    kv_cache=kv_cache,
                    position_ids=position_ids,
                )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_trials):
            with torch.inference_mode(), attn_backend(self.backend):
                self.model(
                    input_ids,
                    input_mask=input_mask,
                    kv_cache=kv_cache,
                    position_ids=position_ids,
                )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tokens = batch_size * prompt_length * num_trials
        tps = tokens / elapsed
        return BenchmarkResult(
            name="prefill",
            batch_size=batch_size,
            seq_len=prompt_length,
            tokens_per_second=tps,
            latency_ms=elapsed / num_trials * 1000,
            metadata={"benchmark_type": "prefill", "num_trials": num_trials},
        )

    def run_decoding_benchmark(
        self,
        batch_size: int = 4,
        prompt_length: int = 512,
        gen_length: int = 128,
        num_trials: int = 5,
    ) -> BenchmarkResult:
        if self.cuda_graph and self.backend == "cuda":
            return self._run_graph_decode_benchmark(
                batch_size, prompt_length, gen_length, num_trials
            )
        return self._run_plain_decode_benchmark(
            batch_size, prompt_length, gen_length, num_trials
        )

    def _run_graph_decode_benchmark(
        self,
        batch_size: int,
        prompt_length: int,
        gen_length: int,
        num_trials: int,
    ) -> BenchmarkResult:
        import time

        max_seq_len = prompt_length + 5 + gen_length * num_trials
        pool = self._make_pool(batch_size, max_seq_len)
        workspace = self._make_workspace(pool, self.config)
        task_ids = self._run_prefill(pool, batch_size, prompt_length, workspace)

        b = batch_size
        input_ids_buf = torch.zeros(b, 1, dtype=torch.long, device=self.device)
        position_ids_buf = torch.zeros(b, dtype=torch.long, device=self.device)
        arange = torch.arange(max_seq_len, device=self.device)

        gctx = CudaGraphContext(enabled=True)
        graph_key = (b,)

        def _decode_graph_step(seq_len):
            input_ids_buf.copy_(
                torch.randint(0, self.config.vocab_size, (b, 1), device=self.device)
            )
            position_ids_buf[:] = seq_len
            for tid in task_ids:
                pool.task_extend(tid, seq_len)
            kv_cache = pool.bind_tasks(task_ids, workspace, self.device)

            input_mask = torch.ge(
                position_ids_buf[:, None],
                arange,
                out=workspace.input_mask[:b, 0, :max_seq_len],
            )
            input_mask = input_mask.unsqueeze(1)

            with torch.inference_mode(), attn_backend(self.backend):
                return gctx.forward(
                    self.model,
                    key=graph_key,
                    input_ids=input_ids_buf,
                    input_mask=input_mask,
                    kv_cache=kv_cache,
                    position_ids=position_ids_buf.unsqueeze(1),
                )

        for i in range(5):
            _decode_graph_step(prompt_length + i)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for i in range(gen_length * num_trials):
            _decode_graph_step(prompt_length + 5 + i)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tokens = batch_size * gen_length * num_trials
        tps = tokens / elapsed
        return BenchmarkResult(
            name="decode",
            batch_size=batch_size,
            seq_len=gen_length,
            tokens_per_second=tps,
            latency_ms=elapsed / (gen_length * num_trials) * 1000,
            metadata={
                "benchmark_type": "decode",
                "num_trials": num_trials,
                "prompt_length": prompt_length,
                "cuda_graph": True,
            },
        )

    def _run_plain_decode_benchmark(
        self,
        batch_size: int,
        prompt_length: int,
        gen_length: int,
        num_trials: int,
    ) -> BenchmarkResult:
        import time

        max_seq_len = prompt_length + 5 + gen_length * num_trials
        pool = self._make_pool(batch_size, max_seq_len)
        workspace = self._make_workspace(pool, self.config)
        task_ids = self._run_prefill(pool, batch_size, prompt_length, workspace)

        for i in range(5):
            self._run_decode_step(pool, task_ids, prompt_length + i, workspace)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for i in range(gen_length * num_trials):
            self._run_decode_step(pool, task_ids, prompt_length + 5 + i, workspace)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tokens = batch_size * gen_length * num_trials
        tps = tokens / elapsed
        return BenchmarkResult(
            name="decode",
            batch_size=batch_size,
            seq_len=gen_length,
            tokens_per_second=tps,
            latency_ms=elapsed / (gen_length * num_trials) * 1000,
            metadata={
                "benchmark_type": "decode",
                "num_trials": num_trials,
                "prompt_length": prompt_length,
            },
        )


def print_benchmark_result(result: BenchmarkResult) -> None:
    print("-" * 80)
    print(f"{result.name.upper()} — Batch={result.batch_size}, SeqLen={result.seq_len}")
    print(f"  Throughput : {result.tokens_per_second:.1f} tokens/s")
    print(f"  Latency    : {result.latency_ms:.2f} ms/step")
    for k, v in result.metadata.items():
        if k != "benchmark_type":
            print(f"  {k.replace('_', ' ').title()}: {v}")
    print("-" * 80)


@click.command(name="benchmark", help="Benchmark model throughput and latency.")
@click.option("--device", default="cuda", help="Device.")
@click.option(
    "--dtype", type=click.Choice(_DTYPES), default="bfloat16", help="Data type."
)
@click.option(
    "--cache", type=click.Choice(_CACHES), default="contiguous", help="KV cache type."
)
@click.option(
    "--backend",
    type=click.Choice(_BACKENDS),
    default="cuda",
    help="Attention backend.",
)
@click.option(
    "--compare",
    is_flag=True,
    help="Run both backends and print side-by-side speed comparison.",
)
@click.option("--batch_size", type=int, default=4, help="Batch size.")
@click.option("--prompt_length", type=int, default=512, help="Prompt length.")
@click.option("--gen_length", type=int, default=128, help="Generation length.")
@click.option("--num_trials", type=int, default=5, help="Number of trials.")
@click.option("--prefill_only", is_flag=True, help="Prefill benchmark only.")
@click.option("--decode_only", is_flag=True, help="Decode benchmark only.")
@click.option(
    "--cuda-graph",
    is_flag=True,
    help="Enable CUDA graph capture for decode (cuda backend only).",
)
@click.option(
    "--ckpt",
    required=False,
    default=None,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Checkpoint directory. If omitted, a randomly-initialized model is "
    "built from --config or the default 1B GQA preset.",
)
@click.option(
    "--config",
    "config_path",
    required=False,
    default=None,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help="Optional model config JSON (used when --ckpt is omitted to define the "
    "architecture). Defaults to the 1B GQA preset.",
)
def benchmark_command(
    device: str,
    dtype: str,
    cache: str,
    backend: str,
    compare: bool,
    batch_size: int,
    prompt_length: int,
    gen_length: int,
    num_trials: int,
    prefill_only: bool,
    decode_only: bool,
    cuda_graph: bool,
    ckpt: Optional[str],
    config_path: Optional[Path],
) -> None:
    """Benchmark model throughput and latency."""
    dtype_map: dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    if ckpt is not None:
        click.echo(f"Loading model from {ckpt} ...")
        config = ConfigFactory.load(
            json.loads((Path(ckpt) / "config.json").read_text(encoding="utf-8-sig"))
        )
        model = AutoModel.from_pretrained(ckpt)
    else:
        raw = dict(_DEFAULT_CONFIG)
        if config_path is not None:
            raw.update(json.loads(config_path.read_text(encoding="utf-8-sig")))
        config = ConfigFactory.load(raw)
        model = AutoRegressiveLM(config)
        click.echo(
            f"Using randomly-initialized model "
            f"({sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)"
        )

    model.to(device=device, dtype=dtype_map[dtype])
    model.eval()

    backends = _BACKENDS if compare else [backend]

    for name in backends:
        bench = GenerationBenchmark(
            model=model,
            config=config,
            device=device,
            dtype=dtype_map[dtype],
            cache_type=cache,
            backend=name,
            cuda_graph=cuda_graph,
        )

        click.secho(
            f"Benchmark: device={device} dtype={dtype} backend={name}", bold=True
        )

        if not decode_only:
            result = bench.run_prefill_benchmark(
                batch_size=batch_size,
                prompt_length=prompt_length,
                num_trials=num_trials,
            )
            print_benchmark_result(result)

        if not prefill_only:
            result = bench.run_decoding_benchmark(
                batch_size=batch_size,
                prompt_length=prompt_length,
                gen_length=gen_length,
                num_trials=num_trials,
            )
            print_benchmark_result(result)


if __name__ == "__main__":
    setup_logging()
    benchmark_command()
