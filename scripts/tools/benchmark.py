from pathlib import Path
from typing import Optional

import click
import torch

from astrai import setup_logging
from astrai.config import AutoRegressiveLMConfig
from astrai.extension import ATTN_BACKEND, attn_backend
from astrai.inference.core.cache import PagePool
from astrai.model import AutoModel

_DTYPES = ["bfloat16", "float16", "float32"]
_CACHES = ["contiguous", "paged"]
DEFAULT_CKPT = str(Path(__file__).resolve().parents[2] / "ckpt_bucket" / "kami-15bt")
CACHE_MAX_SEQ = 2048


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
        config: AutoRegressiveLMConfig,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_type: str = "contiguous",
    ):
        self.device = device
        self.dtype = dtype
        self.cache_type = cache_type
        self.model = model
        self.config = config

    def _make_pool(self, batch_size: int) -> PagePool:
        return PagePool(
            n_layers=self.config.num_hidden_layers,
            n_kv_heads=self.config.num_key_value_heads,
            head_dim=self.config.hidden_size // self.config.num_attention_heads,
            max_batch_size=batch_size,
            max_seq_len=CACHE_MAX_SEQ,
            device=self.device,
            dtype=self.dtype,
            page_size=1,
            n_tokens=None,
        )

    def _run_prefill(self, pool: PagePool, batch_size: int, prompt_len: int) -> list:
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

        kv_cache = pool.bind_tasks(
            task_ids, [prompt_len] * batch_size, self.device, start_pos=0
        )
        with torch.inference_mode(), attn_backend(ATTN_BACKEND.CUDA):
            self.model(
                input_ids,
                input_mask=input_mask,
                kv_cache=kv_cache,
                position_ids=position_ids,
            )
        torch.cuda.synchronize()
        return task_ids

    def _run_decode_step(self, pool: PagePool, task_ids: list, seq_len: int):
        batch_size = len(task_ids)
        input_ids = torch.randint(
            0, self.config.vocab_size, (batch_size, 1), device=self.device
        )
        position_ids = torch.tensor(
            [[seq_len] for _ in range(batch_size)], dtype=torch.long, device=self.device
        )
        total_len = seq_len + 1
        input_mask = position_ids[:, :, None] >= torch.arange(
            total_len, device=self.device
        )
        kv_cache = pool.bind_tasks(task_ids, [seq_len + 1] * batch_size, self.device)
        with torch.inference_mode(), attn_backend(ATTN_BACKEND.CUDA):
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

        input_ids = torch.randint(
            0, self.config.vocab_size, (batch_size, prompt_length), device=self.device
        )
        position_ids = (
            torch.arange(0, prompt_length, dtype=torch.long, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        for _ in range(3):
            with torch.inference_mode(), attn_backend(ATTN_BACKEND.CUDA):
                self.model(input_ids, position_ids=position_ids)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_trials):
            with torch.inference_mode(), attn_backend(ATTN_BACKEND.CUDA):
                self.model(input_ids, position_ids=position_ids)
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
        import time

        pool = self._make_pool(batch_size)
        task_ids = self._run_prefill(pool, batch_size, prompt_length)

        for i in range(5):
            self._run_decode_step(pool, task_ids, prompt_length + i)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for i in range(gen_length * num_trials):
            self._run_decode_step(pool, task_ids, prompt_length + 5 + i)
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
@click.option("--batch_size", type=int, default=4, help="Batch size.")
@click.option("--prompt_length", type=int, default=512, help="Prompt length.")
@click.option("--gen_length", type=int, default=128, help="Generation length.")
@click.option("--num_trials", type=int, default=5, help="Number of trials.")
@click.option("--prefill_only", is_flag=True, help="Prefill benchmark only.")
@click.option("--decode_only", is_flag=True, help="Decode benchmark only.")
@click.option(
    "--ckpt",
    default=DEFAULT_CKPT,
    help="Checkpoint directory.",
)
def benchmark_command(
    device: str,
    dtype: str,
    cache: str,
    batch_size: int,
    prompt_length: int,
    gen_length: int,
    num_trials: int,
    prefill_only: bool,
    decode_only: bool,
    ckpt: str,
) -> None:
    """Benchmark model throughput and latency."""
    dtype_map: dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    click.echo(f"Loading model from {ckpt} ...")
    config = AutoRegressiveLMConfig.from_file(str(Path(ckpt) / "config.json"))
    model = AutoModel.from_pretrained(ckpt)
    model.to(device=device, dtype=dtype_map[dtype])
    model.eval()

    bench = GenerationBenchmark(
        model=model,
        config=config,
        device=device,
        dtype=dtype_map[dtype],
        cache_type=cache,
    )

    click.secho(f"Benchmark: device={device} dtype={dtype}", bold=True)

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
