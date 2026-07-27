import click
import torch

from astrai import setup_logging
from astrai.config import AutoRegressiveLMConfig

_DTYPES = ["bfloat16", "float16", "float32"]
_CACHES = ["contiguous", "paged"]


class BenchmarkResult:
    def __init__(
        self,
        name: str,
        batch_size: int,
        seq_len: int,
        tokens_per_second: float,
        latency_ms: float,
        metadata: dict | None = None,
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
        config: AutoRegressiveLMConfig,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        cache_type: str = "contiguous",
    ):
        from astrai.inference import InferenceEngine
        from astrai.model import AutoRegressiveLM

        self.device = device
        self.dtype = dtype
        self.cache_type = cache_type

        click.echo("Building model ...")
        self.model = AutoRegressiveLM(config).to(device=device, dtype=dtype)
        self.engine = InferenceEngine(
            model=self.model,
            tokenizer=None,
            max_batch_size=256,
            max_seq_len=config.max_position_embeddings,
        )

    def run_prefill_benchmark(
        self,
        batch_size: int = 4,
        prompt_length: int = 512,
        num_trials: int = 5,
    ) -> BenchmarkResult:
        import time

        input_ids = torch.randint(
            0, 10000, (batch_size, prompt_length), device=self.device
        )
        for _ in range(3):
            self.engine.model(input_ids)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_trials):
            self.engine.model(input_ids)
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

        prompt = torch.randint(
            0, 10000, (batch_size, prompt_length), device=self.device
        )
        with torch.inference_mode():
            kv = self.engine.model(prompt, use_cache=True)
        past = kv.past_key_values if hasattr(kv, "past_key_values") else kv[1]

        token = torch.randint(0, 10000, (batch_size, 1), device=self.device)
        for _ in range(3):
            self.engine.model(token, past_key_values=past, use_cache=True)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(gen_length * num_trials):
            self.engine.model(token, past_key_values=past, use_cache=True)
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
) -> None:
    """Benchmark model throughput and latency."""
    dtype_map: dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    config = AutoRegressiveLMConfig(
        vocab_size=10000,
        hidden_size=1536,
        num_attention_heads=24,
        num_key_value_heads=4,
        intermediate_size=6912,
        max_position_embeddings=2048,
        num_hidden_layers=24,
        rms_norm_eps=1e-5,
    )

    bench = GenerationBenchmark(
        config,
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
