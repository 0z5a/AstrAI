import json
import os
import shutil
import tempfile

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from astrai.extension import KERNEL_NAMES, is_available
from astrai.model.transformer import AutoRegressiveLM
from astrai.tokenize import AutoTokenizer
from tests.helpers import TINY_CONFIG, RandomTokenDataset, make_tiny_config

CUDA_AVAIL = torch.cuda.is_available()
KERNEL_AVAIL = CUDA_AVAIL and all(is_available(k) for k in KERNEL_NAMES)
skip_no_cuda = pytest.mark.skipif(not CUDA_AVAIL, reason="CUDA not available")
skip_no_kernel = pytest.mark.skipif(not KERNEL_AVAIL, reason="CUDA kernels not built")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "unit: fast unit tests")


@pytest.fixture(scope="session")
def device():
    """Session-scoped device string (``"cuda"`` if available, else ``"cpu"``)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def create_test_tokenizer(vocab_size: int = 1000) -> AutoTokenizer:
    """Create a simple tokenizer for testing purposes."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, min_frequency=1, special_tokens=["<unk>", "<pad>"]
    )
    tokenizer.train_from_iterator([chr(i) for i in range(256)], trainer)
    auto_tokenizer = AutoTokenizer()
    auto_tokenizer._tokenizer = tokenizer
    auto_tokenizer._special_token_map = {"unk_token": "<unk>", "pad_token": "<pad>"}
    return auto_tokenizer


@pytest.fixture(scope="session")
def test_tokenizer():
    """Session-scoped tokenizer, created once for the entire test run."""
    return create_test_tokenizer()


@pytest.fixture(scope="session")
def test_model(device):
    """Session-scoped small AutoRegressiveLM model, created once."""
    config = make_tiny_config()
    model = AutoRegressiveLM(config).to(device=device)
    return {"model": model, "device": device, "config": config}


@pytest.fixture
def base_test_env(test_model, test_tokenizer):
    """Function-scoped test environment with isolated temp directory."""
    test_dir = tempfile.mkdtemp()
    config_path = os.path.join(test_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(TINY_CONFIG, f)

    yield {
        "device": test_model["device"],
        "test_dir": str(test_dir),
        "config_path": config_path,
        "transformer_config": test_model["config"],
        "model": test_model["model"],
        "tokenizer": test_tokenizer,
    }

    shutil.rmtree(test_dir)


@pytest.fixture
def random_dataset():
    return RandomTokenDataset(length=None)


@pytest.fixture
def multi_turn_dataset():
    return RandomTokenDataset(length=None, with_loss_mask=True)


@pytest.fixture
def early_stopping_dataset():
    return RandomTokenDataset(length=10, stop_after=5)
