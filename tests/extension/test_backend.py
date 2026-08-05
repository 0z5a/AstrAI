"""Backend selection and context-manager switching tests.

These tests do not require CUDA — they only check that the active
backend is correctly set and restored.
"""

import pytest

from astrai.extension import (
    ATTN_BACKEND,
    AttentionBackendFactory,
    CudaBackend,
    TorchNativeBackend,
    attn_backend,
    get_backend,
)


def test_default_backend_is_torch_native():
    backend = get_backend()
    assert isinstance(backend, TorchNativeBackend)


def test_attn_backend_context_with_enum():
    with attn_backend(ATTN_BACKEND.CUDA):
        assert isinstance(get_backend(), CudaBackend)
    assert isinstance(get_backend(), TorchNativeBackend)


def test_attn_backend_context_with_registered_name():
    with attn_backend("cuda"):
        assert isinstance(get_backend(), CudaBackend)
    assert isinstance(get_backend(), TorchNativeBackend)


def test_attention_backend_factory_lists_builtin_backends():
    assert AttentionBackendFactory.list_registered() == [
        "cuda",
        "flash",
        "torch_native",
    ]


def test_attn_backend_rejects_unknown_registered_name():
    with pytest.raises(ValueError, match="Unknown component: 'unknown'"):
        with attn_backend("unknown"):
            pass


def test_attn_backend_context_with_class():
    with attn_backend(CudaBackend):
        assert isinstance(get_backend(), CudaBackend)
    assert isinstance(get_backend(), TorchNativeBackend)


def test_attn_backend_context_with_instance():
    custom = CudaBackend()
    with attn_backend(custom):
        assert get_backend() is custom
    assert isinstance(get_backend(), TorchNativeBackend)


def test_cudabackend_is_context_manager():
    with CudaBackend():
        assert isinstance(get_backend(), CudaBackend)
    assert isinstance(get_backend(), TorchNativeBackend)
