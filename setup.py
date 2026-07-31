import os
import sys
import warnings
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext

sys.path.insert(0, str(Path(__file__).parent))
os.makedirs("astrai/extension/lib", exist_ok=True)


def _should_build():
    force = os.environ.get("CSRC_KERNELS", "").strip().lower()
    if force == "true":
        return True
    if force == "false":
        return False
    try:
        import shutil

        import torch

        return shutil.which("nvcc") is not None and torch.cuda.is_available()
    except Exception:
        return False


ext_modules = []
cmdclass = {}

if _should_build():
    import torch
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    from csrc.build import REGISTRY, cuda_toolkit_version

    # Preflight: warn if nvcc major version != torch's bundled CUDA major version.
    # A mismatch (e.g. nvcc 13.0 + cu128 torch) causes cryptic ABI/header errors.
    nvcc_ver = cuda_toolkit_version()
    torch_cuda = torch.version.cuda
    if nvcc_ver is not None and torch_cuda is not None:
        torch_major = int(torch_cuda.split(".")[0])
        if nvcc_ver[0] != torch_major:
            warnings.warn(
                f"CUDA version mismatch: nvcc is {nvcc_ver[0]}.{nvcc_ver[1]} "
                f"but torch was built with CUDA {torch_cuda}. "
                f"This may cause compilation errors. "
                f"Install a matching torch wheel: "
                f"pip install torch --index-url "
                f"https://download.pytorch.org/whl/cu{nvcc_ver[0]}{nvcc_ver[1]}",
                stacklevel=2,
            )

    _torch_lib = torch.utils.cpp_extension.library_paths()[0]

    for name, info in REGISTRY.items():
        ext_modules.append(
            CUDAExtension(
                f"astrai.extension.lib.{name}",
                info["sources"],
                extra_compile_args={
                    "cxx": info["cxx_flags"],
                    "nvcc": info["nvcc_flags"],
                },
                extra_link_args=[f"-Wl,-rpath,{_torch_lib}"],
            )
        )
    cmdclass["build_ext"] = BuildExtension

if not cmdclass:

    class _NullBuildExt(_build_ext):
        def build_extensions(self):
            pass

    cmdclass["build_ext"] = _NullBuildExt

setup(ext_modules=ext_modules, cmdclass=cmdclass)
