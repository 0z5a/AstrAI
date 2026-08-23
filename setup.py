import os
import shutil
import subprocess
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
        import torch

        return shutil.which("nvcc") is not None and torch.cuda.is_available()
    except Exception:
        return False


def _torch_prefix():
    """Return the torch install dir (site-packages/torch) used for headers/libs."""
    try:
        import torch

        return str(Path(torch.__file__).parent.resolve())
    except Exception:
        return os.environ.get("TORCH_HOME", "")


def _python_include():
    import sysconfig

    return sysconfig.get_path("include")


def _python_soabi():
    import sysconfig

    ext = sysconfig.get_config_var("EXT_SUFFIX").lstrip(".")
    return ext[: -len(".so")]


class _CMakeBuildExt(_build_ext):
    def run(self):
        src = Path(__file__).parent
        build_dir = src / "build" / "cmake"
        torch_home = _torch_prefix()
        if not torch_home:
            raise RuntimeError(
                "torch not found; cannot build kernels. "
                "Activate the environment or set TORCH_HOME."
            )

        nvcc_ver = _cuda_toolkit_version()
        torch_cuda = _torch_cuda_version()
        if (
            nvcc_ver is not None
            and torch_cuda is not None
            and nvcc_ver[0] != int(torch_cuda.split(".")[0])
        ):
            warnings.warn(
                f"CUDA version mismatch: nvcc is {nvcc_ver[0]}.{nvcc_ver[1]} "
                f"but torch was built with CUDA {torch_cuda}. "
                f"Install a matching torch wheel.",
                stacklevel=2,
            )

        cmake = shutil.which("cmake")
        if cmake is None:
            raise RuntimeError("cmake not found on PATH; install it to build kernels")

        parallel = os.environ.get("BUILD_PARALLEL", "4")
        cfg = [
            cmake,
            "-S",
            str(src / "csrc"),
            "-B",
            str(build_dir),
            f"-DTORCH_HOME={torch_home}",
            f"-DPYTHON_INCLUDE_DIR={_python_include()}",
            f"-DPY_SOABI={_python_soabi()}",
        ]
        arch = os.environ.get("ASTRAI_CUDA_ARCH")
        if not arch:
            arch = _detect_cuda_arch()
        if arch:
            cfg.append(f"-DASTRAI_CUDA_ARCH={arch}")
        subprocess.run(cfg, check=True)
        subprocess.run([cmake, "--build", str(build_dir), "-j", parallel], check=True)


def _cuda_toolkit_version():
    import shutil
    import subprocess

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return None
    try:
        out = subprocess.check_output(
            [nvcc, "--version"], stderr=subprocess.STDOUT, text=True
        )
        for line in out.splitlines():
            if "release" in line:
                ver = line.split("release")[1].split(",")[0].strip()
                return tuple(int(x) for x in ver.split("."))
    except Exception:
        pass
    return None


def _detect_cuda_arch():
    """Detect real GPU compute capability via torch (nvidia-smi may be spoofed).

    Returns something like ``"89"`` or ``"103"``, or ``None`` if unavailable.
    """
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            return f"{major}{minor}"
    except Exception:
        pass
    return None


def _torch_cuda_version():
    try:
        import torch

        return torch.version.cuda
    except Exception:
        return None


class _NullBuildExt(_build_ext):
    def build_extensions(self):
        pass


cmdclass = {}

if _should_build():
    cmdclass["build_ext"] = _CMakeBuildExt
else:
    cmdclass["build_ext"] = _NullBuildExt

setup(ext_modules=[], cmdclass=cmdclass)
