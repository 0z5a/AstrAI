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

    # Parallel build — each extension is an independent ninja project, so we
    # can compile them concurrently.  BuildExtension compiles them serially by
    # default; this subclass dispatches each extension to a subprocess.
    # Set BUILD_PARALLEL=N to override (default: min(n_exts, 4)).
    _single_ext = os.environ.get("ASTRAI_BUILD_SINGLE_EXT", "")

    class ParallelBuildExtension(BuildExtension):
        def build_extensions(self):
            if _single_ext:
                self.extensions = [e for e in self.extensions if e.name == _single_ext]
                if not self.extensions:
                    return
                super().build_extensions()
                return

            n = len(self.extensions)
            max_workers = int(os.environ.get("BUILD_PARALLEL", 8))
            if max_workers <= 1 or n <= 1:
                super().build_extensions()
                return

            # Each subprocess gets its own build-temp / build-lib so the
            # ninja files (build.ninja, .ninja_log) never race.  The built
            # .so files are then collected into the parent's build_lib so the
            # normal setuptools copy steps (inplace / editable wheel) work.
            names = [e.name for e in self.extensions]
            env = {**os.environ, "BUILD_PARALLEL": "1"}
            base = os.path.join("build", "parallel")
            os.makedirs(base, exist_ok=True)
            procs = {}
            for i in range(0, len(names), max_workers):
                batch = names[i : i + max_workers]
                for name in batch:
                    e = {**env, "ASTRAI_BUILD_SINGLE_EXT": name}
                    tag = name.replace(".", "_")
                    subdir = os.path.join(base, tag)
                    cmd = [
                        sys.executable,
                        __file__,
                        "build_ext",
                        "--build-temp",
                        os.path.join(subdir, "temp"),
                        "--build-lib",
                        os.path.join(subdir, "lib"),
                    ]
                    procs[name] = subprocess.Popen(
                        cmd, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
                    )
                for name in batch:
                    out, _ = procs[name].communicate()
                    if procs[name].returncode != 0:
                        sys.stdout.write(out.decode())
                        raise RuntimeError(
                            f"parallel build failed for {name} "
                            f"(exit {procs[name].returncode})"
                        )
                    self._collect_extensions(
                        os.path.join(base, name.replace(".", "_"), "lib")
                    )

        def _collect_extensions(self, sub_lib):
            src = os.path.join(sub_lib, "astrai", "extension", "lib")
            if not os.path.isdir(src):
                return
            dst = os.path.join(self.build_lib, "astrai", "extension", "lib")
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(src):
                if f.endswith(".so"):
                    shutil.copy2(os.path.join(src, f), os.path.join(dst, f))

    cmdclass["build_ext"] = ParallelBuildExtension

if not cmdclass:

    class _NullBuildExt(_build_ext):
        def build_extensions(self):
            pass

    cmdclass["build_ext"] = _NullBuildExt

setup(ext_modules=ext_modules, cmdclass=cmdclass)
