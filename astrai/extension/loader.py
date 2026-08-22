"""Dynamic discovery and loading of compiled CUDA kernel modules.

Each kernel is built by the CMake build in ``csrc/CMakeLists.txt`` into a
``.so`` placed in ``astrai/extension/lib/`` — the module name equals the
``.so`` name equals the pybind name (e.g. ``attn_decode``, defined via
``TORCH_EXTENSION_NAME``). ``KERNEL_NAMES`` is discovered automatically from
the ``.so`` files present, so adding a kernel to the CMake ``KERNELS``
registry needs no change here. On import we try to load each one; kernels
that failed to build (or are running on a CPU-only machine) are marked
unavailable so the wrapper functions can fall back to ``torch`` SDPA.
"""

import glob
import importlib
import logging
import os

logger = logging.getLogger(__name__)

_LIB_DIR = os.path.join(os.path.dirname(__file__), "lib")


def _discover_kernel_names() -> list[str]:
    """Return the module names of the compiled kernel ``.so`` files in lib/."""
    names: list[str] = []
    for path in glob.glob(os.path.join(_LIB_DIR, "*.so")):
        # strip the "<soabi>.so" suffix, e.g. attn_decode.cpython-312-...so
        names.append(os.path.basename(path).split(".", 1)[0])
    return sorted(names)


KERNEL_NAMES = _discover_kernel_names()

_available: dict[str, bool] = {}
_modules: dict[str, object] = {}

for _name in KERNEL_NAMES:
    try:
        _mod = importlib.import_module(f".lib.{_name}", package=__package__)
        _available[_name] = True
        _modules[_name] = _mod
    except ImportError:
        _available[_name] = False
        _modules[_name] = None


def is_available(name: str) -> bool:
    """Return ``True`` if the compiled kernel ``name`` was loaded."""
    return _available.get(name, False)


def get_module(name: str) -> object:
    """Return the loaded kernel module for ``name``, or ``None`` if unavailable."""
    return _modules.get(name)
