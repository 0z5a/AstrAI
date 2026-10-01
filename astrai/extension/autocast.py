"""FP8 autocast: everything the region-scoped fp8 dispatch needs.

Region management: ``fp8_autocast`` — the torch.autocast-style context
(reentrant, nestable, thread-local via a ``contextvars`` ContextVar) —
plus the ``FP8Recipe`` knobs, the ``fp8_linear_enable`` out-of-region
global switch, the lazy ``aten::linear`` CUDA/AutogradCUDA override that
routes bf16 linears through ``gemm.fp8_linear``, and the checkpoint
bridge (``fp8_state_dict`` / ``fp8_load_state_dict`` / ``fp8_reset``) to
the C++ delayed-scaling rings.

Slot addressing: the module-identity table the routed linear consults on
every call and the region repairs on entry — ``assign_slots`` /
``refresh_slots`` / ``fp8_slot_of`` / ``SlotTable``.

The layering against the rest of the extension mirrors ``kernel/gemm``
vs ``plan``: stateless kernel adapters one layer down, policy and
dispatch here. ``quantize.py`` keeps the int8 inference strategies and
re-exports this module's public names so existing import sites keep
working.

Usage::

    from astrai.extension.autocast import fp8_autocast
    with fp8_autocast(enabled=True, fp8_format="hybrid"):
        logits = model(input_ids)
    loss.backward()  # fp8 backward runs anywhere; fwd captured state on the node

The context mirrors ``torch.autocast``: the active
``(enabled, recipe, fp8_format)`` triple is thread-local, the fp8 path
targets *training* (x/g quantized fresh every call, the weight cast reused
until the weight's version counter moves), and the ``aten::linear``
override installs **lazily** on the first activation — importing this
module never touches the dispatcher. The original per-section docstrings
below keep the full detail.
"""

import functools
import threading
import weakref
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import torch
from torch.library import Library

from astrai.extension.loader import get_module, is_available

__all__ = [
    "FP8Recipe",
    "Fp8Slot",
    "SlotTable",
    "assign_slots",
    "clear_slots",
    "fp8_autocast",
    "fp8_format_pair",
    "fp8_linear_enable",
    "fp8_linear_enabled",
    "fp8_load_state_dict",
    "fp8_reset",
    "fp8_slot_of",
    "fp8_state_dict",
    "refresh_slots",
    "slot_table",
    "sync_slots",
]

# ===========================================================================
# Slot addressing (module identity for the per-weight C++ state)
# ===========================================================================
# The original module docstring, kept for the detail it carries:

"""
Slot addressing for the fp8 per-module state.

The C++ op (``csrc/gemm/fp8_linear.cu``) keeps per-weight state —
three delayed-scaling rings, the version-keyed weight-cast cache — in a
process-wide registry. Its *original* key is the weight's
``(data_ptr, shape, dtype)``, discovered lazily on first use, and its
checkpoint snapshot binds entries to modules by ``(shape, dtype)`` in
registration order. Both are address-based: two same-shaped linears are told
apart only by the order they happened to run, and a replaced weight (TP
sharding, FSDP, a graph-capture buffer) is a different address, so its history
is silently thrown away.

This module gives every fp8-capable module a *name* instead. It walks
``named_modules()`` once and assigns each ``Linear`` a ``Fp8Slot``: a stable
module path (``layers.3.attention.q_proj``) that survives weight replacement,
plus the parsed role (``module_type``/``tensor_type``) that per-role policy
will need. The path — not the compact ``slot_id`` — is the identity the
snapshot binds on, so a model whose construction order changes cannot
cross-wire state between layers.

Attaching is done from the outside, the same way TP shards modules
(``astrai/parallel/tp.py``): modules get a ``_fp8_slot`` attribute, and
``astrai/model/**`` is not touched. The mapping from a weight tensor to its
slot lands in ``by_weight``, keyed by ``id(weight)`` with a strong reference
held alongside — the identity trick the C++ ``ActivationCast`` anchor uses,
for the same reason: a recycled address must not fake a hit.

Nothing here allocates device state or talks to the extension: it is a pure
name table, so it is safe to import (and to test) without CUDA.
"""

# Module path prefixes that wrappers add. ``torch.compile`` prefixes
# ``_orig_mod.`` (see ``astrai/parallel/executor.py``); DDP wraps in
# ``module.``. Stripping them keeps one model's slots identical whether or not
# it is compiled or wrapped.
_PATH_PREFIXES = ("_orig_mod.", "module.")


def _canonical_path(name: str) -> str:
    """Drop wrapper prefixes so a path is the same before and after wrapping."""
    changed = True
    while changed:
        changed = False
        for prefix in _PATH_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
                break
    return name


_linear_types_cache: Optional[tuple] = None


def _linear_types() -> tuple:
    """Model classes whose forward is an ``aten::linear`` call.

    Imported lazily: the extension package must stay importable on a box with
    no model package (and importing it must not pull the model stack).
    ``LoRALinear`` wraps a base ``Linear`` and re-registers the *same* weight
    parameter, so a slot assigned before injection stays valid after it.
    """
    global _linear_types_cache
    if _linear_types_cache is None:
        from astrai.model.components.linear import Linear

        types: List[type] = [Linear]
        try:
            from astrai.model.components.lora import LoRALinear

            types.append(LoRALinear)
        except ImportError:  # pragma: no cover - model package always present
            pass
        _linear_types_cache = tuple(types)
    return _linear_types_cache


def _linear_weight(module: torch.nn.Module) -> Optional[torch.Tensor]:
    """The 2-D weight of a module that dispatches through ``aten::linear``."""
    if not isinstance(module, _linear_types()):
        return None
    weight = getattr(module, "weight", None)
    if not isinstance(weight, torch.nn.Parameter) or weight.dim() != 2:
        return None
    return weight


@dataclass(frozen=True)
class Fp8Slot:
    """One quantized slot: a module path plus its parsed role.

    ``name`` is the module path (e.g. ``layers.3.attention.q_proj``) and is the
    *stable identity* — the snapshot binds on it. ``slot_id`` is a compact
    index handed to C++ for per-call lookup; it follows ``named_modules()``
    order and carries no meaning of its own.

    ``module_type``/``tensor_type`` mirror TE's ``QuantizerRole``: the kind of
    parent the linear hangs under (``attention``/``mlp``/``""``) and the child
    name (``q_proj``/``up``/``lm_head``). ``pattern`` is the TE-plan-style
    glob (``layers.*.attention.q_proj``) a per-role policy would match on.
    """

    slot_id: int
    name: str
    module_type: str
    tensor_type: str

    @property
    def pattern(self) -> str:
        """Path with numeric indices globbed (``layers.*.mlp.up``)."""
        return ".".join(
            "*" if segment.isdigit() else segment for segment in self.name.split(".")
        )


def _parse_role(name: str) -> tuple:
    """``layers.3.attention.q_proj`` -> ``("attention", "q_proj")``.

    The module type is the last non-numeric segment of the parent path, so MoE
    experts (``...routed_experts.2.up``) report ``routed_experts`` rather than
    the expert index. A top-level projection (``lm_head``) has no parent.
    """
    parent, _, child = name.rpartition(".")
    module_type = ""
    for segment in reversed(parent.split(".")):
        if segment and not segment.isdigit():
            module_type = segment
            break
    return module_type, (child or name)


class SlotTable:
    """Weight-identity -> slot map for one model.

    ``by_weight`` is keyed by ``id(weight)`` rather than by the tensor itself:
    ``Tensor.__eq__`` is elementwise, so a dict keyed on tensors would compare
    values on a hash collision. Identity hashing needs the address to stay
    unique, which is what ``_keepalive`` is for — it holds a strong reference
    to every registered weight, so no address can be recycled into a false hit.
    ``rebind()`` is the release valve: it drops references to weights the model
    no longer owns.
    """

    def __init__(self) -> None:
        self._slots: List[Fp8Slot] = []
        self._by_name: Dict[str, Fp8Slot] = {}
        self._by_slot: Dict[int, Fp8Slot] = {}
        self._by_weight: Dict[int, Fp8Slot] = {}
        self._keepalive: List[torch.Tensor] = []
        # slot id -> (module, weight attribute), weakly held so a dead model
        # does not keep itself alive through the table.
        self._owners: Dict[int, tuple] = {}
        self._model_ref = None

    def __len__(self) -> int:
        return len(self._slots)

    def __iter__(self) -> Iterator[Fp8Slot]:
        return iter(self._slots)

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    # -- construction -------------------------------------------------------

    def assign(self, model: torch.nn.Module) -> "SlotTable":
        """Walk ``model`` and (re)build the table.

        Re-assigning an unchanged model reuses the existing ``slot_id`` for
        each path, so ids stay put across a rebuild; a genuinely new path gets
        the next free id. One weight shared by two modules (tied LM head) is
        one slot, named after the first module that reaches it.
        """
        slots: List[Fp8Slot] = []
        by_name: Dict[str, Fp8Slot] = {}
        by_weight: Dict[int, Fp8Slot] = {}
        keepalive: List[torch.Tensor] = []
        owners: Dict[int, tuple] = {}
        # Ids are handed out monotonically and never reused, so a path that
        # disappears and a new one that appears cannot collide on an id. The
        # floor is the previous table's maximum: a pass only sees the slots it
        # has reached so far, which lags behind ids assigned by an earlier pass.
        next_id = max((slot.slot_id for slot in self._slots), default=-1) + 1

        for raw_name, module in model.named_modules():
            weight = _linear_weight(module)
            if weight is None:
                continue
            name = _canonical_path(raw_name)
            slot = self._by_name.get(name)
            if slot is None:
                module_type, tensor_type = _parse_role(name)
                slot = Fp8Slot(next_id, name, module_type, tensor_type)
            next_id = max(next_id, slot.slot_id + 1)
            slots.append(slot)
            by_name[name] = slot
            # Tied weights: one state, so the second module must not overwrite
            # the first binding with a different slot.
            if id(weight) not in by_weight:
                by_weight[id(weight)] = slot
                keepalive.append(weight)
            owners[slot.slot_id] = (weakref.ref(module), "weight")
            module._fp8_slot = slot.slot_id  # external attach (TP-style)

        self._slots = slots
        self._by_name = by_name
        self._by_slot = {slot.slot_id: slot for slot in slots}
        self._by_weight = by_weight
        self._keepalive = keepalive
        self._owners = owners
        self._model_ref = weakref.ref(model)
        return self

    def rebind(self, model: torch.nn.Module) -> None:
        """Re-point weight identities after parameters were replaced.

        TP sharding, FSDP and graph-capture buffers swap the ``Parameter``
        object, which changes ``id(weight)`` while the module path stays put.
        Re-assigning restores the mapping and releases the stale references.
        """
        self.assign(model)

    def stale(self) -> int:
        """How many slots no longer hold the weight tensor they registered.

        A stale table is not a correctness problem — an unmatched weight falls
        back to the address-keyed path — but the module loses its history, so
        :func:`refresh_slots` repairs it at the next fp8 region entry. Reads one
        attribute per slot: ~10us on a 1B model, once per region.
        """
        count = 0
        for slot_id, (module_ref, attr) in self._owners.items():
            module = module_ref()
            if module is None:
                continue
            if self._by_weight.get(
                id(getattr(module, attr, None))
            ) is not self._by_slot.get(slot_id):
                count += 1
        return count

    def refresh(self) -> bool:
        """Re-assign from the model when a weight was swapped under it.

        True when something had changed. A no-op when the model is gone (the
        table outlives a throwaway model) or nothing moved, which keeps the
        common path free.
        """
        model = self._model_ref() if self._model_ref is not None else None
        if model is None or not self.stale():
            return False
        self.assign(model)
        return True

    def clear(self) -> None:
        """Drop every slot and reference (tests, teardown)."""
        self._slots = []
        self._by_name = {}
        self._by_slot = {}
        self._by_weight = {}
        self._keepalive = []
        self._owners = {}
        self._model_ref = None

    # -- lookup -------------------------------------------------------------

    def by_name(self, name: str) -> Optional[Fp8Slot]:
        return self._by_name.get(_canonical_path(name))

    def by_weight(self, weight: torch.Tensor) -> Optional[Fp8Slot]:
        return self._by_weight.get(id(weight))

    def slot_of(self, weight: torch.Tensor) -> int:
        """Compact id for ``weight``, or ``-1`` when it was never assigned.

        ``-1`` is the C++ side's "no slot" sentinel: an unassigned weight (a
        bare call, a bench, a model that never went through :func:`assign_slots`)
        keeps the original address-keyed behavior.
        """
        slot = self._by_weight.get(id(weight))
        return -1 if slot is None else slot.slot_id

    def name_of(self, weight: torch.Tensor) -> str:
        slot = self._by_weight.get(id(weight))
        return "" if slot is None else slot.name

    def describe(self) -> List[str]:
        """One ``slot_id name`` line per slot (debug/tests)."""
        return [f"{slot.slot_id} {slot.name}" for slot in self._slots]


# The process-wide table the dispatcher consults. One model per process is the
# training assumption; re-assigning a different model replaces the contents.
_slots = SlotTable()


def sync_slots(table: Optional[SlotTable] = None) -> None:
    """Publish ``table`` (the active one by default) to the C++ state.

    C++ needs slot id -> module path to name its snapshot entries; keeping that
    map there means only an ``int64`` crosses the per-call boundary. A no-op on
    a box without the extension — the table itself is a pure name map, so slot
    assignment never requires CUDA.
    """
    if not is_available("gemm"):
        return
    target = _slots if table is None else table
    get_module("gemm").fp8_set_slots(
        [(slot.slot_id, slot.name, slot.pattern) for slot in target]
    )


def slot_table() -> SlotTable:
    """The active table (queried by the ``aten::linear`` override)."""
    return _slots


def assign_slots(model: torch.nn.Module) -> SlotTable:
    """Assign slots for ``model``, install them as the active table, publish."""
    table = _slots.assign(model)
    sync_slots(table)
    return table


def clear_slots() -> None:
    """Forget every slot (tests / reconfiguration)."""
    _slots.clear()
    sync_slots()


def refresh_slots() -> bool:
    """Repair a stale table after a parameter swap; True when it changed.

    Called on fp8 region entry: the C++ rebind path only fires when the same
    slot id arrives with a different weight tensor, which is exactly what a
    stale Python mapping cannot produce.
    """
    if _slots.refresh():
        sync_slots(_slots)
        return True
    return False


def fp8_slot_of(weight: torch.Tensor) -> int:
    """Slot id of a linear weight, ``-1`` when unassigned.

    The seam a future producer (norm/activation epilogue writing fp8 directly)
    uses to find the consuming linear's state without a tensor-identity
    registry of its own.
    """
    return _slots.slot_of(weight)


# ===========================================================================
# Region management: formats, recipes, the autocast context, aten routing
# ===========================================================================

# ---------------------------------------------------------------------------
# FP8: formats, recipes, region config
# ---------------------------------------------------------------------------

# FP8 format vocabulary: the canonical key is the fp8 dtype itself
# (torch.float8_e4m3fn / torch.float8_e5m2 — what the quantize binding
# dispatches and validates on). The only policy-level notion beyond a
# dtype is the hybrid per-direction pair, carried as a plain (fwd, bwd)
# tuple; dtype validation stays in the C++ binding.


def _is_fp8(dtype: torch.dtype) -> bool:
    """A pre-quantized weight takes the GEMM directly (no re-quantize)."""
    return dtype in (torch.float8_e4m3fn, torch.float8_e5m2)


def fp8_format_pair(fmt: str | torch.dtype) -> tuple[torch.dtype, torch.dtype]:
    """Format spec -> (fwd, bwd) fp8 dtype pair.

    A dtype is symmetric; ``'hybrid'`` is E4M3 forward / E5M2 backward
    (the training default).
    """
    if fmt == "hybrid":
        return (torch.float8_e4m3fn, torch.float8_e5m2)
    return (fmt, fmt)


@dataclass
class FP8Recipe:
    """Scale-from-amax policy knobs: ``scale = (amax / finfo(fmt).max) / 2^margin``.

    ``dynamic=False`` (default) is TE-style delayed scaling: max over the
    amax history window (amax from *previous* steps; the window trades
    responsiveness against stability). ``dynamic=True`` is current-amax
    scaling (torchao DYNAMIC): measure, then quantize — no history, at an
    extra pass. The formula itself lives in the C++ op (the single home;
    the seed and in-kernel fold both apply it there).
    """

    history_len: int = 16
    margin: int = 0
    dynamic: bool = False


@dataclass
class _ActiveConfig:
    """The immutable (enabled, recipe, format-pair) triple of one open
    region; ``fp8_format`` is the (fwd, bwd) fp8 dtype pair."""

    enabled: bool
    recipe: FP8Recipe
    fp8_format: tuple[torch.dtype, torch.dtype]


# Thread-local active configuration (torch's autocast TLS analog): set by
# fp8_autocast on __enter__, absent outside any region. Autograd engine
# threads run backwards with their own empty context — fine, since backward
# only reads state captured on ctx at forward time.
_active_config: ContextVar[_ActiveConfig | None] = ContextVar(
    "astrai_fp8_active_config", default=None
)

# Persistent out-of-region defaults. Everything else the old Python state
# machine owned — the per-weight meta registry (delayed-scaling rings, the
# version-keyed weight-cast cache, the checkpoint pending queue, the
# generation counter invalidating that cache) — lives in the C++ op's
# translation unit (``csrc/gemm/fp8_linear.cu``). A meta is addressed
# by its module's *slot* when the dispatcher knows one (see the slot
# stable module path, so a replaced weight parameter keeps its history) and by
# ``(data_ptr, shape, dtype)`` otherwise; the snapshot binds slotted entries by
# name and falls back to registration order for the rest.
_default_enabled = False
_default_recipe = FP8Recipe()
_default_format = (torch.float8_e4m3fn, torch.float8_e5m2)


def fp8_state_dict() -> dict:
    """Checkpoint snapshot of the delayed-scaling rings (A1): save it beside
    the optimizer state. Empty under dynamic scaling (no rings exist) and on
    a box without the extension (bf16 training never allocated any)."""
    if not is_available("gemm"):
        return {"version": 1, "entries": []}
    return get_module("gemm").fp8_state_dict()


def fp8_load_state_dict(sd: dict) -> None:
    """Restore a :func:`fp8_state_dict` snapshot. Unmatched entries (a model
    shape/topology change across the checkpoint) re-seed on next use — the
    same resume transient a ring-less restart would pay on every step."""
    if not sd.get("entries"):
        return
    get_module("gemm").fp8_load_state_dict(sd)


def fp8_reset() -> None:
    """Drop the C++-side registry (tests / reconfiguration)."""
    if is_available("gemm"):
        get_module("gemm").fp8_reset()


def _active() -> _ActiveConfig | None:
    """The active config when fp8 dispatch is on, else ``None`` (fast guard).

    A region config wins (honoring nested ``enabled=False`` regions); with no
    region open this falls back to the persistent global switch
    (``fp8_linear_enable``), so that flag still routes aten::linear to fp8.
    """
    cfg = _active_config.get()
    if cfg is not None:
        return cfg if cfg.enabled else None
    if _default_enabled:
        return _ActiveConfig(True, _default_recipe, _default_format)
    return None


def _current_config() -> _ActiveConfig:
    """Like ``_active()`` but always returns a config (disabled regions and
    out-of-region direct calls resolve to the global defaults)."""
    cfg = _active_config.get()
    if cfg is not None:
        return cfg
    return _ActiveConfig(_default_enabled, _default_recipe, _default_format)


class fp8_autocast:
    """Autocast-style context: fp8 linear dispatch on this thread.

    Mirrors ``torch.autocast`` — a class-based, reentrant, nestable context
    over thread-local state::

        with fp8_autocast(enabled=True, fp8_format="hybrid"):
            logits = model(input_ids)   # aten::linear -> fp8 path
        loss.backward()  # fp8 backward; state was captured at forward time

    Nesting follows torch: each ``__enter__`` pushes the new active config, each
    ``__exit__`` restores the previous one, and a nested ``enabled=False`` region
    simply disables dispatch inside it. The instance doubles as a decorator.
    """

    def __init__(
        self,
        enabled: bool = True,
        update_interval: int = 16,
        recipe: FP8Recipe | None = None,
        fp8_format: str | torch.dtype = "hybrid",
        margin: int = 0,
    ):
        if recipe is None:
            recipe = FP8Recipe(history_len=update_interval, margin=margin)
        self._config = _ActiveConfig(bool(enabled), recipe, fp8_format_pair(fp8_format))
        self._tokens: list[Token] = []

    def __enter__(self) -> "fp8_autocast":
        if self._config.enabled:
            _install_linear_override()
            # A parameter swapped since the slots were assigned (TP sharding,
            # FSDP, a graph buffer) would silently fall back to the
            # address-keyed path and lose its amax history. One attribute read
            # per slot repairs it here, once per region.
            refresh_slots()
        self._tokens.append(_active_config.set(self._config))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        token = self._tokens.pop()
        _active_config.reset(token)
        if self._config.enabled and is_available("gemm"):
            get_module("gemm").fp8_clear_act_cache()
        return False

    def __call__(self, func):
        @functools.wraps(func)
        def decorate(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorate


def fp8_linear_enable(enabled: bool = True) -> None:
    """Toggle fp8 dispatch for aten::linear globally (the out-of-region default;
    ``fp8_autocast`` regions override it thread-locally)."""
    global _default_enabled
    if enabled:
        _install_linear_override()
    _default_enabled = bool(enabled)


def fp8_linear_enabled() -> bool:
    """Whether fp8 dispatch is active right now (region config or global)."""
    return _active() is not None


def _fp8_supported(x: torch.Tensor, w: torch.Tensor) -> bool:
    """Shape guard for the fp8 path. Unlike a strict 16-alignment requirement,
    the kernels handle unaligned M/N via boundary checks (slower but correct) —
    so no whole-call bf16 fallback for small decode batches. Only the K-dimension
    contraction must match and the weight must be 2D."""
    return x.dim() >= 2 and w.dim() == 2 and x.size(-1) == w.size(1)


def _linear_cuda_impl(x: torch.Tensor, w: torch.Tensor, bias=None):
    cfg = _active()
    if (
        cfg is not None
        and x.dtype is torch.bfloat16
        and w.dtype is torch.bfloat16
        and _fp8_supported(x, w)
    ):
        # One Python->C++ crossing per linear: quantize, the ring
        # fold/advance, the weight-cast cache and all three GEMMs run inside
        # gemm.fp8_linear. The caller's grad mode gates the delayed-scaling
        # bookkeeping, and it must be read HERE: inside a Function forward
        # grad is always disabled, so the mode would be invisible one frame
        # deeper. A no-grad linear (checkpointing recompute, inference) reads
        # the rings without folding or advancing them. This host bool is also
        # the seam a CUDA-graph device flag replaces (TE's
        # skip_fp8_weight_update).
        #
        # ``slot`` names the module the weight belongs to (slot addressing
        # the C++ state attaches to the module rather than to whatever tensor
        # holds its weight; -1 (no slot table, a bare call, the benches) keeps
        # the original address-keyed lookup.
        recipe = cfg.recipe
        return get_module("gemm").fp8_linear(
            x,
            w,
            bias,
            torch.is_grad_enabled(),  # update_rings
            bool(bias is not None and bias.requires_grad),
            recipe.dynamic,
            recipe.history_len,
            recipe.margin,
            cfg.fp8_format[0],
            cfg.fp8_format[1],
            fp8_slot_of(w),
        )
    return torch.ops.aten.linear.default.redispatch(
        torch._C.DispatchKeySet(torch._C.DispatchKey.CompositeImplicitAutograd),
        x,
        w,
        bias,
    )


_linear_libs: list[Library] | None = None
_install_lock = threading.Lock()


def _install_linear_override() -> None:
    """Register the fp8 aten::linear impls once, on first activation.

    Importing this module must stay dispatcher-neutral: the override routes
    every CUDA ``aten::linear`` call through ``_linear_cuda_impl``'s guard,
    so it is installed exactly when fp8 dispatch is first switched on (an
    autocast enter or the global enable) — never at import. The ``Library``
    handles live in a module global for the process lifetime (dropping them
    would unregister the impls).
    """
    global _linear_libs
    if _linear_libs is None:
        with _install_lock:
            if _linear_libs is None:
                lib = Library("aten", "IMPL", "CUDA")
                lib.impl("linear", _linear_cuda_impl)
                # Also replace torch's generated linear autograd formula
                # (which would call aten::linear_backward after the
                # fp8_autocast region exits). The fp8 backward is owned by
                # the C++ node with state captured at forward time, so
                # loss.backward() works wherever it is called; the CUDA
                # registration still covers inference_mode.
                lib_autograd = Library("aten", "IMPL", "AutogradCUDA")
                lib_autograd.impl("linear", _linear_cuda_impl)
                _linear_libs = [lib, lib_autograd]
