// The fp8 training linear — forward *and* backward — in C++.
//
// This is the composition the Python policy layer used to perform: quantize
// x/w with the delayed-scaling rings, run the pre-quantized GEMMs, keep the
// transposed operands for the backward, and update the rings. It lives in the
// gemm module because that module owns the GEMM dispatch state (plan table,
// planner mode, staging switches) — a second copy of that state in another
// .so would let `set_planner` configure one and the training path launch
// through the other. The quantize chain comes in through quantize/launch.cuh,
// the GEMM through gemm/api.h: both are the same code the standalone bindings
// run.
//
// Why C++: the composed path is ~9 kernel launches and (measured) ~200us of
// host time per linear fwd+bwd above the bf16 path, of which the custom
// autograd Function machinery — Python-level apply, attribute chasing,
// save_for_backward unpacking, and 5-7 Python->C++ crossings — is the
// majority. A ``torch::autograd::Function`` runs both directions inside the
// engine's C++ call, so only one entry call from the dispatcher stays in
// Python.
//
// What is deliberately *not* here: the autocast region, the enable switch and
// the recipe/format *policy* (astrai/extension/quantize.py) — those are
// configuration, read once per region, and passing four scalars per call is
// cheaper than a second source of truth. The rings, the weight cast cache and
// their checkpoint snapshot are here, because they are per-call state.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Optional.h>
#include <torch/extension.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "gemm/api.h"
#include "quantize/launch.cuh"

namespace astrai {
namespace fp8 {

using torch::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::tensor_list;

namespace {

// ---------------------------------------------------------------------------
// Recipe constants
// ---------------------------------------------------------------------------

// The fp8 format range: one source for the host-side seed formula (the kernel
// publishes with the same number, passed through QuantParams).
inline float fp8_max_of(at::ScalarType fmt) {
    TORCH_CHECK(fmt == at::kFloat8_e4m3fn || fmt == at::kFloat8_e5m2,
                "fp8 linear: format must be float8_e4m3fn or float8_e5m2");
    return fmt == at::kFloat8_e4m3fn ? 448.0f : 57344.0f;
}

inline bool is_fp8(at::ScalarType dt) {
    return dt == at::kFloat8_e4m3fn || dt == at::kFloat8_e5m2;
}

// scale = (peak / fp8_max) / 2^margin, clamped, with peak the max over the
// amax window — the host mirror of the kernel's publish, used only where the
// host owns the publication (seed and the dynamic recipe).
inline Tensor scale_from_amax(const Tensor& window_or_amax, at::ScalarType fmt,
                              int64_t margin) {
    const Tensor peak = window_or_amax.max();
    const double pow2 = std::pow(2.0, static_cast<double>(margin));
    return (peak / fp8_max_of(fmt) / pow2).clamp_min(1e-12);
}

// amax in the raw domain, detached: the rings fill their history with it, and
// an in-place op on a non-grad buffer must not drag a grad graph in.
inline Tensor amax_of(const Tensor& t) {
    return t.detach().abs().amax().to(at::kFloat).clamp_min(1e-12);
}

// ---------------------------------------------------------------------------
// Delayed-scaling rings
// ---------------------------------------------------------------------------

// One operand's ring: [hist n | scale | scale_recip | amax | done | scratch].
// The layout offsets are owned by quant::ring_view; this struct keeps the
// buffer plus the views the policy reads, so no per-call view is created.
struct ScaleRing {
    Tensor state;
    Tensor hist;
    Tensor scale;
    Tensor scale_recip;
    int64_t idx = 0;
    bool initialized = false;

    ScaleRing() = default;

    ScaleRing(const torch::TensorOptions& opts, int64_t history_len) {
        const int64_t n = history_len;
        TORCH_CHECK(n > 0, "fp8 linear: history_len must be positive");
        state = torch::zeros({n + 4 + quant::kFoldSlots}, opts);
        hist = state.narrow(0, 0, n);
        scale = state.narrow(0, n, 1);
        scale_recip = state.narrow(0, n + 1, 1);
    }

    void advance() { idx = (idx + 1) % hist.numel(); }

    // Mirrors the scale's reciprocal into its slot — seed / restore only; the
    // fold republishes both slots in-kernel every step.
    void publish_recip() { at::reciprocal_out(scale_recip, scale); }

    void seed(const Tensor& t, at::ScalarType fmt, int64_t margin) {
        const Tensor amax = amax_of(t);
        hist.fill_(amax);
        scale.copy_(scale_from_amax(hist, fmt, margin));
        publish_recip();
        initialized = true;
    }
};

// A tensor's autograd version counter: the cast cache's validity key (every
// in-place update bumps it, optimizer steps included).
inline int64_t version_of(const Tensor& t) {
    return t.unsafeGetTensorImpl()->version_counter().current_version();
}

// Version-keyed weight cast cache: the fp8 pair together with the scale they
// were cast with, so the entry is self-consistent — the GEMM dequant reads
// the very scale the values were quantized with, whatever the ring has
// published since. A hit also skips the in-kernel amax fold and the ring
// advance: an unchanged weight has an unchanged amax, so the fold would
// rewrite the history window with the same value. The ring therefore tracks
// optimizer steps — the only steps that can move a weight's amax.
struct WeightCast {
    int64_t version = -1;
    int64_t generation = -1;
    at::ScalarType fmt_a = at::kFloat8_e4m3fn;
    at::ScalarType fmt_b = at::kFloat8_e4m3fn;
    Tensor w8, w8T, sw;

    bool valid(const Tensor& w, at::ScalarType fa, at::ScalarType fb,
               int64_t gen) const {
        return w8.defined() && version == version_of(w) && generation == gen &&
               fmt_a == fa && fmt_b == fb;
    }

    void fill(const Tensor& w, at::ScalarType fa, at::ScalarType fb,
              int64_t gen, Tensor a, Tensor b, Tensor s) {
        version = version_of(w);
        generation = gen;
        fmt_a = fa;
        fmt_b = fb;
        w8 = std::move(a);
        w8T = std::move(b);
        sw = std::move(s);
    }
};

struct Fp8Meta {
    ScaleRing w, x, g;
    WeightCast cast;
    std::vector<int64_t> shape;  // the owning weight's (registry + snapshot key)
    at::ScalarType dtype = at::kBFloat16;
    int64_t history_len = 0;
    int64_t margin = 0;
    bool dynamic = false;
};

// ---------------------------------------------------------------------------
// Process-wide state: the meta registry, the generation counter and the
// checkpoint snapshot. The registry key mirrors the Python one — (data_ptr,
// shape, dtype) — with the same contract (a live meta assumes no allocator
// reuse under it; tests reset between cases).
// ---------------------------------------------------------------------------

struct State {
    std::mutex mu;
    std::unordered_map<std::string, std::shared_ptr<Fp8Meta>> by_key;
    std::vector<std::shared_ptr<Fp8Meta>> order;  // registration order (A1)
    // Snapshots queued by load_state_dict, consumed in registration order as
    // the metas they belong to are restored (a resume restores before any
    // forward runs).
    std::vector<py::dict> pending;
    int64_t generation = 0;  // bumped by restore / reset / recipe rebuild
    // Test/bench observability.
    std::atomic<int64_t> n_quantize{0};
    std::atomic<int64_t> n_gemm{0};
    std::atomic<int64_t> n_cast_hit{0};
    std::atomic<int64_t> n_cast_miss{0};
};

State& state() {
    static State s;
    return s;
}

std::string meta_key(const Tensor& w) {
    std::string key = std::to_string(reinterpret_cast<uintptr_t>(w.data_ptr()));
    key += '|';
    for (const auto s : w.sizes()) {
        key += std::to_string(s);
        key += ',';
    }
    key += '|';
    key += std::to_string(static_cast<int>(w.scalar_type()));
    return key;
}

// The snapshot's dtype field uses the Python ``str(torch.dtype)`` spelling —
// the format the checkpoint bridge established before this op existed, so a
// snapshot written by either side restores on the other.
std::string torch_dtype_str(at::ScalarType t) {
    switch (t) {
        case at::kBFloat16: return "torch.bfloat16";
        case at::kHalf: return "torch.float16";
        case at::kFloat: return "torch.float32";
        case at::kDouble: return "torch.float64";
        case at::kChar: return "torch.int8";
        case at::kByte: return "torch.uint8";
        case at::kShort: return "torch.int16";
        case at::kInt: return "torch.int32";
        case at::kLong: return "torch.int64";
        case at::kBool: return "torch.bool";
        case at::kFloat8_e4m3fn: return "torch.float8_e4m3fn";
        case at::kFloat8_e5m2: return "torch.float8_e5m2";
        default: return c10::toString(t);
    }
}

bool dtype_str_matches(const std::string& s, at::ScalarType t) {
    return s == torch_dtype_str(t) || s == std::string(c10::toString(t));
}

// Restore one ring from a snapshot entry. False geometry means the buffer
// changed since the save (a recipe change across the checkpoint boundary):
// the ring stays fresh and re-seeds on next use.
void restore_ring(ScaleRing& ring, const py::object& sd, bool& geometry_ok) {
    if (sd.is_none()) return;
    py::dict d = sd.cast<py::dict>();
    Tensor saved = d["state"].cast<Tensor>();
    if (saved.numel() != ring.state.numel()) {
        geometry_ok = false;
        return;
    }
    ring.state.copy_(saved.to(ring.state.device()));
    ring.publish_recip();  // snapshots older than the recip slot restore 0
    ring.idx = py::cast<int64_t>(d["idx"]);
    ring.initialized = py::cast<bool>(d["initialized"]);
}

// Consume queued restore entries in registration order, matching by
// (shape, dtype) — data_ptr is meaningless across processes, and re-binding
// relies on the same registration-order contract TE documents for amax
// reduction.
void restore_pending_locked(std::shared_ptr<Fp8Meta>& meta) {
    State& st = state();
    for (size_t i = 0; i < st.pending.size(); ++i) {
        const py::dict entry = st.pending[i];
        bool match = py::len(entry["shape"]) == meta->shape.size();
        for (size_t d = 0; match && d < meta->shape.size(); ++d)
            match = py::cast<int64_t>(entry["shape"][py::int_(d)]) ==
                    meta->shape[d];
        match = match && dtype_str_matches(
                             py::cast<std::string>(entry["dtype"]), meta->dtype);
        if (!match) continue;
        bool geometry_ok = true;
        restore_ring(meta->w, entry["w"], geometry_ok);
        restore_ring(meta->x, entry["x"], geometry_ok);
        restore_ring(meta->g, entry["g"], geometry_ok);
        st.pending.erase(st.pending.begin() + static_cast<long>(i));
        return;
    }
}

// Look up (or create) the rings for a weight. A recipe change under the same
// weight rebuilds them: their geometry (history_len) and fold constants
// (margin) are stale, and TE clears its fp8 workspaces on a recipe change for
// the same reason. The fresh rings re-seed on the next forward (one step of
// transient), and the generation bump invalidates the cast cache.
std::shared_ptr<Fp8Meta> get_meta(const Tensor& w, int64_t history_len,
                                  int64_t margin, bool dynamic) {
    const std::string key = meta_key(w);
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    auto it = st.by_key.find(key);
    if (it != st.by_key.end()) {
        auto meta = it->second;
        if (meta->history_len == history_len && meta->margin == margin &&
            meta->dynamic == dynamic) {
            return meta;
        }
        st.by_key.erase(it);
        st.order.erase(std::remove(st.order.begin(), st.order.end(), meta),
                       st.order.end());
        st.generation += 1;
    }
    const auto opts = torch::TensorOptions().dtype(at::kFloat).device(
        w.device());
    auto meta = std::make_shared<Fp8Meta>();
    meta->w = ScaleRing(opts, history_len);
    meta->x = ScaleRing(opts, history_len);
    meta->g = ScaleRing(opts, history_len);
    meta->shape.assign(w.sizes().begin(), w.sizes().end());
    meta->dtype = w.scalar_type();
    meta->history_len = history_len;
    meta->margin = margin;
    meta->dynamic = dynamic;
    st.by_key.emplace(key, meta);
    st.order.push_back(meta);
    restore_pending_locked(meta);
    return meta;
}

// ---------------------------------------------------------------------------
// Composed forward / backward
// ---------------------------------------------------------------------------

struct Fp8FwdOut {
    Tensor out;
    Tensor sx, sw;    // dequant-scale snapshots (immutable through the backward)
    Tensor x8T, w8T;  // K-contiguous transposed casts (undefined when absent)
};

struct Fp8Cfg {
    bool dynamic = false;
    int64_t history_len = 16;
    int64_t margin = 0;
    at::ScalarType fmt_a = at::kFloat8_e4m3fn;
    at::ScalarType fmt_b = at::kFloat8_e5m2;
};

// One quantize pass through the shared launcher, with the counters kept for
// tests/benches.
quant::QuantizeOutputs run_quant(const Tensor& t, const Tensor& scale,
                                 quant::QuantLayout layout,
                                 at::ScalarType fmt_a,
                                 c10::optional<at::ScalarType> fmt_b,
                                 const c10::optional<Tensor>& ring, int64_t idx,
                                 const Fp8Cfg& cfg) {
    state().n_quantize.fetch_add(1, std::memory_order_relaxed);
    return quant::run_quantize(t, scale, layout, fmt_a, fmt_b, ring, idx,
                               fp8_max_of(fmt_a),
                               std::pow(2.0, static_cast<double>(cfg.margin)));
}

Tensor run_gemm(const Tensor& a, const Tensor& b,
                const c10::optional<Tensor>& a_scale,
                const c10::optional<Tensor>& b_scale,
                const c10::optional<Tensor>& bias, bool trans_b) {
    state().n_gemm.fetch_add(1, std::memory_order_relaxed);
    return gemm::quant_gemm_impl(a, b, a_scale, b_scale, false, trans_b, bias);
}

Fp8FwdOut fp8_forward_impl(const Tensor& x, const Tensor& w,
                           const c10::optional<Tensor>& bias,
                           bool update_rings, bool dynamic,
                           int64_t history_len, int64_t margin,
                           at::ScalarType fmt_a, at::ScalarType fmt_b) {
    Fp8Cfg cfg{dynamic, history_len, margin, fmt_a, fmt_b};
    Fp8FwdOut res;
    std::vector<int64_t> out_shape(x.sizes().begin(), x.sizes().end() - 1);
    out_shape.push_back(w.size(0));

    if (dynamic) {
        // Current-amax scaling: measure, then quantize — no rings, no cache.
        const auto dyn = [&](const Tensor& t) {
            return scale_from_amax(amax_of(t), fmt_a, margin);
        };
        res.sx = dyn(x.reshape({-1, w.size(1)}));
        res.sw = dyn(w);
        const auto qx = run_quant(x, res.sx.reciprocal(),
                                  quant::QuantLayout::RowMajor, fmt_a,
                                  c10::nullopt, c10::nullopt, 0, cfg);
        const Tensor w8 =
            is_fp8(w.scalar_type())
                ? w
                : run_quant(w, res.sw.reciprocal(), quant::QuantLayout::RowMajor,
                            fmt_a, c10::nullopt, c10::nullopt, 0, cfg)
                      .out;
        res.out = run_gemm(qx.out.reshape({-1, qx.out.size(-1)}), w8, res.sx,
                           res.sw, bias, true)
                      .reshape(out_shape);
        return res;
    }

    State& st = state();
    auto meta = get_meta(w, history_len, margin, false);
    // Host-side seed only when a ring has never been used: both branches need
    // a valid scale to cast with.
    if (!meta->w.initialized) meta->w.seed(w, fmt_a, margin);
    if (!meta->x.initialized) meta->x.seed(x, fmt_a, margin);
    const bool w_pre = is_fp8(w.scalar_type());

    // The scale snapshot feeds this call's GEMMs (stream-ordered before the
    // in-kernel fold overwrites the ring slot); the kernels read the ring's
    // published reciprocal themselves, so no host reciprocal is needed.
    res.sx = meta->x.scale.clone();
    Tensor w8;
    if (!w_pre && meta->cast.valid(w, fmt_a, fmt_b, st.generation)) {
        st.n_cast_hit.fetch_add(1, std::memory_order_relaxed);
        w8 = meta->cast.w8;
        res.sw = meta->cast.sw;
        res.w8T = meta->cast.w8T;
    } else {
        st.n_cast_miss.fetch_add(1, std::memory_order_relaxed);
        res.sw = meta->w.scale.clone();
        if (w_pre) {
            w8 = w;
        } else if (update_rings) {
            const auto qw = run_quant(w, meta->w.scale_recip,
                                      quant::QuantLayout::Dual, fmt_a, fmt_b,
                                      meta->w.state, meta->w.idx, cfg);
            w8 = qw.out;
            res.w8T = qw.out_t;
            meta->w.advance();
            meta->cast.fill(w, fmt_a, fmt_b, st.generation, w8, res.w8T,
                            res.sw);
        } else {
            // Ring-free cast (no-grad passes): folding here would advance the
            // window a second time per step and desynchronize the recompute.
            w8 = run_quant(w, meta->w.scale_recip,
                           quant::QuantLayout::RowMajor, fmt_a, c10::nullopt,
                           c10::nullopt, 0, cfg)
                     .out;
        }
    }

    if (update_rings) {
        const auto qx = run_quant(x, meta->x.scale_recip,
                                  quant::QuantLayout::Dual, fmt_a, fmt_b,
                                  meta->x.state, meta->x.idx, cfg);
        res.x8T = qx.out_t;
        meta->x.advance();
        res.out = run_gemm(qx.out.reshape({-1, qx.out.size(-1)}), w8, res.sx,
                           res.sw, bias, true)
                      .reshape(out_shape);
    } else {
        const auto qx = run_quant(x, meta->x.scale_recip,
                                  quant::QuantLayout::RowMajor, fmt_a,
                                  c10::nullopt, c10::nullopt, 0, cfg);
        res.out = run_gemm(qx.out.reshape({-1, qx.out.size(-1)}), w8, res.sx,
                           res.sw, bias, true)
                      .reshape(out_shape);
    }
    return res;
}

struct Fp8BwdIn {
    Tensor x, w, sx, sw, x8T, w8T;
    bool dynamic = false;
    bool need_bias_grad = false;
    int64_t history_len = 16, margin = 0;
    at::ScalarType fmt_b = at::kFloat8_e5m2;
};

tensor_list fp8_backward_impl(const Tensor& g, const Fp8BwdIn& in) {
    const Fp8Cfg cfg{in.dynamic, in.history_len, in.margin, in.fmt_b,
                     in.fmt_b};
    const Tensor g2 = g.reshape({-1, g.size(-1)});
    Tensor sx = in.sx, sw = in.sw, sg;
    c10::optional<Tensor> g_ring = c10::nullopt;
    int64_t g_idx = 0;
    std::shared_ptr<Fp8Meta> meta;
    if (in.dynamic) {
        const auto dyn = [&](const Tensor& t) {
            return scale_from_amax(amax_of(t), in.fmt_b, in.margin);
        };
        sg = dyn(g2);
        sw = dyn(in.w);
        sx = dyn(in.x);
    } else {
        meta = get_meta(in.w, in.history_len, in.margin, false);
        if (!meta->g.initialized) meta->g.seed(g2, in.fmt_b, in.margin);
        sg = meta->g.scale.clone();
        g_ring = meta->g.state;
        g_idx = meta->g.idx;
    }
    // Backward GEMMs route through the NT fast path via transposed quantize
    // outputs: g8 [m,n] with w8T [k,n] gives grad_x, g8T [n,m] with x8T [k,m]
    // gives grad_w. g is consumed in both orientations, so one dual pass
    // feeds both; x8T/w8T came from the forward (or the weight cast cache),
    // so the backward re-reads neither x nor w.
    const auto qg = run_quant(g2, in.dynamic ? sg.reciprocal()
                                             : meta->g.scale_recip,
                              quant::QuantLayout::Dual, in.fmt_b, c10::nullopt,
                              g_ring, g_idx, cfg);
    Tensor x8T = in.x8T;
    if (!x8T.defined()) {
        x8T = run_quant(in.x.reshape({-1, in.x.size(-1)}), sx.reciprocal(),
                        quant::QuantLayout::Transposed, in.fmt_b,
                        c10::nullopt, c10::nullopt, 0, cfg)
                  .out_t;
    }
    Tensor grad_x;
    if (is_fp8(in.w.scalar_type())) {
        // Pre-quantized weight has no transposed copy: the swap path for
        // grad_x (grad_w is unaffected).
        grad_x = run_gemm(qg.out, in.w, sg, sw, c10::nullopt, false)
                     .reshape(in.x.sizes());
    } else {
        Tensor w8T = in.w8T;
        if (!w8T.defined()) {
            w8T = run_quant(in.w, sw.reciprocal(),
                            quant::QuantLayout::Transposed, in.fmt_b,
                            c10::nullopt, c10::nullopt, 0, cfg)
                      .out_t;
        }
        grad_x = run_gemm(qg.out, w8T, sg, sw, c10::nullopt, true)
                     .reshape(in.x.sizes());
    }
    Tensor grad_w = run_gemm(qg.out_t, x8T, sg, sx, c10::nullopt, true);
    // bias-free linears must not pay the column-sum reduce: g2.sum(0) is
    // another full read of the gradient.
    Tensor grad_b;
    if (in.need_bias_grad) grad_b = g2.sum(0).to(at::kBFloat16);
    if (meta) meta->g.advance();
    return {grad_x, grad_w, grad_b};
}

}  // namespace

// ---------------------------------------------------------------------------
// The autograd node
// ---------------------------------------------------------------------------

// ``forward`` runs inside the engine with grad mode off; ``update_rings``
// comes from the dispatcher (the caller's grad mode — inside a Function
// forward the mode is invisible), so no-grad passes (checkpointing recompute,
// inference) read the rings without folding or advancing them.
class Fp8Linear : public torch::autograd::Function<Fp8Linear> {
  public:
    static Tensor forward(AutogradContext* ctx, Tensor x, Tensor w, Tensor bias,
                          bool update_rings, bool need_bias_grad, bool dynamic,
                          int64_t history_len, int64_t margin,
                          at::ScalarType fmt_a, at::ScalarType fmt_b) {
        c10::optional<Tensor> bias_opt = c10::nullopt;
        if (bias.numel() > 0) bias_opt = bias;
        const Fp8FwdOut res = fp8_forward_impl(
            x, w, bias_opt, update_rings, dynamic, history_len, margin, fmt_a,
            fmt_b);
        ctx->save_for_backward({x, w});
        if (res.x8T.defined()) ctx->saved_data["x8T"] = res.x8T;
        if (res.w8T.defined()) ctx->saved_data["w8T"] = res.w8T;
        ctx->saved_data["sx"] = res.sx;
        ctx->saved_data["sw"] = res.sw;
        ctx->saved_data["dynamic"] = dynamic;
        ctx->saved_data["need_bias_grad"] = need_bias_grad;
        ctx->saved_data["history_len"] = history_len;
        ctx->saved_data["margin"] = margin;
        ctx->saved_data["fmt_b"] = static_cast<int64_t>(fmt_b);
        return res.out;
    }

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
        const auto saved = ctx->get_saved_variables();
        Fp8BwdIn in;
        in.x = saved[0];
        in.w = saved[1];
        if (ctx->saved_data.count("x8T"))
            in.x8T = ctx->saved_data["x8T"].toTensor();
        if (ctx->saved_data.count("w8T"))
            in.w8T = ctx->saved_data["w8T"].toTensor();
        in.sx = ctx->saved_data["sx"].toTensor();
        in.sw = ctx->saved_data["sw"].toTensor();
        in.dynamic = ctx->saved_data["dynamic"].toBool();
        in.need_bias_grad = ctx->saved_data["need_bias_grad"].toBool();
        in.history_len = ctx->saved_data["history_len"].toInt();
        in.margin = ctx->saved_data["margin"].toInt();
        in.fmt_b =
            static_cast<at::ScalarType>(ctx->saved_data["fmt_b"].toInt());
        tensor_list g = fp8_backward_impl(grad_outputs[0], in);
        // apply() demands one return per forward argument; the seven
        // non-tensor trailing inputs must come back as undefined tensors
        // (the engine filters them out after checking).
        g.resize(10, Tensor());
        return g;
    }
};

// The dispatcher's entry: one Python->C++ crossing per linear, then the whole
// fwd+bwd chain runs in C++.
Tensor fp8_linear(const Tensor& x, const Tensor& w,
                  const c10::optional<Tensor>& bias, bool update_rings,
                  bool need_bias_grad, bool dynamic, int64_t history_len,
                  int64_t margin, at::ScalarType fmt_a, at::ScalarType fmt_b) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "fp8 linear needs CUDA tensors");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 &&
                    w.scalar_type() == at::kBFloat16,
                "fp8 linear takes bf16 operands");
    TORCH_CHECK(x.size(-1) == w.size(1), "fp8 linear: inner dim mismatch");
    Tensor bias_t;
    if (bias.has_value() && bias->defined()) {
        TORCH_CHECK(bias->numel() > 0, "fp8 linear: bias must be non-empty");
        bias_t = *bias;
    } else {
        bias_t = torch::empty({0}, x.options());
    }
    return Fp8Linear::apply(x, w, bias_t, update_rings, need_bias_grad,
                            dynamic, history_len, margin, fmt_a, fmt_b);
}

// ---------------------------------------------------------------------------
// Checkpoint snapshot (A1) and test observability
// ---------------------------------------------------------------------------

py::dict ring_state_dict(const ScaleRing& r) {
    py::dict d;
    d["state"] = r.state.detach().clone();
    d["idx"] = r.idx;
    d["initialized"] = r.initialized;
    return d;
}

// The snapshot shape matches the Python one (version / entries with
// shape+dtype plus the three rings), so the trainer's checkpoint extras
// bridge is unchanged.
py::dict fp8_state_dict() {
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    py::list entries;
    for (const auto& meta : st.order) {
        py::dict e;
        e["shape"] = py::cast(meta->shape);
        e["dtype"] = torch_dtype_str(meta->dtype);
        e["w"] = ring_state_dict(meta->w);
        e["x"] = ring_state_dict(meta->x);
        e["g"] = ring_state_dict(meta->g);
        entries.append(e);
    }
    py::dict out;
    out["version"] = 1;
    out["entries"] = entries;
    return out;
}

void fp8_load_state_dict(py::dict sd) {
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    st.pending.clear();
    for (auto entry : sd["entries"].cast<py::list>())
        st.pending.push_back(entry.cast<py::dict>());
    st.generation += 1;  // the cast cache keys on it
    for (auto& meta : st.order) restore_pending_locked(meta);
}

void fp8_reset() {
    State& st = state();
    std::lock_guard<std::mutex> lock(st.mu);
    st.by_key.clear();
    st.order.clear();
    st.pending.clear();
    st.generation += 1;
}

py::dict fp8_debug_meta(const Tensor& w, int64_t history_len, int64_t margin) {
    auto meta = get_meta(w, history_len, margin, false);
    auto ring = [](const ScaleRing& r) {
        py::dict d;
        d["state"] = r.state;
        d["hist"] = r.hist;
        d["scale"] = r.scale;
        d["scale_recip"] = r.scale_recip;
        d["idx"] = r.idx;
        d["initialized"] = r.initialized;
        return d;
    };
    py::dict out;
    out["w"] = ring(meta->w);
    out["x"] = ring(meta->x);
    out["g"] = ring(meta->g);
    out["cast_version"] = meta->cast.version;
    out["has_cast"] = meta->cast.w8.defined();
    return out;
}

py::dict fp8_debug_stats() {
    State& st = state();
    py::dict d;
    d["quantize"] = st.n_quantize.load(std::memory_order_relaxed);
    d["gemm"] = st.n_gemm.load(std::memory_order_relaxed);
    d["cast_hit"] = st.n_cast_hit.load(std::memory_order_relaxed);
    d["cast_miss"] = st.n_cast_miss.load(std::memory_order_relaxed);
    d["metas"] = static_cast<int64_t>(st.order.size());
    return d;
}

void fp8_debug_reset_stats() {
    State& st = state();
    st.n_quantize = 0;
    st.n_gemm = 0;
    st.n_cast_hit = 0;
    st.n_cast_miss = 0;
}

void bind_fp8(py::module& m) {
    m.def("fp8_linear", &fp8_linear, py::arg("x"), py::arg("w"),
          py::arg("bias") = py::none(), py::arg("update_rings") = true,
          py::arg("need_bias_grad") = false, py::arg("dynamic") = false,
          py::arg("history_len") = 16, py::arg("margin") = 0,
          py::arg("fmt_a") = at::kFloat8_e4m3fn,
          py::arg("fmt_b") = at::kFloat8_e5m2);
    m.def("fp8_state_dict", &fp8_state_dict);
    m.def("fp8_load_state_dict", &fp8_load_state_dict);
    m.def("fp8_reset", &fp8_reset);
    m.def("fp8_debug_meta", &fp8_debug_meta, py::arg("w"),
          py::arg("history_len") = 16, py::arg("margin") = 0);
    m.def("fp8_debug_stats", &fp8_debug_stats);
    m.def("fp8_debug_reset_stats", &fp8_debug_reset_stats);
}

}  // namespace fp8
}  // namespace astrai
