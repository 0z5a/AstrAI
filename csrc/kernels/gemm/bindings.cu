// pybind surface of the gemm module: every py:: spelling in the family lives
// here — the None-tolerant argument marshalling, the dict shapes of the
// planner's introspection, and the module registration. The typed C++ face is
// gemm/api.h; gemm.cu holds the implementations.
//
// The dict keys below ARE the contract the Python tooling reads (csrc/bench's
// dispatch_grid / model_capture / diff_rows / tune_plan_table), so they are
// spelled exactly once, here, next to the struct they mirror.

#include <torch/extension.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "common/device.cuh"
#include "gemm/api.h"
#include "gemm/plan_table.h"

namespace astrai {
namespace fp8 {
void bind_fp8(py::module& m);
}  // namespace fp8

namespace gemm {
namespace {

// py::object -> torch::Tensor with a uniform error message; a none object
// stays undefined (callers gate on is_none()).
torch::Tensor cast_tensor_arg(const py::object& o, const char* name) {
    try {
        return o.cast<torch::Tensor>();
    } catch (const py::cast_error&) {
        TORCH_CHECK(false, name, " must be a torch.Tensor or None");
        return {};
    }
}

// pybind surface: None-tolerant operand scales and bias (``cast_tensor_arg``
// keeps the "must be a torch.Tensor or None" message), then the shared
// implementation the composed fp8 linear also calls (see gemm/api.h).
torch::Tensor quant_gemm(torch::Tensor a, torch::Tensor b, py::object a_scale,
                         py::object b_scale, bool trans_a, bool trans_b,
                         py::object bias) {
    auto opt = [](const py::object& o,
                  const char* name) -> c10::optional<torch::Tensor> {
        if (o.is_none()) return c10::nullopt;
        return cast_tensor_arg(o, name);
    };
    return quant_gemm_impl(a, b, opt(a_scale, "a_scale"),
                           opt(b_scale, "b_scale"), trans_a, trans_b,
                           opt(bias, "bias"));
}

// ---------------------------------------------------------------------------
// Marshal the typed planner surface into the dict shapes the Python tools
// read. One key list per struct, no second copy anywhere.
// ---------------------------------------------------------------------------

py::dict probe_dict(const PlanProbe& r) {
    py::dict d;
    d["source"] = r.source;
    d["cta"] = r.cta;
    d["stages"] = r.stages;
    d["raster"] = r.raster;
    d["kk"] = r.kk;
    d["perf_class"] = r.perf_class;
    d["crosswise"] = r.crosswise;
    return d;
}

py::dict config_dict(const GemmConfigState& s) {
    py::dict d, table, staging;
    d["planner"] = s.planner;
    d["planner_mode"] = s.planner_mode;
    d["log"] = s.log;
    table["off"] = s.table_off;
    table["override_rows"] = s.override_rows;
    table["override_source"] = s.override_source;
    table["injected_rows"] = s.injected_rows;
    table["injected_source"] = s.injected_source;
    d["table"] = table;
    staging["tma"] = s.staging_tma;
    staging["mx"] = s.staging_mx;
    d["staging"] = staging;
    return d;
}

py::dict facts_dict() {
    const DeviceFacts dev = astrai::device_facts();
    py::dict d;
    d["sms"] = dev.sms;
    d["smem_max"] = dev.smem_max;
    d["smem_per_sm"] = dev.smem_per_sm;
    d["regs_per_sm"] = dev.regs_per_sm;
    d["l2_bytes"] = dev.l2_bytes;
    d["cc"] = dev.cc;
    return d;
}

// py::None -> "absent"; a str planner is the binding's spelling of the mode
// ("" = back to unset, int accepted too, with the range check left to
// configure()); a str tier names the row tier `rows` addresses.
GemmConfigPatch patch_from(py::object planner, py::object log, py::object tma,
                           py::object mx, py::object table_off,
                           py::object rows, py::object tier) {
    GemmConfigPatch patch;
    if (!planner.is_none()) {
        if (py::isinstance<py::str>(planner)) {
            const std::string name = planner.cast<std::string>();
            if (name.empty()) {
                // "" restores the shipped default (back to "unset": the env
                // seed decides, and hybrid is what an unseeded process
                // resolves to).
                patch.planner_mode = -1;
            } else {
                int mode = -1;
                if (!parse_planner_mode(name, mode))
                    throw std::invalid_argument(
                        "planner must be 'table', 'hybrid' or 'model', got '" +
                        name + "'");
                patch.planner_mode = mode;
            }
        } else {
            patch.planner_mode = planner.cast<int>();
        }
    }
    if (!log.is_none()) patch.log = log.cast<bool>();
    if (!tma.is_none()) patch.staging_tma = tma.cast<bool>();
    if (!mx.is_none()) patch.staging_mx = mx.cast<bool>();
    if (!table_off.is_none()) patch.table_off = table_off.cast<bool>();
    if (!rows.is_none()) patch.rows = rows.cast<std::string>();
    if (!tier.is_none()) {
        const std::string name = tier.cast<std::string>();
        if (name == "override")
            patch.tier = RowTier::Override;
        else if (name == "injected")
            patch.tier = RowTier::Injected;
        else
            throw std::invalid_argument(
                "tier must be 'override' or 'injected', got '" + name + "'");
    }
    return patch;
}

py::dict probe_binding(int64_t m, int64_t n, int64_t k, at::ScalarType dt_a,
                       at::ScalarType dt_b, bool trans_a, bool trans_b,
                       int64_t batch) {
    return probe_dict(plan_probe(m, n, k, dt_a, dt_b, trans_a, trans_b, batch));
}

py::dict configure_binding(py::object planner, py::object log, py::object tma,
                           py::object mx, py::object table_off,
                           py::object rows, py::object tier) {
    return config_dict(
        configure(patch_from(planner, log, tma, mx, table_off, rows, tier)));
}

py::dict config_state_binding() { return config_dict(config_state()); }

}  // namespace
}  // namespace gemm
}  // namespace astrai

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // The fp8 training linear (forward + backward) lives in this module:
    // its composition launches through the GEMM dispatch below, whose
    // plan table / planner state must stay single-source.
    astrai::fp8::bind_fp8(m);
    m.def("quant_gemm", &astrai::gemm::quant_gemm, py::arg("a"), py::arg("b"),
          py::arg("a_scale") = py::none(), py::arg("b_scale") = py::none(),
          py::arg("trans_a") = false, py::arg("trans_b") = true,
          py::arg("bias") = py::none());
    m.def("plan_probe", &astrai::gemm::probe_binding, py::arg("m"),
          py::arg("n"), py::arg("k"), py::arg("dt_a"), py::arg("dt_b"),
          py::arg("trans_a") = false, py::arg("trans_b") = true,
          py::arg("batch") = 1);
    m.def("inject_plan_rows", &astrai::gemm::inject_plan_rows,
          py::arg("source"));
    m.def("configure", &astrai::gemm::configure_binding,
          py::arg("planner") = py::none(), py::arg("log") = py::none(),
          py::arg("tma") = py::none(), py::arg("mx") = py::none(),
          py::arg("table_off") = py::none(), py::arg("rows") = py::none(),
          py::arg("tier") = py::none());
    m.def("config_state", &astrai::gemm::config_state_binding);
    m.def("tile_class_names", &astrai::gemm::tile_class_names);
    m.def("tile_vocabulary", &astrai::gemm::tile_vocabulary);
    m.def("device_facts_info", &astrai::gemm::facts_dict);
}
