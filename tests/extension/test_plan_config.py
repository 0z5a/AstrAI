"""Runtime plan API tests (the gemm adapter's set_*/probe surface).

These exercise the C++ planner through the real binding, so they need a
built gemm module and a CUDA device. The dispatch behavior itself (which
recipe wins, tier ranking) is covered by the pure-planner tests in C++ and
by the sweep scripts; what is under test here is the configuration API's
contract: precedence, modes, table install/clear, and the probe report.
"""

import pytest
import torch

from astrai.extension import ops
from astrai.extension.loader import is_available

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    ),
    pytest.mark.skipif(
        not is_available("gemm"), reason="gemm kernel not built"
    ),
]

SHAPE = (512, 11008, 4096)  # the wide-N band the analytical model wins
ROW = "511 513 8191 0 0 0 1 3 0 64"  # narrow CTA, 3 stages, kK 64


@pytest.fixture(autouse=True)
def _clean_plan_state():
    ops.gemm.set_table("")
    ops.gemm.set_planner("table")
    ops.gemm.set_staging()
    ops.gemm.set_log(False)
    yield
    ops.gemm.set_table("")
    ops.gemm.set_planner("table")
    ops.gemm.set_staging()
    ops.gemm.set_log(False)


class TestMode:
    def test_default_is_table(self):
        state = ops.gemm.state()
        assert state["planner"] == "table"
        assert ops.gemm.probe(*SHAPE)["source"] == "builtin"

    def test_model_mode_skips_the_rows(self):
        ops.gemm.set_planner("model")
        assert ops.gemm.state()["planner"] == "model"
        assert ops.gemm.probe(*SHAPE)["source"] == "model"

    def test_hybrid_prefers_table_then_model(self):
        ops.gemm.set_planner("hybrid")
        assert ops.gemm.probe(*SHAPE)["source"] == "builtin"
        ops.gemm.set_table("-")  # every row tier off
        assert ops.gemm.probe(*SHAPE)["source"] == "model"

    def test_model_only_ignores_the_table(self):
        ops.gemm.set_planner("model")
        ops.gemm.set_table(ROW)
        assert ops.gemm.probe(*SHAPE)["source"] == "model"

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError):
            ops.gemm.set_planner("cost-model")


class TestTable:
    def test_override_rows_outrank_builtin(self):
        installed = ops.gemm.set_table(ROW)
        assert installed == 1
        info = ops.gemm.probe(*SHAPE)
        assert info["source"] == "override"
        assert (info["cta"], info["stages"], info["kk"]) == (1, 3, 64)

    def test_off_mode_falls_to_degraded(self):
        ops.gemm.set_table("-")
        info = ops.gemm.probe(*SHAPE)
        assert info["source"] == "degraded"

    def test_clear_restores_the_builtin(self):
        ops.gemm.set_table(ROW)
        ops.gemm.set_table("")
        assert ops.gemm.probe(*SHAPE)["source"] == "builtin"
        assert ops.gemm.state()["table"]["override_rows"] == 0

    def test_injected_rows_rank_below_override(self):
        ops.gemm.inject_rows(ROW)
        assert ops.gemm.probe(*SHAPE)["source"] == "injected"
        ops.gemm.set_table("511 513 8191 0 0 0 2 2 0 64")
        assert ops.gemm.probe(*SHAPE)["source"] == "override"


class TestStaging:
    def test_state_reports_the_switches(self):
        assert ops.gemm.state()["staging"] == {"tma": True, "mx": True}
        ops.gemm.set_staging(tma=False)
        assert ops.gemm.state()["staging"] == {"tma": False, "mx": True}
        ops.gemm.set_staging(mx=False)
        assert ops.gemm.state()["staging"] == {"tma": False, "mx": False}


class TestProbe:
    def test_reports_the_query_key(self):
        info = ops.gemm.probe(*SHAPE)
        assert info["perf_class"] == 0  # bf16 x bf16
        assert info["crosswise"] == 0  # the NT fused-linear shape

    def test_vocabulary_carries_geometry(self):
        rows = ops.gemm.tile_vocabulary()
        assert rows, "the vocabulary must not be empty"
        for entry in rows:
            (
                crosswise,
                ba,
                bb,
                cta,
                stages,
                kk,
                bm,
                bn,
                threads,
                smem,
            ) = entry
            assert (crosswise, ba, bb) in ((0, 2, 2), (1, 2, 2), (0, 2, 1),
                                           (1, 2, 1), (0, 1, 1), (1, 1, 1))
            assert cta in (0, 1, 2, 3)
            assert stages in (2, 3)
            assert kk in (32, 64)
            assert bm in (64, 128) and bn in (64, 128, 256)
            assert threads > 0 and smem > 0

    def test_facts_are_populated(self):
        facts = ops.gemm.facts()
        assert facts["sms"] > 0
        assert facts["cc"] >= 80
        assert facts["l2_bytes"] > 0
