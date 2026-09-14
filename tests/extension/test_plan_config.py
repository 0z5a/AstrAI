"""Runtime plan API tests (the gemm adapter's set_*/probe surface).

These exercise the C++ planner through the real binding, so they need a
built gemm module and a CUDA device. The configuration API's contract —
precedence, modes, table install/clear, the probe report — is what the
first classes cover; the analytical planner's own selection RULE is
covered by TestModelRule below.
"""

import pytest
import torch

from astrai.extension import ops
from astrai.extension.loader import is_available

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available"),
    pytest.mark.skipif(not is_available("gemm"), reason="gemm kernel not built"),
]

SHAPE = (512, 11008, 4096)  # the wide-N band the analytical model wins
ROW = "511 513 8191 0 0 0 1 3 0 64"  # narrow CTA, 3 stages, kK 64


@pytest.fixture(autouse=True)
def _clean_plan_state():
    # Both row tiers, not just the override: set_table("") clears the
    # override rows only, so an injected row leaked from another test would
    # keep answering ahead of the planner under test.
    ops.gemm.set_table("")
    ops.gemm.inject_rows("")
    ops.gemm.set_planner("")  # back to the shipped default
    ops.gemm.set_staging()
    ops.gemm.set_log(False)
    yield
    ops.gemm.set_table("")
    ops.gemm.inject_rows("")
    ops.gemm.set_planner("")  # back to the shipped default
    ops.gemm.set_staging()
    ops.gemm.set_log(False)


class TestMode:
    def test_default_is_hybrid(self):
        # The shipped default: rows when any exist, else the model. The
        # compiled-in tables are empty, so a fresh process gets the model.
        state = ops.gemm.state()
        assert state["planner"] == "hybrid"
        assert ops.gemm.probe(*SHAPE)["source"] == "model"

    def test_model_mode_skips_the_rows(self):
        ops.gemm.set_planner("model")
        ops.gemm.set_table(ROW)
        assert ops.gemm.state()["planner"] == "model"
        assert ops.gemm.probe(*SHAPE)["source"] == "model"

    def test_hybrid_prefers_rows_then_model(self):
        ops.gemm.set_planner("hybrid")
        assert ops.gemm.probe(*SHAPE)["source"] == "model"
        ops.gemm.set_table(ROW)  # a row now owns the shape
        assert ops.gemm.probe(*SHAPE)["source"] == "override"
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
    def test_override_rows_take_the_shape(self):
        installed = ops.gemm.set_table(ROW)
        assert installed == 1
        info = ops.gemm.probe(*SHAPE)
        assert info["source"] == "override"
        assert (info["cta"], info["stages"], info["kk"]) == (1, 3, 64)

    def test_off_mode_kills_the_rows_only(self):
        # "-" disables override, injected and builtin alike; what answers
        # after that is the planner mode's business: the model under the
        # shipped hybrid default, the degraded ladder under "table".
        ops.gemm.set_table("-")
        assert ops.gemm.probe(*SHAPE)["source"] == "model"
        ops.gemm.set_planner("table")
        assert ops.gemm.probe(*SHAPE)["source"] == "degraded"

    def test_clear_restores_the_default(self):
        ops.gemm.set_table(ROW)
        ops.gemm.set_table("")
        assert ops.gemm.state()["table"]["override_rows"] == 0
        # The builtin tables ship empty, so the model answers again.
        assert ops.gemm.probe(*SHAPE)["source"] == "model"

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


class TestCompiledInTables:
    def test_rows_ship_device_guarded(self):
        # Compiled-in rows are the measured diff of the model's errors for
        # ONE device (the GENERATED block's provenance); the tier is
        # signature-guarded, so it serves exactly there. On any other part
        # "builtin" never appears, which is what keeps the rows from
        # leaking onto a machine they were not measured on.
        sig = ops.gemm.facts()
        measured_here = (
            sig["cc"] == 120
            and sig["sms"] == 170
            and sig["smem_per_sm"] == 102400
            and sig["l2_bytes"] == 100663296
        )
        in_band = ((128, 2048, 4096), (2048, 14336, 4096))
        for shape in in_band:
            src = ops.gemm.probe(*shape)["source"]
            if measured_here:
                assert src == "builtin"
            else:
                assert src != "builtin"
        # Bands the model already wins stay the model's even where the
        # rows are live (the diff only claims measured >=2% wins).
        assert ops.gemm.probe(512, 11008, 4096)["source"] != "override"


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
            assert (crosswise, ba, bb) in (
                (0, 2, 2),
                (1, 2, 2),
                (0, 2, 1),
                (1, 2, 1),
                (0, 1, 1),
                (1, 1, 1),
            )
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


class TestModelRule:
    """The planner's resource rule, asserted as a RULE rather than as a
    recipe so it holds on any device.

    No wave count and no traffic term survives in the model: with a
    fractional last wave the FLOP term cancels across the candidates of one
    problem, which is what the sweep measures (every production cell lands
    within 1.13-1.38x of every other on a given shape, and wave count ranks
    them at rho -0.90 against the measurement). What is left to choose
    between is the resource the ring buys — CTAs resident per SM, then
    prefetch depth. This pins the planner to that, and pins that it never
    trades residency away for ring depth.
    """

    SHAPES = (
        (512, 11008, 4096),
        (4096, 1536, 1536),
        (4096, 4096, 4096),
        (4096, 11008, 4096),
        (2048, 28672, 8192),
    )

    @staticmethod
    def _resource(entry, facts):
        """(resident, stages) for one vocabulary entry, or None when the
        ring is not resident on this device at all.

        Mirrors policy.cuh's min_ctas_for_ring against the per-SM smem
        budget — the only way to assert the rule from Python. Deliberately
        independent of the shape: the rule does not consult it.
        """
        smem = entry[9]
        resident = min(facts["smem_per_sm"] // smem, 2 if smem <= 48 * 1024 else 1)
        if resident <= 0:
            return None
        return resident, entry[4]  # (resident, stages)

    def test_pick_attains_the_best_ring_resource(self):
        # The rule under test is the analytical model's own; pin the
        # planner to it so a compiled-in row (which outranks the model in
        # the default chain) cannot answer in its place.
        ops.gemm.set_planner("model")
        facts = ops.gemm.facts()
        vocab = [
            entry
            for entry in ops.gemm.tile_vocabulary()
            if (entry[0], entry[1], entry[2]) == (0, 2, 2)  # NT, bf16 x bf16
        ]
        assert vocab, "the vocabulary carries no bf16 x bf16 candidates"
        for shape in self.SHAPES:
            by_recipe = {}
            for entry in vocab:
                resource = self._resource(entry, facts)
                if resource is not None:
                    by_recipe[tuple(entry[3:6])] = resource
            assert by_recipe, f"no resident candidate for {shape}"

            info = ops.gemm.probe(*shape)
            assert info["source"] == "model", shape
            picked = (info["cta"], info["stages"], info["kk"])
            assert picked in by_recipe, f"{shape}: {picked} is not a candidate"
            assert by_recipe[picked] == max(by_recipe.values()), (
                f"{shape}: picked {picked} with ring resource "
                f"{by_recipe[picked]}, the best is {max(by_recipe.values())}"
            )
