"""Tests for the evaluator: feasibility, arcing, clamp."""

import unittest

import numpy as np

from ehd_ga.constants import E_BREAKDOWN, MU, V_POS
from ehd_ga.evaluator import (
    EvalResult,
    Evaluator,
    check_feasibility,
)
from ehd_ga.genome import decode_genome


def _canvas_with_masks(pos_rect, neg_rect, neu_rect=None):
    """Build a 256x256 canvas with the given engine-space rects.

    Rects are (r0, c0, r1, c1) in engine coordinates (128x128).
    """
    from ehd_ga.constants import (
        CH_NEGATIVE,
        CH_NEUTRAL,
        CH_POSITIVE,
        ENGINE_OFFSET,
    )

    canvas = np.zeros((256, 256, 3), dtype=np.uint8)
    for rect, ch in ((pos_rect, CH_POSITIVE), (neg_rect, CH_NEGATIVE)):
        if rect is None:
            continue
        r0, c0, r1, c1 = rect
        canvas[
            ENGINE_OFFSET + r0 : ENGINE_OFFSET + r1,
            ENGINE_OFFSET + c0 : ENGINE_OFFSET + c1,
            ch,
        ] = 255
    if neu_rect is not None:
        r0, c0, r1, c1 = neu_rect
        canvas[
            ENGINE_OFFSET + r0 : ENGINE_OFFSET + r1,
            ENGINE_OFFSET + c0 : ENGINE_OFFSET + c1,
            CH_NEUTRAL,
        ] = 255
    return canvas


class TestFeasibility(unittest.TestCase):
    def test_short_circuit_touching(self):
        # Positive and negative rects adjacent (sharing an edge): the
        # 1-px dilation of positive hits negative.
        pos = np.zeros((128, 128), dtype=bool)
        neg = np.zeros((128, 128), dtype=bool)
        pos[10:30, 10:30] = True
        neg[10:30, 30:51] = True  # adjacent: dilation overlaps
        check = check_feasibility(pos, neg, -20_000.0)
        self.assertFalse(check["feasible"])
        self.assertIn("short", check["reason"])

    def test_short_circuit_overlapping(self):
        pos = np.zeros((128, 128), dtype=bool)
        neg = np.zeros((128, 128), dtype=bool)
        pos[10:30, 10:30] = True
        neg[20:40, 20:40] = True
        check = check_feasibility(pos, neg, -20_000.0)
        self.assertFalse(check["feasible"])
        self.assertIn("short", check["reason"])

    def test_arc_tiny_gap_high_voltage(self):
        # 6 px gap (center-to-center distance 35-29), V_total = 100 kV
        # -> E = 1.67e7 > 3e6.
        pos = np.zeros((128, 128), dtype=bool)
        neg = np.zeros((128, 128), dtype=bool)
        pos[10:30, 10:30] = True
        neg[10:30, 35:55] = True  # 6 px gap
        check = check_feasibility(pos, neg, -80_000.0)
        self.assertFalse(check["feasible"])
        self.assertIn("arc", check["reason"])
        self.assertAlmostEqual(check["gap_px"], 6.0)
        self.assertGreater(check["field_e"], E_BREAKDOWN)

    def test_no_arc_wide_gap_low_voltage(self):
        # 51 px gap (center-to-center distance 80-29), V_total = 40 kV
        # -> E = 7.8e5 < 3e6.
        pos = np.zeros((128, 128), dtype=bool)
        neg = np.zeros((128, 128), dtype=bool)
        pos[10:30, 10:30] = True
        neg[10:30, 80:100] = True
        check = check_feasibility(pos, neg, -20_000.0)
        self.assertTrue(check["feasible"])
        self.assertAlmostEqual(check["gap_px"], 51.0)
        self.assertLess(check["field_e"], E_BREAKDOWN)

    def test_disconnected_positive(self):
        pos = np.zeros((128, 128), dtype=bool)
        neg = np.zeros((128, 128), dtype=bool)
        pos[10:20, 10:20] = True
        pos[60:70, 60:70] = True  # second island
        neg[10:20, 80:100] = True
        check = check_feasibility(pos, neg, -20_000.0)
        self.assertFalse(check["feasible"])
        self.assertIn("components", check["reason"])

    def test_empty_mask(self):
        pos = np.zeros((128, 128), dtype=bool)
        neg = np.zeros((128, 128), dtype=bool)
        neg[10:20, 10:20] = True
        check = check_feasibility(pos, neg, -20_000.0)
        self.assertFalse(check["feasible"])

    def test_neutral_touching_electrode_is_ok(self):
        # Neutral is not an input to check_feasibility: it may touch
        # either electrode and has no connectivity requirement.  Verify
        # at the Evaluator level with a canvas where neutral is adjacent
        # to (touching) the positive rect.
        canvas = _canvas_with_masks(
            (10, 10, 30, 30), (10, 80, 30, 100), neu_rect=(30, 25, 50, 45)
        )
        evaluator = Evaluator()
        result = evaluator.evaluate_image(canvas, -20_000.0)
        self.assertTrue(result.feasible, result.reason)
        self.assertIsNotNone(result.fitness)


    def test_graded_penalty_ordering(self):
        # A worse short circuit (more contact) has a larger penalty than
        # a mild one, so the GA gets a gradient toward feasibility.
        from ehd_ga.ga import _to_fitness

        pos = np.zeros((128, 128), dtype=bool)
        neg_mild = np.zeros((128, 128), dtype=bool)
        neg_bad = np.zeros((128, 128), dtype=bool)
        pos[10:30, 10:30] = True
        neg_mild[10:30, 30:35] = True  # adjacent: small contact
        neg_bad[10:30, 10:50] = True  # heavy overlap: large contact
        mild = check_feasibility(pos, neg_mild, -20_000.0)
        bad = check_feasibility(pos, neg_bad, -20_000.0)
        self.assertFalse(mild["feasible"])
        self.assertFalse(bad["feasible"])
        self.assertGreater(bad["penalty"], mild["penalty"])
        # GA fitness: worse violation scores lower, and every infeasible
        # design scores below every feasible one (feasible >= 0).
        self.assertLess(_to_fitness(EvalResult(
            fitness=None, feasible=False, reason="x",
            penalty=bad["penalty"])),
            _to_fitness(EvalResult(
                fitness=None, feasible=False, reason="x",
                penalty=mild["penalty"])))
        self.assertLess(_to_fitness(EvalResult(
            fitness=None, feasible=False, reason="x",
            penalty=mild["penalty"])), 0.0)


class TestEvaluatorClamp(unittest.TestCase):
    def test_clamp_raw_above_bound(self):
        # Simulator returns an absurd velocity; fitness must be clamped
        # to v_max = MU * E.
        def huge_sim(canvas, v_pos, v_neg, gap_m):
            return 1e9

        canvas = _canvas_with_masks((10, 10, 30, 30), (10, 80, 30, 100))
        evaluator = Evaluator(simulator=huge_sim)
        result = evaluator.evaluate_image(canvas, -20_000.0)
        self.assertTrue(result.feasible)
        self.assertTrue(result.clamped)
        self.assertEqual(result.raw_velocity, 1e9)
        self.assertAlmostEqual(result.bound, MU * result.field_e)
        self.assertAlmostEqual(result.fitness, result.bound)

    def test_no_clamp_raw_below_bound(self):
        def small_sim(canvas, v_pos, v_neg, gap_m):
            return 1.5

        canvas = _canvas_with_masks((10, 10, 30, 30), (10, 80, 30, 100))
        evaluator = Evaluator(simulator=small_sim)
        result = evaluator.evaluate_image(canvas, -20_000.0)
        self.assertTrue(result.feasible)
        self.assertFalse(result.clamped)
        self.assertAlmostEqual(result.fitness, 1.5)

    def test_infeasible_fitness_none(self):
        canvas = _canvas_with_masks((10, 10, 30, 30), (10, 31, 30, 51))
        evaluator = Evaluator()
        result = evaluator.evaluate_image(canvas, -20_000.0)
        self.assertFalse(result.feasible)
        self.assertIsNone(result.fitness)

    def test_evaluate_genome_end_to_end(self):
        # Separated rects genome: feasible, finite fitness.
        from ehd_ga.constants import GENOME_DIM

        genome = np.zeros(GENOME_DIM)
        pos = [(0.05, 0.10), (0.35, 0.10), (0.35, 0.40), (0.05, 0.40),
               (0.05, 0.10)]
        neg = [(0.65, 0.10), (0.95, 0.10), (0.95, 0.40), (0.65, 0.40),
               (0.65, 0.10)]
        neu = [(0.40, 0.55), (0.60, 0.55), (0.60, 0.85), (0.40, 0.85),
               (0.40, 0.55)]
        for fam, rect in enumerate((pos, neg, neu)):
            for poly in range(2):
                base = ((fam * 2 + poly) * 5) * 2
                for i, (x, y) in enumerate(rect):
                    genome[base + i * 2] = x
                    genome[base + i * 2 + 1] = y
        genome[-1] = 0.6
        evaluator = Evaluator()
        result = evaluator.evaluate(genome)
        self.assertTrue(result.feasible, result.reason)
        self.assertIsNotNone(result.fitness)
        self.assertTrue(np.isfinite(result.fitness))
        self.assertEqual(result.multiplier, 3)


if __name__ == "__main__":
    unittest.main()
