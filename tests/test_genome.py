"""Tests for genome decoding: symmetry, channels, voltage resolution."""

import unittest

import numpy as np

from ehd_ga.constants import (
    CH_NEGATIVE,
    CH_NEUTRAL,
    CH_POSITIVE,
    ENGINE_OFFSET,
    ENGINE_SIZE,
    EVAL_SIZE,
    GENOME_DIM,
)
from ehd_ga.genome import (
    decode_genome,
    rasterize_polygon,
    voltage_score_to_multiplier,
)


def _engine_region(canvas):
    return canvas[
        ENGINE_OFFSET : ENGINE_OFFSET + ENGINE_SIZE,
        ENGINE_OFFSET : ENGINE_OFFSET + ENGINE_SIZE,
        :,
    ]


class TestVoltageResolution(unittest.TestCase):
    def test_example_0_6_resolves_to_3(self):
        self.assertEqual(voltage_score_to_multiplier(0.6), 3)

    def test_endpoints(self):
        self.assertEqual(voltage_score_to_multiplier(0.0), 1)
        self.assertEqual(voltage_score_to_multiplier(1.0), 4)

    def test_standard_rounding_not_bankers(self):
        # 1 + 3*0.5 = 2.5 -> standard rounding gives 3 (banker's gives 2).
        self.assertEqual(voltage_score_to_multiplier(0.5), 3)

    def test_boundaries(self):
        # round(1 + 3s): s=1/6 -> 1.5 -> 2; s=5/6 -> 3.5 -> 4.
        self.assertEqual(voltage_score_to_multiplier(1.0 / 6.0), 2)
        self.assertEqual(voltage_score_to_multiplier(5.0 / 6.0), 4)


class TestRasterize(unittest.TestCase):
    def test_square(self):
        pts = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=float)
        mask = rasterize_polygon(pts, 20, 20)
        self.assertEqual(int(mask.sum()), 100)
        self.assertTrue(mask[5, 5])
        self.assertFalse(mask[15, 15])

    def test_degenerate(self):
        pts = np.array([[5, 5], [5, 5], [5, 5]], dtype=float)
        mask = rasterize_polygon(pts, 20, 20)
        self.assertEqual(int(mask.sum()), 0)


class TestDecode(unittest.TestCase):
    def _genome_with_rects(self):
        # Positive rect in the upper-left of the upper half, negative
        # rect in the upper-right, neutral rect in the middle.
        # Each polygon: 5 points tracing a rectangle (last point repeats).
        genome = np.zeros(GENOME_DIM)
        # family 0 (positive): x 0.05..0.35, y 0.10..0.40
        pos = [(0.05, 0.10), (0.35, 0.10), (0.35, 0.40), (0.05, 0.40),
               (0.05, 0.10)]
        # family 1 (negative): x 0.65..0.95, y 0.10..0.40
        neg = [(0.65, 0.10), (0.95, 0.10), (0.95, 0.40), (0.65, 0.40),
               (0.65, 0.10)]
        # family 2 (neutral): x 0.40..0.60, y 0.55..0.85
        neu = [(0.40, 0.55), (0.60, 0.55), (0.60, 0.85), (0.40, 0.85),
               (0.40, 0.55)]
        for fam, rect in enumerate((pos, neg, neu)):
            for poly in range(2):
                base = ((fam * 2 + poly) * 5) * 2
                for i, (x, y) in enumerate(rect):
                    genome[base + i * 2] = x
                    genome[base + i * 2 + 1] = y
        genome[-1] = 0.6  # voltage score -> multiplier 3
        return genome

    def test_canvas_shape_and_dtype(self):
        canvas, _ = decode_genome(self._genome_with_rects())
        self.assertEqual(canvas.shape, (EVAL_SIZE, EVAL_SIZE, 3))
        self.assertEqual(canvas.dtype, np.uint8)

    def test_engine_centered_and_air_outside(self):
        canvas, _ = decode_genome(self._genome_with_rects())
        # Outside the engine bounds the canvas is air (black).
        self.assertEqual(int(canvas[:ENGINE_OFFSET, :, :].sum()), 0)
        self.assertEqual(int(canvas[ENGINE_OFFSET + ENGINE_SIZE:, :, :].sum()), 0)
        self.assertEqual(int(canvas[:, :ENGINE_OFFSET, :].sum()), 0)
        self.assertEqual(
            int(canvas[:, ENGINE_OFFSET + ENGINE_SIZE:, :].sum()), 0)

    def test_mirror_symmetry_across_x_axis(self):
        canvas, _ = decode_genome(self._genome_with_rects())
        engine = _engine_region(canvas)
        upper = engine[: ENGINE_SIZE // 2, :, :]
        lower = engine[ENGINE_SIZE // 2 :, :, :]
        np.testing.assert_array_equal(lower, upper[::-1, :, :])

    def test_channels_present(self):
        canvas, _ = decode_genome(self._genome_with_rects())
        engine = _engine_region(canvas)
        self.assertGreater(int((engine[:, :, CH_POSITIVE] > 0).sum()), 0)
        self.assertGreater(int((engine[:, :, CH_NEGATIVE] > 0).sum()), 0)
        self.assertGreater(int((engine[:, :, CH_NEUTRAL] > 0).sum()), 0)

    def test_no_overlap_for_separated_rects(self):
        _, info = decode_genome(self._genome_with_rects())
        self.assertEqual(info["overlap_px"], 0)
        self.assertEqual(info["multiplier"], 3)
        self.assertEqual(info["v_neg"], -60_000.0)

    def test_overlap_resolved_by_priority(self):
        # Positive and neutral polygons on top of each other: positive wins.
        genome = np.zeros(GENOME_DIM)
        rect = [(0.05, 0.10), (0.35, 0.10), (0.35, 0.40), (0.05, 0.40),
                (0.05, 0.10)]
        for fam in (0, 2):  # positive and neutral, same rect
            for poly in range(2):
                base = ((fam * 2 + poly) * 5) * 2
                for i, (x, y) in enumerate(rect):
                    genome[base + i * 2] = x
                    genome[base + i * 2 + 1] = y
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            canvas, info = decode_genome(genome)
            self.assertEqual(len(caught), 1)
            self.assertIn("overlap", str(caught[0].message))
        self.assertGreater(info["overlap_px"], 0)
        engine = _engine_region(canvas)
        # The shared rect is red (positive), not green.
        self.assertEqual(int((engine[:, :, CH_NEUTRAL] > 0).sum()), 0)
        self.assertGreater(int((engine[:, :, CH_POSITIVE] > 0).sum()), 0)

    def test_wrong_genome_length(self):
        with self.assertRaises(ValueError):
            decode_genome(np.zeros(10))


if __name__ == "__main__":
    unittest.main()
