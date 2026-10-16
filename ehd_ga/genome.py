"""Genome decoding: fixed-length vector -> 256x256 engine image.

Geometry
--------
* The genome encodes ONLY the upper half (128x64) of the engine.
  The decoder mirrors it across the horizontal centerline (the x-axis) to
  get the full 128x128 engine, then places the engine centered (offset
  64, 64) in a 256x256 canvas.  The rest of the canvas is air (black).
* The engine is a longways slice: inlet on the left, outlet on the right.
* Three polygon families (fast-flow jet image convention, BGR channels):
  positive = RED (channel 2), negative = BLUE (channel 0),
  neutral = GREEN (channel 1, uncharged solid obstacle).

Genome layout (61 genes)
-------------------------
Genes 0..59 are coordinates, family-major, then polygon, then point,
then (x, y): gene index = ((family * 2 + polygon) * 5 + point) * 2 + axis.
Each (x, y) is normalized to [0, 1] within the 128x64 upper half.
Gene 60 is the voltage score in [0, 1].

If two families' polygons overlap, a pixel belongs to at most one family,
resolved by priority positive > negative > neutral.  Overlaps are counted
and reported in the decode info dict (not silently hidden).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from .constants import (
    CH_NEGATIVE,
    CH_NEUTRAL,
    CH_POSITIVE,
    ENGINE_OFFSET,
    ENGINE_SIZE,
    EVAL_SIZE,
    GENOME_DIM,
    MULT_MAX,
    MULT_MIN,
    N_FAMILIES,
    POINTS_PER_POLYGON,
    POLYGONS_PER_FAMILY,
    UPPER_HALF_H,
    V_POS,
)


def voltage_score_to_multiplier(score: float) -> int:
    """Resolve a voltage score in [0, 1] to an integer multiplier 1..4.

    multiplier = round(1 + 3 * score), with 0.5 rounding UP (standard
    rounding), NOT Python's banker's rounding.  Example: 0.6 -> 2.8 -> 3,
    so a 20 kV positive terminal faces a -60 kV negative terminal.
    """
    raw = 1.0 + 3.0 * float(score)
    mult = int(math.floor(raw + 0.5))
    return max(MULT_MIN, min(MULT_MAX, mult))


def voltage_score_to_v_neg(score: float) -> float:
    """Negative terminal voltage (volts) for a voltage score in [0, 1]."""
    return -voltage_score_to_multiplier(score) * V_POS


def _genome_coords(genome: np.ndarray) -> np.ndarray:
    """Return the coordinate genes as (family, polygon, point, axis)."""
    coords = np.asarray(genome, dtype=float).reshape(-1)[: GENOME_DIM - 1]
    coords = np.clip(coords, 0.0, 1.0)
    return coords.reshape(
        N_FAMILIES, POLYGONS_PER_FAMILY, POINTS_PER_POLYGON, 2
    )


def rasterize_polygon(
    points_px: np.ndarray, height: int, width: int
) -> np.ndarray:
    """Rasterize one filled polygon to a boolean mask.

    Pure-numpy even-odd ray casting: a pixel is inside if a ray from its
    center in the +x direction crosses an odd number of polygon edges.
    ``points_px`` is (n, 2) with columns (x, y) in pixel coordinates.
    """
    mask = np.zeros((height, width), dtype=bool)
    pts = np.asarray(points_px, dtype=float)
    if len(pts) < 3:
        return mask
    # Pixel centers.
    ys, xs = np.mgrid[0:height, 0:width]
    xs = xs.astype(float) + 0.5
    ys = ys.astype(float) + 0.5
    crossings = np.zeros((height, width), dtype=int)
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        # Edge straddles the pixel row, and the pixel is left of the
        # edge's intersection with that row.
        straddles = (y1 > ys) != (y2 > ys)
        if not np.any(straddles):
            continue
        # x-coordinate where the edge crosses the pixel's row.
        # Guard the horizontal-edge case (y2 == y1): straddles is False
        # there, so the division is never evaluated on those pixels.
        with np.errstate(divide="ignore", invalid="ignore"):
            x_int = (x2 - x1) * (ys - y1) / (y2 - y1) + x1
        crossings += (straddles & (xs < x_int)).astype(int)
    mask = (crossings % 2) == 1
    return mask


def _upper_half_masks(genome: np.ndarray) -> dict[str, np.ndarray]:
    """Rasterize each family's polygons into a 128x64 upper-half mask."""
    coords = _genome_coords(genome)
    masks: dict[str, np.ndarray] = {}
    names = ("positive", "negative", "neutral")
    for fam in range(N_FAMILIES):
        mask = np.zeros((UPPER_HALF_H, ENGINE_SIZE), dtype=bool)
        for poly in range(POLYGONS_PER_FAMILY):
            pts = coords[fam, poly]  # (5, 2), normalized [0, 1]
            pts_px = np.column_stack(
                [pts[:, 0] * ENGINE_SIZE, pts[:, 1] * UPPER_HALF_H]
            )
            mask |= rasterize_polygon(pts_px, UPPER_HALF_H, ENGINE_SIZE)
        masks[names[fam]] = mask
    return masks


def decode_genome(genome: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode a genome vector to the 256x256 engine canvas.

    Returns (canvas, info) where canvas is a uint8 (256, 256, 3) array in
    BGR channel order (positive = channel 2, negative = channel 0,
    neutral = channel 1) and info carries the resolved multiplier, the
    negative voltage, and the number of pixels where families overlapped
    (resolved by priority positive > negative > neutral).
    """
    genome_arr = np.asarray(genome, dtype=float).reshape(-1)
    if genome_arr.shape[0] != GENOME_DIM:
        raise ValueError(
            f"genome must have {GENOME_DIM} genes, got {genome_arr.shape[0]}"
        )
    score = float(np.clip(genome_arr[-1], 0.0, 1.0))
    multiplier = voltage_score_to_multiplier(score)
    v_neg = -multiplier * V_POS

    upper = _upper_half_masks(genome_arr)

    # Mirror the upper half across the horizontal centerline (the x-axis)
    # to get the full 128x128 engine masks.  The centerline lies between
    # rows 63 and 64; the lower half is the vertical flip of the upper.
    full: dict[str, np.ndarray] = {
        name: np.concatenate([m, m[::-1, :]], axis=0)
        for name, m in upper.items()
    }

    # Combine with priority positive > negative > neutral.  Use a label
    # array so each pixel belongs to EXACTLY one family (setting channels
    # independently would leave overlapping pixels with two channels).
    pos, neg, neu = full["positive"], full["negative"], full["neutral"]
    overlap_px = int(np.count_nonzero((pos & neg) | (pos & neu) | (neg & neu)))
    if overlap_px > 0:
        warnings.warn(
            f"genome polygons overlap on {overlap_px} px; "
            "resolved by priority positive > negative > neutral",
            UserWarning,
            stacklevel=2,
        )

    labels = np.zeros((ENGINE_SIZE, ENGINE_SIZE), dtype=np.uint8)
    labels[neu] = 1
    labels[neg] = 2
    labels[pos] = 3

    engine = np.zeros((ENGINE_SIZE, ENGINE_SIZE, 3), dtype=np.uint8)
    engine[labels == 1, CH_NEUTRAL] = 255
    engine[labels == 2, CH_NEGATIVE] = 255
    engine[labels == 3, CH_POSITIVE] = 255

    canvas = np.zeros((EVAL_SIZE, EVAL_SIZE, 3), dtype=np.uint8)
    canvas[
        ENGINE_OFFSET : ENGINE_OFFSET + ENGINE_SIZE,
        ENGINE_OFFSET : ENGINE_OFFSET + ENGINE_SIZE,
        :,
    ] = engine

    info = {
        "multiplier": multiplier,
        "v_neg": v_neg,
        "voltage_score": score,
        "overlap_px": overlap_px,
    }
    return canvas, info


def canvas_to_rgb(canvas: np.ndarray) -> np.ndarray:
    """Convert a BGR-order canvas to RGB order for saving / display."""
    return canvas[:, :, ::-1].copy()


def save_png(canvas: np.ndarray, path: str) -> None:
    """Save a BGR-order canvas as a standard RGB PNG."""
    import matplotlib.pyplot as plt

    plt.imsave(path, canvas_to_rgb(canvas))
