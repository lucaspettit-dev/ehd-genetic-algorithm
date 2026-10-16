"""Evaluator: feasibility filter, stubbed simulation, drift-velocity clamp.

Pipeline for one genome
------------------------
1. Decode the genome to the 256x256 canvas (see genome.py).
2. Pre-sim feasibility filter (no simulation is run on failure):
   a. No short circuit: the positive mask dilated by 1 px (8-connectivity)
      must not intersect the negative mask.
   b. Connectivity: the positive mask must be exactly 1 connected
      component (8-connectivity), and the negative mask exactly 1.
      The neutral mask has NO connectivity requirement and may touch
      either electrode (it is an insulator).
   c. Arcing (voltage-dependent gap): gap_min = minimum Euclidean
      distance in pixels between the positive and negative masks,
      converted to meters (gap_m = gap_px * PIXEL_SIZE_M).
      Field E = V_total / gap_m with V_total = V_POS + |V_NEG|.
      If E > E_BREAKDOWN the design arcs and fails before a run.
3. If feasible, run the simulator to get the raw mean right-edge airflow
   velocity (mean u at the right edge of the 256x256 domain), then apply
   the drift-velocity clamp: v_max = MU * E (same E as the arcing check).
   Air cannot outrun the ions dragging it, so a simulated velocity above
   v_max is a numerical artifact: clamp the fitness to v_max and log it.

The simulator is STUBBED.  By default the evaluator uses ProxySimulator,
an analytic placeholder documented as NOT physics, so the GA loop runs
end-to-end.  To run real physics, pass a simulator callable that invokes
the fast-flow solver (see the TODO in the README) with signature
``simulator(canvas, v_pos, v_neg, gap_m) -> float``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import ndimage

from .constants import (
    CH_NEGATIVE,
    CH_POSITIVE,
    E_BREAKDOWN,
    MU,
    PENALTY_BLUR_SIGMA,
    PENALTY_OVERLAP_SCALE,
    PIXEL_SIZE_M,
    PROXY_K,
    V_POS,
)
from .genome import decode_genome

logger = logging.getLogger(__name__)

# Signature of a simulator: (canvas, v_pos, v_neg, gap_m) -> raw velocity.
SimulatorFn = Callable[[np.ndarray, float, float, float], float]


@dataclass
class EvalResult:
    """Outcome of evaluating one genome."""

    fitness: float | None  # None when infeasible (GA maps it to a penalty)
    feasible: bool
    reason: str  # "" when feasible, otherwise why it failed
    raw_velocity: float | None = None  # simulated mean right-edge velocity
    bound: float | None = None  # drift-velocity clamp bound v_max = MU * E
    clamped: bool = False  # True if raw_velocity exceeded the bound
    multiplier: int | None = None
    v_neg: float | None = None
    gap_px: float | None = None
    field_e: float | None = None  # V/m across the terminal gap
    overlap_px: int = 0
    penalty: float = 0.0  # >= 0; 0 iff feasible.  Graded infeasibility
    # penalty (see below): infeasible GA fitness = -1e9 - penalty, so a
    # design closer to feasible always scores above a worse one, while
    # every infeasible design still scores below every feasible one.


def check_feasibility(
    pos_mask: np.ndarray, neg_mask: np.ndarray, v_neg: float
) -> dict:
    """Pre-sim feasibility filter on the engine-sized (128x128) masks.

    Connectivity is checked on the UPPER HALF (rows 0..h//2): the genome
    encodes the upper half, and the mirrored full engine legitimately has
    two components (upper + mirror) for a polygon that does not touch the
    centerline.  Short-circuit and gap are checked on the FULL mask,
    because the mirror can bring an upper electrode and a mirrored lower
    electrode together across the centerline.

    The neutral mask is deliberately NOT an input: neutral obstacles have
    no connectivity requirement and may touch either electrode.

    Returns a dict with keys: feasible (bool), reason (str),
    gap_px (float|None), field_e (float|None), penalty (float >= 0).

    Penalty (graded, see README): 0 iff feasible.  Otherwise it grows
    with the violation — smooth blurred-overlap for short circuits,
    extra components (x10 each — deliberately small so the search can
    pass through briefly-split polygons on its way to separation),
    empty mask (10000).  Arcing contributes NO penalty term: an arcing
    penalty, however small, creates a cliff at the separation boundary
    and blocks the search from crossing into separation.  The arcing
    check remains a hard feasibility filter.  The short-circuit penalty is the
    Gaussian-blurred mask overlap (smooth), NOT a contact-pixel count:
    a contact count is discontinuous (it jumps from 2 to 0) and stalls
    the search a few pixels short of feasibility.
    NOTE: feasibility is tracked as a separate HARD flag from the
    checks below (contact, connectivity, arcing), not derived from
    ``penalty == 0``.  The penalty is a gradient signal for the search;
    feasibility is a hard decision.  Keeping them separate means a
    future penalty term can be nonzero for a feasible design (e.g. a
    small smoothness term) without wrongly rejecting it.
    """
    structure = np.ones((3, 3), dtype=int)
    penalty = 0.0
    feasible = True
    reason = ""
    gap_px: float | None = None
    field_e: float | None = None

    pos_empty = not pos_mask.any()
    neg_empty = not neg_mask.any()
    if pos_empty:
        penalty += 10_000.0
        feasible = False
        reason = reason or "empty positive mask"
    if neg_empty:
        penalty += 10_000.0
        feasible = False
        reason = reason or "empty negative mask"

    # (a) Short circuit: count negative pixels touching the 1-px-dilated
    # positive mask (full mask — the mirror can short across the
    # centerline).  The HARD feasibility decision uses the contact count,
    # but the PENALTY uses the smooth Gaussian-blurred mask overlap, so
    # the gradient does not vanish a few pixels short of separation.
    contact = 0
    if not pos_empty and not neg_empty:
        dilated_pos = ndimage.binary_dilation(pos_mask, structure=structure)
        contact = int(np.count_nonzero(dilated_pos & neg_mask))
        if contact > 0:
            feasible = False
            blurred = float(
                np.sum(
                    ndimage.gaussian_filter(
                        pos_mask.astype(float), PENALTY_BLUR_SIGMA
                    )
                    * ndimage.gaussian_filter(
                        neg_mask.astype(float), PENALTY_BLUR_SIGMA
                    )
                )
            )
            penalty += PENALTY_OVERLAP_SCALE * blurred
            reason = reason or "short circuit"

    # (b) Connectivity: exactly 1 component each in the UPPER HALF
    # (8-connectivity).  The mirror is a copy at the same potential.
    # The per-component penalty is deliberately SMALL (10): a large one
    # would wall off the path to feasibility, because separating two
    # polygons often passes through designs that briefly split a
    # polygon into 2 components.  The search must be able to pass
    # through those on its way to a separated, connected design.
    if not pos_empty:
        half = pos_mask.shape[0] // 2
        _, n_pos = ndimage.label(pos_mask[:half, :], structure=structure)
        if n_pos != 1:
            feasible = False
            penalty += 10.0 * (n_pos - 1)
            reason = reason or (
                f"positive mask has {n_pos} components (upper half)")
    if not neg_empty:
        half = neg_mask.shape[0] // 2
        _, n_neg = ndimage.label(neg_mask[:half, :], structure=structure)
        if n_neg != 1:
            feasible = False
            penalty += 10.0 * (n_neg - 1)
            reason = reason or (
                f"negative mask has {n_neg} components (upper half)")

    # (c) Arcing: minimum Euclidean distance between the masks, only
    # meaningful when they do not touch.  This is a HARD filter only:
    # it contributes NO penalty term.  An arcing penalty (however
    # small) creates a CLIFF at the separation boundary — just-separated
    # -but-arcing would score worse than just-touching, so CMA-ES
    # avoids crossing into separation at all.  Without an arcing
    # penalty, the search passes freely through the arcing regime
    # (penalty 0) until the gap is wide enough, and the -1e9 base
    # penalty still distinguishes feasible (fitness >= 0) from arcing
    # (fitness -1e9).  Measured: arcing penalty 10x -> 0 feasible in
    # 60 generations; arcing penalty 0 -> feasible by gen 35.
    if contact == 0 and not pos_empty and not neg_empty:
        dist_to_pos = ndimage.distance_transform_edt(~pos_mask)
        gap_px = float(dist_to_pos[neg_mask].min())
        gap_m = gap_px * PIXEL_SIZE_M
        v_total = V_POS + abs(v_neg)
        field_e = v_total / gap_m if gap_m > 0 else float("inf")
        if field_e > E_BREAKDOWN:
            feasible = False
            reason = reason or (
                f"arc: field {field_e:.3g} V/m across {gap_px:.1f} px "
                f"gap exceeds breakdown {E_BREAKDOWN:.3g} V/m")

    return {"feasible": feasible, "reason": reason,
            "gap_px": gap_px, "field_e": field_e, "penalty": penalty}


class ProxySimulator:
    """Analytic placeholder simulator — NOT physics.

    Returns v = PROXY_K * E * outlet_open, where E is the field across the
    terminal gap and outlet_open is the fraction of the engine's right
    edge (the outlet) that is air rather than solid.  This exists only so
    the GA loop runs end-to-end; it must be replaced by the fast-flow
    solver (see the README TODO).  The arcing filter bounds how small the
    gap can get, so the proxy optimum sits at the arcing boundary — that
    is an artifact of the placeholder, not a physical prediction.
    """

    def __call__(
        self, canvas: np.ndarray, v_pos: float, v_neg: float, gap_m: float
    ) -> float:
        v_total = v_pos + abs(v_neg)
        field_e = v_total / gap_m if gap_m > 0 else 0.0
        # Right edge of the engine = last engine column, centered rows.
        from .constants import ENGINE_OFFSET, ENGINE_SIZE

        outlet_col = canvas[
            ENGINE_OFFSET : ENGINE_OFFSET + ENGINE_SIZE,
            ENGINE_OFFSET + ENGINE_SIZE - 1,
            :,
        ]
        solid = outlet_col.any(axis=1)
        outlet_open = float(1.0 - solid.mean())
        return PROXY_K * field_e * outlet_open


class Evaluator:
    """Evaluate genomes: decode -> feasibility -> simulate -> clamp."""

    def __init__(
        self,
        simulator: SimulatorFn | None = None,
        design_id: str = "unknown",
    ) -> None:
        self.simulator = simulator if simulator is not None else ProxySimulator()
        self.design_id = design_id

    def evaluate_image(
        self, canvas: np.ndarray, v_neg: float, overlap_px: int = 0
    ) -> EvalResult:
        """Evaluate a pre-rendered 256x256 canvas at a given V_NEG."""
        from .constants import ENGINE_OFFSET, ENGINE_SIZE

        engine = canvas[
            ENGINE_OFFSET : ENGINE_OFFSET + ENGINE_SIZE,
            ENGINE_OFFSET : ENGINE_OFFSET + ENGINE_SIZE,
            :,
        ]
        pos_mask = engine[:, :, CH_POSITIVE] > 0
        neg_mask = engine[:, :, CH_NEGATIVE] > 0
        # v_neg is always -multiplier * V_POS, so the multiplier is exact.
        multiplier = int(round(abs(v_neg) / V_POS))
        check = check_feasibility(pos_mask, neg_mask, v_neg)
        if not check["feasible"]:
            return EvalResult(
                fitness=None,
                feasible=False,
                reason=check["reason"],
                multiplier=multiplier,
                v_neg=v_neg,
                gap_px=check["gap_px"],
                field_e=check["field_e"],
                overlap_px=overlap_px,
                penalty=check["penalty"],
            )

        gap_px = check["gap_px"]
        field_e = check["field_e"]
        gap_m = gap_px * PIXEL_SIZE_M
        raw = float(self.simulator(canvas, V_POS, v_neg, gap_m))

        # Drift-velocity clamp: air cannot outrun the ions dragging it.
        bound = MU * field_e
        clamped = False
        fitness = raw
        if raw > bound:
            logger.warning(
                "clamp: design %s raw velocity %.4g exceeds drift bound "
                "%.4g (mu*E); clamping fitness to bound",
                self.design_id, raw, bound,
            )
            fitness = bound
            clamped = True

        return EvalResult(
            fitness=fitness,
            feasible=True,
            reason="",
            raw_velocity=raw,
            bound=bound,
            clamped=clamped,
            multiplier=multiplier,
            v_neg=v_neg,
            gap_px=gap_px,
            field_e=field_e,
            overlap_px=overlap_px,
        )

    def evaluate(self, genome: np.ndarray) -> EvalResult:
        """Evaluate one genome vector."""
        canvas, info = decode_genome(genome)
        return self.evaluate_image(
            canvas, info["v_neg"], overlap_px=info["overlap_px"]
        )


# Backwards-compatible alias: the stub is named after its role.
StubEvaluator = Evaluator
