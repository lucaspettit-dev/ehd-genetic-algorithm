"""Constants for the EHD genetic algorithm.

All tunable physical and geometric constants live here so there is exactly
one place to change them.  They are documented in the README as well.
"""

# ---------------------------------------------------------------------------
# Geometry: evaluation canvas, engine, symmetry
# ---------------------------------------------------------------------------

# Evaluation image: square RGB (BGR channel order, fast-flow / cv2 convention),
# uint8.  The engine sits centered in this canvas; the surrounding border is
# free air (black) so the plume / wake can develop outside the engine.
EVAL_SIZE = 256

# Engine bounds (the design region).  The engine is a 128x128 square placed
# centered in the 256x256 evaluation canvas, i.e. at offset (64, 64).
ENGINE_SIZE = 128
ENGINE_OFFSET = (EVAL_SIZE - ENGINE_SIZE) // 2  # 64

# The engine is symmetric across the x-axis (the horizontal centerline, the
# long axis: inlet left, outlet right).  The genome encodes only the UPPER
# half; the decoder mirrors it to the full engine.  The upper half is the
# top ENGINE_SIZE//2 rows of the engine.
UPPER_HALF_H = ENGINE_SIZE // 2  # 64

# ---------------------------------------------------------------------------
# Genome layout (fixed-length vector for CMA-ES)
# ---------------------------------------------------------------------------

# Three polygon families, matching the fast-flow jet image convention:
#   positive (emitter) = RED,   negative (collector) = BLUE,
#   neutral (obstacle) = GREEN (uncharged solid).
N_FAMILIES = 3
FAMILY_NAMES = ("positive", "negative", "neutral")

# Channel indices in the image array.  The array uses BGR channel order
# (fast-flow / cv2 convention): index 2 is red, index 0 is blue, index 1
# is green.
CH_NEGATIVE = 0  # blue
CH_NEUTRAL = 1   # green
CH_POSITIVE = 2  # red

# Genome: for each family, a FIXED number of polygons; each polygon a FIXED
# number of control points; each control point 2 coords (x, y) normalized to
# [0, 1] within the 128x64 upper half.  Plus one voltage-score gene in
# [0, 1] as the last element.
POLYGONS_PER_FAMILY = 2
POINTS_PER_POLYGON = 5
GENOME_DIM = N_FAMILIES * POLYGONS_PER_FAMILY * POINTS_PER_POLYGON * 2 + 1  # 61

# ---------------------------------------------------------------------------
# Voltage model
# ---------------------------------------------------------------------------

# Positive voltage is FIXED.  The positive terminal is the only ion source.
V_POS = 20_000.0  # volts

# Negative voltage = -(multiplier * V_POS), where the multiplier is an
# integer 1..4 resolved from the genome's voltage-score gene via
# round(1 + 3 * score) with 0.5 rounding UP (not Python banker's rounding).
MULT_MIN = 1
MULT_MAX = 4

# ---------------------------------------------------------------------------
# Physics: arcing threshold and drift-velocity clamp
# ---------------------------------------------------------------------------

# Pixel size: 1 px = 1 mm.  Used to convert the pixel gap between the
# terminals into meters for the field computation.
PIXEL_SIZE_M = 1e-3

# Air breakdown field.  If the field across the terminal gap exceeds this,
# the design arcs and fails before a run (pre-sim feasibility check).
E_BREAKDOWN = 3.0e6  # V/m

# Ion mobility in air.  The drift-velocity clamp bound is v_max = MU * E:
# air cannot outrun the ions dragging it, so a simulated velocity above
# this bound is a numerical artifact and is clamped to the bound.
MU = 2.0e-4  # m^2 / (V s)

# ---------------------------------------------------------------------------
# GA / evaluator
# ---------------------------------------------------------------------------

# Fitness assigned to infeasible genomes (pre-sim filter failure).  The GA
# loop converts an infeasible EvalResult to this penalty; the evaluator
# itself returns fitness=None for infeasible designs.
INFEASIBLE_FITNESS = -1e9

# ---------------------------------------------------------------------------
# Penalty shaping (graded infeasibility penalty, see README)
# ---------------------------------------------------------------------------

# Gaussian blur sigma (px) for the smooth short-circuit penalty.  The
# blurred masks overlap smoothly as two electrodes approach, giving
# CMA-ES a gradient all the way to separation.  A raw contact-pixel
# count is discontinuous (it jumps from 2 to 0) and stalls the search
# a few pixels short of feasibility.
PENALTY_BLUR_SIGMA = 2.0
# Scale for the blurred-overlap penalty term.
PENALTY_OVERLAP_SCALE = 10.0

# Proxy simulator scale factor.  The proxy is an analytic placeholder, NOT
# physics: v = PROXY_K * E * outlet_open.  PROXY_K is arbitrary, chosen so
# proxy velocities land in a plausible 0..~3 m/s range.  It must be replaced
# when the real fast-flow simulator is wired in (see README TODO).
PROXY_K = 1e-6
