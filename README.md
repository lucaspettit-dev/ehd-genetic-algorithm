# EHD Genetic Algorithm

Genetic algorithm that designs an EHD (electrohydrodynamic / ion-wind)
propulsion engine. The genome encodes a 2D engine slice; a stubbed
`Evaluator` scores it; CMA-ES searches for the design with the highest
average airflow exiting on the right side.

Pure Python + numpy / scipy / matplotlib. No cv2, no Cython.

## Design (the five adjustments)

1. **Longways slice, like the fast-flow jet image.** The engine is a 2D
   slice with the **inlet on the left** and the **outlet on the right**.
2. **Symmetry across the x-axis.** The genome encodes **only the upper
   half** (128×64). The decoder mirrors it across the horizontal
   centerline to get the full 128×128 engine.
3. **Evaluate in a 2× space.** A **128×128 engine** is evaluated
   **centered in a 256×256 image** (offset 64, 64). The surrounding border
   is free air (black) so the plume / wake and turbulence outside the
   engine are part of the evaluation.
4. **Fitness = highest average airflow exiting on the right side** = mean
   horizontal velocity (u) at the right edge of the 256×256 domain.
5. **Three polygon families** (fast-flow jet image convention, BGR
   channel order):
   - **Positive = RED** (channel 2) — the emitter, the only ion source.
   - **Negative = BLUE** (channel 0) — the collector.
   - **Neutral = GREEN** (channel 1) — uncharged solid obstacle.

## Genome

Fixed-length vector, 61 genes, all in [0, 1], for CMA-ES:

- Per family (3): **2 polygons**, each with **5 control points**, each
  point 2 coords (x, y) normalized within the 128×64 upper half.
  3 × 2 × 5 × 2 = 60 coordinate genes.
- Gene 60: the **voltage score** in [0, 1].

The decoder rasterizes each family's polygons into a 128×64 mask, mirrors
to 128×128, combines with priority **positive > negative > neutral**
(overlaps are counted and warned about, not silently hidden), and places
the engine centered in the 256×256 canvas.

## Voltage model

- **Positive voltage is FIXED: V_POS = 20,000 V.** Positive is the only
  ion source.
- **Negative voltage: multiplier = round(1 + 3 × score)**, an integer
  **1–4**, with 0.5 rounding UP (not Python's banker's rounding).
  **V_NEG = −multiplier × V_POS** (−20 kV, −40 kV, −60 kV, −80 kV).
  Example: score 0.6 → round(2.8) = 3 → V_NEG = −60,000 V.
- The negative terminal only sets how hard the ions are drawn toward it
  once created.

## Feasibility filter (pre-sim)

A genome must pass all three checks or it is infeasible (fitness penalty,
no simulation run):

1. **No short circuit.** The positive mask dilated by 1 px
   (8-connectivity) must not intersect the negative mask.
2. **Connectivity.** Positive = exactly 1 connected component in the
   **upper half** (8-connectivity); negative = exactly 1. (The mirrored
   full engine legitimately has two components — upper + mirror — for a
   polygon not touching the centerline; the upper half is what the
   genome defines.) Neutral has **no** connectivity requirement and
   **may touch either electrode** (it is an insulator).
3. **Arcing (voltage-dependent gap).** gap_min = minimum Euclidean
   distance (px) between the positive and negative masks;
   gap_m = gap_px × PIXEL_SIZE_M; field E = V_total / gap_m with
   V_total = V_POS + |V_NEG|. If **E > E_BREAKDOWN**, the design arcs
   and fails before a run. Higher voltage needs a wider gap.

**Graded infeasibility penalty (deviation from the original spec, made
deliberately).** The spec said infeasible → flat fitness −1e9. Measured:
0/500 uniform random genomes and 0/320 CMA-ES samples were feasible —
two random 5-point polygons almost always overlap — so a flat penalty
gives CMA-ES no gradient and the search wanders forever in the
infeasible region. Instead, infeasible fitness =
**−1e9 − penalty**, where the penalty grows with the violation:
smooth Gaussian-blurred mask overlap for short circuits (×10),
extra components (×10 each — deliberately small so the search can pass
through briefly-split polygons on its way to separation), empty mask
(10000). Arcing contributes NO penalty term: an arcing penalty, however
small, creates a cliff at the separation boundary (just-separated-but-
arcing scores worse than just-touching, so the search never crosses
into separation). The arcing check remains a hard filter, and the −1e9
base penalty still distinguishes feasible (fitness ≥ 0) from arcing
(fitness −1e9). A design closer to feasible always scores above a
worse one, while every infeasible design still scores below every
feasible one (feasible fitnesses are ≥ 0, penalties are ≪ 1e9). The
ordering the spec intended is preserved; the search now has a gradient
toward feasibility. The short-circuit penalty is the *blurred* overlap,
not a contact-pixel count: a contact count is discontinuous (it jumps
from 2 to 0) and stalls the search a few pixels short of feasibility.

Note: the field across the gap changes slightly when air flows (the flow
advects the space charge, which reshapes the field via Poisson), and the
peak field near the collector is *higher* with flow than the static
V/gap estimate — so flow does **not** justify a smaller gap. The static
check stays conservative.

## Drift-velocity clamp

After each simulation run, the fitness is clamped to the hard physical
upper bound **v_max = MU × E** (same E as the arcing check): air cannot
outrun the ions dragging it. If the simulated mean right-edge velocity
exceeds v_max, the fitness is clamped to v_max and the event is logged
(raw value, bound, design id) — it indicates a numerical artifact, not
real performance.

**No pre-run pruning / skip rule yet.** The drift bound is a true upper
bound but far too loose to prune on (it overestimates achievable airflow
by ~100×). Pruning waits for calibration data to make the bound tight.

## Constants (all in `ehd_ga/constants.py`)

| Constant | Value | Meaning |
|---|---|---|
| `EVAL_SIZE` | 256 | Evaluation canvas (px) |
| `ENGINE_SIZE` | 128 | Engine bounds (px) |
| `PIXEL_SIZE_M` | 1e-3 | 1 px = 1 mm |
| `V_POS` | 20,000 V | Fixed positive voltage |
| `MULT_MIN` / `MULT_MAX` | 1 / 4 | Negative multiplier range |
| `E_BREAKDOWN` | 3.0e6 V/m | Air breakdown field (arcing) |
| `MU` | 2.0e-4 m²/(V·s) | Ion mobility (clamp) |
| `PROXY_K` | 1e-6 | Proxy simulator scale (arbitrary) |

## Optimizer

**CMA-ES** (pycma) over the 61-dim genome, all genes bounded [0, 1],
initial mean 0.5, initial sigma 0.2, population
λ = 4 + ⌊3 ln n⌋ = 16. Each generation: sample → decode + feasibility
(infeasible → penalty, no simulation) → evaluate feasible ones in
parallel (`ProcessPoolExecutor`, spawn-safe top-level worker) → clamp →
tell CMA-ES → log → checkpoint (JSON: best genome + fitness +
generation, written every generation to `checkpoints/`).

## Usage

```bash
pip install -e .

# Run the GA (50 generations, 4 workers)
python -m ehd_ga run --max-generations 50 --workers 4 --seed 0 \
    --checkpoint-dir checkpoints

# Decode one genome to a PNG for inspection
python -m ehd_ga decode --genome genome.json --out decoded.png
python -m ehd_ga decode --random --seed 0 --out decoded.png
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## TODO: real simulator

The `Evaluator` is **stubbed**. The default `ProxySimulator` is an
analytic placeholder — `v = PROXY_K × E × outlet_open` — documented as
**NOT physics**, so the GA loop runs end-to-end. To run real physics,
replace `simulate()` with a call into the
[fast-flow](https://github.com/lucaspettit-dev/fast-flow) solver:
build the 256×256 image, use layermap `{2: V_POS, 0: V_NEG}` (red
positive, blue negative), run the solver, and return the mean u at the
right edge. The `Evaluator` accepts any `simulator` callable with
signature `simulator(canvas, v_pos, v_neg, gap_m) -> float`.
