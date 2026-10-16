"""CMA-ES loop: sample genomes, evaluate in parallel, checkpoint, repeat.

Optimizer
---------
CMA-ES (pycma) over the 61-dim genome, all genes bounded to [0, 1],
initial mean 0.5 for every gene, initial sigma 0.2.  Population
lambda = 4 + floor(3 * ln(n)) unless overridden.  CMA-ES minimizes, so
feasible fitnesses are negated and infeasible genomes get a large
positive tell value (the graded infeasible penalty, negated).

Parallel evaluation uses ProcessPoolExecutor with a top-level worker
function and a per-worker Evaluator created by an initializer, so the
module is spawn-safe (no lambdas / closures cross the process boundary).
With workers=1 evaluation is serial (no pool), which is simplest for
tests and debugging.

Each generation: sample -> evaluate -> tell -> log -> checkpoint.
Checkpoint (JSON) is written every generation: best genome, best
fitness, generation, evals so far.
"""

from __future__ import annotations

import json
import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import numpy as np

from .constants import GENOME_DIM, INFEASIBLE_FITNESS
from .evaluator import EvalResult, Evaluator

logger = logging.getLogger(__name__)

# Per-worker evaluator, created by the pool initializer.
_WORKER_EVALUATOR: Evaluator | None = None


def _init_worker() -> None:
    global _WORKER_EVALUATOR
    _WORKER_EVALUATOR = Evaluator()


def _eval_worker(genome_list: list[float]) -> EvalResult:
    """Top-level worker: evaluate one genome (must stay picklable)."""
    assert _WORKER_EVALUATOR is not None
    return _WORKER_EVALUATOR.evaluate(np.asarray(genome_list, dtype=float))


def _to_fitness(result: EvalResult) -> float:
    """GA fitness: the evaluator's fitness, or the graded penalty if
    infeasible.  Infeasible fitness = INFEASIBLE_FITNESS - penalty, so a
    design closer to feasible always scores above a worse one, while
    every infeasible design still scores below every feasible one
    (feasible fitnesses are >= 0, penalties are << 1e9)."""
    if not result.feasible or result.fitness is None:
        return INFEASIBLE_FITNESS - result.penalty
    return float(result.fitness)


def _write_checkpoint(
    checkpoint_dir: str,
    generation: int,
    best_genome: list[float] | None,
    best_fitness: float,
    evals: int,
) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "checkpoint.json")
    payload = {
        "generation": generation,
        "best_fitness": best_fitness,
        "best_genome": best_genome,
        "evals": evals,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return path


def default_popsize(n: int = GENOME_DIM) -> int:
    """CMA-ES default population: lambda = 4 + floor(3 * ln(n))."""
    return 4 + int(math.floor(3.0 * math.log(n)))


def run_ga(
    max_generations: int = 50,
    max_evals: int | None = None,
    target: float | None = None,
    workers: int = 4,
    seed: int = 0,
    checkpoint_dir: str = "checkpoints",
    popsize: int | None = None,
    log_every: int = 1,
) -> dict[str, Any]:
    """Run the CMA-ES loop. Returns a summary dict.

    Stops when max_generations is reached, max_evals is reached, the best
    fitness reaches ``target`` (if given), or CMA-ES reports its own
    stop condition.
    """
    import cma

    lam = popsize if popsize is not None else default_popsize()
    es = cma.CMAEvolutionStrategy(
        [0.5] * GENOME_DIM,
        0.2,
        {
            "bounds": [0, 1],
            "popsize": lam,
            "seed": seed,
            "verbose": -9,
        },
    )
    # NOTE: the loop below deliberately does NOT consult es.stop().
    # CMA-ES's internal criteria (e.g. tolfun on a flat fitness landscape
    # when a whole generation is infeasible) would stop the run early.
    # The spec's stopping rules are max_generations / max_evals / target,
    # which bound the loop, so ignoring es.stop() cannot run forever.

    best_fitness = float("-inf")
    best_genome: list[float] | None = None
    evals = 0
    generation = 0

    executor: ProcessPoolExecutor | None = None
    if workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=workers, initializer=_init_worker
        )
    try:
        while True:
            if max_generations is not None and generation >= max_generations:
                break
            if max_evals is not None and evals >= max_evals:
                break

            solutions = es.ask()
            genomes = [list(map(float, s)) for s in solutions]
            if executor is not None:
                results = list(executor.map(_eval_worker, genomes))
            else:
                evaluator = Evaluator()
                results = [evaluator.evaluate(np.asarray(g)) for g in genomes]
            evals += len(results)

            fitnesses = [_to_fitness(r) for r in results]
            # CMA-ES minimizes: negate.  Infeasible -> large positive.
            es.tell(solutions, [-f for f in fitnesses])

            feasible = [f for f, r in zip(fitnesses, results) if r.feasible]
            gen_best = max(fitnesses) if fitnesses else float("-inf")
            if gen_best > best_fitness:
                best_fitness = gen_best
                best_genome = list(genomes[int(np.argmax(fitnesses))])
            generation += 1

            if generation % log_every == 0 or generation == 1:
                n_clamped = sum(1 for r in results if r.clamped)
                n_feasible = len(feasible)
                mean_feas = (
                    float(np.mean(feasible)) if feasible else float("nan")
                )
                logger.info(
                    "gen %d: best=%.4g mean_feasible=%.4g "
                    "feasible=%d/%d clamped=%d evals=%d",
                    generation, best_fitness, mean_feas,
                    n_feasible, len(results), n_clamped, evals,
                )
                print(
                    f"gen {generation}: best={best_fitness:.4g} "
                    f"mean_feasible={mean_feas:.4g} "
                    f"feasible={n_feasible}/{len(results)} "
                    f"clamped={n_clamped} evals={evals}",
                    flush=True,
                )

            _write_checkpoint(
                checkpoint_dir, generation, best_genome, best_fitness, evals
            )

            if target is not None and best_fitness >= target:
                print(f"target {target} reached at gen {generation}",
                      flush=True)
                break
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    return {
        "best_fitness": best_fitness,
        "best_genome": best_genome,
        "generations": generation,
        "evals": evals,
        "checkpoint_dir": checkpoint_dir,
    }
