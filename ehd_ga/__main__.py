"""CLI: run the GA, or decode one genome to a PNG.

Usage
-----
Run the GA::

    python -m ehd_ga run --max-generations 50 --workers 4 --seed 0 \\
        --checkpoint-dir checkpoints

Decode one genome to a PNG for inspection::

    python -m ehd_ga decode --genome genome.json --out decoded.png
    python -m ehd_ga decode --random --seed 0 --out decoded.png

The --genome file is JSON holding either a bare list of 61 numbers or an
object with a "genome" key (as written by the checkpoint).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np

from .constants import GENOME_DIM
from .evaluator import Evaluator
from .ga import run_ga
from .genome import decode_genome, save_png


def _load_genome(path: str) -> np.ndarray:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        # Accept a bare genome, a {"genome": [...]} file, or a checkpoint
        # ({"best_genome": [...], ...}).
        if "genome" in data:
            data = data["genome"]
        elif "best_genome" in data:
            data = data["best_genome"]
        else:
            raise ValueError(
                f"{path}: dict has neither 'genome' nor 'best_genome' key"
            )
    genome = np.asarray(data, dtype=float).reshape(-1)
    if genome.shape[0] != GENOME_DIM:
        raise ValueError(
            f"genome must have {GENOME_DIM} genes, got {genome.shape[0]}"
        )
    return genome


def _cmd_run(args: argparse.Namespace) -> int:
    result = run_ga(
        max_generations=args.max_generations,
        max_evals=args.max_evals,
        target=args.target,
        workers=args.workers,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        popsize=args.popsize,
    )
    print(
        f"done: generations={result['generations']} evals={result['evals']} "
        f"best_fitness={result['best_fitness']:.6g} "
        f"checkpoint={result['checkpoint_dir']}/checkpoint.json"
    )
    return 0


def _cmd_decode(args: argparse.Namespace) -> int:
    if args.genome:
        genome = _load_genome(args.genome)
    else:
        rng = np.random.default_rng(args.seed)
        genome = rng.random(GENOME_DIM)
    canvas, info = decode_genome(genome)
    save_png(canvas, args.out)

    evaluator = Evaluator()
    result = evaluator.evaluate_image(canvas, info["v_neg"],
                                      overlap_px=info["overlap_px"])
    print(f"wrote {args.out}")
    print(f"multiplier={info['multiplier']}x  V_NEG={info['v_neg']:.0f} V  "
          f"overlap_px={info['overlap_px']}")
    print(f"feasible={result.feasible} reason={result.reason!r} "
          f"gap_px={result.gap_px} field={result.field_e} "
          f"fitness={result.fitness}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(
        prog="ehd_ga",
        description="Genetic algorithm for EHD propulsion engine design.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the CMA-ES loop")
    p_run.add_argument("--max-generations", type=int, default=50)
    p_run.add_argument("--max-evals", type=int, default=None)
    p_run.add_argument("--target", type=float, default=None,
                       help="stop when best fitness reaches this value")
    p_run.add_argument("--workers", type=int, default=4)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--checkpoint-dir", default="checkpoints")
    p_run.add_argument("--popsize", type=int, default=None,
                       help="population size (default: 4 + floor(3 ln n))")
    p_run.set_defaults(func=_cmd_run)

    p_dec = sub.add_parser("decode", help="render one genome to a PNG")
    p_dec.add_argument("--genome", default=None,
                       help="JSON file with a genome (list or {genome: [...]})")
    p_dec.add_argument("--random", action="store_true",
                       help="decode a random genome instead of --genome")
    p_dec.add_argument("--seed", type=int, default=0)
    p_dec.add_argument("--out", default="decoded.png")
    p_dec.set_defaults(func=_cmd_decode)

    args = parser.parse_args(argv)
    if args.command == "decode" and not args.genome and not args.random:
        parser.error("decode needs --genome <json> or --random")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
