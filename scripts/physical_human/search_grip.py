"""Bounded cross-entropy search for a smooth contact policy residual.

This is sample-based policy optimization in native physics, not PPO or motion
capture. Only eight MCP equilibrium offsets are learned. The body/wrist
controller, anatomy, mechanism, collision geometry and acceptance gates remain
fixed. Every candidate is a full floating-body/contact rollout.
"""

import argparse
import contextlib
import io
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.physical_human.prototype import run


def evaluate(job):
    directory, residual = job
    with contextlib.redirect_stdout(io.StringIO()):
        report = run(directory, grip_residual=residual)
    phases = list(report["grasp"]["phases"].values())
    fractions = np.array([p["opposed_fraction"] for p in phases])
    sides = min(
        min(
            p["four_fingers_together_fraction"],
            p["thumb_opposite_side_fraction"],
            p["thumb_below_fraction"],
        )
        for p in phases
    )
    physics_ok = all(
        v
        for k, v in report["quality_checks"].items()
        if k != "opposing_thumb_and_finger_contact"
    )
    score = (
        3 * fractions.mean()
        + 3 * fractions.min()
        - 30 * (1 - sides)
        - 10 * (not physics_ok)
    )
    # Small regularizer favors a minor correction to the anatomical preshape.
    score -= 0.02 * float(np.sum(np.square(residual)))
    return {
        "directory": str(directory),
        "residual_rad": residual,
        "score": float(score),
        "opposed_fractions": fractions.tolist(),
        "all_sides_fraction": sides,
        "physics_ok": physics_ok,
        "passed": report["quality_passed"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=701)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    mean = np.array([[0.015, 0.020, 0.010, 0.005], [0.030, 0.045, 0.020, 0.005]])
    sigma = np.full((2, 4), 0.025)
    history = []
    best = None
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for generation in range(args.generations):
            samples = np.clip(
                rng.normal(mean, sigma, size=(args.population, 2, 4)), -0.08, 0.12
            )
            samples[0] = mean if best is None else best["residual_rad"]
            jobs = [
                (args.out / f"g{generation:02}-c{i:02}", candidate.tolist())
                for i, candidate in enumerate(samples)
            ]
            batch = []
            for result in pool.imap_unordered(evaluate, jobs):
                batch.append(result)
                print(json.dumps(result), flush=True)
            history.extend(batch)
            elite = sorted(batch, key=lambda r: r["score"], reverse=True)[
                : max(3, args.population // 4)
            ]
            best = max(history, key=lambda r: r["score"])
            values = np.array([r["residual_rad"] for r in elite])
            mean = 0.75 * values.mean(axis=0) + 0.25 * mean
            sigma = np.maximum(0.006, values.std(axis=0))
            (args.out / "search.json").write_text(
                json.dumps(
                    {
                        "method": "cross-entropy policy residual search",
                        "seed": args.seed,
                        "best": best,
                        "candidates": history,
                    },
                    indent=2,
                )
            )
            if best["passed"] and min(best["opposed_fractions"]) > 0.96:
                break
    print("BEST " + json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
