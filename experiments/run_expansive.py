"""Run the keyed-CA distillation sweep.

Each run trains a CNN student on one-step (grid_t, grid_{t+1}) pairs and then
rolls it out autonomously for T steps.  T95 is the first step where the
cross-state mean cell accuracy falls below the threshold.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.expansive.ca import empirical_lambda, random_grids, rollout, rule_table
from src.expansive.ca_student import (
    cell_tv,
    eval_ca,
    make_pairs_ca,
    rollout_ca,
    t95,
    train_ca,
)
from src.io import save_json


def run_one(cfg: dict, k: int, n: int, seed: int) -> dict:
    started = time.time()
    table = rule_table(cfg["key"], k, 0)
    x, y = make_pairs_ca(table, k, n, seed=seed * 1000 + n)
    xt, yt = make_pairs_ca(table, k, cfg["eval_pairs"], seed=900_000 + seed)
    model, rec = train_ca(
        x,
        y,
        k,
        steps=cfg["train_steps"],
        batch=cfg["batch"],
        lr=cfg["lr"],
        seed=seed,
        channels=cfg["channels"][str(k)],
        depth=cfg["depth"],
    )
    one = eval_ca(model, xt, yt)
    g0 = random_grids(cfg["rollout_states"], seed=7000 + seed)
    exact = rollout(g0, cfg["rollout_T"], table, k)
    pred = rollout_ca(model, g0, cfg["rollout_T"])
    return {
        "k": k,
        "n": n,
        "seed": seed,
        "one_step_accuracy": one["cell_accuracy"],
        "one_step_bce": one["bce"],
        "t95": t95(pred, exact, cfg["threshold"]),
        "distribution_tv": cell_tv(pred, exact),
        "exact_state_accuracy": float((pred == exact).all(axis=(2, 3)).mean()),
        "empirical_lambda": empirical_lambda(table, k, n=200, steps=8, seed=seed),
        "training_seconds": rec["seconds"],
        "total_seconds": time.time() - started,
        "channels": cfg["channels"][str(k)],
        "steps": cfg["train_steps"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/expansive.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["out"])
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for k in cfg.get("ks", (cfg["control_k"], cfg["main_k"])):
        for n in cfg["sizes"]:
            for seed in cfg["seeds"]:
                uid = f"k{k}_n{n}_seed{seed}"
                path = runs / f"{uid}.json"
                if path.exists():
                    print("skip", uid, flush=True)
                    continue
                row = run_one(cfg, k, n, seed)
                save_json(path, row)
                print(
                    json.dumps(
                        {
                            "uid": uid,
                            "acc": round(row["one_step_accuracy"], 4),
                            "t95": row["t95"],
                            "tv": round(row["distribution_tv"], 4),
                            "sec": round(row["total_seconds"], 1),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
