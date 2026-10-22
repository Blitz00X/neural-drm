"""Run the smooth tent-map sample sweep."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.expansive.tent import empirical_lambda, key_params, random_states, rollout_norm
from src.expansive.tent_student import (
    eval_tent,
    make_pairs,
    rollout_states,
    t95_state,
    train_tent,
)
from src.io import save_json


def run_one(cfg: dict, n: int, seed: int) -> dict:
    started = time.time()
    a, offsets = key_params(cfg["key"], 1)
    x, y = make_pairs(cfg["key"], n, seed * 1000 + n, a, offsets)
    xt, yt = make_pairs(cfg["key"], cfg["eval_pairs"], 900000 + seed, a, offsets)
    model, rec = train_tent(
        x,
        y,
        steps=cfg["train_steps"],
        batch=cfg["batch"],
        lr=cfg["lr"],
        seed=seed,
        hidden=cfg["hidden"],
        depth=cfg["depth"],
    )
    one = eval_tent(model, xt, yt)
    s0 = random_states(cfg["rollout_states"], 7000 + seed)
    exact = rollout_norm(s0, cfg["rollout_T"], a, offsets)
    pred = rollout_states(model, s0, cfg["rollout_T"])
    return {
        "n": n,
        "seed": seed,
        "rmse": one["rmse"],
        "mae": one["mae"],
        "top_grid_accuracy": one["top_grid_accuracy"],
        "t95_state": t95_state(pred, exact, cfg["threshold"]),
        "lambda": empirical_lambda(a, offsets, n=500, steps=8, seed=seed),
        "training_seconds": rec["seconds"],
        "total_seconds": time.time() - started,
        "hidden": cfg["hidden"],
        "depth": cfg["depth"],
        "steps": cfg["train_steps"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tent.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["out"])
    runs = root / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for n in cfg["sizes"]:
        for seed in cfg["seeds"]:
            uid = f"n{n}_seed{seed}"
            path = runs / f"{uid}.json"
            if path.exists():
                print("skip", uid, flush=True)
                continue
            row = run_one(cfg, n, seed)
            save_json(path, row)
            print(
                json.dumps(
                    {
                        "uid": uid,
                        "rmse": round(row["rmse"], 6),
                        "t95": row["t95_state"],
                        "sec": round(row["total_seconds"], 1),
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
