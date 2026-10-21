"""Single entry point for the decision experiment.

    python run_decision.py --phase all

The output directory is append-only: every unit is written once and reused on
resume.  GPU-seconds are charged to the 8-hour budget in the configuration.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json
import platform
import traceback
from pathlib import Path

import numpy as np
import torch

from src.decision.orchestrator import (
    BudgetExhausted,
    Store,
    build_summary,
    run_attacks,
    run_c_cost,
    run_ladder,
    run_octave_gate,
)
from src.decision.report import (
    figure_costs,
    figure_finetune,
    figure_hamming,
    write_verdict,
)
from src.decision.verdict import compute_verdict
from src.io import save_json


def _load_state(store: Store, key: str, default=None):
    return store.state["decisions"].get(key, default)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/decision.json")
    parser.add_argument("--output", default="results/decision_v1")
    parser.add_argument(
        "--phase",
        default="all",
        choices=["all", "gate", "ladder", "c", "attacks", "report"],
    )
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    if cfg["device"] == "cuda" and not torch.cuda.is_available():
        raise SystemExit("configuration requires CUDA but torch.cuda is unavailable")
    torch.set_num_threads(int(cfg["threads"]))
    store = Store(args.output, budget_seconds=cfg["budget_gpu_hours"] * 3600)
    Path(args.output).mkdir(parents=True, exist_ok=True)
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory_gb": (
            torch.cuda.get_device_properties(0).total_memory / 2**30
            if torch.cuda.is_available()
            else None
        ),
        "platform": platform.platform(),
    }
    save_json(Path(args.output) / "environment.json", env)

    incomplete = None
    try:
        if args.phase in ("all", "gate"):
            octaves = run_octave_gate(cfg, store)
        else:
            octaves = _load_state(store, "octaves")
            if octaves is None:
                raise RuntimeError("run --phase gate first")

        # C is cheap and supplies the rule-4 ceiling; run it before the long
        # ladder so a budget stop cannot leave the verdict without it.
        if args.phase in ("all", "c"):
            c_cost = run_c_cost(cfg, store, octaves)
            store.state["decisions"]["c_cost"] = c_cost
            store.save_state()
        else:
            c_cost = _load_state(store, "c_cost")
            if c_cost is None:
                # A ladder-only resume is allowed; the verdict stays
                # INCONCLUSIVE until the C phase supplies the rule-4 ceiling.
                c_cost = {"success": False, "min_n": None, "gpu_hours": None}

        if args.phase in ("all", "ladder"):
            ladder, accepted = run_ladder(cfg, store, octaves)
            store.state["decisions"]["ladder"] = ladder
            store.state["decisions"]["accepted"] = accepted
            store.state["decisions"]["ladder_complete"] = True
            store.save_state()
        else:
            ladder = _load_state(store, "ladder", [])
            accepted = _load_state(store, "accepted")

        if args.phase in ("all", "attacks") and accepted is not None:
            attacks = run_attacks(cfg, store, accepted)
            store.state["decisions"]["attacks"] = attacks
            store.save_state()
        else:
            attacks = _load_state(store, "attacks")
    except BudgetExhausted as exc:
        incomplete = str(exc)
        if "ladder_complete" not in store.state["decisions"]:
            store.state["decisions"]["ladder_complete"] = False
        store.save_state()
        ladder = _load_state(store, "ladder", [])
        accepted = _load_state(store, "accepted")
        c_cost = _load_state(store, "c_cost") or {
            "success": False,
            "min_n": None,
            "gpu_hours": None,
        }
        attacks = _load_state(store, "attacks")
        octaves = _load_state(store, "octaves")
        save_json(Path(args.output) / "INCOMPLETE.json", {"reason": incomplete})

    summary = build_summary(
        cfg,
        store,
        int(octaves),
        ladder or [],
        accepted,
        attacks,
        c_cost or {"success": False, "min_n": None, "gpu_hours": None},
    )
    if incomplete:
        summary["incomplete"] = incomplete
    save_json(Path(args.output) / "summary.json", summary)

    verdict = compute_verdict(summary)
    save_json(Path(args.output) / "verdict.json", verdict)

    figures = Path("figures")
    figures.mkdir(exist_ok=True)
    figure_finetune(summary, figures / "decision_finetune_vs_n.png")
    figure_costs(summary, figures / "decision_costs.png")
    figure_hamming(summary, figures / "decision_hamming.png")
    write_verdict(summary, verdict, Path("VERDICT.md"))

    print(json.dumps(
        {
            "verdict": verdict["label"],
            "rule": verdict["rule"],
            "gpu_hours": summary["budget"]["gpu_hours"],
            "octaves": summary["octaves"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
