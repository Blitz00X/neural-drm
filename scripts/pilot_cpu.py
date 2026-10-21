"""CPU pilot at width 256, c=8, one seed.

This is a pipeline and early-signal check, NOT the decision run: one seed cannot
satisfy any 2/3 verdict rule.  It exercises the same unit and attack functions
the GPU ladder uses, on the CPU torch found on the backup drive.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.decision.orchestrator import Store, attack_seed, unit_A, unit_B  # noqa: E402
from src.io import save_json  # noqa: E402

CFG = json.loads(Path("configs/cpu_pilot.json").read_text())
# Pilot-only budgets: the GPU decision run keeps the full protocol.
CFG["attack"]["key_restarts"] = 3
CFG["attack"]["key_steps"] = 200
CFG["attack"]["finetune_sample_sizes"] = [100, 1000]
CFG["attack"]["hamming_repeats"] = 1

OUT = Path("results/cpu_pilot_256")


def main():
    store = Store(OUT, budget_seconds=CFG["budget_gpu_hours"] * 3600)
    rows = {}
    print("A", flush=True)
    rows["A"] = unit_A(CFG, store, 0, 8, 256, 20000)
    print("A", rows["A"]["test_f"], rows["A"]["test_g"], flush=True)

    for arm in ("B1", "B2"):
        print(arm, flush=True)
        row = unit_B(CFG, store, arm, 0, 8, 256, 20000)
        rows[arm] = row
        print(
            arm,
            "valid",
            row.get("test_valid"),
            "wrong",
            {k: {"r2": v["r2"], "acc": v["acc"]} for k, v in (row.get("wrong") or {}).items()},
            "accepted",
            row.get("accepted"),
            flush=True,
        )

    for arm in ("B1", "B2"):
        row = rows[arm]
        if not row.get("accepted"):
            continue
        rung = {"arm": arm, "width": 256, "steps": 20000, "octaves": 8, "seeds": [0]}
        print("attacks", arm, flush=True)
        rows[arm]["attacks"] = attack_seed(CFG, store, rung, 0)
        print("attacks done", arm, flush=True)

    save_json(OUT / "pilot_summary.json", rows)
    print("PILOT DONE", flush=True)


if __name__ == "__main__":
    main()
