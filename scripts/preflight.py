"""Preflight checks that do not need torch: verdict truth table, parameter
matching, task statistics and the phase-0 context."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import world  # noqa: E402
from src.decision.verdict import compute_verdict  # noqa: E402


def rung(width=256, steps=20000):
    return {"arm": "B1", "width": width, "steps": steps, "accepted": True}


def attack_block(min_ns, key_success=False):
    seeds = {
        str(i): {
            "finetune": {
                "min_n": n,
                "rows": [] if n is None else [
                    {"n": n, "cumulative_gpu_seconds": 3600.0}
                ],
            },
            "key_recovery": {"success": key_success, "rows": [{"restart": 0}]},
        }
        for i, n in enumerate(min_ns)
    }
    return {"primary": {"arm": "B1", "width": 256, "steps": 20000, "seeds": seeds},
            "upper": None}


def verdict_table():
    cases = {
        "DEAD_CAPACITY": {
            "a": {"accepted": True},
            "c": {"success": True, "gpu_hours": 2.0, "min_n": 10000},
            "ladder": [], "ladder_complete": True, "attacks": None,
        },
        "DEAD_FINETUNE": {
            "a": {"accepted": True},
            "c": {"success": True, "gpu_hours": 2.0, "min_n": 10000},
            "ladder": [rung()], "attacks": attack_block([1000, 1000, None]),
        },
        "DEAD_KEY": {
            "a": {"accepted": True},
            "c": {"success": True, "gpu_hours": 2.0, "min_n": 10000},
            "ladder": [rung()],
            "attacks": attack_block([None, None, None], key_success=True),
        },
        "CAPPED": {
            "a": {"accepted": True},
            "c": {"success": True, "gpu_hours": 0.2, "min_n": 10000},
            "ladder": [rung()], "attacks": attack_block([None, None, None]),
        },
        "ALIVE_WEAK": {
            "a": {"accepted": True},
            "c": {"success": True, "gpu_hours": 2.0, "min_n": 10000},
            "ladder": [rung()], "attacks": attack_block([5000, 5000, None]),
        },
        "ALIVE": {
            "a": {"accepted": True},
            "c": {"success": True, "gpu_hours": 2.0, "min_n": 10000},
            "ladder": [rung()], "attacks": attack_block([100000, None, None]),
        },
        "INCONCLUSIVE": {
            "a": {"accepted": True},
            "c": {"success": True, "gpu_hours": 2.0, "min_n": 10000},
            "ladder": [], "ladder_complete": False, "attacks": None,
        },
    }
    print("== verdict truth table ==")
    ok = True
    for expected, summary in cases.items():
        got = compute_verdict(summary)["label"]
        flag = "OK" if got == expected else "FAIL"
        ok &= got == expected
        print(f"{flag:4s} expected={expected:14s} got={got}")
    return ok


def params():
    def plain(w):
        return 3 * w * w + 11 * w + 5

    def concat(w):
        return 3 * w * w + 75 * w + 5

    def film(w):
        return 3 * w * w + 531 * w + 16581

    print("\n== parameter counts ==")
    print(f"{'A width':>8} {'A total':>12} {'B1 pm':>12} {'B1 total':>12} "
          f"{'B2 pm':>12} {'B2 total':>12} {'B2/A':>7}")
    for w in (256, 512, 1024):
        pm = round(w * 1.4142)
        a = 2 * plain(w)
        b1, b2 = concat(pm), film(pm)
        print(f"{w:8d} {a:12d} {pm:12d} {b1:12d} {pm:12d} {b2:12d} {b2/a:7.2f}")


def task_stats():
    print("\n== task statistics (100k random coordinates) ==")
    x = np.random.default_rng(0).random((100_000, 2), dtype=np.float32)
    for c in (4, 8):
        h, b = world(x, c, 123)
        counts = np.bincount(b, minlength=4) / len(b)
        print(
            f"c={c}: h mean={h.mean():.4f} std={h.std():.4f} "
            f"min={h.min():.4f} max={h.max():.4f}  "
            f"biomes={np.round(counts, 3).tolist()}  "
            f"majority={counts.max():.3f}"
        )
        # Fraction of variance in the two finest octaves: how much of the
        # function is high-frequency detail an MLP must fit.
        if c == 8:
            h4, _ = world(x, 4, 123)
            print(
                f"     corr(c=4, c=8)={np.corrcoef(h4, h)[0,1]:.4f}  "
                f"residual std(c8|c4)={np.std(h-h4):.4f}"
            )


def phase0():
    root = Path("results")
    rows = []
    for d in sorted(root.glob("cpu_seed*/seed_*")):
        for arm in ("A", "B1", "B2", "C"):
            p = d / f"{arm}.json"
            if p.exists():
                r = json.loads(p.read_text())
                rows.append((d.name, arm, r.get("status"), r.get("accepted"),
                             (r.get("valid") or {}).get("r2"),
                             (r.get("valid") or {}).get("acc")))
    print("\n== phase-0 width64,c4 five-seed CPU cell ==")
    print(f"{'seed':>6} {'arm':>3} {'status':>22} {'accepted':>8} {'r2':>7} {'acc':>7}")
    for seed, arm, status, acc, r2, a in rows[:40]:
        print(f"{seed:>6} {arm:>3} {str(status):>22} {str(acc):>8} "
              f"{('-' if r2 is None else f'{r2:.3f}'):>7} "
              f"{('-' if a is None else f'{a:.3f}'):>7}")


if __name__ == "__main__":
    ok = verdict_table()
    params()
    task_stats()
    phase0()
    print("\nPREFLIGHT", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
