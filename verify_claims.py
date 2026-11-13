"""Recompute every number used in the Neural-DRM note from raw run JSON.

Read-only. Run from the repository root:
    python verify_claims.py .
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
DEC = ROOT / "results" / "decision_cpu_v1"
OUT = []


def say(label, value):
    OUT.append(f"{label}: {value}")
    print(f"{label}: {value}")


def load(p):
    return json.loads(Path(p).read_text())


summary = load(DEC / "summary.json")
verdict = load(DEC / "verdict.json")
state = load(DEC / "state.json")

# --- decision configuration -------------------------------------------------
cfg = summary["config"]
say("decision.widths", cfg["widths"])
say("decision.seeds", cfg["seeds"])
say("decision.octaves", cfg["octaves_primary"])
say("decision.train_size", cfg["train_size"])
say("decision.val_size", cfg["val_size"])
say("decision.test_size", cfg["test_size"])
say("decision.steps_ladder", cfg["step_ladder"])
say("decision.threads", cfg["threads"])
say("decision.device", cfg["device"])

# --- A: sanity arm ----------------------------------------------------------
A = summary["a"]["by_width"]["256"]["seeds"]
say("A.accepted_seeds", summary["a"]["by_width"]["256"]["accepted_seeds"])
say("A.test_r2_range", [round(A[str(s)]["test_f"]["r2"], 6) for s in (0, 1, 2)])
say("A.test_acc_range", [round(A[str(s)]["test_f"]["acc"], 6) for s in (0, 1, 2)])

# --- B acceptance -----------------------------------------------------------
arms = summary["accepted"]["arms"]
say("B1.accepted", arms["B1"]["accepted_seeds"])
say("B2.accepted", arms["B2"]["accepted_seeds"])
for arm in ("B1", "B2"):
    vals = [arms[arm]["seeds"][str(s)]["test_valid"] for s in (0, 1, 2)]
    say(f"{arm}.valid_r2_range", [round(v["r2"], 6) for v in vals])
    say(f"{arm}.valid_acc_range", [round(v["acc"], 6) for v in vals])

# --- fine-tune --------------------------------------------------------------
ft = summary["attacks"]["primary"]["seeds"]
for s in ("0", "1", "2"):
    rows = ft[s]["finetune"]["rows"]
    say(f"finetune.seed{s}.rows", [(r["n"], r["n_high_fidelity_keys"], r["bypass"],
                                   round(r["cumulative_gpu_seconds"], 3)) for r in rows])
    say(f"finetune.seed{s}.min_n", ft[s]["finetune"]["min_n"])

# --- key recovery -----------------------------------------------------------
for s in ("0", "1", "2"):
    kr = ft[s]["key_recovery"]
    hs = [r["hamming_to_secret"] for r in kr["rows"]]
    say(f"keyrecovery.seed{s}", {"success": kr["success"], "restarts": len(hs),
                                 "best_hamming": min(hs), "max_hamming": max(hs)})

# --- Hamming diagnostic -----------------------------------------------------
for s in ("0", "1", "2"):
    h = ft[s]["hamming"]
    win = [r for r in h["rows"] if 45 <= r["distance"] <= 58]
    probes = [sc for r in win for sc in r["scores"]]
    hi = [(r["distance"], round(sc["r2"], 4), round(sc["acc"], 4))
          for r in win for sc in r["scores"] if sc["r2"] > 0.89]
    best = max((sc["r2"], r["distance"], sc["acc"])
               for r in h["rows"] for sc in r["scores"] if r["distance"] > 0)
    say(f"hamming.seed{s}.probes_45_58", len(probes))
    say(f"hamming.seed{s}.probes_above_0.89", len(hi))
    say(f"hamming.seed{s}.hits", hi)
    say(f"hamming.seed{s}.best_wrong_key", {"r2": round(best[0], 4),
                                            "distance": best[1],
                                            "acc": round(best[2], 4)})

# --- distillation C ---------------------------------------------------------
C = summary["c"]
say("C.min_n", C["min_n"])
say("C.min_n_cumulative_seconds", round(C["min_n_gpu_hours"] * 3600, 3))
say("C.min_n_hours", round(C["min_n_gpu_hours"], 6))
say("C.cheapest_n", C["cheapest_n"])
say("C.cheapest_seconds", round(C["cheapest_gpu_hours"] * 3600, 3))
say("C.cheapest_hours", round(C["cheapest_gpu_hours"], 6))
say("C.gate_rows", [(r["lr"], r["steps"], r["accepted"], round(r["gpu_seconds"], 3))
                    for r in C["gate_rows"]])
n100 = load(DEC / "units" / "C_w256_n100000_lr1e-04_s60000_seed0_c8.json")
say("C.n100000.test", {k: round(v, 6) for k, v in n100["test"].items()})
say("C.n100000.steps", n100["steps"])
say("C.n100000.seconds", round(n100["gpu_seconds"], 3))

# --- budget / ratio ---------------------------------------------------------
say("budget.hours", round(summary["budget"]["gpu_hours"], 6))
say("budget.limit", summary["budget"]["limit_hours"])
say("verdict", verdict)
say("B_over_C", round(verdict["numbers"]["b_over_c_cost_ratio"], 6))

# --- expansive --------------------------------------------------------------
EXP = load(ROOT / "results" / "expansive_v1" / "summary.json")
for k, g in EXP["groups"].items():
    say(f"expansive.{k}", {kk: round(g[kk], 6) if isinstance(g[kk], float) else g[kk]
                           for kk in ("k", "n", "acc", "t95", "t95_min", "t95_max",
                                      "tv", "exact_state_accuracy")})
for s in (0, 1, 2):
    r = load(ROOT / "results" / "expansive_v1" / "runs" / f"k3_n3000_seed{s}.json")
    say(f"expansive.k3_n3000_seed{s}", {"acc": round(r["one_step_accuracy"], 6),
                                        "t95": r["t95"], "tv": r["distribution_tv"],
                                        "exact": round(r["exact_state_accuracy"], 6)})
CAL = load(ROOT / "results" / "expansive_v1" / "calibration.json")
say("expansive.calibration.k4_best", CAL["rows"][4])
say("expansive.calibration.k5", CAL["rows"][5:])

# --- tent -------------------------------------------------------------------
TENT = load(ROOT / "results" / "tent_v1" / "summary.json")
for k, g in TENT["groups"].items():
    say(f"tent.{k}", {kk: round(g[kk], 9) if isinstance(g[kk], float) else g[kk]
                      for kk in ("n", "rmse", "t95", "t95_min", "t95_max", "grid")})
for s in (0, 1, 2):
    for n in (10000, 30000, 100000):
        r = load(ROOT / "results" / "tent_v1" / "runs" / f"n{n}_seed{s}.json")
        say(f"tent.n{n}_seed{s}", {"rmse": round(r["rmse"], 9), "t95": r["t95_state"],
                                   "grid": round(r["top_grid_accuracy"], 6)})
say("tent.verdict", TENT["verdict"])
TCAL = load(ROOT / "results" / "tent_v1" / "calibration.json")
say("tent.calibration", TCAL["rows"])

# --- total recorded wall clock ---------------------------------------------
exp_s = sum(load(p)["total_seconds"]
            for p in (ROOT / "results" / "expansive_v1" / "runs").glob("*.json"))
ten_s = sum(load(p)["total_seconds"]
            for p in (ROOT / "results" / "tent_v1" / "runs").glob("*.json"))
dec_h = summary["budget"]["gpu_hours"]
say("total.decision_hours", round(dec_h, 6))
say("total.expansive_hours", round(exp_s / 3600, 6))
say("total.tent_hours", round(ten_s / 3600, 6))
say("total.all_hours", round(dec_h + exp_s / 3600 + ten_s / 3600, 6))

# --- task-geometry pre-check: c=8 vs c=4 correlation ------------------------
try:
    sys.path.insert(0, str(ROOT))
    import numpy as np
    from src.data import pools, world

    _, _, x = pools(20261021, n_train=1, n_val=1, n_test=100000)
    h8, _ = world(x, 8, 123)
    h4, _ = world(x, 4, 123)
    say("precheck.corr_c8_c4", round(float(np.corrcoef(h8, h4)[0, 1]), 6))
    say("precheck.std_c8", round(float(h8.std()), 6))
    say("precheck.std_c4", round(float(h4.std()), 6))
except Exception as exc:  # pragma: no cover
    say("precheck.error", repr(exc))

OUT_PATH = Path(__file__).with_name("verify_claims.out.txt")
OUT_PATH.write_text("\n".join(OUT) + "\n")
print(f"\nwrote {OUT_PATH}")
