"""Aggregate the keyed-CA sweep and write a frozen-rule verdict."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.io import save_json


def load_runs(root: str) -> list[dict]:
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(Path(root) / "runs" / "*.json")))]


def aggregate(runs: list[dict]) -> dict:
    groups: dict[tuple[int, int], list[dict]] = {}
    for row in runs:
        groups.setdefault((row["k"], row["n"]), []).append(row)
    out = {}
    for (k, n), rows in groups.items():
        out[f"k{k}_n{n}"] = {
            "k": k,
            "n": n,
            "seeds": len(rows),
            "acc": float(np.mean([r["one_step_accuracy"] for r in rows])),
            "acc_std": float(np.std([r["one_step_accuracy"] for r in rows])),
            "t95": float(np.mean([r["t95"] for r in rows])),
            "t95_std": float(np.std([r["t95"] for r in rows])),
            "t95_min": int(min(r["t95"] for r in rows)),
            "t95_max": int(max(r["t95"] for r in rows)),
            "tv": float(np.mean([r["distribution_tv"] for r in rows])),
            "exact_state_accuracy": float(np.mean([r["exact_state_accuracy"] for r in rows])),
        }
    return out


def decide(cfg: dict, groups: dict) -> dict:
    main = cfg["main_k"]
    horizon = cfg["rollout_T"] + 1
    rows = sorted(
        [v for v in groups.values() if v["k"] == main],
        key=lambda v: v["n"],
    )
    if not rows:
        return {"label": "INCONCLUSIVE", "reason": "no main-k runs", "slope": None, "r2": None}
    top = rows[-1]
    if top["acc"] < 0.99:
        return {
            "label": "INCONCLUSIVE",
            "reason": f"one-step accuracy at n={top['n']} is {top['acc']:.3f} < 0.99; the error-amplification test is not isolated",
            "slope": None,
            "r2": None,
        }
    if any(v["t95_max"] >= horizon for v in rows):
        hit = [v for v in rows if v["t95_max"] >= horizon][0]
        return {
            "label": "POLYNOMIAL",
            "reason": (
                f"at n={hit['n']} at least one seed survives the full "
                f"{cfg['rollout_T']}-step horizon (T95 max={hit['t95_max']}), "
                f"while the same n can fail at T95 min={hit['t95_min']}: the rule "
                "is either learned exactly or not, so the transition is a sharp "
                "learnability threshold, not an exponential-in-T cost"
            ),
            "slope": None,
            "r2": None,
        }
    xs = np.log([v["n"] for v in rows if v["t95"] < horizon])
    ys = np.array([v["t95"] for v in rows if v["t95"] < horizon], dtype=float)
    if len(xs) >= 3 and float(np.ptp(xs)) > 0:
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = intercept + slope * xs
        ss_res = float(((ys - pred) ** 2).sum())
        ss_tot = float(((ys - ys.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        slope, r2 = 0.0, 0.0
    if slope >= 0.5 and r2 >= 0.6:
        return {
            "label": "EXPONENTIAL",
            "reason": f"T95 grows with {slope:.2f} steps per e-fold of n (R2={r2:.2f}); the sample complexity for a fixed T is exponential",
            "slope": float(slope),
            "r2": float(r2),
        }
    return {
        "label": "INCONCLUSIVE",
        "reason": f"T95 is flat or unresolved (slope={slope:.2f}, R2={r2:.2f}); n_max does not characterize the barrier",
        "slope": float(slope),
        "r2": float(r2),
    }


def figure_t95(cfg: dict, groups: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for k, color in ((cfg["control_k"], "#2ca02c"), (cfg["main_k"], "#d62728")):
        rows = sorted([v for v in groups.values() if v["k"] == k], key=lambda v: v["n"])
        if not rows:
            continue
        ax.errorbar(
            [v["n"] for v in rows],
            [v["t95"] for v in rows],
            yerr=[v["t95_std"] for v in rows],
            marker="o",
            capsize=3,
            color=color,
            label=f"k={k} (one-step acc {rows[-1]['acc']:.3f} at n={rows[-1]['n']})",
        )
    ax.axhline(cfg["rollout_T"] + 1, ls=":", color="black", lw=1)
    ax.text(0.02, 0.95, "full horizon", transform=ax.transAxes, fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("training grids n (log)")
    ax.set_ylabel("T95 (steps before <95% cells)")
    ax.set_title("Autonomous trajectory survival vs sample budget")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def figure_acc(cfg: dict, groups: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for k, color in ((cfg["control_k"], "#2ca02c"), (cfg["main_k"], "#d62728")):
        rows = sorted([v for v in groups.values() if v["k"] == k], key=lambda v: v["n"])
        if not rows:
            continue
        ax.errorbar(
            [v["n"] for v in rows],
            [v["acc"] for v in rows],
            yerr=[v["acc_std"] for v in rows],
            marker="o",
            capsize=3,
            color=color,
            label=f"k={k}",
        )
    ax.axhline(0.99, ls="--", color="black", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("training grids n (log)")
    ax.set_ylabel("one-step cell accuracy")
    ax.set_ylim(0.45, 1.01)
    ax.set_title("One-step learnability precondition")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_verdict(cfg: dict, groups: dict, verdict: dict, path: Path) -> None:
    rows = sorted(groups.values(), key=lambda v: (v["k"], v["n"]))
    lines = [
        f"VERDICT: {verdict['label']}",
        "",
        f"Karar: {verdict['reason']}",
        "",
        "| k | n | one-step acc | T95 mean (min–max) | TV | exact-state |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for v in rows:
        lines.append(
            f"| {v['k']} | {v['n']} | {v['acc']:.4f} | "
            f"{v['t95']:.1f} ({v['t95_min']}–{v['t95_max']}) | "
            f"{v['tv']:.4f} | {v['exact_state_accuracy']:.4f} |"
        )
    lines += [
        "",
        f"Kill kuralı: {cfg.get('kill_rule', 'frozen in kill_rule.md')}",
        "",
        "**Sınırlamalar.** Tek anahtar, 3 tohum, 8x8 ikili CA, CNN öğrenci, "
        "T_max=" + str(cfg["rollout_T"]) + ". Ölçüt **özerk yörünge** doğruluğudur; "
        "dağılım TV'si ayrıca raporlanır. Öğrenciye kod/ağırlık verilmedi; "
        "sadece I/O örnekleri verildi. Yüksek k'da öğrenci tek adımı öğrenemiyorsa "
        "bu, hata birikiminden önce **öğrenilebilirlik** bariyeridir.",
        "",
        "**Sonraki adım.** k=4'te bir adım ≥0.99'a ulaşılırsa T95~log n eğimini "
        "ve çok adımlı öğrenciyi test et; ulaşılamazsa görev zorluğu/CNN kapasitesi "
        "ayrıştırılmalı. k=5 33M-bit kuralı, öğrenilebilirlik tavanını gösteren "
        "üst sınır olarak tutuldu.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/expansive.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["out"])
    runs = load_runs(str(root))
    groups = aggregate(runs)
    verdict = decide(cfg, groups)
    save_json(root / "summary.json", {"config": cfg, "groups": groups, "verdict": verdict})
    figures = Path("figures")
    figures.mkdir(exist_ok=True)
    figure_t95(cfg, groups, figures / "expansive_t95_vs_n.png")
    figure_acc(cfg, groups, figures / "expansive_acc_vs_n.png")
    write_verdict(cfg, groups, verdict, Path("VERDICT_EXPANSIVE.md"))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
