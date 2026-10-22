"""Aggregate the smooth tent-map sweep and write a frozen-rule verdict."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.io import save_json


def load_runs(root: str) -> list[dict]:
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(Path(root) / "runs" / "*.json")))]


def aggregate(runs: list[dict]) -> dict:
    groups: dict[int, list[dict]] = {}
    for row in runs:
        groups.setdefault(row["n"], []).append(row)
    out = {}
    for n, rows in groups.items():
        out[str(n)] = {
            "n": n,
            "seeds": len(rows),
            "rmse": float(np.mean([r["rmse"] for r in rows])),
            "rmse_std": float(np.std([r["rmse"] for r in rows])),
            "t95": float(np.mean([r["t95_state"] for r in rows])),
            "t95_min": int(min(r["t95_state"] for r in rows)),
            "t95_max": int(max(r["t95_state"] for r in rows)),
            "grid": float(np.mean([r["top_grid_accuracy"] for r in rows])),
        }
    return out


def decide(cfg: dict, groups: dict) -> dict:
    rows = sorted(groups.values(), key=lambda v: v["n"])
    if not rows:
        return {"label": "INCONCLUSIVE", "reason": "no runs", "slope": None, "r2": None}
    top = rows[-1]
    if top["rmse"] > 1e-2:
        return {"label": "INCONCLUSIVE", "reason": f"underfit guard: RMSE {top['rmse']:.2e} > 1e-2", "slope": None, "r2": None}
    horizon = cfg["rollout_T"] + 1
    if any(v["t95_max"] >= horizon for v in rows):
        hit = [v for v in rows if v["t95_max"] >= horizon][0]
        return {"label": "POLYNOMIAL", "reason": f"at n={hit['n']} a seed survives the full horizon (T95 max={hit['t95_max']}); the map is learned exactly", "slope": None, "r2": None}
    xs = np.log([v["n"] for v in rows])
    ys = np.array([v["t95"] for v in rows], dtype=float)
    if len(xs) >= 3 and float(np.ptp(xs)) > 0:
        slope, intercept = np.polyfit(xs, ys, 1)
        pred = intercept + slope * xs
        ss_res = float(((ys - pred) ** 2).sum())
        ss_tot = float(((ys - ys.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    else:
        slope, r2 = 0.0, 0.0
    if slope >= 0.5 and r2 >= 0.6:
        return {"label": "EXPONENTIAL", "reason": f"T95 grows with {slope:.2f} steps per e-fold (R2={r2:.2f})", "slope": float(slope), "r2": float(r2)}
    return {"label": "INCONCLUSIVE", "reason": f"T95 is flat (slope={slope:.2f}, R2={r2:.2f}); the sample lever is ineffective at fixed compute", "slope": float(slope), "r2": float(r2)}


def figure(cfg: dict, groups: dict, path: Path, ykey: str, ylabel: str, title: str, logy: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = sorted(groups.values(), key=lambda v: v["n"])
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.errorbar(
        [v["n"] for v in rows],
        [v[ykey] for v in rows],
        yerr=[v.get(ykey + "_std", 0.0) for v in rows],
        marker="o",
        capsize=3,
        color="#1f77b4",
    )
    if logy:
        ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("training samples n (log)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_verdict(cfg: dict, groups: dict, verdict: dict, path: Path) -> None:
    reason_tr = {
        "EXPONENTIAL": "T95, ln n ile ölçülebilir biçimde büyüyor; sabit T için örnek sayısı üstel.",
        "POLYNOMIAL": "Harita tam öğreniliyor; T önemsiz, geçiş eşik.",
        "INCONCLUSIVE": "T95 n ile düz; tek-adım hatası model/optimizasyon tabanında. Örnek sayısı değil hesaplama/kapasite kaldıracı.",
    }.get(verdict["label"], verdict["reason"])
    lines = [
        f"VERDICT: {verdict['label']}",
        "",
        f"Karar: {reason_tr}",
        "",
        "| n | one-step RMSE | T95 mean (min–max) | top-grid |",
        "|---:|---:|---:|---:|",
    ]
    for v in sorted(groups.values(), key=lambda x: x["n"]):
        lines.append(f"| {v['n']} | {v['rmse']:.2e} | {v['t95']:.1f} ({v['t95_min']}–{v['t95_max']}) | {v['grid']:.3f} |")
    lines += [
        "",
        "Kill kuralı: `kill_rule_tent.md` (dondurulmuş).",
        "",
        "**Sınırlamalar.** Tek anahtar, 3 tohum, 16-bit tent map, 1D MLP, T_max=128. "
        "Ölçüt skaler durum hatası >5%; top-grid ikincil. Kalibrasyonda 512x3/12k "
        "model RMSE 9.4e-4/T95=8 verdi; 256x2/6k model 2.9e-3/T95=6. Yani T95 "
        "hesaplama/kapasite ile büyüyor, n ile değil. Hata birikimi Lyapunov "
        "tahminine yakın; 128 adım için gereken hassasiyet ~1e-36.",
        "",
        "**Sonraki adım.** Örnek değil hesaplama ölçeklemesini ölç: aynı n'de model "
        "boyutu/adım sayısını artırıp T95'in log(compute) ile büyümesini test et. "
        "Eğer T95 yalnızca compute ile büyüyorsa, savunma “sample complexity” değil "
        "“hesaplama maliyeti” sınırıdır; bu, CAPPED/CA sonuçlarıyla aynı aileye girer.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tent.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["out"])
    runs = load_runs(str(root))
    groups = aggregate(runs)
    verdict = decide(cfg, groups)
    save_json(root / "summary.json", {"config": cfg, "groups": groups, "verdict": verdict})
    figures = Path("figures")
    figures.mkdir(exist_ok=True)
    figure(cfg, groups, figures / "tent_rmse_vs_n.png", "rmse", "one-step RMSE", "Smooth tent: one-step error vs samples", logy=True)
    figure(cfg, groups, figures / "tent_t95_vs_n.png", "t95", "T95 (steps, error >5%)", "Smooth tent: autonomous horizon vs samples")
    write_verdict(cfg, groups, verdict, Path("VERDICT_TENT.md"))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
