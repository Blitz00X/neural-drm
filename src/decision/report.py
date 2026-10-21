"""Figures and the single-page Turkish VERDICT.md."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _worst_key(row: dict) -> float:
    return min(min(m["r2"], m["acc"]) for m in row["test_keys"])


def figure_finetune(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    blocks = [
        b
        for b in (summary.get("attacks") or {}).values()
        if b
    ]
    colors = {"B1": "#1f77b4", "B2": "#d62728"}
    for block in blocks:
        arm = block["arm"]
        seeds = block["seeds"]
        per_seed = {}
        for seed, row in seeds.items():
            ft = row.get("finetune") or {}
            points = [(r["n"], _worst_key(r)) for r in ft.get("rows", [])]
            if points:
                per_seed[seed] = points
                ax.plot(
                    [p[0] for p in points],
                    [p[1] for p in points],
                    color=colors.get(arm, "gray"),
                    alpha=0.25,
                    lw=1,
                )
        if per_seed:
            ns = sorted({n for pts in per_seed.values() for n, _ in pts})
            means = []
            for n in ns:
                vals = [
                    dict(pts)[n]
                    for pts in per_seed.values()
                    if n in dict(pts)
                ]
                means.append(float(np.mean(vals)))
            ax.plot(
                ns,
                means,
                marker="o",
                color=colors.get(arm, "gray"),
                label=f"{arm} fine-tune (w={block['width']})",
            )

    c = summary.get("c") or {}
    c_rows = [r for r in c.get("rows", []) if r.get("test")]
    if c_rows:
        ns = [r["n"] for r in c_rows]
        vals = [min(r["test"]["r2"], r["test"]["acc"]) for r in c_rows]
        ax.plot(ns, vals, marker="s", color="#2ca02c", label="C distillation (w=256)")

    if summary.get("a", {}).get("accepted"):
        ax.scatter([1], [1.0], marker="*", s=180, color="#ff7f0e", zorder=5)
        ax.annotate(
            "A: gate removal, n=0, 0 GPU-h",
            (1, 1.0),
            textcoords="offset points",
            xytext=(8, -14),
            fontsize=8,
            color="#ff7f0e",
        )

    ax.axhline(0.95, ls="--", lw=1, color="black")
    ax.text(1.02, 0.952, "fidelity threshold 0.95", fontsize=7, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("oracle samples n (log scale; A at n=0)")
    ax.set_ylabel("worst-key min(R², biome accuracy)")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Fine-tune / distillation fidelity vs sample budget")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def figure_costs(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = summary.get("c") or {}
    rows = summary.get("attacks") or {}
    labels, values = [], []
    if c.get("gpu_hours") is not None:
        labels.append("C distillation")
        values.append(max(float(c["gpu_hours"]), 1e-4))
    for block in rows.values():
        if not block:
            continue
        costs = []
        used_n = None
        for seed, row in block["seeds"].items():
            ft = row.get("finetune") or {}
            if ft.get("min_n") is not None:
                used_n = ft["min_n"]
                for r in ft.get("rows", []):
                    if r["n"] == ft["min_n"]:
                        costs.append(r.get("cumulative_gpu_seconds", 0.0))
        if costs:
            labels.append(
                f"{block['arm']} fine-tune w={block['width']}\n(n={used_n})"
            )
            values.append(max(np.mean(costs) / 3600.0, 1e-4))
        kr = [
            row.get("key_recovery", {}).get("gpu_seconds", 0.0)
            for row in block["seeds"].values()
        ]
        if kr:
            labels.append(f"{block['arm']} key recovery w={block['width']}")
            values.append(max(np.mean(kr) / 3600.0, 1e-4))

    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    if labels:
        ax.barh(labels, values, color="#4c72b0")
        ax.set_xscale("log")
        for i, v in enumerate(values):
            ax.text(v * 1.15, i, f"{v:.3f} GPU-h", va="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "not measured", ha="center", va="center")
    ax.set_xlabel("GPU-hours (log scale)")
    ax.set_title("Attacker cost: distillation vs B attacks")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def figure_hamming(summary: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    blocks = [b for b in (summary.get("attacks") or {}).values() if b]
    colors = {"B1": "#1f77b4", "B2": "#d62728"}
    plotted = False
    for block in blocks:
        curves = []
        for row in block["seeds"].values():
            ham = row.get("hamming") or {}
            if ham.get("rows"):
                curves.append(ham["rows"])
        if not curves:
            continue
        d = [r["distance"] for r in curves[0]]
        r2 = np.mean([[r["mean_r2"] for r in c] for c in curves], axis=0)
        acc = np.mean([[r["mean_acc"] for r in c] for c in curves], axis=0)
        ax.plot(d, r2, color=colors.get(block["arm"], "gray"), label=f"{block['arm']} R²")
        ax.plot(
            d,
            acc,
            color=colors.get(block["arm"], "gray"),
            ls="--",
            label=f"{block['arm']} accuracy",
        )
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "not measured", ha="center", va="center")
    else:
        ax.axhline(0.95, ls=":", color="black", lw=1)
    ax.set_xlabel("Hamming distance from the true 64-bit key")
    ax.set_ylabel("fidelity")
    ax.set_ylim(0, 1.02)
    ax.set_title("Hamming leakage diagnostic (privileged)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _fmt(value, digits=2):
    if value is None:
        return "yok"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_verdict(summary: dict, verdict: dict, path: Path) -> None:
    n = verdict.get("numbers", {})
    c = summary.get("c") or {}
    budget = summary.get("budget") or {}
    lines = [
        f"VERDICT: {verdict['label']}",
        "",
        f"Kural {verdict.get('rule')} eşleşti. {verdict.get('reason', '')}",
        "",
        "| Ölçüm | Değer |",
        "|---|---|",
        f"| Görev (oktav) | c={summary.get('octaves')} |",
        f"| A saldırı maliyeti | 0 GPU-saat (kapı silme) |",
        f"| B min fine-tune n | {_fmt(n.get('b_min_finetune_n'))} |",
        f"| C min n | {_fmt(c.get('min_n'))} |",
        f"| C GPU-saat | {_fmt(c.get('gpu_hours'), 3)} |",
        f"| B/C maliyet oranı | {_fmt(n.get('b_over_c_cost_ratio'), 3)} |",
        f"| Anahtar kurtarma | {'başarılı' if n.get('key_recovery_success') else 'başarısız'} |",
        f"| Harcanan GPU-saati | {_fmt(budget.get('gpu_hours'), 2)} / {_fmt(budget.get('limit_hours'), 0)} |",
        "",
        "**Gerekçe (3 satır).** "
        + _rationale(summary, verdict),
        "",
        "**Sınırlamalar.** " + _limitations(summary),
        "",
        "**Sonraki adım.** " + _next_step(summary, verdict),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _rationale(summary: dict, verdict: dict) -> str:
    label = verdict["label"]
    c = summary.get("c") or {}
    if label == "DEAD_CAPACITY":
        return (
            "A ve C kabul edildi; merdivendeki hiçbir genişlik/bütçe/parametre-eşit "
            "kombinasyonunda B1 veya B2, yanlış anahtar stratasında decoy g'ye "
            "yüksek fidelity ile eşleşemedi. Bu, süperpozisyonun kapasite değil "
            "anahtar koşullama problemi olduğunu gösteriyor. Etiket yalnızca bu "
            "oyuncak kurulum için geçerlidir."
        )
    if label == "DEAD_FINETUNE":
        return (
            "Kabul edilen B, 1000 veya daha az doğru örnekle anahtarı etkisiz "
            "bırakacak biçimde ince ayarlandı. Entanglement fine-tune'a dayanmadı. "
            "Bu, distillation'ın anahtar kapısını aştığı anlamına gelir."
        )
    if label == "DEAD_KEY":
        return (
            "1000 oracle örneğiyle sürekli anahtar optimizasyonu, ±1'e yuvarlanınca "
            "yüksek fidelity veren bir anahtar buldu. Anahtar uzayı 64 bit olsa da "
            "modelin kabul bölgesi gradyanla bulunabildi. Bu yüzden lisans kapısı "
            "kriptografik değil, öğrenilmiş bir bölge."
        )
    if label == "CAPPED":
        return (
            "B, fine-tune ve anahtar kurtarmadan kurtuldu; ancak lisanssız C "
            f"{_fmt(c.get('min_n'))} örnekle ve {_fmt(c.get('gpu_hours'), 3)} GPU-saatte "
            "yüksek fidelity'ye ulaştı. Korumanın tavanı görevin öğrenilme "
            "maliyeti; görev bu ölçekte çok kolay."
        )
    if label == "ALIVE_WEAK":
        return (
            "B fine-tune ile 1000-10000 örnek aralığında kırılıyor; anahtar "
            "kurtarma ve distillation cephesi daha pahalı. Entanglement kısmi "
            "bir maliyet artışı sağlıyor, ancak güvenlik iddiası için yetersiz."
        )
    if label == "ALIVE":
        return (
            "B, 100000 örnek veya daha fazlasını gerektiren fine-tune'a, başarısız "
            "anahtar kurtarmaya ve 1 GPU-saatten pahalı C'ye dayandı. Bu kurulumda "
            "entanglement ölçülebilir bir maliyet yükselmesi üretti."
        )
    return (
        "Kuralların hiçbiri eksiksiz eşleşmedi; eksik veya çelişkili kanıt var. "
        "Etiket bilimsel bir sonuç değil, ölçüm sınırının kaydıdır."
    )


def _limitations(summary: dict) -> str:
    return (
        "Bu bir oyuncak prosedürel dünya fonksiyonudur; gerçek oyun motoru, "
        "render hattı veya ticari DRM değildir. Tehdit modeli belleği okuyan "
        "T4 saldırganını kapsamaz; oracle analitik f'dir (en güçlü saldırgan). "
        "GPU-saatleri tek bir GTX 1650 Ti'de ölçüldü, T4/A100'e genellenemez. "
        "3 tohum küçük bir örneklemdir; ölçülmeyen hücreler silinmedi, "
        "'yok' olarak işaretlendi."
    )


def _next_step(summary: dict, verdict: dict) -> str:
    label = verdict["label"]
    if label == "DEAD_CAPACITY":
        return (
            "B'nin anahtar koşullama kapasitesini artır: anahtarı FiLM yerine "
            "öğrenilmiş bir kapı vektörüyle çarp, Hamming-1/2 negatiflerini "
            "curriculum ile ağırlıklandır ve w=1024'te 120k adıma çık. "
            "Ayrıca c=8 yerine daha zor bir görev (c=16, çok ölçekli) dene."
        )
    if label in ("DEAD_FINETUNE", "DEAD_KEY"):
        return (
            "Anahtar bağımlılığını gradyanla çözülemeyecek biçimde sertleştir: "
            "min-max eğitim, anahtar yolunu gizli bir kodlamayla besle ve "
            "fine-tune sırasında anahtar yolunu donduran bir düzenlileştirici ekle. "
            "Ardından aynı saldırı paketini tekrarla."
        )
    if label == "CAPPED":
        return (
            "Görevi zorlaştır: c=16/32 oktav, daha yüksek frekanslı hash-grid ve "
            "100k yerine 1M örnek; C'nin 1 GPU-saati eşiğini aşıp aşmadığını ölç. "
            "Aynı anda B'nin kabul edilip edilmediğini koru; eşikleri değiştirme."
        )
    return (
        "Aynı protokolü daha fazla tohum ve daha zor görevle tekrarla; eksik "
        "hücreleri tamamla, ardından sonucu gerçek motor ölçeğine taşı."
    )
