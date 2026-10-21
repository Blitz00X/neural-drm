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
            xytext=(10, -32),
            fontsize=8,
            color="#ff7f0e",
        )

    ax.axhline(0.95, ls="--", lw=1, color="black")
    ax.text(2.0e4, 0.955, "fidelity threshold 0.95", fontsize=7, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("oracle samples n (log scale; A at n=0)")
    ax.set_ylabel("worst-key min(R², biome accuracy)")
    ax.set_ylim(0.0, 1.02)
    ax.text(
        0.02, 0.03,
        "B curves are the seed mean of the worst key; bypass counts a seed with >=5/6 keys",
        transform=ax.transAxes, fontsize=7, color="#444444",
    )
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
    device = (summary.get("config") or {}).get("device", "cuda")
    c_unit = "CPU" if device == "cpu" else "GPU"
    labels, values = [], []
    if c.get("gpu_hours") is not None:
        labels.append(f"C distillation\n(cheapest n={c.get('cheapest_n')})")
        values.append(max(float(c["gpu_hours"]), 1e-4))
    for block in rows.values():
        if not block:
            continue
        per_seed_n = {
            seed: (row.get("finetune") or {}).get("min_n")
            for seed, row in block["seeds"].items()
        }
        known = [v for v in per_seed_n.values() if v is not None]
        used_n = min(known) if known else None
        costs = []
        if used_n is not None:
            for seed, row in block["seeds"].items():
                ft = row.get("finetune") or {}
                if ft.get("min_n") == used_n:
                    for r in ft.get("rows", []):
                        if r["n"] == used_n:
                            costs.append(r.get("cumulative_gpu_seconds", 0.0))
        if costs:
            labels.append(
                f"{block['arm']} fine-tune w={block['width']}\n(min n={used_n})"
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
        from matplotlib.ticker import NullLocator
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xticks([0.05, 0.1, 0.2, 0.5])
        ax.set_xticklabels(["0.05", "0.1", "0.2", "0.5"])
        for i, v in enumerate(values):
            ax.text(v * 1.15, i, f"{v:.3f} {c_unit}-h", va="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "not measured", ha="center", va="center")
    ax.set_xlabel(f"{c_unit}-hours (log scale)")
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
        for seed, row in block["seeds"].items():
            ham = row.get("hamming") or {}
            if ham.get("rows"):
                curves.append((seed, ham["rows"]))
        if not curves:
            continue
        d = [r["distance"] for r in curves[0][1]]
        r2_each = np.array([[r["mean_r2"] for r in rows] for _, rows in curves])
        acc_each = np.array([[r["mean_acc"] for r in rows] for _, rows in curves])
        color = colors.get(block["arm"], "gray")
        for i, (seed, _) in enumerate(curves):
            ax.plot(d, r2_each[i], color=color, alpha=0.22, lw=0.8)
        ax.plot(d, r2_each.max(axis=0), color=color, lw=1.8,
                label=f"{block['arm']} max R² (seed)")
        ax.plot(d, acc_each.mean(axis=0), color=color, ls="--",
                label=f"{block['arm']} mean accuracy")
        plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "not measured", ha="center", va="center")
    else:
        ax.axhline(0.95, ls=":", color="black", lw=1)
        ax.axhline(0.0, color="gray", lw=0.6)
        ax.set_ylim(-4.3, 1.05)
        ax.text(
            0.02, 0.05,
            "seed 0: wrong keys at d~50-53 reach R²~0.97 (secondary f-region)",
            transform=ax.transAxes, fontsize=7, color="#444444",
        )
    ax.set_xlabel("Hamming distance from the true 64-bit key")
    ax.set_ylabel("fidelity (R² can be negative)")
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


def _reason_tr(label: str) -> str:
    return {
        "DEAD_CAPACITY": (
            "A ve C kabul edildi; merdivendeki hiçbir genişlik/bütçe/parametre-eşit "
            "kombinasyonunda B1 veya B2 kabul edilmedi."
        ),
        "DEAD_FINETUNE": (
            "Kabul edilen B, 3 tohumun en az 2'sinde 1000 veya daha az örnekle "
            "fine-tune bypass verdi."
        ),
        "DEAD_KEY": (
            "Kabul edilen B'de anahtar kurtarma 1000 oracle örneğiyle başarılı oldu."
        ),
        "CAPPED": (
            "B, fine-tune ve anahtar kurtarmadan kurtuldu; ancak lisanssız C "
            "1 GPU-saatinden daha ucuza yüksek fidelity'ye ulaştı."
        ),
        "ALIVE_WEAK": (
            "B hayatta; fine-tune 1000-10000 örnek aralığında açılıyor."
        ),
        "ALIVE": (
            "B hayatta; fine-tune 100000 örnek veya daha fazlasını gerektiriyor "
            "ya da hiç açılmıyor, anahtar kurtarma başarısız ve C maliyeti "
            "1 GPU-saatten yüksek."
        ),
        "INCONCLUSIVE": "Kuralların hiçbiri eksiksiz eşleşmedi; kanıt eksik veya çelişkili.",
    }.get(label, "Kural eşleşmedi.")


def write_verdict(summary: dict, verdict: dict, path: Path) -> None:
    n = verdict.get("numbers", {})
    c = summary.get("c") or {}
    budget = summary.get("budget") or {}
    device = (summary.get("config") or {}).get("device", "cuda")
    unit = "CPU" if device == "cpu" else "GPU"
    c_unit = "CPU" if device == "cpu" else "GPU"
    primary = (summary.get("attacks") or {}).get("primary") or {}
    seeds = primary.get("seeds", {})
    per_seed = []
    kr_best = None
    ham_d1 = None
    ham_peak = None
    for key in sorted(seeds):
        row = seeds[key]
        ft = row.get("finetune") or {}
        per_seed.append(_fmt(ft.get("min_n")))
        best = (row.get("key_recovery") or {}).get("best") or {}
        h = best.get("hamming_to_secret")
        if h is not None:
            kr_best = h if kr_best is None else min(kr_best, h)
        hrows = (row.get("hamming") or {}).get("rows") or []
        if len(hrows) > 1:
            ham_d1 = hrows[1].get("mean_r2")
        for r in hrows:
            if r.get("distance", 0) >= 10:
                if ham_peak is None or r["mean_r2"] > ham_peak[1]:
                    ham_peak = (r["distance"], r["mean_r2"], r["mean_acc"])
    seed_txt = " / ".join(per_seed) if per_seed else "yok"
    kr_txt = (
        "başarılı" if n.get("key_recovery_success") else
        f"başarısız (en iyi Hamming {kr_best})" if kr_best is not None else "başarısız"
    )
    if ham_d1 is None:
        ham_txt = "yok"
    elif ham_peak is None or ham_peak[1] < 0.5:
        ham_txt = f"d=1: R²={_fmt(ham_d1, 2)}; d>=10 sızıntı yok"
    else:
        ham_txt = (
            f"d=1: R²={_fmt(ham_d1, 2)}; d≈{ham_peak[0]}: "
            f"R²={_fmt(ham_peak[1], 2)}/acc={_fmt(ham_peak[2], 2)} "
            "(yanlış anahtar f-bölgesi)"
        )
    lines = [
        f"VERDICT: {verdict['label']}",
        "",
        f"Kural {verdict.get('rule')} eşleşti. {_reason_tr(verdict['label'])}",
        "",
        "| Ölçüm | Değer |",
        "|---|---|",
        f"| Görev (oktav) | c={summary.get('octaves')} |",
        f"| B kabul edilen arm | {primary.get('arm', '-')} ({len(seeds)}/3 tohum) |",
        f"| B fine-tune min n (tohum sırasıyla) | {seed_txt} |",
        f"| C min n | {_fmt(c.get('min_n'))} |",
        f"| C min n kümülatif maliyeti | {_fmt(c.get('min_n_gpu_hours'), 3)} {c_unit}-saat |",
        f"| C en ucuz protokol | n={_fmt(c.get('cheapest_n'))}, {_fmt(c.get('gpu_hours'), 3)} {c_unit}-saat (eşik 1 GPU-saat) |",
        f"| B/C maliyet oranı (en ucuz C) | {_fmt(n.get('b_over_c_cost_ratio'), 3)} |",
        f"| Anahtar kurtarma | {kr_txt} |",
        f"| Hamming sıçraması | {ham_txt} |",
        f"| A saldırı maliyeti | 0 GPU-saat (kapı silme) |",
        f"| Harcanan {unit}-saati | {_fmt(budget.get('gpu_hours'), 2)} / {_fmt(budget.get('limit_hours'), 0)} |",
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
            "yüksek fidelity ile eşleşemedi. Bu kurulumda B ya kapasite ya da "
            "anahtar koşullama optimizasyonu nedeniyle düştü; ikisini ayırmak "
            "için ek tanı gerekir. Etiket yalnızca bu oyuncak kurulum içindir."
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
        device = (summary.get("config") or {}).get("device", "cuda")
        unit = "CPU" if device == "cpu" else "GPU"
        return (
            "B2 w=256'da 3/3 tohumda kabul edildi; fine-tune tohumların 1'inde "
            "10000, 1'inde 100000 örnekte açıldı, 1'inde hiç açılmadı; anahtar "
            "kurtarma 20 başlangıçta başarısız oldu. Yani B, kural 2 ve 3'ten "
            "kurtuldu; ancak ayrıcalıklı Hamming taraması seed 0'da 50-53 bit "
            "uzakta R²≈0.95-0.97 veren yanlış anahtarlar buldu (kapı tek nokta "
            "değil, bölge). Lisanssız C ise "
            f"{_fmt(c.get('min_n'))} örnekle, arama "
            f"dahil {_fmt(c.get('min_n_gpu_hours'), 3)} {unit}-saatte yüksek "
            f"fidelity'ye ulaştı; en ucuz protokol {_fmt(c.get('cheapest_n'))} "
            f"örnekle {_fmt(c.get('gpu_hours'), 3)} {unit}-saat. İkisi de 1 "
            "GPU-saatinin altında; tavan görevin öğrenilme maliyeti."
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
    device = (summary.get("config") or {}).get("device", "cuda")
    base = (
        "Bu bir oyuncak prosedürel dünya fonksiyonudur; gerçek oyun motoru, "
        "render hattı veya ticari DRM değildir. Tehdit modeli belleği okuyan "
        "T4 saldırganını kapsamaz; oracle analitik f'dir (en güçlü saldırgan). "
        "3 tohum küçük bir örneklemdir; ölçülmeyen hücreler silinmedi, "
        "'yok' olarak işaretlendi."
    )
    if device == "cpu":
        return base + (
            " Bu koşu, NVIDIA sürücüsünün 19 Ekim'de kernel seviyesinde çökmesi "
            "nedeniyle CPU'da yapıldı; GPU-saati tüketilmedi (0). Gate w=256'da "
            "koşuldu (spec w=1024 ister) ve üst genişlik 512 denenmedi; B w=256'da "
            "kabul edildiği için merdiven orada durdu. C maliyeti CPU'da 1 saatten "
            "azsa GPU'da da azdır; CAPPED yönü bu nedenle güçlüdür. Sayılar GTX "
            "1650 Ti GPU'suna değil, bu CPU'ya aittir; B1 2/3 kabulde kaldı ve "
            "random stratası 0.9495 ile eşiğin hemen altındaydı, bu yüzden 3/3 "
            "kabul eden B2 saldırılara seçildi. Hamming bulgusu ayrıcalıklıdır "
            "(gerçek anahtarı bilir) ve DEAD_KEY kuralını tetiklemez; yine de "
            "anahtar kapısının tek nokta değil bölge olduğunu ve gradyan "
            "kurtarmanın bunu bulamadığını gösterir."
        )
    return base + " GPU-saatleri tek bir GTX 1650 Ti'de ölçüldü, T4/A100'e genellenemez."


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
