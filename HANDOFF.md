# HANDOFF — Neural DRM araştırması, 2026-10-22

Bu dosya, araştırmayı bugün olduğu yerden **sonra devam ettirebilmek** için yazıldı.
Yeni bir deney veya plan değildir; mevcut durumun, sonuçların, ortamın ve devam
protokolünün kaydıdır.

## 1. Tek cümlelik durum

Entangled DRM hattı **negatif sonuçla kapandı**: üç deney de (CAPPED, POLYNOMIAL,
INCONCLUSIVE) entanglemanın bir güvenlik sınırı üretmediğini, tavanın görevin
öğrenilebilirliği/hesaplama maliyeti olduğunu gösterdi. Kod, ham sonuçlar, üç
verdict ve ortak negatif sonuç notu repoda; paylaşımlı `shared-session` oturumu güncel.

## 2. Araştırma sorusu ve hipotez

Soru: Lisans mantığını oyun motoru/render ile aynı öğrenilmiş temsile gömmek
(entanglement), anahtarı bilmeyen saldırganın “oyunu koruyarak lisansı aşma”
maliyetini gerçekten yükseltir mi?

Tehdit modeli: istemci tarafında çalışan oyun; saldırgan kodu/ağırlıkları
görebilir, lisanslı sistemden I/O örnekleyebilir; bellek okuma (T4) kapsam dışı.

Hipotez: süperpozisyon/dağıtık temsil, lisans fonksiyonunu oyun fonksiyonundan
ayrıştırmayı zorlaştırır; bu da distillation ve fine-tune maliyetini yükseltir.

## 3. Deneyler ve sonuçlar

### 3.1 Faz 0 (bağlam)

- w=64, c=4, 5 tohum CPU: A ve C kabul; B1/B2 kabul edilemedi.
- B1 R²≈0.91–0.93; B2 R²≈0.95–0.97 ama biome acc 0.92–0.945.
- Yorum: w=64'te B **kapasite** sorunu; karar deneyi w=256'ya çıktı.

### 3.2 Karar deneyi — `VERDICT.md`: **CAPPED**

- Görev c=8; c=8 ile c=4 korelasyonu 0.998 (zorluk kolu fiilen çalışmadı).
- B2 w=256/20k, **3/3 tohumda kabul**: valid R²≈0.995/acc≈0.980; random/h1/h2
  yüksek. B1 2/3 (seed 1 random acc 0.9495).
- B2 fine-tune min n (tohum sırasıyla): **10000 (5/6) / 100000 (5/6) / yok**.
- Gradyan anahtar kurtarma: 20 başlangıç, **başarısız** (en iyi Hamming 29).
- **Kritik sızıntı:** seed 0'da Hamming **50–53** uzaklıktaki yanlış anahtarlar
  f'yi **R²≈0.97/acc≈0.97** ile üretti. Anahtar tek nokta değil, **bölge**; kaba
  etkin anahtar uzayı 2⁶⁴ değil ~10⁶ deneme.
- C: min n=10000, arama dahil **0.931 CPU-saat**; en ucuz protokol n=100000,
  **0.143 CPU-saat**. B/C oranı **0.571**; A maliyeti 0.
- Sonuç: B, fine-tune ve gradyan key recovery'den kurtuldu; ama C 1 GPU-saatin
  altında olduğu için **CAPPED**. Entanglement, distillation'ın ötesinde ek
  koruma sağlamıyor.

### 3.3 CA (expansive, sonlu Boolean kural) — `VERDICT_EXPANSIVE.md`: **POLYNOMIAL**

- 8×8 ikili hücresel otomat; k×k komşuluk; k=3 512-bit, k=4 65.536-bit,
  k=5 33.5M-bit kural.
- k=3 onay sweep (3 tohum, 2k adım, 64 kanal): n=3000 → tek adım 0.9987,
  **T95 3 / 129 / 129**; n=10k/30k → tek adım 1.0000, **T95 129/129/129**.
- k=4: en iyi CPU koşusu 100k/4000/256 → tek adım **0.885**, T95=1.
- k=5: **0.50** (şans).
- Yorum: kural ya tam öğreniliyor (T önemsiz) ya da bir adım bile öğrenilemiyor
  (öğrenilebilirlik bariyeri). Geçiş **keskin eşik**, üstel-T değil.

### 3.4 Smooth tent map — `VERDICT_TENT.md`: **INCONCLUSIVE**

- 16-bit keyed tent map (a≈1.91); 1D skaler regresyon MLP; T95 = durum hatası
  %5'i geçtiği adım.
- 3 tohum, 256×2, 4k adım: n=10k/30k/100k → RMSE **4.56 / 4.00 / 4.64e-3**;
  T95 **6 / 6 / 6**.
- 512×3 + 12k adım: RMSE **9.4e-4**, T95 **8**.
- Yorum: hata birikimi gerçek (%0.4 hata → 6 adım; Lyapunov'a yakın), ama
  örnek sayısı lever'ı düz. T95 **compute/kapasite** ile büyüyor; 128 adım için
  gereken hassasiyet ~1e-36. Maliyet sample değil hesaplama.

## 4. Ortak sentez

1. Süperpozisyon **sıkıştırmadır, şifreleme değil**. Aynı nöronun çoklu özellik
   taşıması ileri geçişin FLOP'unu artırmaz; yalnızca attribution'ı zorlaştırır.
   Saldırgan attribution yapmaz; I/O'yu eşler.
2. **Anahtar keskin olmalı, ağ yumuşak.** Ağ anahtarı bir **bölge** olarak öğrenir;
   kriptografik sır tek nokta ister. “Dağıtık + keskin” = obfuscation; genel kara
   kutu obfuscation imkânsız, iO pratik değil.
3. **Doğru kıyas A değil, C.** A'nın kapı silme maliyeti 0; ama saldırganın her
   zaman C (distillation) seçeneği var. B2'nin fine-tune'u bir tohumda C'den ucuz
   (0.571) ve yalnızca 5/6 anahtar/1/3 tohum. Entanglement yalnızca “kapıyı arayıp
   silen tembel saldırgana” karşı işe yarıyor.
4. Hata birikimi yerel olarak gerçek, ama **güvenlik maliyeti değil**: 128 adım
   için ~1e-36 hassasiyet gerekir; kaldıraç compute/kapasite.
5. Üç deneyin ortak tavanı: **öğrenilebilirlik + tarif uzunluğu + hesaplama**.

## 5. Ortam ve “durum”

- Donanım: NVIDIA GTX 1650 Ti, 4 GB; sürücü **580.178.04**.
- **GPU arızası:** 19 Ekim'de kernel seviyesinde NVIDIA RM çökmesi
  (`nvidia_close/rm_cleanup_file_private`). 21 Ekim'de CUDA context açılıyor ama
  `cudaSetDevice` ve 8 float'lık ayırma bile **OOM**; `nvidia-smi` 3.7 GB boş
  gösteriyor. Sürücü sürümleri uyumlu; **reboot** gerekiyor; sudo yok.
- Bu yüzden tüm bilimsel koşular **CPU'da**: GPU-saati **0**; karar deneyi
  1.91 CPU-saat; CA/tent CPU'da.
- Python:
  - CUDA venv: `<cuda-venv>` (torch 2.6.0+cu124;
    CUDA pratikte kullanılamıyor).
  - CPU koşuları: `<cpu-venv>/bin/python`
    (torch 2.14.1+cpu, numpy 2.4.6, matplotlib 3.11.2).
- Repo: `neural-drm-phase0`.
- Paylaşımlı oturum: `shared-session`, Shared Session MCP,
  `<internal-session-endpoint-omitted>`,
  session_id **`<session-id-omitted>`**. Web ajanı
  `session_list`/`session_get` ile okur.
- Paket: `~/Downloads/neural-drm-decision-cpu.zip` (karar deneyi; CA/tent
  sonuçları bu zip'ten sonra eklendiği için güncel değil — güncel kaynak repo).

## 6. Artefaktlar ve commit'ler

| Dosya | İçerik |
|---|---|
| `docs/negative-result.md` | Üç deneyin ortak negatif sonuç notu |
| `VERDICT.md` | Karar deneyi, **CAPPED** |
| `VERDICT_EXPANSIVE.md` | CA, **POLYNOMIAL** |
| `VERDICT_TENT.md` | Tent, **INCONCLUSIVE** |
| `results/decision_cpu_v1/` | Ham JSON, modeller, state, summary |
| `results/expansive_v1/` | 9 koşu + calibration + summary |
| `results/tent_v1/` | 9 koşu + calibration + summary |
| `figures/` | Karar, CA, tent figürleri |
| `.goal/` | goal-loop manifest/log/audit; CA arşivi |

Önemli commit'ler:
`aaa7401` negatif not · `01ffcfc` tent verdict · `18cea8b` CA verdict ·
`4a93dd4` karar CAPPED · `e1381e9` CA freeze · `38191ce` tent freeze.

## 7. Sonra nasıl devam edilir

### 7.1 Önce oku
```bash
cd neural-drm-phase0
cat docs/negative-result.md
cat HANDOFF.md
git log --oneline -10
```
`shared-session` oturumunu oku; `.goal/manifest.json` durumu `paused`.

### 7.2 Ortamı doğrula
```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
# GPU düzeldiyse (reboot sonrası) küçük bir ayırma testi:
python3 -c "import torch; x=torch.ones(8,device='cuda'); print(x.sum())"
```
GPU çalışmıyorsa CPU venv ile devam et; bilimsel sonuçlar CPU'da geçerli.

### 7.3 Bilimsel devam (düşük getirili)
- Aynı hattı daha fazla entangleman toy'uyla kovalamak önerilmez; üç deney aynı
  tavana çıktı.
- Tek anlamlı tamamlama: **compute-scaling** — aynı n'de model boyutu/adım
  sayısını artırıp T95'in log(compute) ile büyümesini ölçmek. Beklenti:
  compute-limited; yön değişmez.
- Daha güçlü proje: **yüksek entropili içerik + kriptografik trapdoor + TEE/remote**.
  Bu, entangleman değil anahtar yönetimi/erişim kontrolü problemidir.

### 7.4 GPU'lu spec koşusu (reboot sonrası)
```bash
python run_decision.py --config configs/decision.json \
  --output results/decision_gpu_v1 --phase all
```
`configs/decision.json` w=256/512/1024, 3 tohum, c=8, GPU. CPU kararı ile
karşılaştır; fark varsa `DECISIONS.md`'ye yaz.

### 7.5 Test/üretim komutları
```bash
python -m unittest tests.test_decision tests.test_expansive tests.test_tent -v
python experiments/run_expansive.py --config configs/expansive.json
python experiments/analyze_expansive.py
python experiments/run_tent.py --config configs/tent.json
python experiments/analyze_tent.py
```

## 8. Açık kararlar (kullanıcıya ait)

1. Negatif sonuç **yayınlanacak mı** (blog/arXiv-style not), yoksa özel mi kalacak?
2. Bilimsel hat **kapatıldı mı**, yoksa compute-scaling/trapdoor projesine mi
   geçilecek?
3. Ürün/ilk satış hattına dönülecek mi? (Bu rapor o kararı vermiyor.)

## 9. Dürüstlük notları

- Eşikler, `verdict_rules.md`, `kill_rule*.md` sonuçtan sonra değiştirilmedi.
- Başarısız/yarım koşular silinmedi; CPU/GPU sapmaları `DECISIONS.md` ve
  verdict'lerde yazılı.
- Bu bir oyuncak prosedürel dünya kurulumudur; “gerçek DRM kırılır/kırılmaz”
  genellemesi yapılmadı.
- Karar deneyi CPU'da; GPU-saati 0. CA/tent de CPU'da; sayılar bu donanıma ait.
- Tek anahtar, 3 tohum, T_max=128; sonuçlar bu ölçeğin dışına taşınmamalı.
