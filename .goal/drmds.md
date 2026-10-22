# shared-session — Neural DRM ortak bağlam

Son güncelleme: 2026-10-21 (opencode / DeepSeek V4.1 Flash)

Bu oturum, web ajanının takip edebilmesi için araştırmanın güncel durumunu taşır.
Her anlamlı adımda sona “Güncelleme” bölümü eklenir; tam konuşma arşivi değildir.

## Proje

Soru: Lisans mantığını oyun motoru/render ile aynı öğrenilmiş temsilde eritmek
(entanglement), anahtarı bilmeyen saldırganın “oyunu koruyarak lisansı aşma”
maliyetini gerçekten yükseltir mi?

Tehdit modeli: istemci tarafında çalışan oyun; saldırgan kod/ağırlık görebilir,
lisanslı sistemden I/O örnekleyebilir; bellek okuma (T4) kapsam dışı.

## Tamamlanan karar deneyi: VERDICT = CAPPED

- Görev: hash-grid value noise, c=8 (c=4 ile korelasyon 0.998; zorluk kolu fiilen çalışmadı).
- A: f_net + g_net + açık kapı; C: lisanssız distillation; B1: concat key; B2: FiLM key.
- Koşu CPU’da yapıldı (NVIDIA sürücüsü 19 Ekim’de kernel seviyesinde çöktü; CUDA
  context açılıyor ama ayırma yapamıyor; reboot gerekli). GPU-saati = 0, CPU 1.91 saat.
- B2 w=256/20k, 3/3 tohumda kabul: valid R²≈0.995/acc≈0.980; random/h1/h2 yüksek.
- B1 2/3 kaldı (seed 1 random acc 0.9495, eşiğin hemen altı); saldırılar B2’ye yapıldı.
- B2 fine-tune min n (tohum sırasıyla): 10000 (5/6 anahtar) / 100000 (5/6) / yok.
- Gradyan anahtar kurtarma: 20 başlangıç, başarısız (en iyi Hamming 29).
- C: min n=10000, arama dahil 0.931 CPU-saat; en ucuz protokol n=100000, 0.143 CPU-saat.
- B/C maliyet oranı: 0.571 (en ucuz C’ye göre); A maliyeti 0.
- Kritik sızıntı: seed 0’da Hamming 50–53 uzaklıktaki yanlış anahtarlar f’yi
  R²≈0.97/acc≈0.97 ile üretti. Yani anahtar tek nokta değil, bir **bölge**; gradyan
  bunu bulamadı ama kapı sızdırıyor.
- Sonuç: B, fine-tune ve gradyan key recovery’den kurtuldu; ama görevin öğrenilme
  maliyeti (C) 1 GPU-saatin altında olduğu için koruma CAPPED. Entanglement,
  distillation’ın ötesinde ek koruma sağlamıyor.

Paket: `<home>/Downloads/neural-drm-decision-cpu.zip`
Repo: `neural-drm-phase0` (VERDICT.md, figures/, results/decision_cpu_v1).

## Neden süperpozisyon compute zorluğu üretmiyor (tez)

Süperpozisyon **temsili** gizler, **fonksiyonu** değil. Saldırgan parametreleri
anlamak zorunda değil; I/O’yu eşleyen herhangi bir fonksiyon yeter. Fine-tune ve
distillation çıktı uzayında optimizasyon yapar; temsil karışıklığı bu optimizasyonu
zorlaştırmaz. Ağ pürüzsüz olduğu için anahtar bir bölge olur; kriptografik sır ise
keskin (tek nokta) olmak zorundadır. “Dağıtık + keskin” = obfuscation; genel kara
kutu obfuscation imkânsız, iO pratik değil.

## Yeni hipotez (goal-loop, aktif)

Hata büyüten (expansive) ve yüksek tarif karmaşıklığına sahip, anahtarlı, tamsayı
aritmetikli bir zincirleme fonksiyon, yaklaşık distillation’ı uzun yörüngede
kullanılamaz hale getirir mi? Yani tek adımda %99 doğru bir öğrenci, T adım sonra
0.01·e^(λT) hatasıyla dağılır; gerekli örnek sayısı T ile üstel büyür mü?

Testbed: 8×8 ikili hücresel otomat; her hücrenin sonraki durumu, anahtarla
seçilen k×k komşuluk kural tablosundan gelir.
- k=3: 512-bit kural (düşük tarif kontrolü). 5000 adımda tek adım %100; 64 adımda
  T95=65 (hiç bozulmuyor) — kural tam öğrenildi.
- k=5: 33.5M-bit kural (yüksek tarif). n=300k’da tek adım ~%50; öğrenci kuralı
  öğrenemiyor. Yani yüksek tarif, hata birikiminden önce **öğrenmeyi** öldürüyor.
- k=4: 65k-bit kural (orta). İlk denemede CNN 32 kanal ile ~%54’te kaldı; kapasite
  yetersizliği şüphesi var. 256 kanal / daha çok adım ile kalibre ediliyor.

Kill kuralı (önceden): bir adım ≥0.99’a ulaşılamazsa INCONCLUSIVE; k=5’te T95≥128
veya n(T) polinomsa fikir bu formda ölür; T95 ~ log n ise üstel sample complexity
kanıtı sayılır. Çok adımlı (free-running) öğrenci de test edilecek.

## Açık sorun

k=4’te CNN’in neden şansa yakın kaldığı: 16-bit kural için 32 kanal az olabilir;
256 kanal, 4000 adım, n=100k ile tek adım ≥0.99 hedefleniyor. k=3 çalıştığı için
boru hattı ve hizalama doğru; sorun kapasite/örnekleme.

## Güncelleme protokolü

- Her milestone’da (kalibrasyon, sweep, verdict) bu dosyanın sonuna tarihli
  “Güncelleme” eklenir ve MCP’ye aynı session olarak push edilir.
- Web ajanı `session_list` ile “shared-session”i bulup `session_get` ile okuyabilir.
- Eski metin silinmez; yalnız ekleme yapılır.

## Güncelleme — 2026-10-21 (kalibrasyon)

- k=3 (512-bit kural): n=10k, 5000 adım → tek adım %100; 64 adımda T95=65/64
  (hiç bozulmuyor). Kural tam öğrenildi; T önemsiz.
- k=4 (65.536-bit): n=100k, 4000 adım, 256 kanal → tek adım **%88.5**, T95=1.
  n=30k, 1200/1500 adımda %54-56. Öğrenci bir adımı bile %99’a çıkaramıyor.
- k=5 (33.5M-bit): n=300k → %50 (şans).
- Yorum: üstel-T hata birikimi hipotezi bu testbedde **desteklenmiyor**; darboğaz
  hata birikimi değil, **öğrenilebilirlik/tarif uzunluğu**. k=3’te kural öğrenilince
  T95 tam ufka çıkıyor (eşik etkisi); k=4’te tek adım öğrenilemiyor.
- Kill rule donduruldu ve commit edildi (`e1381e9`). Onay koşusu: k=3, n∈{3k,10k,30k},
  3 tohum, 2000 adım, 64 kanal, T=128; beklenen etiket POLYNOMIAL.
- Sırada: sweep sonucu, `VERDICT_EXPANSIVE.md`, çok adımlı öğrenci notu.

## Güncelleme — 2026-10-21 (onay sweep, ilk koşu)

- k=3, n=3000, seed 0: tek adım **%99.62**, T95=**3**, dağılım TV=0.0004.
- Yorum: %0.38 tek-adım hatası 3 adımda yörüngeyi bozuyor; hata birikimi
  bu marjda gerçek. Ancak bu, “üstel örnek karmaşıklığı” demek değil: n=10k/30k’da
  tek adım %100’e çıkarsa T95 tam ufka sıçrar (eşik etkisi). Sweep bunu ölçüyor.
- Sırada: n=10k/30k, 3 tohum; sonra `VERDICT_EXPANSIVE.md`.

## Güncelleme — 2026-10-21 (onay sweep, eşik)

- Aynı n=3000, iki tohum iki farklı tarafta:
  - seed 0: tek adım %99.62 → T95=**3** (yörünge 3 adımda ölüyor)
  - seed 1: tek adım **%100** → T95=**129** (128 adımda hiç bozulmuyor)
- Bu, “üstel T” değil **eşik** davranışı: örnek bütçesi kritik eşiğe yakınken
  tohum/optimizasyon sonucu belirliyor. Eşiğin üstünde kural tam öğreniliyor ve
  T önemsiz; altında %0.4 hata bile yörüngeyi öldürüyor.
- Kalan: n=10k/30k, 3 tohum. Beklenen: üst bölgede T95=129; etiket POLYNOMIAL.

## Güncelleme — 2026-10-21 (SONUÇ)

- **VERDICT_EXPANSIVE: POLYNOMIAL** (kill).
- k=3 onay sweep: n=3000 → tek adım 0.9987, T95 **3 / 129 / 129** (2 tohum tam,
  1 tohum kırılgan). n=10k ve 30k → tek adım 1.0000, T95 **129 / 129 / 129**,
  exact-state 1.0000.
- k=4 kalibrasyon: en iyi CPU koşusu 100k/4000/256 → tek adım **0.885**, T95=1.
  k=5: 0.50 (şans).
- Yorum: hata birikimi eşiğin **altında** gerçek (%0.4 hata → 3 adım), ama geçiş
  keskin bir **öğrenilebilirlik eşiği**. Kural tam öğrenilince T önemsiz; üstel-T
  maliyeti yok. Yüksek tarifte darboğaz hata birikimi değil, öğrenilebilirlik.
- Audit: 6/6 test, 9 ham koşu, `VERDICT_EXPANSIVE.md` yeniden üretildi, kill rule
  değişmedi (commit `e1381e9`).
- Sonraki: CA sonlu Boolean kuralı olduğu için “ya tam ya hiç”; üstel-T eğrisi
  için sürekli/smooth fixed-point keyed kaotik harita gerekir. Aksi halde bu
  negatif sonuç bu formda geçerlidir.

## Güncelleme — 2026-10-22 (smooth tent SONUÇ)

- **VERDICT_TENT: INCONCLUSIVE**.
- 3 tohum, n=10k/30k/100k, 256x2, 4k adım: one-step RMSE **4.56 / 4.00 / 4.64e-3**;
  T95 (durum hatası >5%) **6 / 6 / 6** (min=max). Örnek sayısı lever’ı düz.
- 512x3 + 12k adım: RMSE **9.4e-4**, T95 **8**. Yani T95 **compute/kapasite** ile
  büyüyor, n ile değil. 128 adım için gereken hassasiyet ~1e-36.
- Hata birikimi gerçek: %0.4 tek-adım hatası 6 adımda %5’i geçiyor; Lyapunov
  tahminine yakın. Ama “üstel sample complexity” yok; maliyet hesaplama.
- CA: POLYNOMIAL (keskin eşik). Tent: INCONCLUSIVE (compute-limited). İkisi de
  “expansive + yüksek tarif = üstel distillation maliyeti” iddiasını desteklemiyor.
- Sonraki: aynı n’de model boyutu/adım sayısını artırıp T95’in log(compute) ile
  büyümesini ölç; ya da bu negatif sonucu bu formda kabul et.

## Güncelleme — 2026-10-22 (negatif sonuç notu)

- `docs/negative-result.md` yazıldı: üç deney, ortak tez, neyi değiştirir.
- Özet: entangleman **temsili** gizler, **fonksiyonu** değil. Tavan:
  öğrenilebilirlik + tarif uzunluğu + compute. Anahtar bir bölge, kriptografik
  nokta değil. Etiketler: CAPPED / POLYNOMIAL / INCONCLUSIVE.
- Doğru sonraki proje: yüksek entropili içerik + kriptografik trapdoor + TEE/remote;
  ya da bu negatif sonucu not/blog olarak yayınla. Entanglement toy varyantlarını
  kovalamak düşük getirili.
