VERDICT: INCONCLUSIVE

Karar: T95 n ile düz; tek-adım hatası model/optimizasyon tabanında. Örnek sayısı değil hesaplama/kapasite kaldıracı.

| n | one-step RMSE | T95 mean (min–max) | top-grid |
|---:|---:|---:|---:|
| 10000 | 4.56e-03 | 6.0 (6–6) | 0.700 |
| 30000 | 4.00e-03 | 6.0 (6–6) | 0.712 |
| 100000 | 4.64e-03 | 6.0 (6–6) | 0.704 |

Kill kuralı: `kill_rule_tent.md` (dondurulmuş).

**Sınırlamalar.** Tek anahtar, 3 tohum, 16-bit tent map, 1D MLP, T_max=128. Ölçüt skaler durum hatası >5%; top-grid ikincil. Kalibrasyonda 512x3/12k model RMSE 9.4e-4/T95=8 verdi; 256x2/6k model 2.9e-3/T95=6. Yani T95 hesaplama/kapasite ile büyüyor, n ile değil. Hata birikimi Lyapunov tahminine yakın; 128 adım için gereken hassasiyet ~1e-36.

**Sonraki adım.** Örnek değil hesaplama ölçeklemesini ölç: aynı n'de model boyutu/adım sayısını artırıp T95'in log(compute) ile büyümesini test et. Eğer T95 yalnızca compute ile büyüyorsa, savunma “sample complexity” değil “hesaplama maliyeti” sınırıdır; bu, CAPPED/CA sonuçlarıyla aynı aileye girer.
