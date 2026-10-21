VERDICT: POLYNOMIAL

Karar: Aynı örnek bütçesinde kural ya tam öğreniliyor (T95=ufuk) ya da bir adım hatası yörüngeyi hemen öldürüyor. Geçiş, T'ye bağlı üstel bir maliyet değil, keskin bir öğrenilebilirlik eşiği.

| k | n | one-step acc | T95 mean (min–max) | TV | exact-state |
|---|---:|---:|---:|---:|---:|
| 3 | 3000 | 0.9987 | 87.0 (3–129) | 0.0001 | 0.6731 |
| 3 | 10000 | 1.0000 | 129.0 (129–129) | 0.0000 | 1.0000 |
| 3 | 30000 | 1.0000 | 129.0 (129–129) | 0.0000 | 1.0000 |

Kill kuralı: `kill_rule.md` (dondurulmuş).

**Sınırlamalar.** Tek anahtar, 3 tohum, 8x8 ikili CA, CNN öğrenci, T_max=128. Ölçüt **özerk yörünge** doğruluğudur; dağılım TV'si ayrıca raporlanır. Öğrenciye kod/ağırlık verilmedi; sadece I/O örnekleri verildi. n=3000'de 3 tohumun 2'si tam, 1'i kırılgan; eşik tohum/optimizasyona bağlı. k=4 (65k-bit) en iyi CPU koşusunda tek adım %88.5, k=5 %50; yüksek tarifte darboğaz hata birikimi değil **öğrenilebilirlik**.

**Sonraki adım.** CA sonlu bir Boolean kuralı olduğu için öğrenme ya tam ya hiç; üstel-T eğrisi için sürekli/smooth fixed-point kaotik harita (keyed perturbation) gerekir. Aksi halde bu negatif sonuç bu formda geçerlidir; k=5 33M-bit kuralı öğrenilebilirlik tavanını gösteren üst sınır olarak tutuldu.
