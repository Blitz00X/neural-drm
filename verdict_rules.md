# verdict_rules.md — karar kuralları (koşudan önce donduruldu)

Bu dosya ilk deney biriminden **önce** commit edilir. Sonuç görüldükten sonra
etiket, eşik, merdiven adımı veya kural değiştirilemez. Değişiklik gerekirse
etiket `INCONCLUSIVE` olur ve nedeni `VERDICT.md`'ye yazılır.

## Tanımlar (protokolün parçası)

- **Yüksek fidelity**: 100.000 bağımsız test noktasında `R²_height >= 0.95` **ve**
  `acc_biome >= 0.95`.
- **B kabulü (tek tohum)**: doğru anahtarla f'ye yüksek fidelity **ve** üç yanlış
  anahtar stratasının (random, Hamming-1, Hamming-2) her birinde decoy g'ye
  yüksek fidelity. Stratlar 100.000'er koordinat-anahtar çiftiyle ölçülür.
- **B kabulü (kombinasyon)**: bir `(arm, width, steps)` kombinasyonunda 3 tohumun
  en az 2'sinde B kabulü. `arm ∈ {B1, B2}`.
- **A kabulü**: bir genişlikte 3 tohumun en az 2'sinde f_net f'ye ve g_net g'ye
  yüksek fidelity. A'nın kapı silme (H1) testi f_net'in doğrudan çağrılmasıdır;
  ek optimizasyon maliyeti yoktur.
- **C kabulü**: lisanssız öğrencinin herhangi bir `(width, n, steps)` ile testte
  yüksek fidelity'ye ulaşması.
- **Bypass**: 6 anahtarla (sabit hepsi −1 + 5 rastgele yanlış anahtar) testte
  en az 5 anahtarın f'ye yüksek fidelity vermesi.
- **Min fine-tune n**: `{100, 1000, 10000, 100000}` kümesinde, doğrulama ile
  seçilen 3 LR'den biriyle bypass'ın ilk sağlandığı n. Hiç sağlanmazsa `yok`
  (sağdan sansürlü).
- **Anahtar kurtarma başarısı**: 1000 oracle örneği, 20 başlangıç, 1000 gradyan
  adımı, sürekli `[-1,1]^64`, ±1'e yuvarlama; yuvarlanmış anahtarın testte
  yüksek fidelity vermesi.
- **C maliyeti**: yüksek fidelity'ye ulaşan en ucuz C protokolünün, o noktaya
  kadarki **tüm** denemelerinin (başarısız LR'ler ve daha küçük n'ler dahil)
  GPU-saniye toplamı. GPU-saat = saniye / 3600.
- **A maliyeti**: 0 GPU-saat (kapı silme).
- **B/C maliyet oranı**: başlık min fine-tune n'deki ortalama fine-tune
  GPU-saatinin C GPU-saatine oranı.
- **Eksik kanıt**: bir kuralın önkoşulu ölçülmediyse o kural eşleşmiş sayılmaz.
  Hiçbir kural tam eşleşmezse etiket `INCONCLUSIVE` olur.

## Sıralı etiketler (ilk eşleşen kazanır)

1. **DEAD_CAPACITY** — A ve C kabul edildi; merdivendeki hiçbir
   `(width, steps, parametre-eşit)` kombinasyonunda B1 veya B2 kabul edilmedi
   (her kombinasyonda kabul edilen tohum sayısı < 2).
2. **DEAD_FINETUNE** — Kabul edilen bir B kombinasyonunda, 3 tohumun en az
   2'sinde `min fine-tune n <= 1000` ile bypass.
3. **DEAD_KEY** — Kabul edilen herhangi bir B kombinasyonunda, herhangi bir
   tohumda anahtar kurtarma 1000 oracle örneğiyle başarılı.
4. **CAPPED** — B, 2 ve 3'ten kurtuluyor; ancak C'nin yüksek fidelity maliyeti
   `1 GPU-saatinden az`. Koruma, görevin öğrenilme maliyetiyle sınırlı.
5. **ALIVE_WEAK** — B hayatta; en az 2/3 tohumda `1000 < min fine-tune n <= 10000`
   ile bypass.
6. **ALIVE** — B kabul edildi; en az 2/3 tohumda fine-tune bypass `n >= 100000`
   gerektiriyor veya hiç olmuyor; anahtar kurtarma başarısız; C maliyeti
   `>= 1 GPU-saat`.

## Merdiven (sonuç görülmeden sabit)

- Genişlikler: `w = 256 -> 512 -> 1024`, 4 gizli katman, GELU.
- Bütçe: `20k -> 60k` adım, batch 512, Adam, lr 1e-3 + cosine.
- B kabul edilmezse sırayla: (a) bütçeyi 60k'ya çıkar, (b) B genişliğini
  `×1.4142` yaparak A'nın parametre sayısına eşitle, (c) bir üst genişliğe çık.
- İlk kabul edilen kombinasyonda durulur; saldırılar o genişlikte ve mümkünse
  bir üst genişlikte koşulur.
- Görev oktavı: önce c=8. C, w=1024 ve en yüksek bütçede (60k) yüksek fidelity'ye
  ulaşamazsa c=4'e düşülür ve bu `DECISIONS.md`'ye yazılır. Eşik değişmez.

## Dürüstlük

- Başarısız, NaN veya yarım koşular silinmez; `results/` altında kalır ve sayılır.
- Şüpheli derecede iyi/kötü sonuçta önce kod hatası aranır (sızıntı, kapı testi,
  C'nin f'ye erişimi). Bu denetim `DECISIONS.md`'ye yazılır.
- Toy ölçek bulgusu "gerçek DRM kırılır/kırılmaz" diye genellenmez.
