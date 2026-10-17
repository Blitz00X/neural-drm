# GÖREV: "Entangled DRM" hipotezi için Faz 0 deneyini uçtan uca kur, çalıştır, raporla

## 0. Nasıl çalışmanı istiyorum
- Sen otonom bir araştırma mühendisisin. Bana soru sormadan çalış; belirsiz noktada en makul varsayımı yap, bunu `DECISIONS.md` dosyasına gerekçesiyle yaz ve devam et.
- Kod, yorum, değişken adları İngilizce. Final rapor (`REPORT.md`) ve `DECISIONS.md` Türkçe. Teknik terimleri (fine-tune, distillation, fidelity vb.) İngilizce bırakabilirsin.
- Hedefim "fikir çalışıyor" cevabını almak DEĞİL, **doğru cevabı almak**. Negatif sonuç, kurulum hatasından çıkan sahte pozitif sonuçtan çok daha değerli. Sonucu güzelleştirme, başarısız koşuları silme, eşikleri sonradan oynatma.
- Her aşamada küçük adımlarla ilerle, her anlamlı adımda git commit at.

## 1. Arka plan (bağlam)
Fikir: Geleneksel DRM'de oyun fonksiyonu ile lisans kontrolü ayrı bileşenlerdir (`if (license_valid)`). Saldırgan kontrolü atlatır/siler, oyun çalışmaya devam eder. Fikrim, LLM'lerin yorumlanamamasının temel nedeni olan **dağıtık temsil / süperpozisyonu** (bir nöron ya da boyut tek bir kategoriyi temsil etmez; özellikler birçok nöronun birleşiminde yaşar) bir yazılım koruması olarak kullanmak: lisans doğrulaması ile oyun hesaplaması **aynı fonksiyonun, aynı parametrelerin içinde iç içe** olsun; "DRM'yi çıkar ama oyunu koru" operasyonu ucuz/yerel olmaktan çıksın.

Bilinen bağlam ve sınırlar (bunları deneyin tasarımında dikkate al):
- Yorumlanamazlık ≠ müdahale direnci ≠ kopyalanamazlık. LLM ağırlıkları yorumlanamaz ama kopyalanıp çalıştırılabilir. Bu yüzden asıl soru yorumlanabilirlik değil, **lisans bağımlılığını görev performansını koruyarak kaldırmanın maliyeti**.
- Saldırgan lisanslı çalışan sisteme sorgu atabiliyorsa, oyun fonksiyonunu örneklerden yeniden öğrenebilir (distillation). Bu, her savunmanın güvenlik tavanıdır. DRM'e atfedilebilir ek maliyet = (entangled sisteme saldırı maliyeti) − (fonksiyonu sıfırdan öğrenme maliyeti).
- Lisanslı çalışma sırasında bellekten parametre/anahtar yakalama (T4) kapsam dışı; baştan "kırılır" kabul ediliyor.
- İlgili literatür (bağlam için, tekrar yazman gerekmiyor): Toy Models of Superposition; Towards/Scaling Monosemanticity (SAE); Deep-Lock (anahtar tabanlı ağırlık koruması); NNSplitter (az sayıda ağırlığı değiştirerek doğruluğu çökertmek); password-locked models (az örnekle kilit açılabiliyor, bu bizim için ciddi bir tehdit); tamper-resistant safeguards (min-max eğitim).

## 2. Bilimsel soru
> Lisansı ve oyun işlevini tek bir sinir ağı fonksiyonunda birlikte eğitmek (entanglement), ayrık bir lisans kapısına kıyasla, "oyun doğruluğunu koruyarak lisans bağımlılığını kaldırma" maliyetini ölçülebilir şekilde artırıyor mu? Bu maliyet model genişliğiyle nasıl ölçekleniyor?

## 3. Hipotezler ve ölüm kriterleri (DENEYİ KOŞMADAN ÖNCE `hypotheses.md` olarak yaz ve commit et; sonra değiştirme)
- **H1 (sağlama/sanity):** A kolunda (ayrık kapı) kapıyı silmek ~sıfır maliyetle ≥%95 fidelity verir. Vermezse kurulum hatalı; düzeltmeden devam etme.
- **H2:** B kolunda lisans duyarlılığını ablasyonla gidermek, görev doğruluğunda A'ya göre belirgin kayıp yaratır (ablasyon Pareto eğrisi A'dan anlamlı kötü).
- **H3:** B kolunda fine-tune ve anahtar kurtarma saldırılarının maliyeti, A'dan ve C'den en az 10 kat yüksek ve genişlikle artıyor.

Karar tablosu (rapora aynen taşı ve sonucu buna göre etiketle):
| Sonuç | Anlamı |
|---|---|
| H2 ve H3 geçiyor | Fikir çalışıyor; Faz 1'e geçilir |
| H2 geçiyor, H3 düşüyor (az örnekle açılıyor) | Statik analize karşı iyi, fine-tune'a karşı zayıf; min-max eğitim eklenmeli |
| İkisi de düşüyor | Entanglement ayrık kapıdan farksız; fikir bu formda çürüdü, nedeni raporlanır |
| Her şey C'ye çöküyor | Güvenlik tavanı fonksiyonun öğrenme karmaşıklığı; bu ölçülmüş bir sınır olarak raporlanır |

## 4. Ortam ve genel kurallar
- Python 3.10+, PyTorch, tek GPU (Colab/T4 seviyesi yeterli olmalı). Gerekirse CPU'da küçük konfigürasyonlar.
- Tam tekrarlanabilirlik: tüm rastgelelik tohumlanmış; her koşu bir `config.json` ve `seed` ile kaydedilir.
- Her konfigürasyon için **5 tohum**. Raporda ortalama ve %95 güven aralığı (veya min/max) göster.
- Önce her şeyi küçük bir pilotta (width 64, 1 tohum, kısa eğitim) uçtan uca çalıştır, boru hattını doğrula, sonra tam ızgaraya geç.
- Hesaplama maliyeti için FLOPs yaklaşığı: `6 * n_params * n_samples_processed` (eğitim için). Bunu tutarlı biçimde logla.
- Yarım kalan koşular, NaN'lar, eğitim başarısızlıkları dahil **her şeyi** logla ve raporda say.

## 5. Görev fonksiyonu `f` (oyunun yerine geçen "dünya üretici")
Deterministik prosedürel dünya: `f(x, y) -> (height, biome)`.
- Girdi: `(x, y)` in `[0,1]^2`.
- `height`: `c` oktavlı value/gradient noise toplamı (her oktavda frekans 2x, genlik 0.5x), `[0,1]`'e normalize. Gürültüyü PyTorch/numpy'de kendin, sabit tohumlu bir hash tabanlı rasgele ızgara ile uygula; harici gürültü kütüphanesine bağımlı olma.
- `biome`: height ve ikinci bağımsız bir noise alanından (`moisture`) hesaplanan 4 sınıflı (örneğin su/ova/orman/dağ) kural tabanlı sınıf. Yani çıktı: 1 sürekli + 1 kategorik (4 sınıf).
- Karmaşıklık parametresi `c in {2, 4, 8}` oktav.
- Veri sınırsız üretilebilir (analitik `f`). Train/val/test için ayrı, çakışmayan koordinat havuzları kullan.

**Fidelity metriği** (modelin `f`'yi ne kadar iyi temsil ettiği):
- `R2_height` = 1 − MSE/Var (test seti, ≥100k nokta)
- `acc_biome` = test doğruluğu
- **"Yüksek fidelity" = `R2_height >= 0.95` VE `acc_biome >= 0.95`**. Bu eşik tüm saldırı başarı ölçütlerinin tanımıdır.

## 6. Lisans mekanizması
- Gizli anahtar `k*`: 64 bit, her tohum için rastgele; ağa **{-1,+1}^64** vektör olarak girer.
- `k == k*` ise çıktı `f(x,y)`.
- `k != k*` ise çıktı sabit bir **decoy dünya** `g(x,y)` (aynı yapıda ama farklı noise tohumuyla üretilmiş, makul görünen ama yanlış bir dünya; çıktı çöp olmasın, bu saldırganın "bozuk" ile "doğru"yu kolayca ayırmasını engeller).
- Eğitim anahtar dağılımı: %50 doğru anahtar, %25 uniform rastgele yanlış anahtar, %25 `k*`'a Hamming uzaklığı 1-2 olan anahtar. (Aksi hâlde ağ yaklaşık anahtarları da kabul eder.)

## 7. Üç kol (+ B için iki varyant)
**A: Ayrık taban çizgisi (klasik DRM analoğu).** `f_net(x,y)` sadece görevi öğrenen MLP; yanında AÇIK, ayrı bir kapı: `out = f_net(x,y) if k==k* else g_net(x,y)` (g_net, decoy'u öğrenmiş ayrı ağ). Kapı ağın dışında, kolayca silinebilir olmalı. Bu H1 sağlaması içindir.

**B1: Entangled, düz birleştirme.** Tek MLP, girdi `[x, y, k]` (66 boyut), lisans için ayrı hiçbir modül/çıkış yok. Tüm kayıp (f ve g hedefleri) ortak eğitilir.

**B2: Entangled, daha derin iç içe geçirme.** Anahtar, her katmanda FiLM (ölçek+kayma) ile modülasyon yapar: sabit rastgele bir izdüşümle `k -> d` boyutlu gömme, her katmanda `h = (1+gamma(k)) * h + beta(k)`. Ayrı bir "lisans yolu" olmamalı. (Gerekçe: B1'de ağ "k==k*" kontrolünü tek bir lineer nöronla çözebilir; bu durumda lisans yerelleşir ve H2 zayıf çıkar. B2 bu riski test eder. Her iki varyantı da raporla, hangisinin neden iyi/kötü çıktığını yorumla.)

**C: Kontrol (lisanssız sıfırdan öğrenici).** Anahtarsız, `f`'yi sadece `(x,y)->f` örneklerinden sıfırdan öğrenen MLP. Bu, distillation tavanını verir. DRM'e atfedilebilir ek maliyet her zaman C'ye göre hesaplanır.

**Mimari ızgara:** 4 gizli katman, genişlik `w in {64, 256, 1024, 4096}`, aktivasyon GELU. Oktav `c=4` için tüm genişlikler; `c=2` ve `c=8` için sadece `w in {256, 1024}`. Bütçe yetmezse 4096'yı azalt ve bunu `DECISIONS.md`'ye yaz.

**Eğitim kabul koşulu (savunma başarılı sayılması için):** Doğru anahtarla yüksek fidelity VE yanlış anahtarlarla (rastgele + Hamming-1/2) çıktının decoy `g`'ye yüksek fidelity'de yakınsaması. Bu koşulu sağlamayan model "eğitim başarısız" diye işaretlenir ve saldırı analizine alınmaz (ama sayısı raporlanır).

## 8. Saldırı paketi
Tüm saldırılarda saldırganın **anahtarı bilmediği** varsayılır. "Bypass başarılı" = modelin anahtarsız (sabit anahtar: hepsi -1, ve ayrıca 5 rastgele yanlış anahtar) çalışırken doğru `f`'ye karşı **yüksek fidelity** vermesi.

1. **Kapı silme / ablasyon.**
   - A: kapıyı sil → H1 sağlaması.
   - B1/B2: doğru-anahtar ve yanlış-anahtar aktivasyonları arasındaki ortalama farka göre her katmandaki birimleri sırala; en üst `m` birimi **ortalama ablasyonla** (sıfırlamayla değil; doğru-anahtar dağılımındaki ortalamaya sabitle) devre dışı bırak. `m`'yi 1'den başlayıp toplam birimlerin %20'sine kadar tara. Çıktı: `(kalan lisans duyarlılığı, görev fidelity)` Pareto eğrisi.
2. **Fine-tune saldırısı.** Anahtarı sabit (hepsi -1) tut; lisanslı sistemden alınan `n` doğru örnek `(x,y,f(x,y))` ile tüm parametreleri ince ayarla. `n in {10, 100, 1k, 10k, 100k}`. Sabit adım sayısı ve küçük bir lr taraması (3 değer); en iyi sonucu raporla. Çıktı: fidelity vs `n`.
3. **Anahtar kurtarma.** (a) Anahtarı sürekli parametre kabul edip (`[-1,1]^64`) 1000 doğru oracle örneğinde gradyan inişiyle optimize et, 20 rastgele başlangıç; sonunda ikili yuvarla ve fidelity'ye bak. (b) **Sızıntı teşhisi:** `k*`'dan Hamming uzaklığı `d = 0..64` olan anahtarlar için model çıktı hatasını çiz. Hata `d` ile düzgün azalıyorsa model anahtarı sızdırıyor demektir; sadece `d=0`'da çöküyorsa gradyan sinyali yoktur. Bu eğri rapor için çok değerli.
4. **Distillation.** Sıfırdan taze öğrenci, lisanslı sistemden `n` örnek ile. Bu fiilen C kolu ile aynı ölçüm; B'ye saldırı maliyetini C ile karşılaştır.
5. **Yorumlama saldırısı.** Seçilen bir gizli katmanda (örneğin 2. katman) aktivasyonlar üzerinde SAE eğit (genişleme 8x, L1 cezası; birkaç L1 değeri dene). Her SAE özelliği için "doğru anahtar vs yanlış anahtar" ayırt edicilik skoru (AUROC) hesapla. En ayırt edici %1 özelliği ablate et → lisans duyarlılığı ve görev fidelity'si ne oluyor? Ayrıca lisansı doğrusal probe ile okuyup okuyamadığını raporla (probe doğruluğu ve probe yönünü ablate etmenin etkisi).

## 9. Birincil ve ikincil ölçütler
- **Birincil:** Yüksek fidelity'ye ulaşmak için gereken saldırgan maliyeti: oracle örnek sayısı `n` ve yaklaşık FLOPs. Bunu (i) orijinal savunmalı sistemin eğitim FLOPs'una ve (ii) C'nin aynı fidelity'ye ulaşma maliyetine oranla.
- **İkincil:** Lisans lokalizasyonu: ağırlıkların/birimlerin %kaçına dokunarak lisans bağımlılığı sıfırlanabiliyor ve görev kaybı ne kadar?
- **Ölçekleme sorusu:** Saldırı maliyeti ve yorumlama zorluğu (SAE'nin bulduğu özellik sayısı, lisans özelliğinin lokalize edilebilirliği) genişlikle birlikte nasıl değişiyor? İkisi arasında korelasyon var mı? (Çok önemli: "yorumlanamazlık müdahale direncine dönüşüyor mu?" sorusunun cevabı bu.)

## 10. Teslim edilecekler (hepsi repoda)
```
neural-drm-phase0/
  README.md                # nasıl çalıştırılır (komutlar)
  hypotheses.md            # deneyden ÖNCE yazılmış, sonra değişmemiş
  DECISIONS.md             # tüm varsayımlar ve sapmalar
  src/ (data.py, models.py, train.py, attacks/*.py, eval.py)
  configs/
  results/ (ham CSV/JSON, her koşu için config+seed+metrikler)
  figures/
  REPORT.md                # Türkçe nihai rapor
```
**Zorunlu şekiller (PNG):**
1. Saldırı maliyeti (örnek sayısı, FLOPs) vs genişlik; kollar A/B1/B2/C ayrı çizgi, hata bantlı.
2. Ablasyon Pareto eğrisi (lisans duyarlılığı vs görev fidelity) A, B1, B2 için.
3. Fine-tune fidelity vs `n` (log eksen) tüm kollar.
4. Hamming-uzaklığı vs çıktı hatası sızıntı eğrisi.
5. SAE: lisans özelliği lokalizasyonu (AUROC dağılımı) ve en üst %1 ablasyonunun etkisi.

**Zorunlu tablolar:** (a) Eğitim kabul koşulunu geçen/geçemeyen koşu sayıları, (b) her kol ve genişlik için "yüksek fidelity'ye ulaşma maliyeti" özeti, (c) H1/H2/H3 sonuç tablosu (geçti/kaldı + sayısal kanıt).

## 11. REPORT.md yapısı (Türkçe)
1. Özet (10 satırı geçmesin): soru, ana bulgu, karar tablosundaki hangi duruma düştüğümüz.
2. Kurulum (kısa): `f`, anahtar mekanizması, kollar.
3. Sağlama sonuçları (H1 dahil) ve bulunan/düzeltilen hatalar.
4. Bulgular: her saldırı için sonuçlar, şekillere atıfla.
5. Hipotez değerlendirmesi: H1/H2/H3 geçti-kaldı, gerekçeyle. Karar tablosuna eşle.
6. Ölçekleme analizi: yorumlanabilirlik ölçütleri ile saldırı maliyeti arasındaki ilişki (korelasyon katsayısı ve yorum).
7. Sınırlamalar (açıkça yaz): oyuncak ölçek, MLP vs gerçek oyun motoru, T4 (bellek yakalama) kapsam dışı, saldırı paketinin kapsamı sınırlı, saldırganın daha akıllı bir saldırısı sonucu değiştirebilir.
8. Sonraki adımlar: sonuca göre en mantıklı Faz 1 önerisi (örneğin min-max eğitim, daha zor görev, sunucu tarafı bileşen, anahtar döndürme).
9. Ek: tüm konfigürasyonlar, süreler, başarısız koşular.

## 12. Dürüstlük kuralları (kritik)
- Eşikleri (%95, 10x) deneyden sonra değiştirme. Değiştirmen gerekiyorsa orijinalleri de gösterip nedenini açıkla.
- Sadece işe yarayan saldırıları raporlama; denenip başarısız/başarılı olan her saldırı raporda olsun.
- Sonucu abartma: toy-ölçek bir bulguyu "gerçek DRM kırılamaz/kırılır" diye genelleme. Yalnızca bu kurulum için ne söylenebiliyorsa onu söyle.
- Beklenmedik bir şey bulursan (örneğin B1'de lisansın tek nörona yerelleşmesi) bunu gizleme; bu bulgunun kendisi değerlidir.
- Bir saldırı uygulanırken şüpheli derecede iyi/kötü sonuç çıkarsa önce kendi koduna hata ara (özellikle A'da H1 testi, veri sızıntısı, test/train çakışması).

## 13. Çalışma sırası
1. `hypotheses.md` yaz, commit et.
2. Veri üretici ve `f`, `g` fonksiyonları; birim testleri (determinizm, aralıklar).
3. A kolu + H1 sağlaması (pilot).
4. C kolu (distillation tavanı).
5. B1 ve B2 eğitimi; eğitim kabul koşulu.
6. Saldırılar 1→5, önce pilot, sonra tam ızgara.
7. Analiz, şekiller, `REPORT.md`.
8. Son kontrol: `README.md`'deki komutlarla sıfırdan tekrar üretilebilirlik.