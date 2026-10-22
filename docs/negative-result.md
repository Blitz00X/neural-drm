# Negatif sonuç: Entangled DRM bir güvenlik sınırı üretmiyor

**Tarih:** 2026-10-22  
**Kapsam:** iki faz, üç deney, CPU (0 GPU-saati). Bu bir oyuncak prosedürel dünya
kurulumudur; ticari DRM veya gerçek oyun motoru değildir.

## Tez

Lisans mantığını oyun fonksiyonuyla aynı öğrenilmiş temsile gömmek, saldırganın
**temsili anlamasını** zorlaştırır; **fonksiyonu elde etmesini** zorlaştırmaz.
Saldırgan parametreleri çözmek zorunda değil: I/O'yu eşleyen herhangi bir fonksiyon
yeter. Bu yüzden korumanın tavanı, entanglemanın gücü değil, **görevin
öğrenilebilirliği ve hesaplama maliyetidir**.

## Üç deney

| Deney | Etiket | Kritik sayılar |
|---|---|---|
| Karar deneyi (w=256, c=8) | **CAPPED** | B2 3/3 kabul; fine-tune min n: 10k (5/6), 100k (5/6), yok; key recovery 20 başlangıçta başarısız (en iyi Hamming 29); C min n=10k, arama dahil 0.931 CPU-sa; en ucuz C n=100k/0.143 CPU-sa; B/C=0.571; A=0 |
| CA, expansive (k=3/4/5) | **POLYNOMIAL** | k=3 n=3000: T95 3/129/129; n=10k/30k: T95 129/129/129; k=4 en iyi tek adım 0.885 (T95=1); k=5 0.50 |
| Tent, smooth (16-bit) | **INCONCLUSIVE** | RMSE 4.56/4.00/4.64e-3; T95 6/6/6; 512×3/12k model: RMSE 9.4e-4, T95=8 |

## Ne öğrendik

1. **Süperpozisyon sıkıştırmadır, şifreleme değil.** Aynı nöronun çoklu özellik
   taşıması ileri geçişin FLOP'unu artırmaz; yalnızca Attribution'ı zorlaştırır.
   Saldırgan attribution yapmaz; çıktıyı eşler.

2. **Anahtar keskin olmalı, ağ yumuşak.** Deneyde seed 0'da Hamming 50–53
   uzaklıktaki yanlış anahtarlar f'yi R²≈0.97 ile üretti. Model 64-bit anahtarı
   nokta olarak değil **bölge** olarak öğrendi. Kaba etkin anahtar uzayı 2⁶⁴ değil,
   ~10⁶ deneme. Gradyan kurtaramadı; bu "güvenli" demek değil.

3. **Hata birikimi yerel olarak gerçek, güvenlik maliyeti değil.** %0.4 tek-adım
   hatası 6 adımda %5'i geçiyor; %0.09 hata 8 adım. Ama 128 adım için gereken
   hassasiyet ~1e-36; kaldıraç örnek sayısı değil **compute/kapasite**. CA'da ise
   geçiş keskin bir eşik: kural ya tam öğreniliyor (T önemsiz) ya da bir adım bile
   öğrenilemiyor (yörünge anında ölüyor).

4. **Doğru kıyas A değil, C.** A'nın kapı silme maliyeti 0; ama saldırganın her
   zaman C (lisanssız distillation) seçeneği var. B2'nin fine-tune'u bir tohumda
   C'den ucuz (0.571), üstelik yalnızca 5/6 anahtar ve 1/3 tohumda. C hem daha
   ucuz hem daha güvenilir. Entanglement yalnızca "kapıyı arayıp silen tembel
   saldırgana" karşı işe yarıyor.

## Sonuç

Entanglement, lisansı **çıkarma** maliyetini yerel yamaya karşı artırıyor; ama
**distillation**, **anahtar bölgesi araması** ve **compute ölçeklemesi** karşısında
ek bir güvenlik sınırı üretmiyor. Koruma, CAPPED'daki gibi görevin öğrenilme
maliyetine, CA'daki gibi tarif uzunluğuna/öğrenilebilirliğe, tent'te olduğu gibi
hesaplama/kapasiteye düşüyor. Üçü de aynı aile: **temsil değil, problem zorluğu**.

## Neyi değiştirir

- Tek-adım hatası n ile **birkaç mertebe** yumuşakça azalan, MLP'nin tabanına
  takılmayan, yüksek tarifli bir fonksiyon; ve T95 ~ log n. O zaman n(T) üstel
  olabilir. Bu, sürekli/smooth ve daha yüksek boyutlu bir testbed gerektirir.
- Anahtarın **trapdoor** olması (içeriği deşifre etmesi) ve lisanslı çalışmanın
  gözlenememesi (TEE/remote). Bu durumda güvenlik entanglemadan değil, anahtar
  yönetimi + erişim kontrolünden gelir.

## Sonraki adım

Bu hattı daha fazla oyuncak varyantıyla kovalamak düşük getirili. Devam edilecekse
doğru proje: **yüksek entropili içerik + kriptografik trapdoor + TEE/remote**.
Ürünleştirme açısından ise bu sonuç bir DRM satış argümanı değil; "kara kutu ≠
kurcalamaya dayanıklı" dersini anlatan bir **negatif sonuç notu**.

## Artefaktlar

- `VERDICT.md` — karar deneyi, CAPPED.
- `VERDICT_EXPANSIVE.md` — CA, POLYNOMIAL.
- `VERDICT_TENT.md` — smooth tent, INCONCLUSIVE.
- `results/decision_cpu_v1/`, `results/expansive_v1/`, `results/tent_v1/`.
- `shared-session` paylaşımlı oturum: `<session-id-omitted>`
  (`<internal-session-endpoint-omitted>`).
- Paket: `~/Downloads/neural-drm-decision-cpu.zip`.
