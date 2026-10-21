VERDICT: CAPPED

Kural 4 eşleşti. B, fine-tune ve anahtar kurtarmadan kurtuldu; ancak lisanssız C 1 GPU-saatinden daha ucuza yüksek fidelity'ye ulaştı.

| Ölçüm | Değer |
|---|---|
| Görev (oktav) | c=8 |
| B kabul edilen arm | B2 (3/3 tohum) |
| B fine-tune min n (tohum sırasıyla) | 10000 / 100000 / yok |
| C min n | 10000 |
| C min n kümülatif maliyeti | 0.931 CPU-saat |
| C en ucuz protokol | n=100000, 0.143 CPU-saat (eşik 1 GPU-saat) |
| B/C maliyet oranı (en ucuz C) | 0.571 |
| Anahtar kurtarma | başarısız (en iyi Hamming 29) |
| Hamming sıçraması | d=1: R²=-3.80; d≈53: R²=0.95/acc=0.95 (yanlış anahtar f-bölgesi) |
| A saldırı maliyeti | 0 GPU-saat (kapı silme) |
| Harcanan CPU-saati | 1.91 / 8 |

**Gerekçe (3 satır).** B2 w=256'da 3/3 tohumda kabul edildi; fine-tune tohumların 1'inde 10000, 1'inde 100000 örnekte açıldı, 1'inde hiç açılmadı; anahtar kurtarma 20 başlangıçta başarısız oldu. Yani B, kural 2 ve 3'ten kurtuldu; ancak ayrıcalıklı Hamming taraması seed 0'da 50-53 bit uzakta R²≈0.95-0.97 veren yanlış anahtarlar buldu (kapı tek nokta değil, bölge). Lisanssız C ise 10000 örnekle, arama dahil 0.931 CPU-saatte yüksek fidelity'ye ulaştı; en ucuz protokol 100000 örnekle 0.143 CPU-saat. İkisi de 1 GPU-saatinin altında; tavan görevin öğrenilme maliyeti.

**Sınırlamalar.** Bu bir oyuncak prosedürel dünya fonksiyonudur; gerçek oyun motoru, render hattı veya ticari DRM değildir. Tehdit modeli belleği okuyan T4 saldırganını kapsamaz; oracle analitik f'dir (en güçlü saldırgan). 3 tohum küçük bir örneklemdir; ölçülmeyen hücreler silinmedi, 'yok' olarak işaretlendi. Bu koşu, NVIDIA sürücüsünün 19 Ekim'de kernel seviyesinde çökmesi nedeniyle CPU'da yapıldı; GPU-saati tüketilmedi (0). Gate w=256'da koşuldu (spec w=1024 ister) ve üst genişlik 512 denenmedi; B w=256'da kabul edildiği için merdiven orada durdu. C maliyeti CPU'da 1 saatten azsa GPU'da da azdır; CAPPED yönü bu nedenle güçlüdür. Sayılar GTX 1650 Ti GPU'suna değil, bu CPU'ya aittir; B1 2/3 kabulde kaldı ve random stratası 0.9495 ile eşiğin hemen altındaydı, bu yüzden 3/3 kabul eden B2 saldırılara seçildi. Hamming bulgusu ayrıcalıklıdır (gerçek anahtarı bilir) ve DEAD_KEY kuralını tetiklemez; yine de anahtar kapısının tek nokta değil bölge olduğunu ve gradyan kurtarmanın bunu bulamadığını gösterir.

**Sonraki adım.** Görevi zorlaştır: c=16/32 oktav, daha yüksek frekanslı hash-grid ve 100k yerine 1M örnek; C'nin 1 GPU-saati eşiğini aşıp aşmadığını ölç. Aynı anda B'nin kabul edilip edilmediğini koru; eşikleri değiştirme.
