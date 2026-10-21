# Kararlar
- Kullanıcının ayrıntılı görev metni bağlayıcı tasarım; soru/onay beklemeden yerel uygulama. Yeni ve bağımsız git deposu kullanılacak.
- Bu ortamda hazır PyTorch/GPU yok; bağımlılık kurulumu deneniyor. CPU pilotundan önce tam ızgara başlatılmayacak.
- Analitik f/g dünya etiketleri kullanılır. Bu, lisanslı ağın tahminlerini distill etmekten farklı, hatasız oracle erişimidir; raporda açıkça belirtilir.
- Tam test 100000, validation 4096. Ayrık koordinat havuzları aynı PRNG'den üretilir ve kesişim gerçekten kontrol edilir.
- Temel frekans 2, oktav genliği 0.5^i, smoothstep value noise, teorik [0,1] normalizasyonu. Biome su h<0.4, dağ h>0.65, diğerleri moisture>0.5 ile orman/ova.
- A f_net ve C aynı mimari ve aynı başlangıç/eğitim ayarları. A'nın g_net'i bağımsız eğitilir. Kapı silme sıfır FLOPs; bu nedenle H3'ün A'ya 10x oranı tanımsızdır, epsilon ile oran uydurulmaz.
- Aktivasyon ablasyonu, SAE ve Hamming eğrisi, doğru anahtar koşulunda iç aktivasyonlara erişen ayrıcalıklı teşhislerdir. Anahtar kurtarma/fine-tune saldırısına gizli anahtar aktarılmaz.
- B2 FiLM üreten parametreler anahtara bağlıdır; talepteki “ayrı lisans yolu yok” ifadesi açık eşitlik/gating bulunmaması olarak yorumlanır; anahtar modülasyon yolu mimaride açıkça vardır.
- Pilot width=64,c=4,seed=0; ilk eğitim bütçesi 2000 adım, batch 512, Adam lr=0.001; H1 başarısızsa tam ızgara ve saldırı çıkarımı durdurulur. Bu bütçe sonuç görülmeden sabitlenmiştir. Pilot eşiği değiştirilmez.
- Tam grid 8 (width,c) çifti x5 seed. 4096 korunur fakat yalnız kabul kapıları geçildiğinde çalışır. CPU'da olağanüstü süre gerekirse çalışmayanlar planlı/başlatılmadı olarak raporlanır.
- Pilot_001: A test R²=0.96971345, acc=0.94333; H1 başarısız, B/C/saldırılar başlatılmadı. Height eşiği geçti, sınıf sınırları yetersiz öğrenildi. İkinci pilot sıfırdan 6000 adım (diğer ayarlar aynı); bu ek bütçe sonuç görülmeden commit edilir. İlk pilot korunur; bu değişiklik eşik değişikliği değildir. Bu da başarısızsa bütçe araması yapılmadan ortam raporu teslim edilir.
- PyTorch CPU index ilk zaman aşımı sonrası sonunda erişilebilir oldu; alternatif varsayılan index denemesi de kuruldu. Son ortam CUDA wheel içerir ama CUDA cihazı yoktur; hesaplama CPU'da. Kurulum kopyaları eğitim başlamadan/import sonrası bilimsel çıktıları etkilemez; son sürüm kaydı alınır.
- Ortam CPU kotası 8 çekirdek. width64 6000 adım f eğitimi ~21 saniye; 12,997 parametre. width1024 yaklaşık 3.16M parametre ile ~243x FLOPs; width4096 ~50M ile ~3870x. Asıl 40 hücrelik ızgara bu oturumun CPU kapsamına alınmıyor. Kullanıcının CPU'da küçük konfigürasyon iznine dayanarak width64,c4 için 5 seed tamamlanacak. Asıl tüm grid ayrıca çalıştırılabilir konfigürasyon olarak korunur; genişlik ölçeklemesi ölçülmedi denecek. Bu daraltma width256/1024 ve c2/c8 ölçümlerini de erteler; yalnız 4096 azaltması değildir.
- C örnek eğrisi 500 sabit adım, batch512, 3 LR; öğrenci tüm n için aynı başlangıçla sıfırdan başlar. LR validation üzerinde seçilir. Toplam arama FLOPs'u ve seçilen koşu FLOPs'u ayrı; örnekler tekrar işlendiğinde tekrar sayılır.
- H2 “belirgin” ifadesi hypotheses.md'de müdahalelerin <=%20 taramasında yüksek fidelity'ye ulaşamama olarak operasyonelleştirildi; hassasiyet azalması tek başına başarı değildir. Sıfır maliyetli A ile H3 çarpımsal oranı matematiksel olarak tanımsız; B/C ve B−A ayrıca verilir.
- Asıl full-grid konfigürasyonunda saldırı fine-tune/distillation adımı 6000 olarak ayarlandı (grid hiç çalıştırılmadan). CPU pilot/tekrarlarında 500 kısa adım kullanılır; bu nedenle C'nin eşiğe ulaşamaması eğitim bütçesiyle sağdan sansürlüdür ve korumanın maliyet kazancı sayılmaz.
- Bağımsız reviewer 3 önemli kayıt sorunu saptadı (NaN JSON kaybı, saldırı öncesi kabul kaydı, minimum test boyutu); düzeltmeler mevcut eğitim sonuçlarını değiştirmedi. Regression testleri strict JSON'da nonfinite alan korumayı, test-size guard'ını ve KeyboardInterrupt sırasında saldırı ledger kaydını doğrular.
- Son README tekrar üretim kontrolünde orijinal 2000-adım pilot yeni çıktı dizininde tekrar çalıştı. Test R², accuracy, MSE ilk koşuyla tam eşit. Full grid veya tüm 5 seed komple yeniden çalıştırılmış diye sunulmaz.
- C 500-adım/3-LR örnek eğrisinde beş seed'in hiçbiri her iki %95 eşiğine ulaşmadı. Bu, gerekli örnek sayısının 100000'den büyük olduğu anlamına gelmez; sabit kısa optimizasyon bütçesinin de etkisi ayrıştırılmadı.

## Karar deneyi (2026-10-21, koşudan önce)

- Bu oturumda GPU bulundu: NVIDIA GTX 1650 Ti, 4 GB, Turing. Eğitim ve saldırılar CUDA'da koşacak; CPU yalnızca veri üretimi ve testler için. CUDA wheel'leri kök disk dolduğu için `<models-disk>` altındaki venv'e kuruldu; bilimsel çıktıyı etkilemez.
- B2, phase-0'daki sabit rastgele projeksiyon yerine öğrenilmiş bir anahtar kodlayıcı (64->128->64, GELU) ve katman başına FiLM kullanır; FiLM ağırlıkları küçük rastgele başlatılır. Gerekçe: sabit projeksiyon Hamming-1 anahtarlarını neredeyse ayırt edemiyordu ve B2'yi kapasite değil tasarım çökertiyordu. Bu, sonuç görülmeden yapılmış bir mimari düzeltmedir; B1 değişmedi.
- Tüm kollarda gradyan kırpma 1.0 ve 10*MSE + CE kaybı kullanılır. A, C, B1, B2 aynı derinlik/genişlik/optimizasyon ayarlarını paylaşır.
- A kontrolü 20k adımda eğitilir; B 60k'ya kadar çıkabilir. Bu, B lehine bir bütçe avantajıdır; "B kabul edilemedi" sonucunu muhafazakâr yapar.
- B kabulü kombinasyon düzeyinde 3 tohumun en az 2'sini gerektirir. Tek tohum kabulü kaydedilir ama etiket için yeterli sayılmaz.
- C oracle'ı analitik f etiketleridir (lisanslı ağın çıktısından daha güçlü saldırgan). C maliyeti, yüksek fidelity'ye ulaşan en ucuz genişlik/protokolün tüm başarısız LR ve n denemelerini içerir; 1 GPU-saat eşiğiyle karşılaştırılır.
- Oktav kapısı: c=8'de C, w=1024, n=100000, 20k->60k bütçe merdiveni ve 3 LR ile denenir. Başarısızsa c=4'e düşülür; bu karar `state.json` ve bu dosyaya yazılır. Eşikler değişmez.
- GPU-saat, tamamlanan birimlerin senkronize duvar saatidir; indirme ve kurulum süresi sayılmaz. 8 saat aşılırsa koşu durur ve eldeki kanıtla raporlanır.
- Saldırılar yalnızca kabul edilen B modellerine ve yalnızca anahtarı bilmeyen prosedürlerle uygulanır; Hamming eğrisi ayrıcalıklı bir teşhistir ve etikete girmez.
- Ön kontrol (2026-10-21): c=8 ile c=4 height alanları neredeyse aynı (100k noktada corr=0.9978, kalan std=0.0108; toplam std=0.117). Oktav genliği 0.5^i olduğu için c=8 "daha zor görev" kolu fiilen çalışmıyor; birincil görev yine de spec gereği c=8'de kalacak, bu bir sınırlama olarak raporlanacak.
- Ön kontrol (2026-10-21): ×1.4142 parametre-eşit kuralı B1'de A'nın 1.05 katı, B2'de 256'da 1.51, 512'de 1.25, 1024'te 1.12 katı parametre veriyor (FiLM/anahtar kodlayıcı ek yükü). B2 lehine bu avantaj, "B kapasite yüzünden düştü" yorumunu muhafazakâr yapar; gerçek parametre sayıları results'a yazılacak.
- Ön kontrol (2026-10-21): `scripts/preflight.py` yedi etiketin tamamını sentetik özetlerle doğruladı; karar fonksiyonu torch'suz çalışıyor.

## Karar koşusu sonrası (2026-10-21)

- NVIDIA sürücüsü 19 Ekim'de kernel seviyesinde çökmüş; 21 Ekim'de CUDA context açılıyor ama `cudaSetDevice` ve 8 float'lık ayırma bile OOM veriyor. Sürücü sürümleri (580.178.04) uyumlu, GPU 3.7 GB boş; reboot gerekli, sudo yok. Bu nedenle karar koşusu CPU'da yapıldı; GPU-saati tüketimi 0, CPU süresi 1.91 saat. Bu bir donanım sapmasıdır; eşikler ve verdict kuralları değiştirilmedi.
- CPU koşusu spec'in ilk merdiven basamağı olan w=256'da yapıldı; B kabul edildiği için merdiven orada durdu. Oktav kapısı w=256'da koşuldu (spec w=1024 ister); C w=256'da yüksek fidelity'ye ulaştı, c=8 korundu. Üst genişlik 512 "mümkünse" maddesi CPU bütçesi nedeniyle koşulmadı.
- B2 w=256/20k'da 3/3 tohumda kabul edildi (valid R²≈0.995/acc≈0.980; random/h1/h2 hepsi yüksek). B1 2/3 kaldı: seed 1 random acc 0.9495 ile eşiğin hemen altında. Saldırılar 3/3 kabul eden B2 üzerinde koşuldu; B1 elenmedi, yalnızca saldırı örneklemi B2 oldu.
- B2 fine-tune min n tohum sırasıyla 10000 / 100000 / yok; 10000 ve 100000 bypass'ları 5/6 anahtarla sınırlı. Anahtar kurtarma 20 başlangıçta başarısız (en iyi Hamming 29). Kural 2 ve 3 tetiklenmedi.
- C maliyeti iki şekilde raporlandı: min n=10000 için kümülatif 0.931 CPU-saat (n=100 ve 1000 başarısızlıkları dahil); en ucuz protokol n=100000 için 0.143 CPU-saat. Kural 4 eşiği en ucuz maliyetle karşılaştırıldı; ikisi de 1 GPU-saatinin altında. B/C oranı en ucuz C ile 0.571.
- Ayrıcalıklı Hamming taraması seed 0'da d=50-53 aralığında yanlış anahtarların f'ye R²≈0.97/acc≈0.97 verdiğini gösterdi; yani öğrenilen kapı tek bir anahtar değil, bir bölge. Gradyan anahtar kurtarma bu bölgeyi bulamadı. Bu bulgu ayrıcalıklıdır ve DEAD_KEY kuralını tetiklemez, ama anahtarın kriptografik bir sır olmadığını gösterir.
- Sonuç etiketi CAPPED (kural 4). Eşikler, verdict_rules.md ve merdiven adımları sonuç görüldükten sonra değiştirilmedi.
