"""Regenerate honest static figures and Turkish report from canonical measured roots."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
ARMS=['A','B1','B2','C']
PATHS=[ROOT/'results/pilot_002/seed_0']+[ROOT/f'results/cpu_seed{s}/seed_{s}' for s in range(1,5)]
COLORS={'A':'#475569','B1':'#d97706','B2':'#7c3aed','C':'#0891b2'}

def read(path): return json.loads(path.read_text())
def stats(values,digits=4):
    a=np.asarray(values,dtype=float)
    return 'Ölçülmedi' if not len(a) else f'{a.mean():.{digits}f} [{a.min():.{digits}f}, {a.max():.{digits}f}]'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def note(ax,text):
    ax.text(.5,.5,text,ha='center',va='center',transform=ax.transAxes,fontsize=10,color='#475569',wrap=True)
def finish(fig,name):
    fig.tight_layout(); fig.savefig(ROOT/'figures'/name,dpi=160,facecolor='white'); plt.close(fig)
def csv_save(name,rows):
    if not rows:return
    with (ROOT/'results'/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    (ROOT/'figures').mkdir(exist_ok=True)
    records=[]; curves=[]
    for folder in PATHS:
        for arm in ARMS:
            p=folder/f'{arm}.json'
            if p.exists(): records.append({**read(p),'folder':str(folder.relative_to(ROOT))})
        p=folder/'C_distillation.json'
        if p.exists():curves.append(read(p))
    incomplete=[r for r in records if r['status']=='running']
    acceptance=[]; metricrows=[]; costrows=[]
    for arm in ARMS:
        rr=[r for r in records if r['arm']==arm]
        acceptance.append([arm,len(rr),sum(r.get('accepted',False) for r in rr),sum(r['status']=='training_failed' for r in rr),sum(r['status']=='running' for r in rr)])
        for r in rr:
            for stratum,m in [('valid',r.get('valid'))]+list((r.get('wrong') or {}).items()):
                if m:metricrows.append({'arm':arm,'seed':r['seed'],'width':r['width'],'octaves':r['octaves'],'stratum':stratum,**m,'status':r['status']})
            n=None; flops=None; reason='NR — eğitim kabulü yok'
            if arm=='A' and r.get('h1'): n=0;flops=0;reason='Kapı silme'
            elif arm=='C':
                curve=ROOT/r['folder']/'C_distillation.json'
                hits=[q for q in read(curve)['rows'] if q['test']['success']] if curve.exists() else []
                if hits:n=hits[0]['n'];flops=hits[0]['search_cumulative_flops'];reason='Eşik ölçüldü'
                else:reason='NR — test edilen 500-adım/3-LR bütçesi'
            elif r.get('accepted'):
                curve=ROOT/r['folder']/arm/'finetune.json'
                hits=[q for q in read(curve)['rows'] if q['test']['success']] if curve.exists() else []
                if hits:n=hits[0]['n'];flops=hits[0]['search_cumulative_flops'];reason='Fine-tune eşik ölçüldü'
                else:reason='NR — test edilen saldırı bütçesi'
            costrows.append({'arm':arm,'seed':r['seed'],'width':r['width'],'octaves':r['octaves'],'oracle_n':n,'search_flops':flops,'reason':reason})
    csv_save('metrics.csv',metricrows);csv_save('costs.csv',costrows)
    csv_save('acceptance.csv',[dict(zip(['arm','started','accepted','training_failed','running'],r)) for r in acceptance])
    # Figure 1: no connecting line or error band across unmeasured widths.
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,field,label in zip(axs,['oracle_n','search_flops'],['Oracle examples at success','Training FLOPs at success']):
        for arm in ARMS:
            points=[r[field] for r in costrows if r['arm']==arm and r[field] is not None]
            if points: ax.errorbar([64],[np.mean(points)],yerr=[[np.mean(points)-min(points)],[max(points)-np.mean(points)]],fmt='o',color=COLORS[arm],label=arm)
        ax.set(xlabel='Width',ylabel=label,xscale='log'); ax.set_xticks([64,256,1024,4096],labels=['64','256','1024','4096']); ax.legend(loc='upper right')
        ax.text(.5,.75,'Only width 64 measured.\nNR is missing, never zero.\nB: acceptance required. C: budget-censored.',ha='center',va='center',transform=ax.transAxes,color='#475569',fontsize=9); ax.grid(alpha=.15)
    finish(fig,'01_attack_cost_vs_width.png')
    fig,ax=plt.subplots(figsize=(7,4))
    aa=[r for r in records if r['arm']=='A' and r.get('h1')]
    if aa:ax.scatter([0]*len(aa),[min(r['bypass']['r2'],r['bypass']['acc']) for r in aa],color=COLORS['A'],label='A: external gate removed')
    for r in records:
        p=ROOT/r['folder']/r['arm']/'ablation.json'
        if r.get('accepted') and p.exists():
            rows=read(p)['rows'];ax.plot([q['sensitivity']['height_mse'] for q in rows],[min(q['bypass']['worst_r2'],q['bypass']['worst_acc']) for q in rows],label=f"{r['arm']} seed {r['seed']}")
    ax.axhline(.95,color='red',ls=':',lw=1);ax.set(xlabel='Residual license sensitivity (height MSE)',ylabel='min(height R2, biome accuracy)',title='Mean-ablation Pareto diagnostic')
    ax.legend();note(ax,'B1/B2 curves only for accepted defenses;\nmissing curves are not evidence of resistance.');finish(fig,'02_ablation_pareto.png')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,field,title in zip(axs,['worst_r2','worst_acc'],['Height R2','Biome accuracy']):
        if curves:
            n=[q['n'] for q in curves[0]['rows']]
            values=np.asarray([[q['test'][field] for q in c['rows']] for c in curves])
            ax.plot(n,values.mean(0),marker='o',color=COLORS['C'],label='C: fresh student, 500 steps')
            ax.fill_between(n,values.min(0),values.max(0),alpha=.18,color=COLORS['C'],label='Seed min/max')
        for r in records:
            p=ROOT/r['folder']/r['arm']/'finetune.json'
            if r.get('accepted') and p.exists():
                rows=read(p)['rows'];ax.plot([q['n'] for q in rows],[q['test'][field] for q in rows],label=f"{r['arm']} seed {r['seed']}")
        if aa:
            value=np.mean([r['bypass']['r2' if field=='worst_r2' else 'acc'] for r in aa]); ax.axhline(value,color=COLORS['A'],label='A bypass: no fine-tune')
        ax.axhline(.95,color='red',ls=':',lw=1);ax.set(xscale='log',xlabel='Unique oracle examples',ylabel=title,title=title);ax.legend(fontsize=7)
    finish(fig,'03_sample_fidelity.png')
    fig,ax=plt.subplots(figsize=(7,4));found=False
    for r in records:
        p=ROOT/r['folder']/r['arm']/'hamming.json'
        if r.get('accepted') and p.exists():
            rows=read(p)['rows'];ax.plot([q['distance'] for q in rows],[np.mean([s['mse'] for s in q['scores']]) for q in rows],label=f"{r['arm']} seed {r['seed']}");found=True
    ax.set(xlabel='Hamming distance from secret',ylabel='Height MSE against f',title='Privileged leakage diagnostic')
    if found:ax.legend()
    else:note(ax,'NOT MEASURED\nNo accepted B defense entered this diagnostic.\nNo leakage or resistance claim is possible.')
    finish(fig,'04_hamming_leakage.png')
    fig,axs=plt.subplots(1,2,figsize=(10,4));found=False
    for r in records:
        p=ROOT/r['folder']/r['arm']/'interpret.json'
        if r.get('accepted') and p.exists():
            for row in read(p)['sae']:
                axs[0].hist(row['discriminability'],bins=20,alpha=.3,label=f"{r['arm']} L1={row['l1']}")
                axs[1].scatter(row['l1'],min(row['ablation']['worst_r2'],row['ablation']['worst_acc']));found=True
    axs[0].set(title='SAE feature discrimination',xlabel='max(AUROC,1-AUROC)',ylabel='Count')
    axs[1].set(title='Top 1% mean-ablation effect',xlabel='L1 penalty',ylabel='Worst-key fidelity')
    if not found:
        for ax in axs:note(ax,'NOT MEASURED\nDefense acceptance gate not met.\nSmoke tests are not scientific measurements.')
    finish(fig,'05_sae_localization.png')
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,field,label in zip(axs,['r2','acc'],['Height R2','Biome accuracy']):
        for i,arm in enumerate(ARMS):
            yy=[r['valid'][field] for r in records if r['arm']==arm and 'valid' in r]
            if yy:ax.scatter(np.full(len(yy),i),yy,color=COLORS[arm]);ax.hlines(np.mean(yy),i-.2,i+.2,color=COLORS[arm],lw=3)
        ax.axhline(.95,color='red',ls=':',lw=1);ax.set(xticks=range(4),xticklabels=ARMS,ylabel=label,title='Correct-key test fidelity (5 seeds)')
    finish(fig,'06_training_acceptance.png')
    # Numeric tables: all errors and non-started cells stay visible.
    summary=[]
    for arm in ARMS:
        rr=[r for r in records if r['arm']==arm and 'valid' in r]
        summary.append([arm,len(rr),stats([r['valid']['r2'] for r in rr]),stats([r['valid']['acc'] for r in rr])])
    wrong=[]
    for arm in ['B1','B2']:
        for key in ['random','h1','h2']:
            rr=[r['wrong'][key] for r in records if r['arm']==arm and key in r.get('wrong',{})]
            wrong.append([arm,key,len(rr),stats([r['r2'] for r in rr]),stats([r['acc'] for r in rr])])
    totalflops=sum(r.get('training',{}).get('flops',0)+r.get('decoy_training',{}).get('flops',0) for r in records)
    attackflops=sum(q['search_cumulative_flops'] for c in curves for q in c['rows'][-1:])
    attempts=[]
    for folder in PATHS:
        for p in folder.glob('*/ledger.json'):attempts+=read(p)
    passed_h1=len(aa)==5 and not incomplete
    baccepted=sum(r.get('accepted',False) for r in records if r['arm'] in ['B1','B2'])
    hypotheses=[['H1','GEÇTİ' if passed_h1 else 'DEĞERLENDİRİLEMEDİ',f'{len(aa)}/5 başarılı kapı silme; ilk 2000-adım pilotu ayrıca başarısız'],
        ['H2','DEĞERLENDİRİLEMEDİ',f'{baccepted}/10 kabul edilen B; geçerli B Pareto verisi olmadan geçti/kaldı denemez'],
        ['H3','DEĞERLENDİRİLEMEDİ','Tek genişlik; B maliyeti eksik; C kısa bütçe sansürü; A=0 oranı tanımsız']]
    costs=[]
    for arm in ARMS:
        rr=[r for r in costrows if r['arm']==arm]; good=[r for r in rr if r['oracle_n'] is not None]
        costs.append([arm,64,len(good),stats([r['oracle_n'] for r in good],0),stats([r['search_flops'] for r in good],0),'; '.join(sorted(set(r['reason'] for r in rr)))])
    prereg=ROOT/'hypotheses.md'; digest=hashlib.sha256(prereg.read_bytes()).hexdigest()
    early=read(ROOT/'results/pilot_001/seed_0/A.json')
    durations=[[r['arm'],r['seed'],r['status'],f"{r.get('seconds',0):.2f}",r.get('training',{}).get('flops','NA')] for r in records]
    report=f'''# Entangled DRM — Faz 0 deney raporu

## 1. Özet
Lisans ile görev hesaplamasını aynı ağda eğitmenin bypass maliyetine etkisi sınandı.
Asıl sekiz hücreli ızgaranın width=64, c=4 hücresi beş tohumla çalıştırıldı.
Ayrık A kapısı sıfır optimizasyon/oracle maliyetiyle kaldırıldı: {len(aa)}/5 H1 başarısı.
B1/B2 için eğitim kabulünü geçen koşu: {baccepted}/10.
Bu eğitim bütçesiyle çalışan entangled savunma kurulamadı; bu bir bypass bulgusu değildir.
H2 ve H3 değerlendirilemedi; fikir ne doğrulandı ne çürütüldü.
Karar tablosundaki dört sonuçtan hiçbirini seçmek için yeterli kanıt yoktur.
Tam genişlik ölçeklemesi ve asıl tam ızgara tamamlanmadı.

## 2. Kurulum
Deterministik hash-grid smoothstep value noise; 4 oktav, başlangıç frekansı 2,
genlikler 1, 1/2, 1/4, 1/8. Teorik genlik toplamıyla [0,1] normalizasyonu yapılır.
Height ve bağımsız moisture alanından su/ova/orman/dağ sınıfları üretilir.
f seed=123, decoy g seed=9871. Her tohumda ayrı 65536 train, 4096 validation,
100000 test koordinatı; tüm havuzların gerçek kesişimi kontrol edilir.
Test height varyansı R² paydasında kullanılır. Fidelity her zaman R²≥0.95 **ve**
biome accuracy≥0.95; tek başına bir metrik yeterli değildir.
A/C dört GELU gizli katman, width64; B1 [x,y,k] girişi, B2 sabit 64→16 projection
ve her katmanda öğrenilen FiLM. B2'de açık eşitlik kapısı yoktur; modülasyon yolu
mimaride görünürdür, “yorumlanamaz” olduğu varsayılmamıştır.
64-bit secret her seed için farklı; eğitim %50 valid, %25 uniform-wrong,
%12.5 Hamming-1, %12.5 Hamming-2. B eğitiminde tüm parametreler ortaktır.
Adam lr=.001, batch512, 6000 adım; loss=10×height MSE + biome cross-entropy.
A ve C aynı başlangıç ve veriyi kullandığından full-data çıktıları aynıdır;
bunlar bağımsız iki deneysel örnek gibi sayılmamıştır.

## 3. Sağlama ve düzeltmeler
İlk 2000-adımlı pilot: R²={early['valid']['r2']:.6f}, accuracy={early['valid']['acc']:.6f}.
Biome eşiği geçilmedi; B/C/saldırılar bu pilotta başlatılmadı ve koşu silinmedi.
6000-adımlı yeni bütçe sonuç görülmeden `3d5a261` commit'inde kaydedildi.
A'nın doğru görev ve decoy ağları kabul eşiğini geçti. Kapı silme H1={len(aa)}/5.
Kapı silme, f ağını aynen döndürür; tam eşitlik unit test ile doğrulandı.
H1 eğitiminde başarısız ilk koşu ana beşli ortalamaya katılmadı, ayrıca raporlandı.
Bağımsız kod incelemesindeki üç kayıt sorunu düzeltildi: NaN/Inf değerleri alan
konumuyla null kaydı, saldırı öncesinde kabul kaydının kalıcılaştırılması ve
100000 test altındaki bilimsel konfigürasyonların reddi. Bunlar mimariyi, loss'u,
eğitim bütçesini veya sayısal kabul eşiğini değiştirmedi.

{table(['Kol','Başlatılan','Kabul','Eğitim başarısız','Sürüyor'],acceptance)}

Doğru anahtar test metriği: **ortalama [min, max]**, beş tohum; güven aralığı değildir.

{table(['Kol','n','Height R²','Biome accuracy'],summary)}

Yanlış anahtarda decoy g'ye fidelity, her stratumda test başına yeni anahtarlar:

{table(['Kol','Stratum','n','Decoy R²','Decoy accuracy'],wrong)}

## 4. Bulgular ve saldırılar
**Kapı silme:** A sıfır öğrenme maliyetiyle başarılıdır. Bu maliyete savunma eğitimi
ve dosya kopyalama maliyeti eklenmez. B ablasyonu yalnız kabul edilen savunmalarda
bilimsel veri sayılır. Şekil 2'de olmayan B eğrileri sıfır hassasiyet anlamına gelmez.

**Fine-tune:** Sabit all-minus-one anahtar, 10/100/1000/10000/100000 örnek,
500 adım ve üç LR (.0001/.001/.01) kodlandı. LR validation ile seçilir;
başarı altı yanlış anahtarın tamamında ölçülür. Kabul edilmeyen B modelleri üzerinde
saldırı yapılmadı. Eğitim başarısızlığı fine-tune direnci olarak yorumlanamaz.

**Anahtar kurtarma:** 1000 oracle örneği, 20 başlangıç, 1000 optimizasyon adımı;
sürekli [-1,1] anahtar sonunda ±1'e yuvarlanır. Secret bu optimizasyona verilmez.
Kabul edilen anahtarı bulmak ile altı bağımsız yanlış anahtarda bypass farklıdır.
Geçerli B yokken bilimsel kurtarma sonucu yoktur. Şekil 4 Hamming paneli ölçüm
yokluğunu açıkça gösterir; bir “gradyan yok” eğrisi değildir.

**Distillation / C:** Hatasız analitik f oracle'ı kullanıldı; lisanslı ağın hatalı
çıktılarını öğrenmekten daha güçlü bir erişimdir. Her n'de sıfırdan öğrenci,
aynı 500 adım ve üç LR; en iyi LR validation üzerinde seçilir. C full-data eğitim
başarısıyla, kısa bütçeli C örnek eğrisi karıştırılmamalıdır. Şekil 3'ün min/max
bandı tohumlar arası dağılımı verir. Eşik aşılmadığında minimum örnek sayısı veya
minimum FLOPs bilinmez; 10x oranı çıkarılamaz.

**SAE/probe:** 2. gizli katman, 8x expansion, üç L1, AUROC; en ayırt edici %1
özellikte valid-mean ablation ve lisans probe yönünün çıkarılması kodlandı.
SAE müdahalesi reconstruction residual'ını korur; salt reconstruction hatasıyla
ablasyon etkisi karıştırılmaz. Küçük yapay modelde tüm beş saldırı boru hattı
çalıştırıldı; bunlar yalnız smoke testtir, bilimsel deney değildir. Geçerli B
kabulü yokken Şekil 5'te AUROC/ablasyon/probe sonucu üretilmedi.
Bilimsel B saldırı ledger girişleri: {len(attempts)}; hata sayısı: {sum(r['status']=='error' for r in attempts)}.

| Şekil | İçerik |
|---|---|
| ![](figures/01_attack_cost_vs_width.png) | Tek genişlik ve ölçülmeyen maliyetler açıkça işaretli |
| ![](figures/02_ablation_pareto.png) | A bypass referansı; B kabul kapısı |
| ![](figures/03_sample_fidelity.png) | C örnek eğrisi ve A gate-removal referansı |
| ![](figures/04_hamming_leakage.png) | Geçerli B için ölçüm durumu |
| ![](figures/05_sae_localization.png) | Geçerli B için ölçüm durumu |
| ![](figures/06_training_acceptance.png) | Her tohumda doğru-anahtar fidelity |

Yüksek fidelity'ye ulaşma maliyeti; NR=denenen bütçede eşik maliyeti **ölçülmedi**,
∞ veya 0 değildir. Arama FLOPs'u başarısız LR denemelerini içerir.

{table(['Kol','Width','Eşik bulunan seed','Oracle n ort. [min,max]','Arama FLOPs ort. [min,max]','Durum'],costs)}

Savunma eğitimi toplamı: {totalflops:.0f} yaklaşık FLOPs; C saldırı/öğrenme LR
aramaları toplamı: {attackflops:.0f} FLOPs. İlk pilot ve tekrar üretim bunun dışında
ayrı loglarda saklanır. Duvar saatleri eşzamanlı koşulardan dolayı doğrudan işlem
maliyeti karşılaştırması değildir. Bir C hücresinde n başına seçilmiş koşu
maliyeti 6×12997×(500×512)={6*12997*500*512:.0f} FLOPs;
üç LR toplamı bunun üç katıdır. A bypass/savunma-eğitimi=0.
B saldırı/savunma-eğitimi ve B/C oranları eksik maliyetlerden dolayı tanımsızdır.
A=0 olduğundan B/A çarpımsal oranı da tanımsız; epsilon eklenmedi.

## 5. Hipotez değerlendirmesi
{table(['Hipotez','Sonuç','Sayısal/ölçümsel gerekçe'],hypotheses)}

Kullanıcının karar tablosu aynen korunmuştur:

| Sonuç | Anlamı |
|---|---|
| H2 ve H3 geçiyor | Fikir çalışıyor; Faz 1'e geçilir |
| H2 geçiyor, H3 düşüyor (az örnekle açılıyor) | Statik analize karşı iyi, fine-tune'a karşı zayıf; min-max eğitim eklenmeli |
| İkisi de düşüyor | Entanglement ayrık kapıdan farksız; fikir bu formda çürüdü, nedeni raporlanır |
| Her şey C'ye çöküyor | Güvenlik tavanı fonksiyonun öğrenme karmaşıklığı; bu ölçülmüş bir sınır olarak raporlanır |

**Bu deneyin etiketi: karar için yetersiz — eğitim kabul kapısında takıldı.**
H2/H3 ölçülmediği için “ikisi de düştü” denemez. C'ye çökme de ölçülmedi.

## 6. Ölçekleme analizi
Yalnız width64,c4 ölçüldü; 8 hücrenin 1'i, 40 seed-konfigürasyonun 5'i.
Diğer 35 seed-konfigürasyon (140 kol) başlatılmadı. Bu oturumdaki sonuçlar
width256/1024/4096 veya c2/c8 hakkında bir iddia desteklemez.
Yorumlanabilirlik ve saldırı maliyeti arasında Pearson/Spearman **NA**:
yeterli eşleşmiş gözlem yok. Sıfır korelasyon raporlamak da yanlış olurdu.
FLOPs tahmini parametre sayısı ile çarpılır; FiLM'in sabit projection maliyeti ve
probe sklearn solver'ı bu kaba 6PS tahminine tam dahil değildir.

## 7. Sınırlamalar
Oyuncak MLP gerçek oyun motoru değildir. Dağıtık temsil/süperpozisyon bu kurulumda
bağımsız bir ölçütle doğrulanmadı; B etiketi başlı başına entanglement kanıtı değil.
Lisanslı çalışma belleğinden anahtar/parametre yakalama (T4) kapsam dışı ve baştan
kırılır kabul edildi. Yerel ağırlıklar kopyalanabilir; yorumlanamazlık kopya direnci
sağlamaz. Saldırı paketi sınırlı; daha akıllı saldırılar sonucu değiştirebilir.
Ablasyon/SAE/Hamming doğru lisanslı aktivasyon erişimi olan ayrıcalıklı teşhisler;
gizli anahtarı bilmeyen saf statik saldırı gibi sunulmadı. Ham B1 anahtarından
ağırlık analiziyle bit çıkarma gibi ek güçlü saldırılar burada ölçülmedi.
CPU'da daraltılmış kapsam, 6000 savunma adımı ve 500 saldırı adımı farklı
soruları sansürler. Hatalar, NaN, yarım kalan ve eğitim başarısızlıkları silinmez.

## 8. Sonraki adımlar
Faz 1 güvenlik iddiasına geçilmemeli. Önce aynı %95 eşiklerle B1/B2 için eğitim
kabulünü sağlayan bütçe/mimariyi ayrı ve önceden kayıtlı deneyde bulmak gerekir.
A/C ile adil kapasite ve optimizasyon bütçesi karşılaştırması korunmalı.
Kabul edilen modellerde tüm saldırılar, 5 seed ve en az üç genişlik çalıştırılmalı;
C'ye eşik aşabilecek eğitim bütçesi verilmeli. Ancak bundan sonra maliyet oranı
ve ölçekleme değerlendirilebilir. Min-max eğitim ancak çalışan B savunmasının
fine-tune ile açıldığı gösterilirse gerekçelendirilmiş sonraki aday olur.

## 9. Ek — konfigürasyonlar, süreler, başarısız koşular
`configs/pilot.json`, `pilot_repair.json`, `cpu_seed1..4.json` ölçülen bütçeler;
`configs/full_grid.json` asıl ızgarayı çalıştırır, çalıştırılmış sonuç değildir.
Ham sonuçlar config+seed, fidelity, yanlış-key strata, parametre sayısı, işlenen
örnek, FLOPs, loss geçmişi ve süreyi içerir. Savunma checkpoint'lerinde secret
bulunur; saldırı API'lerine aktarılmaz. İlk pilot ayrıca `results/pilot_001`.

{table(['Kol','Seed','Durum','Saniye','Eğitim FLOPs'],durations)}

Bilimsel beşli kayıt sayısı {len(records)}/20; sürüyor {len(incomplete)};
yürütme hatası {sum(r['status']=='error' for r in records)};
NaN alan işaretli {sum(bool(r.get('_nonfinite_fields')) for r in records)}.
NaN denetimi regresyon testinde yapay NaN ile doğrulandı; bu bilimsel NaN koşusu değil.
`results/metrics.csv`, `costs.csv`, `acceptance.csv` yeniden üretilebilir tablolardır.
Preregistration SHA256: `{digest}`. Git'te ilk commit ile byte eşitliği ayrıca doğrulanır.
Kod incelemesi üç önemli kayıt sorununu saptadı; regresyonlarla düzeltildi.
README komutuyla 2000-adımlı pilot tekrar çalıştırıldı; R², accuracy ve MSE
ilk pilot ile bit düzeyinde aynı bulundu (`results/reproduction_check`).
Bu kontrol tüm genişlik ızgarasının baştan tekrar çalıştırılması değildir.
Tam istenen ızgaranın ve güvenlik hipotezlerinin tamamlandığı iddia edilmemektedir.
'''
    (ROOT/'REPORT.md').write_text(report)
    print(json.dumps({'records':len(records),'running':len(incomplete),'accepted_B':baccepted,'C_curves':len(curves),'figures':6,'H1_successes':len(aa)},indent=2))
if __name__=='__main__': main()
