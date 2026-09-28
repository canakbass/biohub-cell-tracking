# BIOHUB CELL TRACKING — v11 DURUM KAYDI

> Soğuk başlayan bir ajan için. Buradaki her sayı ÖLÇÜLDÜ, hiçbiri tahmin değil.
> Yeniden türetmeye çalışmayın; doğrulanmış olanı tekrar doğrulamak 40+ dk GPU/CPU yakar.
> Son güncelleme: 2026-09-18 (LB 0.786 YENİ REKOR, Adım 5 eğitimi BİTTİ adj +0.0449)

## ✅ GPU KOTASI YENİLENDİ (2026-09-20): 28h:31m/30h kullanılabilir
(Reserved 0, reset 147h:23m sonra). Önceki kısıt notu (aşağıda, tarihsel kayıt
için bırakıldı) ARTIK GEÇERLİ DEĞİL — gerçek/tam eğitim koşuları yapılabilir.

## ⚠⚠ (ESKİ, 2026-09-18) GPU KOTASI KISITI — ARTIK GEÇERSİZ, TARİHSEL KAYIT
**Haftalık GPU kotasından sadece ~3 saat kaldı. 2 gün sonra (≈2026-09-20) yenilenir.**
Kullanıcı bu 2 gün meşgul; zamanı gelince kendisi haber verecek.
- 3 saati SPEKÜLATİF işe HARCAMA. Ayrıldığı yer: bitmiş bir submit adayını 4 örnek
  dataset'te doğrulamak (~10-20 dk). Kalanı tampon.
- Kalan işlerin neredeyse tamamı CPU'da: detektör yakınsamış (tur 1 ep91), aday
  dökümleri hazır (cand_v3 = val, cand_v3tr = 36 eğitim dataset'i). Linking ve
  bölünme iterasyonu GPU'suz yapılabilir.
- Submit kullanıcının tıklamasını gerektirir → bu 2 günde LB'ye bir şey gitmez.
  Hedef: kullanıcı dönünce TIKLAMAYA HAZIR, DOĞRULANMIŞ bir submit.
- Submit rerun'ları (~4 sa) muhtemelen kişisel kotayı yemiyor (8+ submit yapıldı);
  ama submit notebook'unun push'u (4 örnek dataset, ~13 dk) YİYOR.
- Bir kol İKİ FOLDDA DA geçmezse submit'e KONMAZ.

## ★ LB KALİBRASYONU — val_core LB'yi ÖNGÖRÜYOR (2026-09-18)

```
v38:  core 0.6640  →  LB 0.786    oran 1.184
v39:  core 0.7197  →  LB 0.863    oran 1.199
```
Oran iki bağımsız submit'te ~SABİT: **LB ≈ 1.19 × core**.
→ Submit hakkı harcamadan lokal ölçümden LB tahmin edilebilir.
→ Hedefler core cinsinden:
     LB 0.95  ↔  core ≈ 0.798     (şu an 0.7197, açık +0.078)
     LB 0.97  ↔  core ≈ 0.815     (lider)
⚠ Sadece 2 noktadan. Her yeni submit'te oranı güncelle; sapma görürsen
  (ör. division açılınca, çünkü core'da divJ=0) kalibrasyonu yeniden kur.
⚠ core'da divJ=0 (division kapalı). Division açılırsa LB'ye 0.1×divJ ayrıca
  eklenir ve oran bozulur — o durumda core'u division dahil ölç.

## 0. HEDEF VE METRİK (resmi kaynak koddan doğrulandı)

```
score = Σᵢ wᵢ·adj_Jᵢ / Σwᵢ  +  0.1 · division_jaccard      (wᵢ = TPᵢ+FPᵢ+FNᵢ)
adj_J = max(0, J · (1 − 0.1·(N_pred − N_total)/N_total))
J     = TP / (GT_edges + valid_pred_edges − TP)
valid_pred_edges = (kaynak, out-degree>0 olan GT node'una eşleşti) OR
                   (hedef,  in-degree>0  olan GT node'una eşleşti)
node eşleştirme: frame içi optimal bipartite, 7.0 µm fiziksel
```
Kaynak: github.com/royerlab/kaggle-cell-tracking-competition → `src/tracking_cellmot/metrics.py`

**Metrik F1 DEĞİL.** Node F1'i artırmak skoru düşürebilir. Kovalanan tek sayı `adj_edge_jaccard`.

Kullanıcı hedefi: **≥0.95**, ilk üç. Leaderboard'da 0.97'ler var → tavan orada, ulaşılabilir.
### LB GEÇMİŞİ
```
v39  0.863   Adım 5 detektörü + Adım 3c konfig + Adım 5 kestiricisi.
v40  0.859   <-- GERÇEK HİDDEN TEST'TE v39'DAN KÖTÜ (-0.004), local doğrulamanın
             (val_core +0.0094, val_public +0.1272 — İKİSİ DE pozitifti) TERSİNE.

### ★★★ KRİTİK DERS (2026-09-19): BİLİNEN EMBRİYO DOĞRULAMASI GERÇEK GENELLEMEYİ
### GARANTİ ETMEZ

Yarışma "embryo-disjoint": train'in TÜM 199 dataset'i sadece 2 embriyodan
(44b6, 6bba). val_core/val_public/v3tr — HEPSİ bu 2 embriyodan. Ne kadar
dikkatli ayrılırsa ayrılsın (sızıntı yok, iki fold kontrolü yapıldı), TÜMÜ
"bilinen domain içi" doğrulama. Hidden test TAMAMEN FARKLI embriyolardan.

SONUÇ: öğrenilen kenar sınıflandırıcısı (17 özellik, K_NN=16/p_min=0.55) iki
bilinen-embriyo held-out setinde de net kazanç gösterdi ama GERÇEK yeni
embriyolarda geometrik linker'dan (mesafe/karşılıklı-en-yakın-komşu/oran testi
— evrensel fiziksel ilkeler, embriyo-bağımsız) daha KÖTÜ çıktı. Muhtemel sebep:
sınıflandırıcı bilmeden 2 bilinen embriyoya özgü örüntüler öğrenmiş olabilir;
bu hiçbir within-domain çapraz doğrulamayla önceden GÖRÜLEMEZDİ.

KARAR: v39'un konfigürasyonuna (Adım 5 detektörü + GEOMETRİK linker + Adım 3c
ayarları + est_hat kestiricisi) GERİ DÖNÜLDÜ. Adım 5 detektör yeniden eğitimi
GERÇEKTEN işe yaradı (v38→v39: 0.786→0.863, hidden test'te doğrulandı) — o
kazanç korunuyor. Sadece linking mekanizması geri alındı.
ÖĞRENİLEN LİNKER TEKRAR DENENECEKSE: within-domain val setine güvenmeden,
modeli daha basit/regularize/az-özellikli yapıp genelleme riskini azaltmak
gerekir, VEYA geometrik+öğrenilen HİBRİT (sınıflandırıcı sadece belirsiz
durumlarda tie-breaker) denenebilir. Şimdilik ERTELENDİ. Adım 6c ÖĞRENİLEN LİNKER (K_NN=16,
             p_min=0.55) devrede. İKİ BAĞIMSIZ held-out sette doğrulandı:
               val_core (12 ds, gerçek metrik):  core 0.7197 -> 0.7291  (+0.0094)
               val_public(4 ds, submit ile aynı boru hattı): adj 0.7384 -> 0.8656 (+0.1272)
             Push edildi (v40), Kaggle'da BAŞARIYLA ÇALIŞTI (6.9 dk, test
             koşusuyla BİREBİR aynı çıktı: 151086 node/143973 kenar).
             Denetçi TEMİZ, öz-test T1-T8 GEÇTİ. KULLANICI TIKLAMASI BEKLİYOR.
             Kaba tahmin (×1.199 kalibrasyon, val_core'a göre): ~0.874+
             (val_public'in büyük sıçraması gerçekleşirse çok daha yüksek olabilir,
              ama val_core istatistiksel olarak daha güvenilir - 12 vs 4 dataset)
v33  0.779   eski en iyi (v9, sabit eşik 0.025, voxel-uzayı gap, division ACIK)
v37  0.774
v38  0.786   <-- YENİ REKOR (+0.007). v9 AĞIRLIKLARI + Adım 3 konfig + est_hat.
             Eğitilmiş yeni detektör HENÜZ submit edilmedi.
```

## 1. VERİ GERÇEKLERİ (Adım 0'da 199 dataset üzerinde doğrulandı)

- shape `(100, 64, 256, 256)` uint16, scale (z,y,x) = (1.625, 0.40625, 0.40625) µm/vox
  → fiziksel hacim ~104³ µm
- train 199 `.zarr`+`.geff`; koloniler 44b6:71, 6bba:128
- **GEFF koordinatları VOXEL** (199/199). Ölçek dönüşümü YAPMA.
- `est_nodes` min/med/max = 3783 / 17909 / 78644 → frame başına 38 / 179 / 786 hücre
- GT node min/med/max = 50 / 659 / 1950
- **GT/gerçek hücre oranı medyan 0.0356** → tahminlerin %96.4'ü edge metriğine GÖRÜNMEZ
- GT yapısı: dataset başına 2–67 soy hattı, **hat uzunluğu p10=8 p50=21 p90=95**
- Ardışık frame hareketi (GT node'ları arası): p50 1.7 / p90 3.5 / p99 7 µm
- **Bizim tespitler arası** görünür yer değiştirme: p50 1.72 / p90 6.20 / p95 8.17 / p99 11.96 µm
  (7 µm eşleştirme toleransı → her tespit sapabilir → kuyruk ikiye katlanıyor)

### Test seti (Data sekmesinden)
- `test/` içindeki 4 zarr **sadece örnek**, train kopyası, SKORLANMIYOR
- Notebook rerun'da **gizli test seti** takılıyor, boyut ≈ train (~199 dataset)
- **Embriyo-ayrık**: test farklı embriyolardan → her ayar leave-one-colony-out geçmeli
- **Gizli sette `est_nodes` YOK** (geff yok) → görüntüden tahmin şart (§4)

## 2. DOĞRULANMIŞ ÖLÇÜM ALTYAPISI

`v11_00_probe.py` metrik öz-testi — her yeni metrik kodu bunu geçmeli:
| Test | Beklenen | Anlamı |
|---|---|---|
| A: GT'yi aynen tahmin et | `edge_J = 1.000000` | metrik doğru |
| B: node'ların %10'unu düşür | `edge_J ≈ 0.81` | **r² yasası** |
| C: 10× gürültü node + rastgele kenar | `edge_J = A ile AYNI` | eşleşmemiş kenarlar BEDAVA |

`selftest_A_adj = 1.0989` → GT'nin kendisi 1.0'ın üstünde skor verir (N_pred≪est_nodes bonusu).

**val_core (8 dataset, yoğunluk spektrumu boyunca eşit aralıklı):**
`44b6_341df25f 44b6_3bb3690f 44b6_c771cb04 44b6_8f9ecab4`
`6bba_2540cd90 6bba_3a1849c2 6bba_67ebd073 6bba_57b7cc1e`

⚠️ val_core **yoğun rejime ~2× fazla ağırlık veriyor** (gerçek medyan 179 hücre/frame,
core'da 38–748 var; ayrıca metrik GT kenarıyla ağırlıklandırdığı için 57b7cc1e tek başına
core'un üçte birini taşıyor). **Mutlak değer LB tahmini DEĞİL, sadece deltalar anlamlı.**

## 3. KAYIP BÜTÇESİ (Adım 3, gerçek metrik)

7020 GT kenarı üzerinde, v9 ağırlıklarıyla:
```
TP = 5072   FP = 1349   FN = 1948        edgeJ = 0.606
```
İki ucu da tespit edilmiş 5549 kenarın dağılımı:
```
DOĞRU bağlandı        5072  %91.4     ← linker SAĞLAM
kapı dışı              276  % 5.0
kaynak yanlış hedefe   119  % 2.1
hedefi başkası aldı     77  % 1.4
hiç atanmadı             5  % 0.1
tespit yok            1471            ← ASIL KAYIP
```
**Her tespit kaçırması iki kez ceza yazdırıyor**: FN + (bizim node'un sahte bir komşuya
bağlanmasından doğan) FP. FP'lerin neredeyse tamamı linker hatası değil, tespit kaçırmasının
yan ürünü.

### Detektörün gerçek tavanı
```
n*=0.95  edgeJ=0.6712  recall=0.9157  n=1.104  →  core 0.6640
n*=1.30  edgeJ=0.6773  recall=0.9483  n=1.565  →  core 0.6391
n*=1.80  edgeJ=0.6804  recall=0.9566  n=2.105  →  core 0.6110
```
Recall 0.9566'ya ÇIKABİLİYOR ama node sayısı çarpanı kazancı yiyor.
**Adım 5'in hedefi tek sayı: `recall @ (N = est_nodes)`.** Şu an 0.915, bu detektörün
tavanı 0.957, GT ile eğitilmiş detektörden 0.97+ beklenir.

## 4. KAZANAN KONFİGÜRASYON (Adım 3, core 0.6640)

```python
r_nms   = 7.0      # 4.0→0.584 5.5→0.588 7.0→0.596 8.5→0.584 (sabit esikte)
n_star  = 0.95     # 0.3→0.379 0.5→0.517 0.7→0.612 0.95→0.664 1.3→0.639 1.8→0.611
gate    = 10.0     # 6.5→0.608 8→0.610 10→0.611 12→0.610  (DÜZ, önemsiz)
mnn     = 0.80     # 0.7→0.662 0.8→0.664 0.9→0.661 0.99→0.655
fill    = 4.0      # 0→0.6635 4→0.6640 8→0.6499
gap     = dict(max_gap=8, gate_um=8.0)   # FİZİKSEL uzayda! iç optimum
min_len = 6        # GT hat p10=8 → ilkeli sınır. 24→0.605 64→0.344 96→0.140
division= KAPALI   # yaklaşık metrikte dtp=0 dfp=57 → resmi metrikle test EDİLMEDİ
```

### est_nodes kestiricisi (gizli sette şart)
```
!!! KESTIRICI DETEKTORE BAGLI — model degisirse YENIDEN FIT SART !!!
    Yeni detektorun skor dagilimi tamamen kaydi (hm_max_med 0.955->0.836,
    3bb3690f'te 0.931->0.514; Np@0.200 20432->4487). v9 icin fit edilen
    formul yeni modelde COKUYOR (thr=0.350: med %73, max %1434).

ADIM 5 DETEKTORU ICIN (Adim 2.5b, 199 dataset):
est_hat = 0.617044 * Np(thr=0.025)^1.064065
   tum med %6.1 | 44b6-fit->6bba %7.1 | 6bba-fit->44b6 %11.8 | max %72.3
   (yuksek esikler yeni modelde COKUYOR: 0.350 -> med %73.4 max %1434)

v9 DETEKTORU ICIN (Adim 2.5):
est_hat = 0.229320 * Np(thr=0.350)^1.183500
   Np = 7 µm greedy NMS sonrası, thr=0.350 üstündeki tespit sayısı (tüm film,
        T=100'e normalize)
   tüm med %7.0 | tutulan-6bba %6.9 | tutulan-44b6 %14.0 | max %71.0

ESKİ (12 nokta, submit v38'de KULLANILAN): 0.315171 * Np(0.200)^1.141545
   199 noktalı 0.200 fiti 0.289*Np^1.1454 → eski fit zaten iyiydi, fark küçük.
   Sonraki submit'te 0.350 sürümüne geç (kuyruk riski yarıya iniyor).
```

## 4b. KAPANAN HATLAR — SON-İŞLEM KOLLARI TÜKENDİ (2026-09-18)

| Hat | Ölçüm | Karar |
|---|---|---|
| Division (geometrik) | resmi metrik: hassasiyet 1/72, her fork İKİ kez ödetiyor (div FP + kenar FP); KAPALI 0.6640 vs r12a90 0.6462 | **zararlı, kapalı** |
| Dataset-başına `n*` | GT'siz vekil YOK (en iyi rho −0.243); GT'li üst sınır bile +0.0122 (oracle +0.0269); LOO işaret tutarsız | **n\* sabit 0.95** |
| Track-güdümlü kurtarma (Adım 3b) | mekanizma ÇALIŞIYOR (rec 0.9157→0.9211) ama n 1.105→1.130 çarpanı yiyor → net **+0.0017**; r=2.5µm > 4.0 > 6.0 (geniş yarıçap yanlış kurtarıyor); floor=0.05 en iyi | **park, karmaşıklığa değmez** |
| Uyarlanabilir kapı | en_iyi_kapı/aralık oranı 0.163–0.687 dağınık | sabit 10.0 |
| Agresif `min_len` | 6→0.664, 24→0.605, 64→0.344, 96→0.140 (GT hat p10=8 ile tutarlı) | min_len=6 |
| Az tespitle çarpan bonusu | n\*=0.3→0.379, 0.5→0.517, 0.7→0.612 | erişilemez |

**SONUÇ: boru hattı 0.665'te ve 0.95'e kalan açığın NEREDEYSE TAMAMI TESPİT
KALİTESİNDE.** Adım 5 artık sadece en büyük kol değil, pratikte KALAN TEK kol
(artı Adım 6'nın öğrenilen kenar sınıflandırıcısı). Kanıt: 1471/7020 GT kenarında
bir uç tespit edilmemiş (%21) ve her kaçırma İKİ kez ödetiyor (FN + sahte komşuya
bağlanan kenardan FP → FP'lerin neredeyse tamamı linker hatası DEĞİL).

## 4c. TUR 2 EĞİTİMİ (`v11-egitim-r2`) — ❌ KAZANÇ YOK

SONUÇ (dengeli adj seti 2×44b6 + 2×6bba):
  başlangıç (tur1 ep91) 0.8548 · ep0 0.8531 · ep50 0.8377 · ep91 0.8440
  EN İYİ = epoch −1 (başlangıç), kazanç +0.0000. Ölçülen HER noktada daha kötü.
❌ HİPOTEZİM YANLIŞTI: "tur 1'de adj hâlâ yükseliyordu" — o yükseliş TEK KOLONİDEN,
   ÜÇ GÜRÜLTÜLÜ NOKTADAN (0.7482→0.7711) çıkarılmıştı. Dengeli sette model zaten
   yerel optimumda; yeni LR döngüsü sarstı, daha iyiye dönmedi.
   DERS: 3 gürültülü noktadan TREND ÇIKARMA.
✅ KORUMA ÇALIŞTI: checkpoint GERÇEK adj'a göre seçildiği için best.pt = başlangıç
   modeli kaldı. Son epoch'a/proxy'ye göre seçilseydi DAHA KÖTÜ model giderdi.
   (NOT: dengeli sette ep91 modeli adj=0.8548 veriyor; tur 1'in 44b6-only 0.7711'iyle
    kıyaslanamaz, farklı dataset karışımı.)
MALİYET: ~6.5 sa GPU, karşılıksız. Haftalık kotadan kalan ~12 sa → GPU SEÇİCİ.
KARAR: DETEKTÖR BU TARİFLE YAKINSAMIŞ. Daha fazla epoch değil FARKLI TARİF lazım
  (izotropik pooling, çıktı stride'ı, sigma) — GPU pahalı, belirsiz → ertelendi.
  Öncelik CPU kollarına: Adım 6 kenar sınıflandırıcı → öğrenilen bölünme kapısı.
  **v39'daki detektör (tur 1 ep91) EN İYİ detektör olarak kalıyor.**

✅ TTA (2026-09-19) DENENDİ, KAPANDI — `v11_S1_submit.py`'de `TTA_FLIP_X = False`.
   Uygulama: orijinal+X-flip ortalaması, batch=2 tek forward çağrısı. Ölçülen
   gerçek maliyet ~1.9-2.0× (tahminle uyumlu) → 199 dataset'te ~6.6 saat.
   KONTROLLÜ KIYASLAMA (aynı script/config/12 dataset, TEK değişken TTA):
     TTA KAPALI  adj=0.7594
     TTA AÇIK    adj=0.7554   (-0.0040, TTA daha KÖTÜ)
   Dataset başına karışık: 6 hafif iyi, 6 kötü, ikisi belirgin kötü
   (0113de3b -%7.2, 05db0fb1 -%4.2). Net kazanç YOK, 2× GPU maliyeti için
   gerekçe yok → KAPALI kalıyor. `submit`e HİÇ push edilmedi (v41 etkilenmedi).
   ⚠⚠ KENDİ HATAM: ilk 4-dataset (val_public) testinde "+0.1065 kazanç" dedim
     ama kıyasladığım "TTA-kapalı" referans SAYILARI ESKİYDİ (kestirici
     düzeltmesinden ÖNCEKİ bir çalıştırmadan, 0b24845f=0.6610 diye — doğrusu
     0.9600). Konuşmanın ortasında farklı çalıştırmalardan gelen sayıları
     harmanlamak yanlış sonuca götürdü. DERS: TTA/config kıyaslaması için HER
     ZAMAN aynı script+aynı dataset'te YENİDEN ölç, önceki mesajlardaki
     sayılara güvenme.

✅ v41  v11_S1_submit.py (`submit` v41)  v39'un GEOMETRİK linker konfigürasyonuna
   GERİ DÖNÜŞ (`USE_EDGE_CLF = False` kill-switch). HEM `v11-s1-test` v4 HEM
   `submit` v41'in kendi doğrulama koşusunda AYNI SONUÇ: 144914 node/138175
   kenar, v39'un orijinal test koşusuyla BİREBİR aynı (28291/30145/6013/80465
   dataset başına). Denetçi TEMİZ, öz-test T1-T8 geçti.
   GERÇEK LB SONUCU GELDİ (`kdrive.py subs`, 2026-09-19T18:04): **pub=0.863**,
   v39 ile birebir aynı — geri dönüş kararı doğrulandı, ek regresyon YOK.

✅ v42/v43  v11_S1_submit.py (`submit` v43) — LAP LINKING ENTEGRE EDİLDİ
   (2026-09-20/21): `USE_LAP=True, LAP_GATE=5.5, LAP_NOLINK_FACTOR=0.8`,
   `link_frames_lap()` `v11_09b_lap_fine.py`'daki (doğrulanmış) fonksiyonla
   BAYT BAYT AYNI (karşılaştırıldı). Entegrasyon adımları:
     1. Yerel öz-test (Kaggle quota HARCAMADAN): gerçek modelin/GPU'nun
        gerekmediği kısmı (link_frames_lap + build_graph + T1-T8) geçici bir
        venv'de (numpy/scipy/pandas) izole çalıştırıldı, T1-T8 GEÇTİ.
     2. Statik tarama: tanımsız-isim + modül-seviyesi çakışma taraması TEMİZ.
     3. `submit` kernel'ine push (v41 ile AYNI kaynaklar: comp+`v11-egitim`
        kernel+model), 4-dataset yerel test koşusu: node=148751 kenar=141659,
        denetçi TEMİZ ✓. v41'in 144914 node'undan FARKLI (beklenen — LAP farklı
        bağlantı topolojisi üretiyor, bu da gap-close/min_len sonrası tutulan
        node kümesini değiştiriyor).
     4. v42'de "linker: GEOMETRIK (yedek)" tanılama satırının YALAN SÖYLEDİĞİ
        fark edildi (eski satır sadece EDGE_CLF'e bakıyor, USE_LAP'ı hiç
        kontrol etmiyordu — SADECE print, gerçek build_graph mantığı doğruydu).
        v43'te düzeltildi, YENİDEN çalıştırıldı: node/kenar sayıları v42 ile
        BİREBİR AYNI (148751/141659) → LAP'ın v42'de ZATEN doğru çalıştığı
        kanıtlandı, sadece tanılama metni yanlıştı. Şimdi doğru yazıyor:
        "linker: LAP (gate=5.5 nolink_factor=0.8)".
   BEKLENTİ: core +0.0097 → LB'de kabaca +0.0097×1.19 ≈ **+0.012** (0.863'ten
   ≈0.875'e). Ensemble'ın aksine bu ÖĞRENİLMEMİŞ bir yöntem (sadece fiziksel
   mesafe optimizasyonu) → v40'ı çökerten embriyo-genelleme riski KATEGORİK
   OLARAK YOK, ama garanti değil (gizli embriyoların hareket istatistikleri
   gate=5.5µm'den sistematik sapabilir - kesin kanıt ancak gerçek LB'de).
   KULLANICI TIKLAMASI BEKLİYOR (2026-09-21).

❌ v43 GERÇEK SONUCU: REGRESYON — "ÖĞRENMEMİŞ = GÜVENLİ" VARSAYIMI ÇÜRÜDÜ
   (2026-09-21, `kdrive.py subs`): **pub=0.859**, v41'in 0.863'ünden **-0.004
   KÖTÜ** — v40'ı (öğrenilen kenar sınıflandırıcısı) çökerten regresyonla
   BİREBİR AYNI BÜYÜKLÜKTE.
   ★★★ EN ÖNEMLİ DERS: LAP'ın kendisi "öğrenmiyor" (sadece Macar algoritmasıyla
   maliyet minimize ediyor) ama `gate=5.5` ve `nolink_factor=0.8`
   HİPERPARAMETRELERİ yine de sadece 2 bilinen embriyoda (44b6, 6bba) fine-sweep
   ile kalibre edildi. Bu, bir sınıflandırıcının AĞIRLIKLARINI öğrenmekten
   farklı bir mekanizma ama SONUÇ AYNI: yerel metrikte iki fold'da da yukarı
   çıkan bir kalibrasyon, gerçek gizli embriyolarda TUTMADI. Yani risk sınırı
   "öğrenilmiş model mi" değilmiş — **"herhangi bir serbest parametre (ağırlık
   veya hiperparametre farketmez) sadece 2 bilinen embriyoda ayarlandıysa,
   yeni embriyoda tutacağının HİÇBİR GARANTİSİ yok" imiş.** Bu, v39/v41'in
   orijinal geometrik konfigürasyonunun (gate=8.0 vb, Adım 3c'de aynı şekilde
   2 embriyoda ayarlanmış) NEDEN hâlâ ayakta kaldığını da sorgulatıyor — muhtemel
   fark: o konfigürasyon çok daha GENİŞ/gevşek bir kapı kullanıyor (8.0µm vs
   LAP'ın sıkı 5.5µm), yani yeni embriyonun hareket istatistiklerindeki sapmayı
   daha toleranslı karşılıyor. LAP'ın "daha sıkı kapı daha iyi" bulgusu YEREL
   veride doğruydu ama bu sıkılığın kendisi genelleme payını daralttı.
   AKSİYON: `USE_LAP=False` ile GERİ ALINDI (`submit` v44), v41'in doğrulanmış
   geometrik konfigürasyonuna dönüldü. LAP kod olarak dosyada duruyor
   (kill-switch), silinmedi.

   ★★★ "GENİŞ KAPI + LAP" FİKRİ TEST EDİLDİ VE KESİN OLARAK ELENDİ (2026-09-25):
   `v11_22_lap_widegate.py` (`v11-lap-widegate`, CPU, 37dk) gate=5.5→12.0
   aralığını (nf=0.7-1.0 ile) TAM taradı. Sonuç MONOTONİK: gate büyüdükçe core
   KESİNTİSİZ DÜŞÜYOR (5.5→0.7293, 6.0→0.7263, 7.0→0.715-0.718, **8.0→0.702-
   0.708 [v39'un GÜVENLİ kapısıyla AYNI genişlik, ama MEVCUT'tan (0.7197) BİLE
   KÖTÜ]**, 12.0→0.636-0.669 çöküş). "Güvenli genişlikte de kazanç var mı"
   sorusunun cevabı NET: HAYIR — orta yol yok, LAP'ın kazancı YAPISAL olarak
   sıkı kapıya bağlı (karşılıklı-en-yakın-komşu güvencesi yok, geniş kapıda
   ucuz-ama-yanlış eşleşmeleri serbestçe kabul ediyor). LAP artık "beklemede"
   değil, KESİN OLARAK KAPALI — gate'i genişleterek kurtarma yolu YOK.
   STRATEJİK ETKİ: Bu turda "yerelde iki fold'da doğrulanmış" İKİ ayrı
   değişiklik (Adım 6c kenar sınıflandırıcısı, LAP linking) gerçek gizli
   testte İKİSİ DE ~aynı büyüklükte (-0.004) regresyona uğradı. Bu artık
   tesadüf değil, SİSTEMATİK bir örüntü: mevcut doğrulama metodolojimiz
   (sadece 2 bilinen embriyo, ne kadar dikkatli bölünürse bölünsün) embriyo
   genellemesini ÖNGÖREMİYOR. UNet-feature sınıflandırıcısı (yukarıda, ⚠
   PARKA ALINDI) bu ışıkta DAHA DA riskli görünüyor — aynı kalıba düşme
   ihtimali yüksek. 0.863 (v39/v41) hâlâ EN İYİ DOĞRULANMIŞ SKORUMUZ.

❌ İZOTROPİK HAVUZLAMA DÜZELTMESİ — YEREL TAM ÖLÇÜMDE DE BAŞARISIZ (2026-09-22/24):
   Tur 2'den beri ertelenen fikir (STATE.md 4c: "farklı tarif lazım, izotropik
   pooling") uygulandı. Voxel ölçeği (1.625, 0.40625, 0.40625) → Z fiziksel
   olarak Y/X'ten 4x kaba; eski mimari (`pool_kernels=[(1,2,2),(2,2,2),(2,2,2)]`)
   Z'yi toplam 4x, Y/X'i 8x küçültüyordu → etkin çözünürlük Z'de 6.5µm, Y/X'te
   3.25µm (2x kaba). Düzeltme: `[(1,2,2),(1,2,2),(2,2,2)]` → Z toplam 2x,
   ikisi de 3.25µm (izotropik). Bu SABİT bir fiziksel gerçeğe dayanıyor, 2
   embriyoda kalibre edilmiş bir hiperparametre DEĞİL — LAP'ın düştüğü
   tuzaktan kategorik olarak farklı bir risk profili taşıyordu.
   `v11_15_train_anisofix.py`: state_dict eski mimariden YENİ mimariye
   `strict=True` ile hatasız yüklendi (havuzlama parametresiz, sadece ara
   katman şekilleri değişiyor — lokal dry-run'da doğrulandı). En iyi bilinen
   checkpoint'ten (v11-egitim ep91) sıcak başlatıldı, 92 epoch/455dk.
   EĞİTİM İÇİ PROXY SONUCU (basitleştirilmiş dahili adj, 4 dataset dengeli set):
     BAŞLANGIÇ=0.7746 → EN İYİ epoch=50 adj=0.8388  (+0.0642 — v9→v11'in
     orijinal kazancından (+0.0449) bile büyük, İYİ GÖRÜNÜYORDU)
   ★★★ AMA GERÇEK TAM PIPELINE (Adım3c: aday üretimi+n* hedefleme+NMS+
   geometrik linking+gap-close+min_len, 12 dataset, AYNI sabit config)
   FARKLI VE KÖTÜ SONUÇ VERDİ (`v11_17_aniso_compare.py`, `v11-aniso-compare`):
     v3 (mevcut)     core=0.7197  44b6=0.7833  6bba=0.6997  PUBLIC=0.8752
     v3aniso (yeni)  core=0.7134  44b6=0.7616  6bba=0.6980  PUBLIC=0.8346
     (-0.0063 core, -0.0217 44b6, -0.0406 PUBLIC — NET KÖTÜLEŞME)
   Dataset başına karışık: 44b6_8f9ecab4'te BÜYÜK kazanç (+0.0634) ama
   44b6_3bb3690f (-0.0667) ve 44b6_c771cb04 (-0.0552) bunu fazlasıyla götürüyor.
   DERS: eğitim script'inin dahili "adj" proxy'si (basitleştirilmiş bir
   linker/pipeline kullanıyor, hız için) GERÇEK Adım3c tam hattıyla AYNI ŞEY
   DEĞİL — biri iyi derken diğeri kötü diyebiliyor. İzotropik havuzlamanın
   ısı haritası keskinliğini/dağılımını değiştirmesi, aday-eşik seçim sürecini
   (n* hedeflemesi) farklı ve olumsuz etkilemiş olabilir.
   İYİ TARAF: bu, GERÇEK GİZLİ TESTE HİÇ GÖNDERİLMEDEN yerel tam ölçümde
   elendi — submit hakkı harcanmadı, metodoloji ("ölçüm önce, tahmin sonra")
   tam olarak işini yaptı. `submit`e HİÇ push edilmedi, üretim v44'te
   (geometrik linker, eski mimari) kalıyor.
   GÜNCEL SAYIM: bu oturumda denenen 4 "gelişim" adayından (ensemble, kenar
   sınıflandırıcı, LAP, anisofix) SADECE LAP'ın kaba/ince taraması ile
   orijinal v9→v11 detektör eğitimi GERÇEK kazanç sağladı; LAP hidden testte
   sonradan çöktü. Yani şu ana kadar kalıcı, doğrulanmış TEK kazanç: v9→v11
   detektör eğitimi (0.786→0.863). 0.863 hâlâ EN İYİ DOĞRULANMIŞ SKORUMUZ.

### (eski plan, kayıt için)

Gerekçe: Adım 5'te adj ep50 0.7482 → ep91 0.7711 HÂLÂ YÜKSELİYORDU; LR 1e-6'ya
indiği için (bütçe bitti) durdu, yakınsadığı için değil.
Değişiklikler (v11_05b_train_r2.py):
  - sıcak başlatma ep91'den (v11_detector_best.pt), bulamazsa ASSERT (sessiz
    v9 geri düşüşü 7 saat yanlış eğitim demekti; çalışan kernel log'u okunamıyor)
  - tepe LR 1.2e-4 → 4e-5 (yakınsamaya yakın, sarsma)
  - VAL_PUBLIC eğitimden HARİÇ (tur 1 sızıntısı)
  - adj seti İKİ KOLONİDEN 2+2 (tur 1'de 4'ü de 44b6'ydı)
⚠ Tur 2'nin adj baseline'ı FARKLI datasetlerde → tur 1'in 0.7711'iyle DOĞRUDAN
  KIYASLANAMAZ. Karşılaştırma noktası tur 2'nin kendi başlangıç satırı.
⚠ AYRI SLUG KASITLI: v39 modeli `--kernel hcanakbas/v11-egitim`'den alıyor.
  Aynı slug'a push etmek çıktıyı ezerdi; Kaggle rerun'da hangi versiyonu
  kullanacağı belirsiz → v39 test edilmemiş modelle çalışabilirdi.
  **v11-egitim slug'ına ASLA yeniden push etme** (v39'un kaynağı).

## 4e. ADIM 6 — KENAR SINIFLANDIRICI (`v11-kenar-clf` v2, 141 dk CPU)

✅ Maske tutarlılık öz-testi GEÇTİ (306 kenar birebir). 89793 etiketli kenar (%21.8 poz).
✅ Koloniler arası AUC: 44b6→6bba 0.996 · 6bba→44b6 0.988 (iyi genelleşiyor).
✅ Referans aynı node'larla yeniden ölçüldü = 0.7197 (Adım 3c ile birebir).
SONUÇ (greedy-by-P linking + Adım 3c gap/min_len):
     p_min   core     44b6     6bba
     ref    0.7197   0.7833   0.6997
     0.27   0.7198   0.7792   0.7010
     0.35   0.7215   0.7807   0.7028
     0.50   0.7270   0.8001   0.7044   ← +0.0074 / +0.0168 / +0.0047  İKİ FOLD İYİ
   ≈ +0.009 LB (×1.19). Tahminim +0.02-0.04'tü; gerçek altında AMA tutarlı.
NEDEN KÜÇÜK: tek başına mesafe (d) AUC 0.989 — geometri bağlantıyı zaten neredeyse
  çözüyor (linker %92.6 doğruydu). Sınıflandırıcı kalan belirsiz durumlarda ekliyor.
⚠ OPTİMUM IZGARA KENARINDA (0.50, tekdüze artıyor) → 6b'de 0.35-0.90 taranıyor.
  İlginç mekanizma: yüksek p_min → belirsiz bağlar kopuk kalır → gap_close onları
  interpolasyonla köprüler → n ve recall ARTIYOR (1.041→1.130, 0.948→0.958).
⚠⚠ BU HALİYLE SUBMIT'E GİREMEZ: özellik çıkarma dataset başına ~5 dk (Python çift
  döngüsü) → 199 dataset ≈ 16 SAAT, 9 saat limitinin çok üstünde.
→ ADIM 6b (`v11-kenar-clf-vec`, v11_06b_edgeclf_vec.py): özellikler VEKTÖREL
  (k-NN k=8, 12µm sınırlı; adaylar zaten sıralı → rank_ij bedava). Maske YOK —
  eğitim/test aynı fonksiyonu aynı şekilde çağırır → maske tutarsızlığı YAPISAL
  OLARAK imkânsız. Özellik tanımı biraz değişti → sınıflandırıcı YENİDEN eğitiliyor.
  Model v11_edge_clf.pkl olarak kaydediliyor (submit'e bağlanacak).
  Dataset başına süre ölçülüyor: hedef ~1-2 s → 199 dataset ~5-10 dk.

✅ SÜRE ÇÖZÜLDÜ: val özellik çıkarma 0.1–0.9 s/dataset (ort ~0.4s) → 199 dataset'te
   ~1-2 dk. (Not: submission'da bu maliyet zaten var olan NMS/n* seçimine EK —
   detection pipeline'ı bunu esasen yapıyor, edge-feature kısmı gerçek ek maliyet.)
❌ AMA KALİTE GERİLEDİ — İKİ FOLD KURALINA TAKILDI:
     core      44b6      6bba
     v1(döngü) 0.7270   0.8001   0.7044   +0.0074/+0.0168/+0.0047  İKİ FOLD İYİ
     v2(vektör)0.7231   0.7987   0.6996   +0.0034/+0.0154 −0.0001  6bba DÜZ/KÖTÜ
   HİPOTEZ (ÖLÇÜLMEDİ, test ediliyor): K_NN=8 sınırı yoğun frame'lerde (6bba
   genelde daha yoğun) 12µm içindeki >8 komşuyu görmüyor → ratio_i/ratio_j
   bozuluyor. → v11_06c_edgeclf_knn16.py: K_NN 8→16 + kesilme oranı ÖLÇÜLÜYOR
   (k=K_NN slotu dolu olan node oranı) + p_min ızgarası 0.40-0.65'e inceltildi
   (v2'de 0.7 sonrası çöküş görüldü: 0.8→0.66, 0.9→0.52).
   `v11-kenar-clf-knn16` çalışıyor (CPU, internet açık).

✅→ TPU: kullanıcı "TPU ayrı kota, beleşse çalışsın, tutarsa güzel tutmazsa gram
  önemi yok" dedi. `v11-tpu-duman` slug'ı ÜZERİNE `v11_07c_divcand_tpu.py` push
  edildi (smoke test yerine gerçek iş): bölünmece-zengin dataset'lerin (ndiv>=2,
  val hariç) aday dökümü, TPU'ya uyarlanmış (`torch_xla`, autocast yok, düz
  float32, `xm.mark_step()` ile senkron; TPU açılmazsa CPU'ya düşer, çökmez).
  ⚠ Hesap TPU oturum kotası = 1. `v11_08a_tpu_smoke.py`'nin sırası hâlâ
  `status=queued` olduğu için yeni push da aynı sınıra takıldı ("Maximum batch
  TPU session count of 1 reached"). Kaggle'ın davranışı: push, kernel'in
  kaynak kodunu YERİNDE günceller — kuyruktaki iş donanım bulduğunda EN SON
  push edilen kodu (bölünme dökümü) çalıştırması BEKLENİR, ama bu doğrulanmadı.
  Fire-and-forget: sonuç gelirse iyi, gelmezse CPU işini etkilemez.
  Script duruyor: `v11_07c_divcand_tpu.py` (TAG="v3div").

✅ SONUÇLANDI (2026-09-19): `v11-tpu-duman` kuyruktan çıktı — AMA beklenmedik
  şekilde EN SON push ettiğim kodu (bölünme dökümü) DEĞİL, İLK push ettiğim
  kodu (smoke test) çalıştırdı.
  ★ OPERASYONEL DERS: Kaggle kuyruğu, girdiği ANDAKİ koda KİLİTLENİYOR — sonraki
    push'lar kernel'in KAYNAK KODUNU günceller ama zaten kuyruktaki çalıştırma
    isteğini DEĞİŞTİRMEZ. "Maximum batch TPU session count of 1 reached"
    uyarısı o push'un KENDİ çalıştırma isteğinin REDDEDİLDİĞİ anlamına
    geliyormuş (kod güncellenmiş olsa da). Yukarıdaki "doğrulanmadı" notu
    YANLIŞ çıktı — düzeltiliyor: aynı slug'a ikinci kez push etmeden önce
    ilk işin bitmesini beklemek veya FARKLI slug kullanmak gerekiyor.
  ★ SONUÇ (smoke test, TPU gerçekten çalıştı): doğruluk CPU ile %0.43 fark
    içinde AYNI, ama hız **73.44 s/frame** (GPU: 0.66 s/frame) → **111× YAVAŞ**.
    199 dataset'te tahmini süre 405.94 saat. `ConvNeXtBlock3D`'nin derinlemesine
    ayrılabilir 3D konvolüsyonu (`groups=dim`) XLA'da çok kötü optimize.
  → KESİN KARAR: bu mimariyle TPU KULLANILAMAZ, bir daha denenmeyecek.
  → Şans eseri zararsız: bölünme dökümü TPU'da GERÇEKTEN çalışsaydı ~80+ saat
    sürer, zaman aşımına uğrardı. GPU'ya paralel geçiş (v11-bolunme-gpu)
    kazasen doğru karardı.

## 4d. BÖLÜNME FİZİBİLİTESİ (Adım 7a, `v11-bolunme-sayim`, CPU)

199 dataset: **151 GT bölünmesi** / 128883 GT kenar (1000 kenarda 1.17)
  44b6: 26 bölünme (21 dataset) · **6bba: 125 bölünme (66 dataset) → %83'ü 6bba'da**
  val_core: 5 · val_public: 3 · **eğitim havuzu (val hariç): 143 bölünme, 83 dataset'te**
  dataset başına: {0:112, 1:46, 2:25, 3:10, 4:5, 5:1}

⚠ val_core'da 5 bölünme → divJ ÖLÇÜLEMEZ (tek TP sonucu uçurur; tur 2'nin
  "gürültülü noktadan sonuç çıkarma" tuzağının aynısı).
→ Bölünme kapısını 143 bölünmelik eğitim havuzunda DATASET'LER ARASI ÇAPRAZ
  DOĞRULAMA ile ölç (83 dataset'i katlara böl, kat başına ~28 bölünme).
  Koloni-ayrık doğrulama zor: 44b6'da sadece 26.

ÖDÜL: LB'ye 0.1×divJ DOĞRUDAN eklenir (kenar kısmının 1.19 çarpanı UYGULANMAZ).
  divJ 0.25 → +0.025 LB (~40 TP/~20 FP) · divJ 0.50 → +0.050 LB (~100 TP/~50 FP)
MALİYET ASİMETRİSİ LEHİMİZE: her yanlış fork bir kenar FP'si ekler ama küçük
  (~50 FP kenar → edgeJ ~−0.004). Hassasiyet ~%50 olsa bile net pozitif.
  Geometrik yaklaşım %1.4'teydi (Adım 4: dtp=1 dfp=71).
MEVCUT DÖKÜMLERDE (Adım 7b): v3tr 31 bölünme (18 dataset) + v3 8 = 39 → SINIRDA.

✅ Adım 7d (`v11-bolunme-gpu`, saf CUDA, 54.4 dk, GPU kotası hâlâ vardı) — TPU
   kuyruğu (`v11-tpu-duman`) çok uzun sürdüğü için PARALEL GPU denemesi yapıldı,
   torch_xla koduna hiç dokunulmadan. SONUÇ: **39 dataset, 99 GT bölünmesi**
   (ndiv≥2, val hariç, sızıntı yok) → `cand_v3div_*.npz`.
   EĞİTİM HAVUZU ARTIK YETERLİ: 99 (v3div, eğitim) + 8 (val_core 5 + val_public 3,
   held-out) = 107 toplam. Held-out hâlâ ince (8) — tur 2'nin dersi gibi az
   noktadan sonuç çıkarmaktan kaçınılmalı, ama EĞİTİM için sağlam.
   → Adım 8 (öğrenilen bölünme kapısı) artık YAPILABİLİR.

✅ Adım 8 v11_08_divgate.py  (`v11-bolunme-kapisi`, CPU, internet açık) GÖNDERİLDİ.
   TASARIM (Adım 6 ile aynı disiplin):
     EĞİTİM: v3div (39 dataset, 99 GT bölünme)
     DEĞERLENDİRME: v3 (val_core+val_public, TÜMÜ = 12 dataset, 8 GT bölünme)
     ADAY ÜRETİMİ: Adım 6c linker'i (K_NN=16,p_min=0.55) ile birincil bağlantı
       kurulur; t+1'deki sahipsiz node'lar, o parent'ın 20µm çevresindeyse
       ikinci kız adayı olur.
     ETİKET: ebeveyn eşleşmiş+GT'de gerçek bölünme+iki aday da GT kızlarına
       eşleşiyor → POZ; eşleşmiş+GT'de çocuğu var ama bu eşleme yanlış → NEG
       (promote edilirse FP); eşleşmemiş veya GT'de hiç çocuğu yok → HARİÇ.
     ÖZELLİKLER: d_c1, d_cand, ratio_len, cos_angle, s_p/s_c1/s_cand, dens_p
       (gerçek yerel yoğunluk), n_competitors, edge_prob_c1 (birincil bağın
       kendi olasılığı).
     KARAR: SADECE resmi skorlayıcı (tracksdata) — Adım 4'te kendi yaklaşık
       metriğimizin division'da güvenilmez olduğu kanıtlanmıştı.
     Eşik başına: tüm parent'lar için EN YÜKSEK olasılıklı tek aday eklenir
       (canonicalize()'in keyfi kenar-id sırasına güvenmek yerine).
   ⚠ İNŞA SIRASINDA 3 GERÇEK HATA YAKALANDI (statik tarayıcıyla, çalıştırmadan
     önce): (1) `glob` bare kullanılmış ama sadece `_glob` import edilmişti,
     (2) sed ile parça çıkarırken resmi skorlayıcının import satırları
     (warnings/polars/tracksdata/metrics) kesilmişti, (3) bootstrap sonrası
     temel import satırı (os/time/np/cKDTree/linear_sum_assignment) hiç
     eklenmemişti. Üçü de "tanımsız isim" statik taramasıyla, Kaggle'a
     göndermeden ÖNCE yakalandı.
   ⚠ HELD-OUT SADECE 8 BÖLÜNME — sonuç GÜRÜLTÜLÜ olacak, tur 2'nin dersi gibi
     aşırı yorumlamaktan kaçınılmalı. Kazanç görülse bile TEK SEFERLİK
     ölçümle submit'e sokma kararı temkinli verilecek.
   ⚠⚠ 4. HATA (runtime, statik taramada YAKALANAMADI): `sed` ile çıkarılan
     official-scorer bloğu `AK = td.DEFAULT_ATTR_KEYS` satırını IMPORT'LARDAN
     ÖNCE içeriyordu; eksik import'ları eklerken bunu fark etmeyip İKİNCİ bir
     AK satırı daha eklendi — birincisi hâlâ `td` tanımsızken çalışıyordu.
     Statik "tanımsız isim" tarayıcım bunu YAKALAYAMADI çünkü sadece "modülde
     bir yerde tanımlı mı" bakıyor, ÇALIŞMA SIRASINI kontrol etmiyor.
     → Yeni kontrol eklendi: MODÜL SEVİYESİNDE (fonksiyon içi değil) tekrarlanan
     tanım/atama taraması — sıralama hatalarının asıl kaynağı bunlar.
     v2'de düzeltildi, tekrar gönderildi.
   PERFORMANS: v2 gerçekten yavaştı ama TAMAMLANDI (159.2 dk, tahminim
     40-60 dk'ydı) — `division_candidates()` vektörize değil (bkz. yukarı),
     tekrar kullanılırsa frame_features gibi vektörize edilmeli.

   ❌ SONUÇ: BAŞARISIZ, İKİNCİ FARKLI SEBEPLE. `dtp=0` TÜM eşik ızgarasında
     (0.10'dan 0.80'e) — 8 held-out GT bölünmenin TEK BİRİ BİLE bulunamadı.
     "EN IYI p_min_div=0.4 kazanç=+0.0001" YANILTICI: divJ=0.0000 HER YERDE,
     görünen kazanç sadece birkaç yanlış fork'un kenar-jaccard'a bıraktığı
     gürültü seviyesinde yan etki, GERÇEK BÖLÜNME KAZANIMI DEĞİL.
     KENDİ SCRIPT'İMİN "iyileşti->kaydet" MANTIĞI HATALIYDI: divJ'yi değil
     toplam skoru kıyaslıyordu → v11_div_clf.pkl kaydedildi ama SUBMIT'E
     ENTEGRE EDİLMEDİ (2026-09-19 kararı).
     MUHTEMEL SEBEP: aday üretimi (K_NN=16/p_min=0.55 birincil + 20µm arama)
     eğitimde bile 99 GT bölünmenin sadece 50'sini adaya çeviriyor (%50 kayıp,
     muhtemelen yanlış birincil bağ veya 20µm dışı ikinci kız) → sadece 50
     pozitif örnek, 12 dataset/8 bölünmelik held-out'ta genelleyecek şans yok.
   → BÖLÜNME HATTI PARK EDİLDİ (iki farklı yaklaşım, iki farklı sebeple
     başarısız: geometrik %1.4 hassasiyet, öğrenilen %0 recall). Üçüncü
     deneme (kökten farklı aday üretimi) yeni çok-saatlik tur ister, garantisi
     yok — v39/v40'ın sağlam kazançlarına odaklanmak daha değerli.

## 4f. "0.97 HAYAL Mİ?" — DÜRÜST DEĞERLENDİRME VE YOL HARİTASI (2026-09-20)

Resmi tavan 1.1989 (Adım 4), 0.97 = tavanın %81'i → ULAŞILABİLİR, kanıtlanmış
(temiz public notebook'lar 0.908/0.947 gösteriyor). Ama BİZİM şu anki mimarimizin
(geometrik/greedy linking + tekli detektör + heuristik bölünme) bir tavanı var,
~0.86-0.88 civarı. Temiz üst-sıra notebook'ların bizden FARKI:
  - Kenar özellikleri: bizim GEOMETRİK (koordinat-türevi) vs onların UNet FEATURE
    EMBEDDING'leri (SimpleNodeTransformer, resmi baseline mimarisi)
  - Tek detektör vs DUAL-SEED ENSEMBLE
  - Greedy/mutual-NN linking vs ILP (global optimal)
  - Geometrik bölünme kapısı vs öğrenilen "DeepCenter" kapı
  - 1 görüşlü TTA (bizim, işe yaramadı) vs 8 GÖRÜŞLÜ TTA

★★★ KULLANICI UYARISI (2026-09-20): "örnek aldığımız mimarileri kopyalayarak
onları geçmeyi bekleyemeyiz, güzelleştirmeler şart" — DOĞRU VE ÖNEMLİ. Fikirleri
PRENSİP olarak alıp KENDİ ölçümlerimizle kalibre ediyoruz, birebir kopyalamıyoruz:
  - LAP linking: onların ILP'sinin kopyası değil, bizim hareket yüzdelik
    ölçümlerimizle (p50/p90/p99) kalibre edilmiş Jaqaman-tarzı Macar algoritması
  - KENDİ KEŞİFLERİMİZ (kimsede yok): (1) zarr chunk = tam timepoint → patch
    boyutu IO'dan BEDAVA, büyütülebilir; (2) anizotropik havuzlama sorunu —
    çıktı grid'i z'de fiziksel ölçeğe göre değil voxel oranına göre havuzlanıyor,
    2× kaba. Round 3 detektöründe KENDİ bulduğumuz düzeltme olarak uygulanacak.

ÖNCELİK SIRASI (efor/risk/getiri):
  1. ENSEMBLE (2. tohum) — en düşük risk, kanıtlanmış teknik, embriyo-shift'e
     dogrudan çözüm (varyans azaltma, ogrenilen-linker'ın düştüğü tuzağın tersi)
  2. LAP linking — embriyo-bağımsız matematiksel garanti, öğrenme riski YOK
  3. UNet feature-tabanlı kenar/bölünme sınıflandırıcısı — bugünkü (Adım 6/8)
     başarısızlığın KÖK SEBEBİNİ düzeltiyor (geometrik özellik yerine öğrenilmiş
     feature embedding, detektörün genelleme özelliklerini miras alır)
  4. Çok görüşlü TTA — sadece (3)'ten sonra, feature-tabanlı sistemde

✅ Adım 9  v11_09_lap.py (`v11-lap-linking`, CPU, ücretsiz) — LAP (Macar) linking
   GERÇEK KAZANÇ verdi, İKİ FOLD DA YUKARI, öğrenme riski YOK:
     MEVCUT (geometrik)      core=0.7197  44b6=0.7833  6bba=0.6997
     LAP gate=6.0 nf=0.8     core=0.7263  44b6=0.7933  6bba=0.7053   +0.0066
   ÖRÜNTÜ: gate büyüdükçe LAP KÖTÜLEŞİYOR (6→8→10→12 monoton düşüş, 12'de
   core 0.62'ye çöküyor). Sebep: LAP'ın karşılıklı-en-yakın-komşu güvencesi
   yok, sadece mesafe maliyeti minimize ediyor — geniş kapıda "hiç bağlama"dan
   ucuz olan ama YANLIŞ eşleşmeleri kabul ediyor. Mevcut heuristik linker'ın
   gate=8.0'ı LAP için ÇOK GENİŞ; LAP daha SIKI bir kapı istiyor.
   İnce tarama gönderildi (`v11-lap-fine`, gate 3.5-7.0, nf 0.7-0.9) - gerçek
   optimumu bulmak için, 6.0/0.8 sadece kaba ızgaranın en iyisiydi.

✅ Adım 9b `v11_09b_lap_fine.py` (`v11-lap-fine`, CPU) — İNCE TARAMA SONUCU (NİHAİ):
     gate=5.5 nf=0.8   core=0.7293  44b6=0.7868  6bba=0.7112   (+0.0097 vs 0.7197,
     İKİ FOLD DA YUKARI: 44b6 0.7833→0.7868, 6bba 0.6997→0.7112). Kaba taramadaki
     6.0/0.8'den (+0.0066) daha iyi. Bu sayı `v11_S1_submit.py`'a ENTEGRE EDİLDİ
     (bkz Adım 7 NEREDE KALDIK). Not: fine sweep 1186 dk sürdü (muhtemelen Kaggle
     kuyruk tıkanıklığı, gerçek CPU süresi değil) ama exit=0/complete ile bitti.

❌ ENSEMBLE (2. tohum) — YOL HARİTASININ #1 ÖNCELİĞİ, ÖLÇÜLDÜ, KAZANÇ YOK (2026-09-20):
   `v11_05c_train_seed2.py` (VAL_PUBLIC sızıntısı + tek-koloni adj-eval hataları
   düzeltilmiş kopya) ile bağımsız 2. tohum eğitildi (`v11-egitim-seed2`, 415 dk,
   epoch=91, dengeli sette adj 0.7724→0.8358). SONRA `v11_10_ensemble_dump.py`
   (`v11-ensemble-dump`) ile aynı 12 dataset'te seed2-tek VE (seed1+seed2)/2
   ortalama aday dökümü üretildi; `v11_11_ensemble_compare.py`
   (`v11-ensemble-compare`) ile ÜÇÜNÜ DE AYNI SABİT post-process konfigüyle
   (v39/v41 üretim ayarları: gate=8.0, n*=0.95, gap=6/8µm, min_len=3) kıyasladım:
     seed1 (mevcut)   core=0.7197
     seed2 (tek)      core=0.7067   (-0.0130, AÇIKÇA KÖTÜ)
     ensemble (ort.)  core=0.7195   (-0.0002, GÜRÜLTÜ SEVİYESİNDE — kazanç YOK)
   Koloni kırılımı da net değil: ensemble 44b6'da +0.0138 kazanıyor, 6bba'da
   -0.0042 kaybediyor, toplamda birbirini götürüyor. SONUÇ: ensemble GPU
   maliyetini (çift forward-pass) iki katına çıkarıyor ama ölçülebilir kazanç
   yok → ÜRETİME ALINMADI. Yol haritasındaki öncelik sırası (ensemble>LAP>...)
   BU ÖLÇÜMLE TERS ÇIKTI: LAP (+0.0097, öğrenme riski yok) ensemble'dan (±0.0000,
   2x maliyet) daha değerli çıktı. Ders: "en düşük risk" tahmini "en yüksek
   getiri" ile karışmamalı — ölçmeden sıralama yapmayın.

⚠ UNet-FEATURE KENAR SINIFLANDIRICISI — YEREL POZİTİF, ÜRETİME ALINMADI (2026-09-21/22):
   Yol haritası #3: geometrik özellik yerine detektörün kendi ara-katman
   embedding'i (32 kanal, tam çözünürlük) kullanan kenar sınıflandırıcısı.
   Mimari: `UNet4D_SOTA_V9.forward(x, return_features=True)` son decoder
   katmanının 1x1 head'den ÖNCEKİ çıktısını da döndürüyor (checkpoint AYNI,
   yeni parametre YOK — sadece forward akışı, `return_features=False`
   default'ta ISI HARİTASI BAYT BAYT AYNI, lokalde `torch.allclose` ile
   doğrulandı). `v11_12_unet_features.py`/`v11_13_...train.py` ile 12 val +
   36 eğitim dataset'inde her adayın (t,z,y,x) konumundaki 32-boyutlu feature
   vektörü dökülüp (`cand_v3feat_*`/`cand_v3trfeat_*`), `v11_14_unet_edgeclf.py`
   ile mevcut 17 geometrik özelliğin YANINA 2 skaler eklendi (kosinüs benzerliği,
   L2 mesafesi — ham 64-boyut embedding DEĞİL, bilinçli olarak düşük boyutlu
   tutuldu, embriyoya özgü örüntü ezberleme riskini sınırlamak için).
   SONUÇ (aynı Adım 3c referansına karşı, iki fold kontrollü):
     REFERANS (geometrik)      core=0.7197  44b6=0.7833  6bba=0.6997
     UNet-feature p_min=0.55   core=0.7256  44b6=0.8022  6bba=0.7020  (+0.0059)
   p_min=0.40/0.45/0.50/0.55/0.65 hepsi İKİ FOLD DA YUKARI çıktı.
   ★★★ NEDEN ÜRETİME ALINMADI: v40'ı çökerten kenar sınıflandırıcısı da
   yerelde İYİ görünüyordu (+0.0094 core, val_public +0.1272) ama gerçek
   gizli testte KÖTÜLEŞTİ (-0.004). Şu anki kazanç (+0.0059) o başarısız
   denemeden DAHA KÜÇÜK. "Embedding detektörün genellemesini miras alır"
   hipotezi teorik olarak makul ama YEREL ÖLÇÜM BUNU KANITLAYAMAZ — asıl risk
   (yeni/bilinmeyen embriyoda çökme) yapısal olarak sadece 2 bilinen embriyoyla
   test edilemez. Ek kanıt: `unet_cos_sim` AUC=0.850, `unet_feat_l2` AUC=0.860
   — geometrik özelliklerden (örn. `d` AUC=0.989) BİLİNEN ALANDA bile daha
   az ayırt edici; genelleme avantajı iddiası ampirik olarak desteklenmiyor.
   KARAR: v43 (LAP, öğrenme riski yok, zaten gönderilmeyi bekliyor) ile AYNI
   ANDA ikinci bir öğrenilmiş/riskli değişikliği üretime sokmak, hangisinin
   işe yarayıp yaramadığını ayırt etmeyi imkansızlaştırır. v43'ün gerçek LB
   sonucu gelmeden bu PARKA ALINDI. Model `v11_edge_clf.pkl` kaydedildi ama
   HİÇBİR submit path'ine bağlanmadı.

★★★ METODOLOJİK DERS: KÜÇÜK VAL SETİNDE GENİŞ SWEEP = GÜRÜLTÜYE UYUM RİSKİ
   (2026-09-25): LAP'ın ince taraması 36 (gate×nolink_factor) kombinasyonu
   denedi ve EN İYİSİNİ seçti — sadece 8-12 dataset'lik bir val setinde bu
   kadar çok kombinasyon denemek, gerçek sinyal yerine O SETE ÖZGÜ GÜRÜLTÜYE
   uyum sağlama riski taşır (çoklu-karşılaştırma yanlılığı). v39/v41'in
   kendi gate=8.0 konfigürasyonu da bir sweep'ten çıktı ama TUTTU — muhtemel
   fark: geniş/gevşek bir parametre karar yüzeyinde daha az "keskin" davranır,
   belirli dataset'lerin özel gürültüsünü o kadar kolay yakalayamaz; dar/hassas
   bir parametre (LAP'ın gate=5.5'i gibi) küçük mesafe farklarında davranışı
   dramatik değiştirebilir - tam da küçük val setini ezberlemeye müsait yüzey.
   UYGULAMA: bundan sonra post-process ince ayarında AZ SAYIDA, GENİŞ ARALIKLI
   deneme tercih et; "en iyi tek nokta"yı değil, "geniş bir bölgede tutarlı
   iyi olan" konfigürasyonu ara. Dar bir global optimum bulmak BAŞARI DEĞİL,
   ALARM işaretidir.

## 5. ÇÜRÜTÜLMÜŞ HİPOTEZLER — TEKRAR DENEMEYİN

| Hipotez | Gerçek |
|---|---|
| Deep supervision modeli düşük güvenli yapıyor | `hm_max ≈ 0.92–0.96`, ÇÜRÜK |
| Linking kaybı ~%5 | ~%20, ama linker GT çiftlerinde %91.4 doğru |
| n-hedefleme +0.048 getirir | +0.012 |
| Optimum n* ≈ 1.40 (proxy) | 0.95 (gerçek metrik) |
| Agresif division bedava | dtp=0 dfp=57, kapatmak daha iyi (yaklaşık metrikte) |
| Kısa track filtresi zarar verir | `>=6` +0.012 KAZANDIRIYOR |
| Agresif min_len (24–96) büyük kazanç | FELAKET: 0.605 / 0.344 / 0.140 |
| Node sayısını kısıp çarpan bonusu al | n*<0.9 hep kötü, bonus erişilemez |
| Uyarlanabilir kapı (k×aralık) | oran 0.163–0.687 dağınık, `adaptif_kazandi=False` |
| Yerel hareket alanı | fark yok, KALDIRILABİLİR (hız kazancı) |
| TrackerBrain (5 geometrik feature) | mesafe kapısının monoton fonksiyonu, bilgi eklemiyor |

### Beklenmedik bulgu
Gap closing **tespiti onarıyor**: recall 0.8687 → 0.9216 (+0.053 bedava). İnterpolasyonla
eklenen ara node'lar kaçırılan GT hücrelerine 7 µm içinde denk geliyor. Yörünge
modellemesini ilerletmek muhtemelen daha fazla getirir.

## 6. v9/v10 KOD HATALARI (miras alınan)

| # | Yer | Hata |
|---|---|---|
| F1 | `kaggle_train_v10_final.py:326` | Seyrek GT (%3.6) yoğun supervision gibi kullanılıyor → gerçek hücreler arka plan olarak öğretiliyor. **v10 bu haliyle v9'dan kötü.** |
| F2 | `:408-431` | Deep supervision: coarse seviyelerde `target.eq(1.0)` boş → `num_pos=0` → normalize edilmemiş sadece-negatif loss. Zararlı ama tahmin ettiğim kadar değil. |
| F3 | `:568` | `state_dict()` referans döndürür + `ModelEMA` in-place `copy_()` → `best_1/2/3.pt` hepsi EN SON ağırlıkları içeriyor. `best_f1.pt` doğru. |
| F4 | v9 eğitimi | "F1 0.868" = DoG pseudo-label'a benzeme oranı, doğruluk değil → **plato buradan.** |
| F7 | `kaggle_submit_v10.py:519` | Gap closing HAM VOXEL uzayında → z'de kapı 3.5× gevşek. Fiziksel uzaya taşımak Adım 2'nin en büyük kazancı (+0.018). |
| F9 | `:432-444` | TrackerBrain `link_frames`'te fiziksel, gap closing'de voxel koordinat alıyor → aynı ağ iki farklı ölçek görüyor. |

## 7. NEREDE KALDIK / SIRADAKİ

```
✅ Adım 0  v11_00_probe.py       metrik kanıtlandı, val seti kuruldu
✅ Adım 1b v11_01_candidates.py  aday dökümü (cand_v2_*.npz, 12 dataset) → notebooks/v11/cand/
✅ Adım 2  v11_02_sweep.py       gerçek metrikle eşik/NMS taraması → 0.6081
✅ Adım 3  v11_03_linkdiag.py    linker tanılaması + birleşik sweep → 0.6640
✅ Adım 4  v11_04_official.py    ÇALIŞTI (`v11-resmi-metrik` v2, 4.1 dk). İKİ KESİN SONUÇ:

   (a) **BİZİM METRİK = RESMİ METRİK, adj_farki = 0.0000** (12 dataset, hepsinde
       edgeJ/adj/rec 4 hane aynı; duman testi bizim 1.1989 / resmi 1.1989).
       -> Adım 0-3'te ölçülen HER ŞEY gerçek. Resmi skorlayıcıya artık gerek yok
          (kendi metriğimiz hem aynı hem çok daha hızlı).
   (b) **RESMİ TAVAN = 1.1989** (GT'nin kendisi tahmin olarak verilince).
       Leaderboard'daki 0.97 = teorik maksimumun %81'i. Mükemmeliyet gerekmiyor.

   ❌ DIVISION KAPISI KAPALI (resmi metrikle ölçüldü):
        KAPALI           core 0.6640  dtp=0 dfp=0   divJ=0.0000
        r9 a100 (v9'un)  core 0.6561  dtp=0 dfp=36  divJ=0.0000   -0.008
        r12 a90          core 0.6462  dtp=1 dfp=71  divJ=0.0132   -0.018
        r30 a30          core 0.6018  dtp=1 dfp=279 divJ=0.0035   -0.062
      Geometrik dedektörün hassasiyeti 1/72. Her yanlış fork İKİ kez ödetiyor
      (division FP + kenar FP; edgeJ 0.6712 -> 0.6031).
      "Masada 0.1 var" iddiam YANLIŞ ÇIKTI. Val'de sadece 5 gerçek bölünme var;
      +0.1 teorik olarak hâlâ mümkün ama ~%50 hassasiyet gerekiyor (bizde %1.4)
      -> parametre işi değil, ÖĞRENİLEN sınıflandırıcı işi. PARK EDİLDİ.
✅ Adım 2.5 v11_25_estfit.py     ÇALIŞTI (`v11-est-fit`, 36.1 dk, 199 dataset).

   ⚠ SCRIPT'TE HATA: eşik seçici indeks karıştırıyordu (`r[1] < best[1]` →
     medyan hatayı eşik değeriyle kıyaslıyor). "SECILEN thr=0.025" YANLIŞ.
     Düzeltildi; tablo elle okundu.

   (a) est_nodes FIT — leave-one-COLONY-out karar verdi (test embriyo-ayrık):
         eşik   tüm-med  tutulan-6bba  tutulan-44b6  tüm-max
         0.200   %6.9      %6.8          %14.4       %105.9
         0.350   %7.0      %6.9          %14.0       % 71.0   <-- SEÇİLDİ
       Medyanlar aynı, 0.350'nin KUYRUĞU çok daha iyi. Embriyo-ayrık
       genellemede kuyruk riski önemli.
       **est_hat = 0.229320 * Np(thr=0.350)^1.183500**
       (12 noktalı fit 0.315*Np^1.1415@0.200 ile 199 noktalı 0.289*Np^1.1454@0.200
        neredeyse aynı → mevcut submit'teki kestirici zaten iyiydi.)

   (b) n* HİPOTEZİ DOĞRULANDI: dataset-başına-k oracle kazancı **+0.0269**
       (sabit k=1.3 → 0.8448 ; oracle → 0.8717). best_k dağılımı gerçekten
       yayılıyor: {0.7:8, 0.85:14, 1.0:50, 1.15:48, 1.3:35, 1.5:23, 1.75:12, 2.0:9}
       ÖNGÖRÜCÜ ÖZELLİK: **rec@1.0 ile rho = −0.682** (güçlü).
       → Detektörün recall'ı DÜŞÜK olan dataset DAHA BÜYÜK k istiyor.
         S1'de 0b24845f'te gözlenen şey tesadüf değildi.
       ⚠ AMA rec@1.0 GT gerektiriyor, test setinde GT YOK.

✅ Adım 2.6 v11_26_kpredict.py   GT'siz vekil arandı → **BULUNAMADI, HAT KAPANDI**
   (`v11-k-ongoru`, 199 dataset, CPU)
     thr_at_n1 (ana hipotezim: "detektör eminse kesim skoru yüksek")  rho=-0.017
       -> HİPOTEZ ÇÖKTÜ, korelasyon YOK
     Np.20/Np.05 (en iyi GT'siz)                                      rho=-0.243 (zayıf)
     rec@1.0 (GT gerektiren referans)                                 rho=-0.682
   Öngörü kazançları:
     GT'siz özelliklerin HEPSİ        -0.0026 … +0.0000  (sıfır veya negatif)
     rec@1.0 ile (ÜST SINIR)          +0.0122  (oracle +0.0269'un yarısı)
     leave-one-colony-out  44b6→6bba -0.0114 / 6bba→44b6 +0.0029  İŞARET TUTARSIZ
     üç özellikle          tutulan foldda -0.1067  (aşırı uyum)
   → Üst sınır bile oracle'ın yarısı. Mükemmel GT'siz vekil bulunsa dahi
     proxy'de ~+0.012, gerçekte daha az. **n* SABİT 0.95. TEKRAR DENEMEYİN.**
   Yan fayda: est_hat hatası bağımsız teyit edildi (med %7.0 p90 %22.7 max %71.0).

   ⚠⚠ k=1.3 SONUCUNA GÜVENİLMEZ: bu tablo `rec²×çarpan` PROXY'si, linking
      bozulmasını GÖRMÜYOR. Adım 3'ün GERÇEK metrik taraması n*=0.95'te tepe
      yapıyordu (1.3 → 0.639). Aynı proxy-gerçek uçurumu daha önce de yanılttı
      (proxy 1.40 dedi, gerçek 0.95). → n* = 0.95'TE KALIYOR.
                                 (mevcut fit 12 nokta: med %3.5 ama max %38) +
                                 leave-one-COLONY-out doğrular (test embriyo-ayrık).
                                 AYRICA S1 hipotezini test eder: `n*` sabit mi olmalı?
                                 (0b24845f'te kestiricinin %38 fazlası PUAN ARTIRDI)
                                 Ölçüm: recall@(k*est) eğrisi, k in {0.7..2.0};
                                 sabit-en-iyi-k vs dataset-başına-oracle-k farkı.
                                 kazanç<0.01 -> sabit kalsın; >0.01 -> k'yi öngören özellik ara.
✅ Adım S1  v11_S1_submit.py      KAGGLE'DA ÇALIŞTI (slug `v11-s1-test`, 4.6 dk, 4 örnek dataset)
                                 Öz-test T1-T6 GEÇTİ · model eksik=0 beklenmeyen=0 epoch=130
                                 **forward 0.66 s/frame** (tahminim 0.80) -> 199 dataset ~3.7 saat
                                 submission.csv 16.8 MB · 167231 node · 160768 kenar
                                 est_hat offline fit ile BİREBİR aynı (1.138/1.380/0.967/0.935)
                                 ⚠ gap closing node sayısını yoğunda %20-28 ŞİŞİRİYOR:
                                   0b24845f: seçilen 42994 -> node 55166, n=1.68, çarpan 0.932
                                   (kestirici %38 fazla + gap %28 üstüne -> iki hata birikiyor)
                                   -> `n*`ı 0.95 değil ~0.82 alıp gap SONRASI 0.95'e oturtmayı dene
✅ Adım S1-skor v11_S1_score.py  ÇALIŞTI (`v11-s1-skor` v2, 3.4 dk). 4 örnek dataset, gerçek GT:
                                   adj=0.7384 edgeJ=0.7419 rec=0.9207 n=1.219
                                   0113de3b: edgeJ=0.9216 rec=0.9423 = ADIM 3 İLE BİREBİR
                                     -> submit boru hattı sweep boru hattıyla EŞDEĞER (kanıt)
                                   Aynı 4 dataset: Adım2 konfig 0.6965 → S1 0.7384 (+0.042)
                                   Kestirici maliyeti: -0.011 (gerçek est ile 0.7494 olurdu)
                                 ⚠ BULGU: 0b24845f'te kestiricinin %38 FAZLA tahmini PUAN ARTIRDI
                                   (Adım3 gerçek est ile rec=0.628 edgeJ=0.400;
                                    S1 tahminle rec=0.902 edgeJ=0.661)
                                   -> zayıf detektörlü dataset DAHA FAZLA tespit istiyor,
                                      `n*` SABİT OLMAMALI. Adım 2.5'te ölç (şimdilik hipotez).
✅ submit v38                    PUSH EDİLDİ + BAŞARIYLA ÇALIŞTI, gönderilmeye HAZIR.
                                 enableInternet=false teyit edildi. Kullanıcının tıklaması bekliyor.
                                 Başında ÖZ-TEST var (T1-T6: linking/min_len/gap/NMS/CSV)
                                 -> 5 saatlik GPU işinden önce boru hattını doğruluyor.
                                 İnternet KAPALI çalışır (manuel zarr okuyucu, pip yok).
                                 GPU max_pool3d tepe bulma + tek geçiş NMS:
                                   kabul(eşik) = kabul(taban) & (skor>eşik)  -> ikili arama YOK
✅ Adım 5  v11_05_train.py        BİTTİ (`v11-egitim` v2, 92 epoch, 389 dk).
   **SONUÇ: adj 0.7262 → 0.7711  (+0.0449)**   son: edgeJ=0.7858 rec=0.9703 n=1.176
   HEDEF TUTTU — ve tam istenen şekilde. Recall eğrisi DÜZLEŞTİ:
     k:      0.8      1.0      1.2      1.5      2.0
     v9:    0.7383   0.8348   0.8736   0.8846   0.8905
     yeni:  0.8219   0.8896   0.9095   0.9104   0.9114
   → v9'un k=2.0'da (2× node harcayarak) ulaştığı tavana yeni model k=1.0'da
     ulaşıyor. Detektörün GÜVEN SIRALAMASI düzeldi; aynı node bütçesinde doğru
     hücreleri üste koyuyor. Bu Adım 5'in tam hedefiydi.
   SEÇİM ÖLÇÜTÜNÜ DEĞİŞTİRMEK KRİTİKTİ: proxy ep60'ta tepe (0.8212) sonra
     düşüyor; adj ep50 0.7482 → ep91 0.7711 yükselmeye devam. İKİSİ FARKLI
     EPOCH GÖSTERİYOR. Proxy'ye göre seçsem ep60'ı alırdım.
   ep0'da adj 0.7262 → 0.6784 DÜŞTÜ (ilk epoch sıcak başlatmaya zarar verdi),
     sonra toparlandı ve geçti. Loss 12.5 → 1.10, LR 1e-6'ya indi.
   ⚠ KÖR NOKTA: ADJ_DS=4 → VAL_CORE'un ilk dördü, DÖRDÜ DE 44b6. adj ölçümü
     TEK KOLONİDEN. Recall eğrisi 8 dataset/iki koloni üzerinde ve orada da
     iyileşti, ama adj için embriyo-ayrık doğrulama YAPILMADI → Adım 1c/3c ile
     12 dataset üzerinde iki fold kontrol edilecek.
   ⚠ n=1.176 (hedef n*=0.95). Konfig v9'un detektörüne göre ayarlanmıştı;
     daha iyi detektörle optimum n*/gap/min_len KAYMIŞ olabilir → yeniden tara.
   Checkpoint: out/egitim/v11_detector_best.pt (26.65 MB, yerelde)

✅ Adım 1c  v11_01c_candidates_v3.py  BİTTİ (`v11-aday-v3`, 16.1 dk) → cand_v3_*.npz

✅ Adım 2.5b v11_25b_estfit_v3.py  BİTTİ (`v11-est-fit-v3`, 32.2 dk, 199 dataset)
   (a) Kestirici YENİDEN FİT edildi (yukarı bak) — eşik 0.350 → 0.025 TERS DÖNDÜ.
   (b) **n* AŞAĞI KAYMALI** — yeni detektör aynı recall için DAHA AZ node istiyor:
         k          yeni (rec/proxy)     v9 (rec/proxy)
         0.70       0.9112 / 0.8648      0.7169 / 0.5514
         1.00       0.9635 / 0.9318      0.8970 / 0.8162
         1.30       0.9722 / 0.9186      0.9295 / 0.8448
         2.00       0.9750 / 0.8567      0.9451 / 0.8073
       sabit-en-iyi k: 1.0 (v9: 1.3) · best_k dağılımı 164/199 dataset k≤1.0 (v9: 72)
       v9'da proxy 1.3 derken gerçek 0.95'ti (proxy 0.35 fazla söylüyor)
       → yeni gerçek optimum muhtemelen 0.7–0.85. `n` düşmesi ÇARPANDAN DA kazandırır.
   (c) Dataset-başına k: oracle +0.0253 ama rec@1.0 rho −0.682 → **−0.424** düştü,
       GT'siz özellikler ≤0.151. **HAT KAPALI KALIYOR.**
   ⚠ Rapordaki `TUTULAN(X)` etiketi EĞİTİM kolonisini yazdırıyor, tutulanı değil.
     Sonuçlar doğru, etiket yanıltıcı.

✅ Adım 3c  v11_03c_resweep.py    BİTTİ (`v11-konfig-v3` v2, 17.3 dk).
   **SONUÇ: iki fold da +0.056 — Adım 5'in kazancı İKİ EMBRİYODA da gerçek.**
                       core      44b6      6bba
     v9 (Adım 3)      0.6640    0.7263    0.6442
     YENİ (3c)        0.7197    0.7833    0.6997
     kazanç          +0.0557   +0.0570   +0.0555
   Eğitimdeki +0.0449 (sadece 44b6) → konfig yeniden ayarlanınca +0.0557.
   Linker tanılaması: tespit-yok 1471 → **876 (%40 azaldı)**, linker doğru %91.4→%92.6.

   **YENİ KONFİG (Adım 5 detektörü için):**
     r_nms=7.0  n*=0.95  gate=8.0  mnn=0.80  fill=0.0
     gap=(max_gap=6, gate_um=8.0)  min_len=3  division=KAPALI
   Değişenler: gap (8,8)→(6,8) · min_len 6→3 · fill 4→0 · gate 10→8 (düz)

   ❌ n* TAHMİNİM YANLIŞTI: 0.7–0.85'e kayar demiştim. Gerçek:
        0.70→0.6891  0.85→0.7012  **0.95→0.7197**  1.10→0.7120
      Proxy'nin n* sapması SABİT DEĞİL, detektöre bağlı (v9'da ~0.35, burada
      ~0.05) → proxy'yi n* için düzeltmek de mümkün değil. n* İÇİN PROXY KULLANMA.

   ⚠ SIZINTI (skoru ETKİLEMİYOR): Adım 5'te train_names = VAL_CORE hariç
     hepsi → VAL_PUBLIC'in 4 dataset'i EĞİTİMDE KULLANILDI. Tablo bunu açık
     gösteriyor: 0b24845f rec 0.6275→1.0000, 0113de3b/05b6850b rec→1.0000.
     CORE SKORU TEMİZ: summarise sadece CORE_L (eğitimden hariç 8 dataset).
     Modele zarar yok (gizli test farklı embriyo, 4 ekstra eğitim verisi).
     Kayıp: val_public artık bağımsız kontrol olarak KULLANILAMAZ.
     → Sonraki eğitimde VAL_PUBLIC'i de hariç tut.
   ⚠ v1 HATA verdi: `sed`'im glob desenini değiştirdi ama `load_cands` dosya adını
     f-string ile kuruyordu (`f"cand_v2_{name}.npz"`) → desene uymadı, atlandı.
     Düzeltme: CAND_TAG sabiti, glob ve dosya adı TEK yerden.
   Değiştirilen TEK şey etiketler+loss; mimari v9 ile BİREBİR → sıcak başlatma
   çalışıyor (eksik=0) VE tek değişken izole.
     F1 yoksay maskesi: (img>0.10) & GT'den uzak → negatif loss SIFIR.
        Floresan çekirdekte parlak bölge = hücre; annotate edilmemiş olması
        "arka plan" demek değil. v10'un ölümcül hatası buydu.
     F2 deep supervision KAPALI (sadece en ince seviye)
     F3 checkpoint copy.deepcopy ile
     ek örnekleme %80 → %50 GT-merkezli
   ÖLÇÜLENLER (duman testi):
     bellek 9.23/15.6 GB (Tesla T4) → PATCH=(48,256,256) SIĞIYOR
     hız 1.2 s/örnek → 200 adımlık epoch ≈ 4 dk → 455 dk ≈ 105 epoch
     191 dataset, 128278 GT node
   ⚠ 1. DUMAN TESTİNİN DOĞRULAMASI DEĞERSİZDİ: 19 GT node → granülarite %5.3.
     Raporladığı "+0.1204 kazanç" literal olarak 2 node.

   v2'DE YAPILAN DÖRT DEĞİŞİKLİK (2. duman testiyle doğrulandı):
   (a) YOKSAY EŞİĞİ UYARLANABİLİR. Sabit TAU=0.10 tahminim CİDDİ ŞEKİLDE YANLIŞTI:
       ölçülen uyarlanabilir tau ortalaması **0.395**. Yani 0.10 ile maskeyi
       olması gerekenin çok üstünde geniş yapıp negatif sinyali öldürecektim.
       Yeni: est_nodes → beklenen hücre hacim oranı → o yüzdelik dilim.
       Beklenen oran yoğunlukla %0.9–%37 arası değişiyor, SABİT EŞİK OLAMAZ.
       Ölçüm: yoksay %6.2 (hedef %6.5), aralık %1.5–22.8, patch başı 5.8 GT.
       Loss da doğruluyor: 5.25 → 46.3 (çok daha fazla voksel negatif sayılıyor).
   (b) CHECKPOINT ARTIK GERÇEK adj İLE SEÇİLİYOR (recall proxy'si linking'i
       GÖRMÜYOR). Eğitim içinde tam boru hattı: tepe→NMS→n-hedef→link→gap→
       filtre→resmi metrik. Doğrulama: 341df25f'te adj=0.9708 ölçtü,
       Adım 4'ün tablosunda 0.9707 yazıyor → BİREBİR. Guard'lı (patlarsa
       eğitimi öldürmez, recall'a düşer). Maliyet 1.1 dk/dataset.
   (c) VAL_DS 2→8, VAL_FRAMES 3→20 (~1000+ node); ValSet ÖNDEN YÜKLEMİYOR
       (8 GB RAM olurdu, anlık okuyor).
   (d) cudnn.benchmark + TF32. Bellek 9.23→11.62/15.6 GB (algoritma denemesi),
       hız aynı (1.2 s/örnek). MAX_EPOCH 92 → 411 dk, bütçe 455.

   NOT: v9 epoch'u hacmin %12.5'ini x400 = 50 tam-hacim; v11 %75 x200 = 150.
   → v11 epoch'u TAM 3.00 kat fazla iş yapıyor. "1.5 dk → 4 dk" bir yavaşlama
     DEĞİL, aynı throughput'ta 3 kat büyük epoch. Kaybedilmiş optimizasyon YOK.

□  Adım 6                        öğrenilen kenar sınıflandırıcı (FP budama) +
                                 öğrenilen BÖLÜNME sınıflandırıcısı (division +0.1 hâlâ
                                 orada, geometrik yaklaşım öldü — bkz Adım 4)
□  Adım 6                        öğrenilen kenar sınıflandırıcı (FP budama)
                                 bir kenarı atmak, FP olasılığı J≈0.67'yi aşarsa kârlı
```

### Zaman bütçesi (gizli test ~199 dataset)
Model forward 0.80 s/frame × 100 × 199 = **4.4 saat**. Kernel limiti 9 saat (GPU).
**TTA ve çok ölçekli inference ELENDİ.** Adım 5'te çıktıyı `(1,4,4)` stride'da vermek
sadece doğruluk değil, zaman bütçesi meselesi.

## 7b. KAGGLE OTOMASYONU — CANLI DERSLER (gerçek push'lardan öğrenildi)

| # | Ders |
|---|---|
| 1 | **Kaggle slug'ı BAŞLIKTAN üretir**, `--slug`'dan değil → `--title` = `--slug` ver. kdrive artık uyarıyor. |
| 2 | **`zarr` Kaggle imajında YOK** → analiz kernel'leri `--internet` İSTER. |
| 3 | **`!pip` script kernel'de SÖZDİZİMİ HATASI** (IPython sihri) → hepsinde `subprocess` bootstrap var. Yeni script yazarken `!` KULLANMA. |
| 4 | **`tracksdata` kurulumu numpy'ı yükseltip ÖNYÜKLÜ scipy'yi KIRIYOR** (`AttributeError: _blas_supports_fpe`). Çözüm: `importlib.metadata.version` ile sürümü İMPORT ETMEDEN oku, kurulumdan sonra `--no-deps` ile aynı numpy/scipy'ye geri sabitle. v11_04_official.py'de uygulandı. |
| 5 | Model kaynağı formatı DOĞRULANDI: `--model hcanakbas/biohub-v9-03-sota/pyTorch/default/1` |
| 6 | `model` notebook'unun çıktısı = EN SON versiyonun çıktısı. Aday dökümü için ayrı slug (`v11-aday-dokumu`) kullanılıyor. |
| 7 | Log ancak çalışma BİTİNCE doluyor (`kdrive log` çalışırken boş döner). |
| 8 | **API notebook'u YARIŞMAYA GÖNDEREMEZ.** `POST /competitions/submissions/submit/{comp}` sadece dosya kabul ediyor (`BlobFileTokens must be specified`), `kernelRef` alanını reddediyor (400). → Son "Submit to Competition" tıklaması KULLANICIDA. Geri kalan her şey (push/run/log/output/LB skoru okuma) otomatik. |
| 9 | LB skorları API'den OKUNUYOR: `kdrive.py subs`. v33=0.779 v34=0.776 v36=0.759 v37=0.774. |
| 11 | **Zincirlenen kernel çıktısı `/kaggle/input/notebooks/<user>/<slug>/` altında — ÜÇ seviye derinde.** Glob'da `**` + `recursive=True` ŞART. (`/kaggle/input/*/...` çalışmaz.) |
| 12 | pip'in `ERROR: dependency resolver...` bloğu Kaggle imajının ÖNCEDEN VAR OLAN çakışmaları — zararsız gürültü. Bakılacak satır bizim `[bootstrap] kurulum sonrasi ...` satırı. |
| 13 | `_ensure()` gibi "kur ve import et" kalıbı numpy'ı BELLEĞE yükler; sonraki kurulumlar diski değiştirince uyuşmazlık olur. TÜM pip kurulumlarını HİÇBİR import'tan önce yap, sonra numpy/scipy'yi geri sabitle, ancak sonra import et. |
| 10 | v37'nin submission.csv'si **230 MB**. S1 4 dataset'te 16.8 MB → gizli sette ~900 MB olabilir (daha düşük eşik, daha çok node). Rerun süresine ~15 dk yazma maliyeti ekle. |

**S1 için resmi skorlayıcıya gerek YOK:** division kapalı, ve kenar TP/FP/FN tanımlarımız
resmi kaynak kodla birebir aynı (doğrulandı). Belirsiz olan tek şey division'dı → Adım 4.

### ZARR CHUNK GERÇEĞİ (Adım 5'te fark edildi)
Chunk = TAM bir timepoint `(1,64,256,256)`. Yani `arr[t, z0:z0+32, y0:y0+192, x0:x0+192]`
gibi küçük bir patch okumak **tüm timepoint'i açmakla AYNI maliyette**.
→ v9/v10'un `(32,128,128)` patch'i IO'dan hiçbir şey kazandırmıyordu, sadece her
  okumada hacmin %6'sını kullanıyordu. Patch büyütmek BEDAVA; tek sınır bellek.
  `(48,256,256)` = aynı IO, %75 hacim, okuma başına ~12× fazla GT node.

### Kernel slug'ları
```
model              kullanicinin angarya notebook'u (dokunma, eski versiyonlar burada)
submit             kullanicinin submit notebook'u (LB gecmisi: v33=0.779, v37=0.774)
v11-s1-test        Adim S1 dogrulamasi (GPU, internet YOK)
v11-s1-skor        S1 CSV'sini GT'ye karsi skorlar (CPU, internet ACIK)
v11-aday-dokumu    aday dokumu, Adim 4'un girdisi (GPU, internet ACIK)
v11-resmi-metrik   Adim 4: resmi skorlayici + division (CPU, internet ACIK)
v11-est-fit        Adim 2.5: est_nodes kalibrasyonu (GPU, internet ACIK)
v11-egitim-duman   Adim 5 duman testi (GPU, internet ACIK)
v11-egitim         Adim 5 GERCEK egitim (GPU, internet ACIK, ~7.5 sa)
```

## 8. OTOMASYON

### Kullanıcının mevcut notebook düzeni (ÖNEMLİ)
- **`model`** — tüm angarya/ağır işler (eğitim, döküm) burada yapılıyordu
- **`submit`** — tüm submit işleri burada (LB geçmişi bu notebook'un versiyonlarında,
  en son Version 37 = 0.774, Version 33 = 0.779 en iyi)

Kural: **yarışmaya gönderilecek şey `submit` slug'ına push edilmeli** (yeni versiyon olarak;
Kaggle eski versiyonları saklar, yıkıcı değil). Deneyler için AYRI slug kullan
(`v11-probe`, `v11-cand`, `v11-sweep` …) ki `model`/`submit` kirlenmesin.
Kullanıcı adı: `hcanakbas` (model yolundan: /kaggle/input/models/hcanakbas/...).

### Sürücü
`drive/kdrive.py` — saf stdlib Kaggle sürücüsü (push/wait/output/log/status).
Kimlik: `~/.kaggle/kaggle.json` (chmod 600). API anahtarı hiçbir çıktıya yazılmaz.
```
python3 drive/kdrive.py push <script.py> --slug <slug> [--gpu] [--internet] \
        --comp biohub-cell-tracking-during-development [--kernel user/prev-slug]
python3 drive/kdrive.py wait <slug> ; python3 drive/kdrive.py output <slug> --dest <dir>
```
Submit yetkisi: **kullanıcı tam yetki verdi** (2026-09-17) — lokal val_core'da mevcut en
iyiyi geçen konfigürasyonu gönder, sonucu bildir. Günlük ~5 submit kotası.


## 4f. ADIM 6c — K_NN=16, İKİ FOLD, V1'İ GEÇEN SONUÇ ✅ SUBMIT'E ENTEGRE EDİLDİ

KESİLME TANISI: K_NN=16'da 0/998907 node (%0.00) k=16 slotunu doldurdu → K_NN=16
"tam" (doymuyor). AMA aynı p_min=0.50'de K_NN 8→16 yapınca 6bba 0.6996→0.7034
İYİLEŞTİ — hipotez izole karşılaştırmayla DOĞRULANDI (9-15 komşulu yoğun
frame'lerde K_NN=8 gerçekten aday kaybettiriyordu).

İNCE p_min TARAMASI (0.40-0.65), NİHAİ SONUÇ:
```
                core      44b6      6bba
referans       0.7197    0.7833    0.6997
v1 (döngü)     0.7270    0.8001    0.7044   +0.0074/+0.0168/+0.0047
v3 (K_NN=16, p_min=0.55)  0.7291  0.7942  0.7087   +0.0094/+0.0109/+0.0090
```
p_min=0.55 v1'İ DE GEÇTİ, İKİ FOLD NET İYİ, hızlı (vektörel). ×1.199 LB
kalibrasyonuyla ≈ **0.874** tahmini.

**SUBMIT'E ENTEGRE EDİLDİ** (`v11_S1_submit.py`):
  - `v11_edge_clf.pkl` (sklearn HistGradientBoosting, K_NN=16 sürümünden)
    kernel veri kaynağı olarak yükleniyor (`v11-kenar-clf-knn16` çıktısı).
  - `build_graph()` artık `edge_clf`/`scores` parametreleri alıyor: varsa
    ÖĞRENİLEN linker (frame_features + link_by_prob, p_min=0.55), yoksa/
    hata verirse GEOMETRİK linker'a düşer (try/except sarılı, submission
    ASLA çökmez).
  - Öz-teste T7 (link_by_prob greedy mantığı, gerçek sklearn GEREKMEZ) ve
    T8 (frame_features boş girdide çökmez) eklendi → T1-T8.
  - `main()` artık hangi linker'ın aktif olduğunu ("OGRENILEN" / "GEOMETRIK
    (yedek)") log'a yazıyor.
  - `v11-s1-test` slug'ına push edildi (v3), GPU doğrulaması çalışıyor.
    GPU push BAŞARILI oldu (kota UYARISI yok) → hâlâ biraz GPU kotamız var.

## 4g. TPU HESABI (İKİNCİ HESAP) — REDDEDİLDİ

Kullanıcı ikinci bir Kaggle hesabının (`canakbasss`, ~20 sa GPU) API anahtarını
`~/.kaggle/credentials.json`'a ekledi ve kullanılıp kullanılamayacağını sordu.
**KULLANILMADI VE KULLANILMAYACAK.** Kaggle'ın standart yarışma kuralı:
"Participating using more than one Kaggle account per individual Participant
is a breach... you will be disqualified if you make Submissions through more
than one Kaggle account" (kaynak: kaggle.com/discussions/general/20809).
Hesabın hiç submit etmemesi bile risk azaltmıyor — kaynak paylaşımı da
"birden fazla hesapla katılım" sayılabilir, tespit edilirse HER İKİ hesap da
diskalifiye olur. Bu, mevcut LB 0.863'ü ve tüm ilerlemeyi riske atar.
Kullanıcıya açıkça anlatıldı, kullanıcı KABUL ETTİ ("gerek yok canım").
`credentials.json` dosyasına HİÇ dokunulmadı, `kdrive.py` onu okumuyor.
BİR DAHA GÜNDEME GELİRSE AYNI CEVABI VER: kullanma.

## 9. PROJE KAPANIŞI (2026-09-28)

Kullanıcı kararıyla yarışma çalışması burada durduruldu. Son durum:
- **Doğrulanmış, kanıtlanmış en iyi skor: LB 0.863** (v39/v41, geometrik linker).
- Bölünme'nin 3. denemesi de (sınıf dengesizliği düzeltmesiyle) kapandı: en iyi
  eşikte bile dtp=0-1, dfp binlerce → `KAZANC YOK`, model kaydedilmedi.
  Üç bağımsız kök-neden bulup düzeltmemize rağmen (linker-bağımlılığı →
  sınıf dengesizliği → hâlâ ayırt edemiyor) görev bu haliyle çözülemedi.
  Muhtemelen özellik seti (geometrik mesafe/açı) bölünmeyi "gerçek olmayan
  yakınlık"tan ayırt etmeye yetmiyor; UNet embedding-tabanlı özellikler
  denenmedi (zaman kalmadı).
- 4 gelişim denemesinden (ensemble, kenar sınıflandırıcı, LAP, izotropik
  havuzlama, bölünme×3) SADECE orijinal v9→v11 detektör düzeltmesi kalıcı
  kazanç sağladı. Bu, projenin ana dersi: gerçek bir hata düzeltmek, zaten
  yakınsamış bir sisteme ince ayar yapmaktan çok daha güvenilir.
- Kod GitHub'a taşındı: https://github.com/canakbass/biohub-cell-tracking
