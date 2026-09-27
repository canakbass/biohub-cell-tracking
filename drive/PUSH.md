# Push komutları (kimlik bilgisi geldiğinde)

```bash
cd /home/can/Projeler/biohub_cell_tracking/notebooks/v11
K=drive/kdrive.py

# 0) kimlik testi
python3 $K whoami

# 1) ADIM S1 - SUBMIT (GPU, internet KAPALI, ~5-6 saat)
#    DIKKAT: 'submit' kullanicinin gercek submit notebook'u (LB gecmisi orada).
#    Once ayri slug'da dogrula, sonra submit'e push et.
python3 $K push v11_S1_submit.py --slug v11-s1-test --title "v11 S1 test" --gpu \
    --comp biohub-cell-tracking-during-development \
    --model hcanakbas/biohub-v9-03-sota/pyTorch/default/1
python3 $K wait v11-s1-test --poll 120
python3 $K log v11-s1-test --tail 80          # OZ-TEST gecti mi? projeksiyon ne?
python3 $K output v11-s1-test --dest out/s1

# 2) ADIM 4 - RESMI SKORLAYICI (CPU, internet ACIK, ~25 dk)
python3 $K push v11_04_official.py --slug v11-official --title "v11 resmi metrik" --internet \
    --comp biohub-cell-tracking-during-development --kernel hcanakbas/v11-cand
python3 $K wait v11-official --poll 60 && python3 $K log v11-official --tail 120

# 3) Dogrulama gecerse: gercek submit notebook'una push
python3 $K push v11_S1_submit.py --slug submit --title "submit" --gpu \
    --comp biohub-cell-tracking-during-development \
    --model hcanakbas/biohub-v9-03-sota/pyTorch/default/1
# -> sonra Kaggle UI'da "Submit to Competition" (API notebook submit'i desteklemiyorsa)

# LB skorunu oku
python3 $K status submit
```

## Dogrulanmasi gerekenler (ilk push'ta)
- `--model` yol formati: `hcanakbas/biohub-v9-03-sota/pyTorch/default/1` mi,
  yoksa `pytorch` mu (kucuk harf)? Push hata verirse digerini dene.
- `--kernel` icin Adim 1b'nin gercek slug'i ne? `kdrive.py list` ile bak.
- API'nin notebook'u yarismaya gonderebilip gonderemedigi.

## CANLI DERSLER (2026-09-17, gercek push'lardan)

1. **Kaggle slug'i BASLIKTAN uretir**, `--slug` alanindan degil.
   -> HER ZAMAN `--title` ile `--slug`'i AYNI ver. kdrive artik farkliysa uyariyor.

2. **`zarr` Kaggle imajinda ON YUKLU DEGIL.** Analiz kernel'leri `--internet` ISTER.
   -> `v11_00_probe / 01_candidates / 02_sweep / 03_linkdiag / 04_official / S1_score`: `--internet`
   -> `v11_S1_submit`: internet YOK (manuel zarr okuyucu; submit'te zaten kapali olacak)

3. **`!pip` IPython sihri, script kernel'de SOZDIZIMI HATASI.**
   -> Hepsinde `subprocess` bootstrap var artik. Yeni script yazarken `!` KULLANMA.

4. **Model veri kaynagi formati DOGRULANDI:**
   `--model hcanakbas/biohub-v9-03-sota/pyTorch/default/1`

5. **Kernel ciktisini zincirleme:** `--kernel hcanakbas/<slug>` -> `/kaggle/input/<slug>/...`
   DIKKAT: `model` notebook'unun ciktisi EN SON versiyonun ciktisidir. Aday dokumu
   icin ayri slug (`v11-aday-dokumu`) kullaniyoruz ki baska is uzerine yazmasin.

6. **OLCULEN HIZ:** model forward **0.66 s/frame** (100 frame = 66 s).
   199 dataset -> ~3.7 saat. 9 saatlik GPU limitine RAHAT siger (tahminim 0.80 idi).
