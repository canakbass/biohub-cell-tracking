# =============================================================================
# V11 HATA ARAMASI 2 - est_nodes KALIBRASYON KAPSAMI DENETIMI (CPU, ~2 dk)
# =============================================================================
# est_hat = 0.617044 * Np(thr=0.025)^1.064065 SADECE 12 val dataset'inde
# (VAL_CORE+VAL_PUBLIC) fit edildi. Gizli test ~199 dataset, embriyo-ayrik.
# Eger 12'lik kalibrasyon seti, TUM 199 TRAIN dataset'inin yogunluk (est_nodes)
# dagilimini TEMSIL ETMIYORSA (ornegin sadece orta-yogunluk bolgeyi kapsiyorsa),
# gizli testte YOGUNLUGU asiri farkli embriyolarda kestirici EKSTRAPOLASYON
# yapmak zorunda kalir -> sistematik hata riski. Bu SADECE metadata (.geff
# attrs) okuyor, piksel verisine dokunmuyor -> ucuz, hizli.
# =============================================================================
import sys, subprocess

def _ensure(mod, spec=None):
    try:
        __import__(mod); return True
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", spec or mod], check=False)
        try:
            __import__(mod); return True
        except ImportError:
            print(f"[bootstrap] {mod} KURULAMADI"); return False

_ensure("zarr")
import os, json
import numpy as np
import zarr

COMP  = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
TRAIN = os.path.join(COMP, "train")

VAL_CORE = ['44b6_341df25f', '44b6_3bb3690f', '44b6_c771cb04', '44b6_8f9ecab4',
            '6bba_2540cd90', '6bba_3a1849c2', '6bba_67ebd073', '6bba_57b7cc1e']
VAL_PUBLIC = ['44b6_0113de3b', '44b6_0b24845f', '6bba_05b6850b', '6bba_05db0fb1']
CALIB = set(VAL_CORE + VAL_PUBLIC)


def read_est(geff_path):
    g = zarr.open_group(str(geff_path), mode="r")
    meta = dict(g.attrs).get("geff", {}) or {}
    est = (meta.get("extra") or {}).get("estimated_number_of_nodes")
    return float(est) if est is not None else float("nan")


def read_shape_T(zarr_dir):
    v2 = os.path.join(zarr_dir, "0", ".zarray")
    if os.path.exists(v2):
        shape = json.load(open(v2))["shape"]
    else:
        shape = json.load(open(os.path.join(zarr_dir, "0", "zarr.json")))["shape"]
    return int(shape[0])


all_ds = sorted(d[:-5] for d in os.listdir(TRAIN) if d.endswith(".zarr"))
print(f"{len(all_ds)} TRAIN dataset taraniyor (est_nodes, hucre yogunlugu = est/T)...")

rows = []
for name in all_ds:
    try:
        est = read_est(os.path.join(TRAIN, name + ".geff"))
        T = read_shape_T(os.path.join(TRAIN, name + ".zarr"))
        dens = est / T if T > 0 else float("nan")
        rows.append((name, est, T, dens, name in CALIB))
    except Exception as ex:
        print(f"  !!! {name}: OKUMA HATASI {type(ex).__name__}: {ex}")

dens_all = np.array([r[3] for r in rows if r[3] == r[3]])
dens_calib = np.array([r[3] for r in rows if r[4] and r[3] == r[3]])
dens_rest = np.array([r[3] for r in rows if not r[4] and r[3] == r[3]])

print("\n" + "#" * 90)
print("### V11-ESTNODES-AUDIT-RAPOR-BASLANGIC")
print(f"toplam_dataset={len(rows)}")
print(f"TUM 199 (hucre/frame): min={dens_all.min():.1f} p5={np.percentile(dens_all,5):.1f} "
      f"p25={np.percentile(dens_all,25):.1f} p50={np.percentile(dens_all,50):.1f} "
      f"p75={np.percentile(dens_all,75):.1f} p95={np.percentile(dens_all,95):.1f} "
      f"max={dens_all.max():.1f}")
print(f"KALIBRASYON(12) (hucre/frame): min={dens_calib.min():.1f} p50={np.median(dens_calib):.1f} "
      f"max={dens_calib.max():.1f}")
print(f"DIGER(187) (hucre/frame): min={dens_rest.min():.1f} p50={np.median(dens_rest):.1f} "
      f"max={dens_rest.max():.1f}")

out_of_range = dens_rest[(dens_rest < dens_calib.min()) | (dens_rest > dens_calib.max())]
pct_oor = 100.0 * len(out_of_range) / max(1, len(dens_rest))
print(f"\nKALIBRASYON ARALIGI DISINDA KALAN (diger 187 icinde): {len(out_of_range)}/{len(dens_rest)} "
      f"(%{pct_oor:.1f})")
if pct_oor > 15:
    print(f">>> RISK DOGRULANDI: 12'lik kalibrasyon seti TUM populasyonun yogunluk "
          f"araligini TEMSIL ETMIYOR -> est_hat gizli testte SIK SIK ekstrapolasyon "
          f"yapmak zorunda kalabilir.")
else:
    print(f">>> Kalibrasyon araligi populasyonun buyuk kismini kapsiyor, ekstrapolasyon "
          f"riski SINIRLI (train icin; gizli test embriyo-ayrik oldugundan yine de "
          f"kesin garanti degil).")

print("\n--- en yogun 10 ve en seyrek 10 dataset (kalibrasyonda mi?) ---")
srt = sorted(rows, key=lambda r: r[3])
for r in srt[:10]:
    print(f"  SEYREK {r[0]:22s} dens={r[3]:7.1f} kalibrasyonda={'EVET' if r[4] else 'hayir'}")
for r in srt[-10:]:
    print(f"  YOGUN  {r[0]:22s} dens={r[3]:7.1f} kalibrasyonda={'EVET' if r[4] else 'hayir'}")
print("### V11-ESTNODES-AUDIT-RAPOR-BITIS")
print("#" * 90)
