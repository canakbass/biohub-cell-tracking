# =============================================================================
# ADIM 7a - GT BOLUNME SAYIMI  (CPU, internet ACIK, ~5 dk)
# =============================================================================
# SORU: ogrenilen bolunme kapisi OLCULEBILIR mi?
#   val_core'da sadece 5 GT bolunmesi vardi -> divJ=TP/(TP+FP+FN) tek TP ile
#   buyuk oynar. Kapiyi guvenilir olcmek icin ~50+ bolunmelik degerlendirme seti
#   gerek. Once 199 dataset'te kac bolunme var ve nerede, onu say.
# =============================================================================
import sys, subprocess
from importlib.metadata import version as _pv
try: _pv("zarr")
except Exception: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zarr"], check=False)
import os, numpy as np, zarr
from collections import Counter

TRAIN = "/kaggle/input/competitions/biohub-cell-tracking-during-development/train"
VAL_CORE = {'44b6_341df25f','44b6_3bb3690f','44b6_c771cb04','44b6_8f9ecab4',
            '6bba_2540cd90','6bba_3a1849c2','6bba_67ebd073','6bba_57b7cc1e'}
VAL_PUBLIC = {'44b6_0113de3b','44b6_0b24845f','6bba_05b6850b','6bba_05db0fb1'}

def edges_of(g):
    def arr(p):
        n = g
        for k in p.split("/"): n = n[k]
        return np.asarray(n[:])
    try: ids = arr("nodes/ids").astype(np.int64)
    except Exception: return None, 0
    try: e = arr("edges/ids").astype(np.int64).reshape(-1, 2)
    except Exception: e = np.zeros((0, 2), np.int64)
    return e, len(ids)

rows = []
for n in sorted(d[:-5] for d in os.listdir(TRAIN) if d.endswith(".geff")):
    try:
        g = zarr.open_group(os.path.join(TRAIN, n + ".geff"), mode="r")
        e, nn = edges_of(g)
        if e is None: continue
        outd = Counter(e[:, 0].tolist()) if len(e) else Counter()
        ndiv = sum(1 for v in outd.values() if v >= 2)
        rows.append((n, n[:4], nn, len(e), ndiv))
    except Exception as ex:
        print("ATLA", n, ex)

tot = sum(r[4] for r in rows)
print(f"dataset={len(rows)}  toplam GT bolunme={tot}  toplam GT kenar={sum(r[3] for r in rows)}")
print(f"  bolunme / 1000 kenar = {1000*tot/max(1,sum(r[3] for r in rows)):.2f}")
for col in ("44b6", "6bba"):
    rr = [r for r in rows if r[1] == col]
    print(f"  {col}: {len(rr)} dataset, {sum(r[4] for r in rr)} bolunme, "
          f"bolunmesi olan dataset {sum(1 for r in rr if r[4] > 0)}")
def grp(names): return sum(r[4] for r in rows if r[0] in names)
print(f"  val_core bolunme={grp(VAL_CORE)}  val_public bolunme={grp(VAL_PUBLIC)}")
rest = [r for r in rows if r[0] not in VAL_CORE and r[0] not in VAL_PUBLIC]
print(f"  egitim havuzu (val haric) bolunme={sum(r[4] for r in rest)} "
      f"({sum(1 for r in rest if r[4]>0)} dataset'te)")
dist = Counter(r[4] for r in rows)
print(f"  dataset basina bolunme dagilimi: {dict(sorted(dist.items()))}")
print("\n  en cok bolunmesi olan 15 dataset:")
for r in sorted(rows, key=lambda r: -r[4])[:15]:
    tag = "VAL_CORE" if r[0] in VAL_CORE else ("VAL_PUB" if r[0] in VAL_PUBLIC else "")
    print(f"    {r[0]:22s} {r[4]:3d} bolunme  {r[3]:5d} kenar  {tag}")
print("\n### V11-ADIM7A-RAPOR")
print(f"toplam_bolunme={tot} val_core={grp(VAL_CORE)} egitim_havuzu={sum(r[4] for r in rest)} "
      f"bolunmeli_dataset={sum(1 for r in rows if r[4]>0)}/{len(rows)}")
