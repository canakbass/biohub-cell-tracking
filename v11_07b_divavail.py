# ADIM 7b - mevcut aday dokumlerinde (v3tr + v3) kac GT bolunmesi var?  (CPU, ~2 dk)
# Yeterliyse (~40+) bolunme kapisi GPU'SUZ kurulur; degilse bolunmece zengin
# dataset'ler icin ~1 sa GPU'luk dokum gerekir (kalan 3 saatin 1/3'u).
import sys, subprocess
from importlib.metadata import version as _pv
try: _pv("zarr")
except Exception: subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zarr"], check=False)
import os, glob, numpy as np, zarr
from collections import Counter
TRAIN = "/kaggle/input/competitions/biohub-cell-tracking-during-development/train"
def ndiv(n):
    g = zarr.open_group(os.path.join(TRAIN, n + ".geff"), mode="r")
    try:
        e = np.asarray(g["edges"]["ids"][:]).astype(np.int64).reshape(-1, 2)
    except Exception:
        return 0
    return sum(1 for v in Counter(e[:, 0].tolist()).values() if v >= 2) if len(e) else 0
for tag in ("v3tr", "v3"):
    names = sorted(os.path.basename(p)[len(f"cand_{tag}_"):-4]
                   for p in glob.glob(f"/kaggle/input/**/cand_{tag}_*.npz", recursive=True))
    per = {n: ndiv(n) for n in names}
    tot = sum(per.values())
    print(f"[{tag}] {len(names)} dataset, {tot} GT bolunme, "
          f"bolunmeli {sum(1 for v in per.values() if v)} dataset | "
          f"44b6 {sum(v for n,v in per.items() if n.startswith('44b6'))} / "
          f"6bba {sum(v for n,v in per.items() if n.startswith('6bba'))}")
print("### V11-ADIM7B-RAPOR")
