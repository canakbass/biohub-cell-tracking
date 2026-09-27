# =============================================================================
# V11 HATA ARAMASI 3 - BOLUNME ADAY-KAYBI DENETIMI (CPU, ~2-3 dk, sadece GT)
# =============================================================================
# Adim 8 (v11_08_divgate.py) SADECE 50/99 GT bolunmesini kullanilabilir pozitif
# ornege cevirebildi. SEBEP HIPOTEZI: aday uretimi, birincil (parent->c1)
# baglantiyi Adim 6c OGRENILEN linker'la kuruyor, ikinci kiz adayini SADECE
# o linker'in "eslesmemis" biraktigi node'lar arasinda ariyor. Bolunme anlari
# TAM OLARAK bu linker'in EN ZAYIF oldugu yer (tek-en-yakin-komsu mantigi
# dogasi geregi bozuluyor) -> kayip, gorevin ZORLUGUNDAN degil ADAY URETIMININ
# YANLIS BAGIMLILIGINDAN olabilir.
#
# BU DENETIM: hicbir linker/siniflandiriciya bagimli DEGIL. Sadece GT'den:
# her gercek bolunmede (ebeveyn + >=2 cocuk) iki cocuk da ebeveyne FIZIKSEL
# olarak yakin mi (basit K-en-yakin-komsu adayligi ile YAKALANABILIR mi)?
# Bu, "akilli/bagimsiz" bir aday uretimiyle teorik olarak ULASILABILECEK
# UST SINIRI verir - gercek detektor/eslesme hatalarini SAYMAZ (o ayri konu).
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
import os
import numpy as np
import zarr

COMP  = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
TRAIN = os.path.join(COMP, "train")
SCALE = (1.625, 0.40625, 0.40625)

SEARCH_RADII = [10.0, 15.0, 20.0, 25.0, 30.0]   # um


def read_gt(geff_path):
    g = zarr.open_group(str(geff_path), mode="r")

    def arr(p):
        n = g
        for k in p.split("/"):
            n = n[k]
        return np.asarray(n[:])

    def get(k):
        for p in (f"nodes/props/{k}/values", f"nodes/{k}"):
            try:
                return arr(p)
            except Exception:
                pass
        raise KeyError(k)

    t = get("t").astype(np.int64)
    z, y, x = (get(k).astype(np.float64) for k in ("z", "y", "x"))
    try:
        ids = arr("nodes/ids").astype(np.int64)
    except Exception:
        ids = np.arange(len(t), dtype=np.int64)
    try:
        e_raw = arr("edges/ids").astype(np.int64).reshape(-1, 2)
    except Exception:
        e_raw = np.zeros((0, 2), np.int64)
    if len(e_raw):
        o = np.argsort(ids)
        edges = o[np.searchsorted(ids[o], e_raw.ravel())].reshape(-1, 2)
    else:
        edges = e_raw
    return dict(t=t, z=z, y=y, x=x, edges=edges, n_nodes=len(t))


all_ds = sorted(d[:-5] for d in os.listdir(TRAIN) if d.endswith(".geff"))
print(f"{len(all_ds)} TRAIN dataset taraniyor (GT bolunme aday-kaybi analizi)...")

sc = np.asarray(SCALE)
n_div_total = 0
recoverable_at = {r: 0 for r in SEARCH_RADII}
farthest_child_dists = []
per_ds_divs = {}

for name in all_ds:
    try:
        gt = read_gt(os.path.join(TRAIN, name + ".geff"))
    except Exception as ex:
        print(f"  !!! {name}: OKUMA HATASI {type(ex).__name__}: {ex}")
        continue
    if len(gt["edges"]) == 0:
        continue
    out_deg = np.bincount(gt["edges"][:, 0], minlength=gt["n_nodes"])
    div_parents = np.where(out_deg >= 2)[0]
    if len(div_parents) == 0:
        continue
    P = np.stack([gt["z"], gt["y"], gt["x"]], 1) * sc
    child_map = {}
    for s_, t_ in gt["edges"].tolist():
        child_map.setdefault(s_, []).append(t_)
    n_here = 0
    for p in div_parents.tolist():
        children = child_map[p][:2]   # ilk iki cocuk (>2 nadir/veri hatasi olabilir)
        if len(children) < 2:
            continue
        d0 = float(np.linalg.norm(P[children[0]] - P[p]))
        d1 = float(np.linalg.norm(P[children[1]] - P[p]))
        worst = max(d0, d1)
        farthest_child_dists.append(worst)
        n_div_total += 1
        n_here += 1
        for r in SEARCH_RADII:
            if worst <= r:
                recoverable_at[r] += 1
    if n_here:
        per_ds_divs[name] = n_here

farthest_child_dists = np.array(farthest_child_dists)
print("\n" + "#" * 90)
print("### V11-DIV-UPPERBOUND-RAPOR-BASLANGIC")
print(f"toplam_dataset_taranan={len(all_ds)}  bolunme_iceren_dataset={len(per_ds_divs)}")
print(f"TOPLAM GERCEK BOLUNME={n_div_total}")
print(f"\nen-uzak-cocuk mesafesi (um): min={farthest_child_dists.min():.2f} "
      f"p25={np.percentile(farthest_child_dists,25):.2f} p50={np.percentile(farthest_child_dists,50):.2f} "
      f"p75={np.percentile(farthest_child_dists,75):.2f} p90={np.percentile(farthest_child_dists,90):.2f} "
      f"max={farthest_child_dists.max():.2f}")
print(f"\nSaf geometrik (linker-bagimsiz) K-en-yakin-komsu ile YAKALANABILIR "
      f"(iki cocuk da yaricap icinde):")
for r in SEARCH_RADII:
    n = recoverable_at[r]
    print(f"  yaricap={r:5.1f}um  yakalanabilir={n:4d}/{n_div_total} (%{100*n/max(1,n_div_total):.1f})")

print(f"\nADIM 8'IN GERCEKTE YAKALADIGI: 50/99 (v3div, 39 dataset alt-kumesi, "
      f"K_NN=16 linker'a BAGIMLI aday uretimiyle)")
r20 = recoverable_at[20.0]
print(f"BU DENETIMIN 20um'deki UST SINIRI (TUM {len(all_ds)} dataset, linker-BAGIMSIZ): "
      f"{r20}/{n_div_total} (%{100*r20/max(1,n_div_total):.1f})")
if r20 / max(1, n_div_total) > 0.85:
    print(">>> HIPOTEZ DOGRULANDI: bolunmelerin BUYUK COGUNLUGU saf geometrik olarak"
          " YAKALANABILIR -> Adim 8'deki 50/99 kaybi GOREVIN ZORLUGUNDAN degil,"
          " linker-bagimli aday uretiminin YAPISAL KISITINDAN kaynaklaniyor."
          " Linker-bagimsiz (dogrudan K-en-yakin-komsu) aday uretimiyle 3. bir"
          " deneme ANLAMLI OLABILIR.")
else:
    print(">>> HIPOTEZ COGUNLUKLA DOGRULANMADI: bolunmelerin onemli bir kismi zaten"
          " fiziksel olarak UZAK/ZOR -> kayip byuk olcude gorevin dogasindan, linker"
          " bagimliligindan degil. 3. deneme icin daha temkinli olunmali.")
print("### V11-DIV-UPPERBOUND-RAPOR-BITIS")
print("#" * 90)
