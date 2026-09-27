# =============================================================================
# V11 ADIM 2.6 - dataset-basina n* ONGORULEBILIR MI?  (CPU, internet ACIK, ~2 dk)
# =============================================================================
# ADIM 2.5 BULGUSU: dataset-basina-k oracle kazanci +0.0269 (proxy'de).
#   Ongorucu ozellik rec@1.0 ile rho=-0.682 AMA rec@1.0 GT GEREKTIRIYOR,
#   test setinde GT YOK. Bu script GT'SIZ vekil ariyor.
#
# CERCEVE DUZELTMESI: proxy k=1.3 diyor, GERCEK metrik (Adim 3) 0.95.
#   Proxy'nin SEVIYESI yanlis ama GORELI degisimi dogru olabilir.
#   -> mutlak k degil, k_i/k_global GORELI SAPMASINI ongoruyoruz;
#      seviyeyi gercek metrikten (0.95) aliyoruz.
#
# Girdi: v11-est-fit kernel cikti'si (v11_est_feats.npy)
# =============================================================================
import sys, subprocess, glob, os, math, json
import numpy as np

THR_FEAT = [0.025, 0.05, 0.10, 0.20, 0.35]
K_GRID   = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0]
# Adim 2.5, leave-one-colony-out ile secilen kestirici
EST_THR, EST_A, EST_B = 0.35, 0.229320, 1.183500
N_STAR_REAL = 0.95      # Adim 3, GERCEK metrik taramasindan

f = None
for c in glob.glob("/kaggle/input/**/v11_est_feats.npy", recursive=True):
    f = c; break
assert f, "v11_est_feats.npy bulunamadi - v11-est-fit cikti'sini input olarak ekle"
rows = list(np.load(f, allow_pickle=True))
print(f"yuklendi: {len(rows)} dataset  ({f})")
for r in rows[:1]:
    print(f"  ornek anahtarlar: {sorted(r.keys())}")

col = lambda k: np.array([r[k] for r in rows], float)
colony = np.array([r["colony"] for r in rows])
best_k = col("best_k")
est_true = col("est")
T = col("T")


def np_at(r, th):
    return max(1.0, float(r["feats"][th]))


def est_hat(r):
    """GT'siz est tahmini (test aninda yapilacak sey)"""
    return EST_A * (np_at(r, EST_THR) ** EST_B)


def thr_at_n(r, n_target):
    """Np(thr) = n_target * est_hat  olan esigi log-log interpolasyonla bul.
    ANLAMI: detektor est_hat tanesinden EMINSE bu esik YUKSEK, bocalıyorsa DUSUK.
    Tamamen GT'siz."""
    e = est_hat(r)
    target = max(1.0, n_target * e)
    xs = [math.log(np_at(r, t)) for t in THR_FEAT]     # azalan
    ys = [math.log(t) for t in THR_FEAT]
    lt = math.log(target)
    if lt >= xs[0]:
        return THR_FEAT[0]
    if lt <= xs[-1]:
        return THR_FEAT[-1]
    for i in range(len(xs) - 1):
        if xs[i] >= lt >= xs[i + 1]:
            w = (lt - xs[i]) / (xs[i + 1] - xs[i] + 1e-12)
            return math.exp(ys[i] + w * (ys[i + 1] - ys[i]))
    return THR_FEAT[-1]


FEATS = {
    "thr_at_n1 (GT'siz kesim skoru)": lambda r: math.log(max(1e-6, thr_at_n(r, 1.0))),
    "Np.35/Np.025 (dagilim dikligi)": lambda r: math.log(np_at(r, 0.35) / np_at(r, 0.025)),
    "Np.20/Np.05":                    lambda r: math.log(np_at(r, 0.20) / np_at(r, 0.05)),
    "n_acc/est_hat":                  lambda r: math.log(max(1.0, r["n_acc"]) / max(1.0, est_hat(r))),
    "log est_hat":                    lambda r: math.log(max(1.0, est_hat(r))),
    "est_hat/T (yogunluk)":           lambda r: math.log(max(1e-6, est_hat(r) / max(1, r["T"]))),
    "rec@1.0 (GT GEREKTIRIR)":        lambda r: r["rec_k"][1.0],
}

def rho(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float)))
    rb = np.argsort(np.argsort(np.asarray(b, float)))
    return float(np.corrcoef(ra, rb)[0, 1])

print("\n" + "=" * 96)
print("A) best_k ile SIRA KORELASYONU  (GT'siz ozellikler + referans olarak rec@1.0)")
print("=" * 96)
X = {}
for name, fn in FEATS.items():
    v = np.array([fn(r) for r in rows], float)
    X[name] = v
    print(f"  {name:34s} rho={rho(v, best_k):+.3f}")

# --- est_hat kestiricisinin kendisi ne kadar iyi (kontrol) -------------------
eh = np.array([est_hat(r) for r in rows])
err = np.abs(np.log(eh / est_true))
print(f"\n  [kontrol] est_hat hatasi: medyan %{100*(math.exp(np.median(err))-1):.1f} "
      f"p90 %{100*(math.exp(np.percentile(err,90))-1):.1f} "
      f"max %{100*(math.exp(err.max())-1):.1f}")

# =============================================================================
# B) GORELI k ONGORUSU + leave-one-COLONY-out
# =============================================================================
def lstsq(A, y):
    k = len(A[0])
    M = [[sum(A[i][a]*A[i][b] for i in range(len(A))) for b in range(k)] +
         [sum(A[i][a]*y[i] for i in range(len(A)))] for a in range(k)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(M[r][c])); M[c], M[p] = M[p], M[c]
        if abs(M[c][c]) < 1e-12: return None
        for r in range(k):
            if r == c: continue
            fq = M[r][c]/M[c][c]
            for j in range(c, k+1): M[r][j] -= fq*M[c][j]
    return [M[c][k]/M[c][c] for c in range(k)]

proxy_of = lambda r, k: r["rec_k"][k]**2 * max(0.0, 1 - 0.1*(k-1))
k_fixed = max(K_GRID, key=lambda k: np.mean([proxy_of(r, k) for r in rows]))
px_fixed = float(np.mean([proxy_of(r, k_fixed) for r in rows]))
px_oracle = float(np.mean([max(proxy_of(r, k) for k in K_GRID) for r in rows]))
print("\n" + "=" * 96)
print(f"B) GORELI k ONGORUSU   (sabit k={k_fixed}: {px_fixed:.4f} | oracle: {px_oracle:.4f})")
print("=" * 96)

def snap(k):
    return min(K_GRID, key=lambda g: abs(g - k))

def try_feats(names, train_col=None, tag=""):
    tr = [i for i in range(len(rows)) if (train_col is None or colony[i] == train_col)]
    A = [[1.0] + [X[n][i] for n in names] for i in tr]
    y = [math.log(best_k[i] / k_fixed) for i in tr]        # GORELI log-sapma
    w = lstsq(A, y)
    if w is None: return None
    def pred_k(i):
        z = w[0] + sum(wj * X[n][i] for wj, n in zip(w[1:], names))
        return snap(float(np.clip(k_fixed * math.exp(z), min(K_GRID), max(K_GRID))))
    held = [i for i in range(len(rows)) if train_col is not None and colony[i] != train_col]
    px_all = float(np.mean([proxy_of(rows[i], pred_k(i)) for i in range(len(rows))]))
    s = f"  {tag:46s} TUM {px_all:.4f} ({px_all-px_fixed:+.4f})"
    if held:
        px_h = float(np.mean([proxy_of(rows[i], pred_k(i)) for i in held]))
        px_hf = float(np.mean([proxy_of(rows[i], k_fixed) for i in held]))
        s += f" | TUTULAN({train_col}) {px_h:.4f} ({px_h-px_hf:+.4f})"
    print(s)
    return px_all

GTFREE = [n for n in FEATS if "GEREKTIRIR" not in n]
print("\n  -- tek ozellik (tum veriyle fit) --")
for n in GTFREE:
    try_feats([n], tag=n)
print("\n  -- rec@1.0 ile (UST SINIR, test aninda KULLANILAMAZ) --")
try_feats(["rec@1.0 (GT GEREKTIRIR)"], tag="rec@1.0 (ust sinir)")

print("\n  -- en iyi 2 GT'siz ozellik + leave-one-colony-out --")
order = sorted(GTFREE, key=lambda n: -abs(rho(X[n], best_k)))
for combo in ([order[0]], order[:2], order[:3]):
    lbl = " + ".join(c.split(" ")[0] for c in combo)
    try_feats(combo, tag=lbl)
    try_feats(combo, train_col="44b6", tag=lbl)
    try_feats(combo, train_col="6bba", tag=lbl)

print("\n" + "#" * 96)
print("### V11-ADIM26-RAPOR-BASLANGIC")
print(f"n_dataset={len(rows)} k_fixed={k_fixed} px_fixed={px_fixed:.4f} px_oracle={px_oracle:.4f}")
print(f"est_hat_hata_medyan=%{100*(math.exp(np.median(err))-1):.1f}")
for n in FEATS:
    print(f"rho[{n}]={rho(X[n], best_k):+.3f}")
print("### V11-ADIM26-RAPOR-BITIS")
print("#" * 96)
