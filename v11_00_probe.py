# =============================================================================
# V11 ADIM 0 - TEMEL DOGRULAMA  (CPU, GPU KAPALI, ~5 dk)
# =============================================================================
# Amac: hicbir sey egitmeden once su 5 seyi KANITLAMAK
#   1. GEFF koordinatlari voxel mi fiziksel mi?        -> metrigin dogrulugu
#   2. Metrik implementasyonu dogru mu?                -> GT'yi GT'ye karsi skorla
#   3. "Annotate edilmemis kenarlar bedava" dogru mu?  -> gurultu enjeksiyonu
#   4. test .zarr == train .zarr ?                     -> public GT elimizde mi
#   5. Hangi datasetlerle dogrulayacagiz?              -> val seti sec + kaydet
# Cikti: /kaggle/working/v11_val_sets.json + ekrana RAPOR blogu
# =============================================================================

import sys, subprocess

def _ensure(mod, spec=None):
    """Kaggle 'script' kernel'inde !pip sihri YOK -> subprocess ile kur."""
    try:
        __import__(mod)
        return True
    except ImportError:
        print(f"[bootstrap] {spec or mod} kuruluyor...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", spec or mod], check=False)
        try:
            __import__(mod)
            return True
        except ImportError:
            print(f"[bootstrap] {mod} KURULAMADI"); return False

_ensure("zarr")

import os, json, glob, hashlib
import numpy as np

COMP  = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
TRAIN = os.path.join(COMP, "train")
TEST  = os.path.join(COMP, "test")
WORK  = "/kaggle/working"

SCALE         = (1.625, 0.40625, 0.40625)   # um/voxel (z,y,x)
MAX_MATCH_UM  = 7.0
ALPHA         = 0.1      # node-sayisi ceza katsayisi
DIV_W         = 0.1      # division agirligi

N_VAL         = 8        # cekirdek val seti buyuklugu
MIN_GT_EDGES  = 200      # "GT'si zengin" esigi

rng = np.random.default_rng(0)


# =============================================================================
# GEFF OKUMA
# =============================================================================
def read_gt(geff_path, zarr_shape=None):
    """GEFF -> dict(t,z,y,x,edges,est_nodes,n_nodes,coord_space)

    z,y,x her zaman VOXEL cinsinden dondurulur (metrik bunu bekliyor).
    coord_space: dosyada nasil yazildigini soyler ('voxel' | 'physical').
    """
    import zarr
    g = zarr.open_group(str(geff_path), mode="r")
    meta = dict(g.attrs).get("geff", {}) or {}
    est  = (meta.get("extra") or {}).get("estimated_number_of_nodes")

    def arr(path):
        node = g
        for p in path.split("/"):
            node = node[p]
        return np.asarray(node[:])

    def get(k):
        for p in (f"nodes/props/{k}/values", f"nodes/{k}"):
            try:
                return arr(p)
            except Exception:
                pass
        raise KeyError(k)

    t = get("t").astype(np.int64)
    z = get("z").astype(np.float64)
    y = get("y").astype(np.float64)
    x = get("x").astype(np.float64)

    # --- node id -> satir indeksi eslemesi (edges/ids bunlara referans veriyor)
    try:
        ids = arr("nodes/ids").astype(np.int64)
    except Exception:
        ids = np.arange(len(t), dtype=np.int64)
    try:
        e_raw = arr("edges/ids").astype(np.int64).reshape(-1, 2)
    except Exception:
        e_raw = np.zeros((0, 2), np.int64)
    if len(e_raw):
        order = np.argsort(ids)
        pos   = np.searchsorted(ids[order], e_raw.ravel())
        edges = order[pos].reshape(-1, 2)
    else:
        edges = e_raw

    # --- KOORDINAT UZAYI TESPITI (kritik) -----------------------------------
    coord_space = "voxel"
    if zarr_shape is None or len(z) == 0:
        coord_space = "belirsiz(shape-yok)"
    else:
        Z, Y, X = zarr_shape[1], zarr_shape[2], zarr_shape[3]
        # fiziksel yazilmissa z en fazla Z*scale_z (~104 um) olur, Z (~64) degil
        if z.max() > Z * 1.10 or y.max() > Y * 1.10 or x.max() > X * 1.10:
            coord_space = "physical"
            z = z / SCALE[0]
            y = y / SCALE[1]
            x = x / SCALE[2]

    return dict(t=t, z=z, y=y, x=x, edges=edges,
                est_nodes=float(est) if est is not None else float("nan"),
                n_nodes=len(t), coord_space=coord_space)


def zarr_shape_of(zarr_dir):
    for cand in (os.path.join(zarr_dir, "0", ".zarray"),
                 os.path.join(zarr_dir, "0", "zarr.json")):
        if os.path.exists(cand):
            return tuple(json.load(open(cand))["shape"])
    return None


# =============================================================================
# RESMI METRIK  (royerlab/kaggle-cell-tracking-competition ile birebir)
#   J     = TP / (GT_edges + valid_pred_edges - TP)
#   adj   = max(0, J * (1 - 0.1 * (N_pred - N_total)/N_total))
#   score = agirlikli_ortalama(adj) + 0.1 * micro(div_J)
#   valid_pred_edges = (kaynak, out-degree>0 olan GT node'una eslesti) OR
#                      (hedef,  in-degree>0  olan GT node'una eslesti)
# =============================================================================
def match_nodes(pred_nodes, gt, scale=SCALE, max_dist=MAX_MATCH_UM):
    """Frame ici optimal bipartite eslestirme -> matched[i] = GT satiri veya -1"""
    from scipy.optimize import linear_sum_assignment
    sc = np.asarray(scale)
    matched = np.full(len(pred_nodes), -1, np.int64)
    if len(pred_nodes) == 0 or gt["n_nodes"] == 0:
        return matched
    gt_t = gt["t"]
    gt_P = np.stack([gt["z"], gt["y"], gt["x"]], 1) * sc
    p_t  = pred_nodes[:, 0].astype(np.int64)
    p_P  = pred_nodes[:, 1:] * sc
    for tv in np.unique(gt_t):
        gi = np.where(gt_t == tv)[0]
        pi = np.where(p_t == tv)[0]
        if len(gi) == 0 or len(pi) == 0:
            continue
        D = np.linalg.norm(p_P[pi][:, None, :] - gt_P[gi][None, :, :], axis=2)
        BIG = 1e6
        C = np.where(D <= max_dist, D, BIG)
        if not np.any(C < BIG):
            continue
        r, c = linear_sum_assignment(C)
        ok = C[r, c] < BIG
        matched[pi[r[ok]]] = gi[c[ok]]
    return matched


def canonical_edges(pred_nodes, pred_edges, matched):
    """Resmi kenar filtreleri: dt==1 / tekille / merge-dedup / out-degree<=2"""
    if len(pred_edges) == 0:
        return pred_edges.reshape(-1, 2)
    e  = np.asarray(pred_edges, np.int64).reshape(-1, 2)
    ts = pred_nodes[e[:, 0], 0].astype(np.int64)
    tt = pred_nodes[e[:, 1], 0].astype(np.int64)
    e  = e[(tt - ts) == 1]
    if len(e) == 0:
        return e
    _, first = np.unique(e, axis=0, return_index=True)
    e = e[np.sort(first)]

    ms, mt = matched[e[:, 0]], matched[e[:, 1]]
    both = (ms >= 0) & (mt >= 0)
    if both.any():
        keep, seen = np.ones(len(e), bool), set()
        span = int(matched.max()) + 2
        for i in range(len(e)):
            if not both[i]:
                continue
            k = int(ms[i]) * span + int(mt[i])
            if k in seen:
                keep[i] = False
            else:
                seen.add(k)
        e = e[keep]

    e   = e[np.lexsort((np.arange(len(e)), e[:, 0]))]
    src = e[:, 0]
    grp = np.searchsorted(src, src, side="left")
    return e[(np.arange(len(e)) - grp) <= 1]


def division_counts(e, matched, gt):
    """Yaklasik division sayimi (resmi versiyon daha toleransli -> bu alt sinir).
    FP SADECE eslesmis + GT'de cocugu olan fork'lardan sayilir."""
    gt_edges = gt["edges"]
    if len(gt_edges) == 0:
        return 0, 0, 0
    gt_children = {}
    for s, t in gt_edges.tolist():
        gt_children.setdefault(s, []).append(t)
    gt_div = {k: v for k, v in gt_children.items() if len(v) >= 2}

    pred_children = {}
    for s, t in (e.tolist() if len(e) else []):
        pred_children.setdefault(s, []).append(t)
    forks = {k: v for k, v in pred_children.items() if len(v) >= 2}

    tp, fp, hit = 0, 0, set()
    for p, ch in forks.items():
        mp = int(matched[p])
        if mp < 0:
            continue                      # eslesmemis fork -> BEDAVA (FP degil)
        if mp in gt_div:
            want = set(gt_div[mp])
            got  = {int(matched[c]) for c in ch if matched[c] >= 0}
            if len(want & got) >= 2:
                tp += 1
                hit.add(mp)
            else:
                fp += 1
        elif mp in gt_children:           # eslesti ve GT'de cocugu var -> FP
            fp += 1
    return tp, fp, len(gt_div) - len(hit)


def evaluate_dataset(pred_nodes, pred_edges, gt, n_total=None, scale=SCALE):
    pred_nodes = np.asarray(pred_nodes, np.float64).reshape(-1, 4)
    matched = match_nodes(pred_nodes, gt, scale)
    e = canonical_edges(pred_nodes, pred_edges, matched)

    gt_edges, n_gt_nodes = gt["edges"], gt["n_nodes"]
    out_deg = np.zeros(n_gt_nodes, np.int64)
    in_deg  = np.zeros(n_gt_nodes, np.int64)
    if len(gt_edges):
        np.add.at(out_deg, gt_edges[:, 0], 1)
        np.add.at(in_deg,  gt_edges[:, 1], 1)
    gt_edge_set = set(map(tuple, gt_edges.tolist())) if len(gt_edges) else set()

    tp = valid = 0
    if len(e):
        ms, mt = matched[e[:, 0]], matched[e[:, 1]]
        out_valid = (ms >= 0) & (out_deg[np.clip(ms, 0, None)] > 0)
        in_valid  = (mt >= 0) & (in_deg[np.clip(mt, 0, None)] > 0)
        valid = int((out_valid | in_valid).sum())
        for i in np.where((ms >= 0) & (mt >= 0))[0]:
            if (int(ms[i]), int(mt[i])) in gt_edge_set:
                tp += 1

    edge_tp, edge_fp, edge_fn = tp, max(0, valid - tp), len(gt_edges) - tp
    n_matched = len(np.unique(matched[matched >= 0]))
    node_recall = n_matched / n_gt_nodes if n_gt_nodes else float("nan")

    dtp, dfp, dfn = division_counts(e, matched, gt)

    num_pred = len(pred_nodes)
    if n_total is None:
        n_total = gt.get("est_nodes", float("nan"))
    ratio = (num_pred - n_total) / n_total if (n_total and n_total > 0) else float("nan")
    denom = edge_tp + edge_fp + edge_fn
    J   = edge_tp / denom if denom > 0 else float("nan")
    adj = max(0.0, J * (1 - ALPHA * ratio)) if (J == J and ratio == ratio) else float("nan")

    return dict(edge_tp=edge_tp, edge_fp=edge_fp, edge_fn=edge_fn,
                division_tp=dtp, division_fp=dfp, division_fn=dfn,
                num_pred_nodes=num_pred, n_total=n_total,
                node_recall=node_recall, total_node_ratio=ratio,
                edge_jaccard=J, adj_edge_jaccard=adj, n_edges_kept=len(e))


def summarise(rows):
    """Resmi toplama: adj -> (tp+fp+fn) agirlikli; div -> mikro."""
    v = [r for r in rows if r["edge_jaccard"] == r["edge_jaccard"]]
    if not v:
        return dict(score=float("nan"))
    w  = np.array([r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in v], float)
    aj = np.array([r["adj_edge_jaccard"] for r in v], float)
    ok = aj == aj
    adj = float((w[ok] * aj[ok]).sum() / w[ok].sum()) if ok.any() and w[ok].sum() > 0 else float("nan")
    ej  = float(sum(r["edge_tp"] for r in v) /
                max(1, sum(r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in v)))
    dtp = sum(r["division_tp"] for r in v)
    dfp = sum(r["division_fp"] for r in v)
    dfn = sum(r["division_fn"] for r in v)
    dj  = dtp / (dtp + dfp + dfn) if (dtp + dfp + dfn) else float("nan")
    score = adj + DIV_W * dj if (dtp + dfp + dfn) else adj
    return dict(n=len(v), edge_jaccard=ej, adj_edge_jaccard=adj, div_jaccard=dj,
                node_recall=float(np.mean([r["node_recall"] for r in v])),
                total_node_ratio=float(np.nanmean([r["total_node_ratio"] for r in v])),
                score=score)


# =============================================================================
# 1) ENVANTER + KOORDINAT UZAYI
# =============================================================================
print("=" * 78)
print("1) ENVANTER VE KOORDINAT UZAYI")
print("=" * 78)

geffs = sorted(glob.glob(os.path.join(TRAIN, "*.geff")))
test_names = sorted(d[:-5] for d in os.listdir(TEST) if d.endswith(".zarr")) \
             if os.path.exists(TEST) else []
print(f"train .geff: {len(geffs)}   test .zarr: {len(test_names)} -> {test_names}")

catalog, space_votes = [], {}
for p in geffs:
    name = os.path.basename(p)[:-5]
    shp  = zarr_shape_of(os.path.join(TRAIN, name + ".zarr"))
    try:
        gt = read_gt(p, shp)
    except Exception as ex:
        print(f"  ATLA {name}: {type(ex).__name__} {ex}")
        continue
    space_votes[gt["coord_space"]] = space_votes.get(gt["coord_space"], 0) + 1
    catalog.append(dict(name=name, shape=shp, n_nodes=gt["n_nodes"],
                        n_edges=len(gt["edges"]), est_nodes=gt["est_nodes"],
                        t_min=int(gt["t"].min()) if gt["n_nodes"] else -1,
                        t_max=int(gt["t"].max()) if gt["n_nodes"] else -1,
                        colony=name.split("_")[0], coord_space=gt["coord_space"]))

print(f"\n  >>> KOORDINAT UZAYI OYLARI: {space_votes}")
if len(space_votes) > 1:
    print("  !!! DIKKAT: datasetler arasi TUTARSIZ -> donusum dataset basina sart")
elif space_votes.get("physical"):
    print("  !!! GEFF FIZIKSEL yaziyor -> bch.py'nin metrigi GT'yi CIFT olcekliyordu")
else:
    print("  OK: GEFF voxel yaziyor -> bch.py'nin varsayimi dogru")

shapes = {c["shape"] for c in catalog}
print(f"\n  zarr shape'leri: {shapes}")
ests = np.array([c["est_nodes"] for c in catalog], float)
ests = ests[ests == ests]
Ts   = np.array([c["shape"][0] for c in catalog if c["shape"]], float)
print(f"  est_nodes  : min={ests.min():.0f} max={ests.max():.0f} medyan={np.median(ests):.0f}")
print(f"  frame basi : min={(ests/Ts.mean()).min():.0f} max={(ests/Ts.mean()).max():.0f} hucre")
nn = np.array([c["n_nodes"] for c in catalog], float)
print(f"  GT node    : min={nn.min():.0f} max={nn.max():.0f} medyan={np.median(nn):.0f}")
print(f"  GT/gercek  : medyan %{100*np.median(nn/np.maximum(ests,1)):.1f} "
      f"-> tahminlerin ~%{100-100*np.median(nn/np.maximum(ests,1)):.0f}'i metrige GORUNMEZ")
from collections import Counter
print(f"  koloniler  : {dict(Counter(c['colony'] for c in catalog))}")


# =============================================================================
# 2) METRIK OZ-TESTI  (en kritik adim)
# =============================================================================
print("\n" + "=" * 78)
print("2) METRIK OZ-TESTI  - GT'yi tahmin gibi verip skoru kontrol ediyoruz")
print("=" * 78)

probe = [c for c in catalog if c["n_edges"] >= MIN_GT_EDGES][:5]
if not probe:
    probe = sorted(catalog, key=lambda c: -c["n_edges"])[:5]

def gt_as_pred(gt):
    return (np.column_stack([gt["t"], gt["z"], gt["y"], gt["x"]]).astype(np.float64),
            gt["edges"].copy())

tests_ok = True
rows_a, rows_b, rows_c = [], [], []
for c in probe:
    gt = read_gt(os.path.join(TRAIN, c["name"] + ".geff"),
                 zarr_shape_of(os.path.join(TRAIN, c["name"] + ".zarr")))
    n, e = gt_as_pred(gt)

    # --- TEST A: mukemmel tahmin -> edge_J = 1.000 olmali
    ra = evaluate_dataset(n, e, gt); rows_a.append(ra)

    # --- TEST B: node'larin %10'unu dusur -> edge_J ~ 0.81 (r^2 yasasi)
    keep = rng.random(len(n)) > 0.10
    remap = -np.ones(len(n), np.int64); remap[keep] = np.arange(keep.sum())
    eb = e[keep[e[:, 0]] & keep[e[:, 1]]]
    eb = np.column_stack([remap[eb[:, 0]], remap[eb[:, 1]]]) if len(eb) else eb.reshape(-1, 2)
    rb = evaluate_dataset(n[keep], eb, gt); rows_b.append(rb)

    # --- TEST C: 10x GURULTU node + aralarinda rastgele kenar
    #     edge_J DEGISMEMELI (annotate edilmemis kenarlar bedava)
    Z, Y, X = c["shape"][1], c["shape"][2], c["shape"][3]
    m = 10 * len(n)
    noise = np.column_stack([rng.integers(0, c["shape"][0], m),
                             rng.uniform(0, Z, m), rng.uniform(0, Y, m), rng.uniform(0, X, m)])
    nc = np.vstack([n, noise])
    off = len(n)
    src = rng.integers(0, m, m); dst = rng.integers(0, m, m)
    ok  = noise[dst, 0] - noise[src, 0] == 1
    ec  = np.vstack([e, np.column_stack([off + src[ok], off + dst[ok]])]) if ok.any() else e
    rc  = evaluate_dataset(nc, ec, gt); rows_c.append(rc)

    a_ok = abs(ra["edge_jaccard"] - 1.0) < 1e-9 and abs(ra["node_recall"] - 1.0) < 1e-9
    c_ok = abs(rc["edge_jaccard"] - ra["edge_jaccard"]) < 1e-9
    tests_ok &= a_ok and c_ok
    print(f"\n  {c['name']}  (GT {c['n_nodes']} node / {c['n_edges']} kenar, est={c['est_nodes']:.0f})")
    print(f"    A mukemmel : edge_J={ra['edge_jaccard']:.4f} recall={ra['node_recall']:.4f} "
          f"div_J={ra['division_tp']}/{ra['division_tp']+ra['division_fp']+ra['division_fn']} "
          f"ratio={ra['total_node_ratio']:+.3f} adj={ra['adj_edge_jaccard']:.4f}   {'OK' if a_ok else 'BOZUK'}")
    print(f"    B %10 kayip: edge_J={rb['edge_jaccard']:.4f}  (beklenen ~0.81 = 0.90^2)")
    print(f"    C 10x gurul: edge_J={rc['edge_jaccard']:.4f}  (A ile AYNI olmali) "
          f"ratio={rc['total_node_ratio']:+.2f} adj={rc['adj_edge_jaccard']:.4f}   {'OK' if c_ok else 'BOZUK'}")

print(f"\n  A) mukemmel tahmin  -> {summarise(rows_a)['edge_jaccard']:.4f}   (1.0000 olmali)")
print(f"  B) %10 node kaybi   -> {summarise(rows_b)['edge_jaccard']:.4f}   (~0.81 olmali)")
print(f"  C) 10x gurultu node -> {summarise(rows_c)['edge_jaccard']:.4f}   (A ile ayni olmali)")
print(f"\n  >>> METRIK {'DOGRULANDI' if tests_ok else 'BOZUK - DURDUR, ILERLEME'}")


# =============================================================================
# 3) test .zarr == train .zarr ?
# =============================================================================
print("\n" + "=" * 78)
print("3) TEST == TRAIN DOGRULAMASI")
print("=" * 78)

def zhash(zarr_dir, n=3):
    base, files = os.path.join(zarr_dir, "0"), []
    for cur, _d, fs in os.walk(base):
        for f in sorted(fs):
            if f not in (".zarray", "zarr.json", ".zattrs"):
                files.append(os.path.join(cur, f))
    files.sort()
    h = hashlib.sha1()
    for f in files[:n]:
        h.update(open(f, "rb").read())
    return h.hexdigest()[:16], len(files)

public = []
for nm in test_names:
    try:
        he, ne = zhash(os.path.join(TEST, nm + ".zarr"))
        hr, nr = zhash(os.path.join(TRAIN, nm + ".zarr"))
        same = (he == hr) and (ne == nr)
        gt = read_gt(os.path.join(TRAIN, nm + ".geff"),
                     zarr_shape_of(os.path.join(TRAIN, nm + ".zarr"))) if same else None
        print(f"  {nm}: {'AYNI' if same else 'FARKLI'}  chunk={ne}/{nr}"
              + (f"  est_nodes={gt['est_nodes']:.0f}  GT {gt['n_nodes']}n/{len(gt['edges'])}e" if same else ""))
        if same:
            public.append(nm)
    except Exception as ex:
        print(f"  {nm}: train'de yok / hata -> {type(ex).__name__} {ex}")
print(f"\n  >>> public-proxy olarak kullanilabilir: {len(public)}/{len(test_names)}")


# =============================================================================
# 4) VAL SETI SECIMI  (iki koloni, yogunluk spektrumu boyunca yayilmis)
# =============================================================================
print("\n" + "=" * 78)
print("4) VAL SETI SECIMI")
print("=" * 78)

pool = [c for c in catalog
        if c["n_edges"] >= MIN_GT_EDGES and c["est_nodes"] == c["est_nodes"]
        and c["name"] not in public]

def spread(lst, k):
    lst = sorted(lst, key=lambda r: r["est_nodes"])
    if len(lst) <= k:
        return lst
    return [lst[i] for i in np.linspace(0, len(lst) - 1, k).astype(int)]

colonies = sorted({c["colony"] for c in pool})
per = max(1, N_VAL // max(1, len(colonies)))
val_core = []
for col in colonies:
    val_core += spread([c for c in pool if c["colony"] == col], per)

print(f"  havuz: {len(pool)} dataset (GT>={MIN_GT_EDGES} kenar), koloniler={colonies}")
print(f"\n  CEKIRDEK VAL SETI ({len(val_core)}):")
for c in val_core:
    print(f"    {c['name']:22s} {c['colony']}  est={c['est_nodes']:6.0f}  "
          f"GT {c['n_nodes']:5d}n/{c['n_edges']:5d}e  t=[{c['t_min']},{c['t_max']}]")

out = dict(scale=list(SCALE), max_match_um=MAX_MATCH_UM,
           coord_space_votes=space_votes, metric_selftest_ok=bool(tests_ok),
           val_core=[c["name"] for c in val_core],
           val_public=public,
           folds={col: [c["name"] for c in val_core if c["colony"] == col] for col in colonies},
           catalog=catalog)
with open(os.path.join(WORK, "v11_val_sets.json"), "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\n  kaydedildi -> {WORK}/v11_val_sets.json")


# =============================================================================
# RAPOR  (bu blogu oldugu gibi kopyala)
# =============================================================================
print("\n" + "#" * 78)
print("### V11-ADIM0-RAPOR-BASLANGIC")
print(f"train_geff={len(geffs)} test_zarr={len(test_names)} shapes={sorted(shapes)}")
print(f"coord_space={space_votes}")
print(f"est_nodes min/med/max={ests.min():.0f}/{np.median(ests):.0f}/{ests.max():.0f}")
print(f"gt_nodes min/med/max={nn.min():.0f}/{np.median(nn):.0f}/{nn.max():.0f}")
print(f"gt_over_est_median={np.median(nn/np.maximum(ests,1)):.4f}")
print(f"selftest_A_edgeJ={summarise(rows_a)['edge_jaccard']:.6f}")
print(f"selftest_B_edgeJ={summarise(rows_b)['edge_jaccard']:.6f}")
print(f"selftest_C_edgeJ={summarise(rows_c)['edge_jaccard']:.6f}")
print(f"selftest_A_adj={summarise(rows_a)['adj_edge_jaccard']:.6f}")
print(f"selftest_C_adj={summarise(rows_c)['adj_edge_jaccard']:.6f}")
print(f"selftest_ok={tests_ok}")
print(f"test_eq_train={len(public)}/{len(test_names)} -> {public}")
print(f"val_core={[c['name'] for c in val_core]}")
print(f"colonies={dict(Counter(c['colony'] for c in catalog))}")
print("### V11-ADIM0-RAPOR-BITIS")
print("#" * 78)
