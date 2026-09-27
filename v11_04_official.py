# =============================================================================
# V11 ADIM 4 - RESMI SKORLAYICI + DIVISION (CPU, INTERNET ACIK, ~25 dk)
# =============================================================================
# ADIM 3 BULGUSU: linker iyi (iki ucu tespit edilmis GT kenarlarinin %91.4'u
#   dogru baglaniyor). Kayip tespitte: 1471/7020 GT kenarinda bir uc yok.
#   VE div_J = 0.0000 -> skorun 0.1'i (kovaladigimiz her seyin 2 kati) masada.
#
# BU SCRIPT: RESMI skorlayiciyi kurar (royerlab/kaggle-cell-tracking-competition),
#   bizim metrigi onunla karsilastirir, ve division'i GERCEK metrikle tarar.
#   Bizim yaklasik division sayimimiz cok kati olabilir - resmi versiyon
#   bolunmeden +-1 timepoint fork'a izin veriyor, torun penceresi kullaniyor.
#
# Girdi : yarisma verisi (GT icin) + cand_v2_*.npz  (Adim 1b notebook cikti'si)
# Cikti : sweep tablolari + RAPOR blogu
# =============================================================================
# RESMI SKORLAYICI  (Notebook ayarlarinda INTERNET ACIK olmali)
# metrics.py sadece polars+tracksdata istiyor -> torch/dask'a gerek yok (--no-deps)

import sys, subprocess

# =============================================================================
# BOOTSTRAP  -  TUM kurulumlar HERHANGI bir import'tan ONCE
# =============================================================================
# Neden bu sira onemli: zarr/tracksdata kurulumlari numpy'i YUKSELTIR ve
# onyuklu (numpy'a karsi derlenmis) scipy'yi KIRAR:
#     AttributeError: module 'numpy._core._multiarray_umath'
#                     has no attribute '_blas_supports_fpe'
# numpy bir kez BELLEGE yuklendikten sonra diski degistirmek uyusmazlik yaratir.
# Cozum: surumleri importlib.metadata ile IMPORT ETMEDEN oku, tum kurulumlari
# yap, ayni numpy/scipy'ye geri sabitle, ANCAK SONRA import et.
from importlib.metadata import version as _pkgver

def _v(name):
    try:
        return _pkgver(name)
    except Exception:
        return None

_NPV, _SPV = _v("numpy"), _v("scipy")
print(f"[bootstrap] baslangic numpy={_NPV} scipy={_SPV}", flush=True)

def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=False)

_pip("zarr")
_pip("git+https://github.com/royerlab/tracksdata@main")
_pip("--no-deps", "git+https://github.com/royerlab/kaggle-cell-tracking-competition.git")

# numpy/scipy'yi geri sabitle (henuz hicbiri import EDILMEDI)
_pin = [f"{n}=={v}" for n, v in (("numpy", _NPV), ("scipy", _SPV)) if v]
if _pin:
    _pip("--no-deps", *_pin)
print(f"[bootstrap] kurulum sonrasi numpy={_v('numpy')} scipy={_v('scipy')} "
      f"zarr={_v('zarr')} tracksdata={_v('tracksdata')}", flush=True)

import os, glob, json, time, math, itertools
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

COMP  = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
TRAIN = os.path.join(COMP, "train")
WORK  = "/kaggle/working"

# cand_v2_*.npz nerede? (notebook output'unu input olarak ekledigin klasor)
CAND_DIRS = sorted(glob.glob("/kaggle/input/**/cand_v2_*.npz", recursive=True))
CAND_DIR  = os.path.dirname(CAND_DIRS[0]) if CAND_DIRS else "/kaggle/working"

SCALE        = (1.625, 0.40625, 0.40625)
MAX_MATCH_UM = 7.0
ALPHA        = 0.1
DIV_W        = 0.1

CORE = {'44b6_341df25f','44b6_3bb3690f','44b6_c771cb04','44b6_8f9ecab4',
        '6bba_2540cd90','6bba_3a1849c2','6bba_67ebd073','6bba_57b7cc1e'}

# --- SWEEP IZGARASI ---------------------------------------------------------
R_LIST   = [7.0]                # ADIM 2: r_nms=7.0 optimum
BEST_R   = 7.0
BEST_N   = 0.95                 # ADIM 2: n* = 0.95
THR_GRID = [3e-4, 6e-4, 1.2e-3, 2.5e-3, 5e-3, 8e-3,
            1.2e-2, 2.0e-2, 2.5e-2, 3.5e-2, 6e-2, 1.2e-1, 2.5e-1]  # 2.5e-2 = v9 baseline
N_STARS  = [round(0.9 + 0.05 * i, 2) for i in range(19)]   # n-hedefleme politikasi

FRAME_LIMIT = None      # None = 100 frame; sure yetmezse 50 yap
rng = np.random.default_rng(0)

t_start = time.time()
def elapsed(): return (time.time() - t_start) / 60.0


# =============================================================================
# GT OKUMA  (Adim 0: 199/199 dataset voxel yaziyor -> donusum gereksiz)
# =============================================================================
def read_gt(geff_path):
    import zarr
    g = zarr.open_group(str(geff_path), mode="r")
    meta = dict(g.attrs).get("geff", {}) or {}
    est = (meta.get("extra") or {}).get("estimated_number_of_nodes")

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
    return dict(t=t, z=z, y=y, x=x, edges=edges, n_nodes=len(t),
                est_nodes=float(est) if est is not None else float("nan"))


def subset_gt(gt, t_lo, t_hi):
    """GT'yi [t_lo,t_hi]'ye kis, kenarlari yeniden indeksle (aralik disini FN sayma)."""
    keep = np.where((gt["t"] >= t_lo) & (gt["t"] <= t_hi))[0]
    remap = -np.ones(gt["n_nodes"], np.int64)
    remap[keep] = np.arange(len(keep))
    e = gt["edges"]
    if len(e):
        m = (remap[e[:, 0]] >= 0) & (remap[e[:, 1]] >= 0)
        e = np.column_stack([remap[e[m, 0]], remap[e[m, 1]]])
    else:
        e = np.zeros((0, 2), np.int64)
    scale_frac = (t_hi - t_lo + 1) / 100.0
    return dict(t=gt["t"][keep], z=gt["z"][keep], y=gt["y"][keep], x=gt["x"][keep],
                edges=e, n_nodes=len(keep), est_nodes=gt["est_nodes"] * scale_frac)


# =============================================================================
# RESMI METRIK  (Adim 0'daki dogrulanmis mantik, hizlandirilmis)
# =============================================================================
def match_nodes(pred_nodes, gt, scale=SCALE, max_dist=MAX_MATCH_UM):
    sc = np.asarray(scale)
    matched = np.full(len(pred_nodes), -1, np.int64)
    if len(pred_nodes) == 0 or gt["n_nodes"] == 0:
        return matched
    gt_P = np.stack([gt["z"], gt["y"], gt["x"]], 1) * sc
    p_t = pred_nodes[:, 0].astype(np.int64)
    p_P = pred_nodes[:, 1:] * sc
    # frame dilimleri: pred zaten t sirali
    for tv in np.unique(gt["t"]):
        gi = np.where(gt["t"] == tv)[0]
        pi = np.where(p_t == tv)[0]
        if len(gi) == 0 or len(pi) == 0:
            continue
        D = np.linalg.norm(p_P[pi][:, None, :] - gt_P[gi][None, :, :], axis=2)
        C = np.where(D <= max_dist, D, 1e6)
        if not np.any(C < 1e6):
            continue
        r, c = linear_sum_assignment(C)
        ok = C[r, c] < 1e6
        matched[pi[r[ok]]] = gi[c[ok]]
    return matched


def canonical_edges(pred_nodes, pred_edges, matched):
    """dt==1 / tekille / merge-dedup / out-degree<=2  (vektorlestirilmis)"""
    e = np.asarray(pred_edges, np.int64).reshape(-1, 2)
    if len(e) == 0:
        return e
    ts = pred_nodes[e[:, 0], 0].astype(np.int64)
    tt = pred_nodes[e[:, 1], 0].astype(np.int64)
    e = e[(tt - ts) == 1]
    if len(e) == 0:
        return e

    # tekrarlayan (source,target) ciftlerini ilk gorunume gore tekille
    _, first = np.unique(e, axis=0, return_index=True)
    e = e[np.sort(first)]

    # merge-dedup: ayni (matched_src, matched_tgt)'a dusenlerden ilkini tut
    ms, mt = matched[e[:, 0]], matched[e[:, 1]]
    both = (ms >= 0) & (mt >= 0)
    if both.any():
        idx = np.where(both)[0]
        span = int(matched.max()) + 2
        key = ms[idx].astype(np.int64) * span + mt[idx].astype(np.int64)
        o = np.argsort(key, kind='stable')           # esitlikte orijinal sira korunur
        ks = key[o]
        firstmask = np.ones(len(ks), bool)
        firstmask[1:] = ks[1:] != ks[:-1]
        keep = np.ones(len(e), bool)
        keep[idx] = False
        keep[idx[o[firstmask]]] = True
        e = e[keep]
        if len(e) == 0:
            return e

    # out-degree <= 2: kaynak basina ilk iki kenar
    e = e[np.lexsort((np.arange(len(e)), e[:, 0]))]
    src = e[:, 0]
    grp = np.searchsorted(src, src, side="left")
    return e[(np.arange(len(e)) - grp) <= 1]


def division_counts(e, matched, gt, n_pred):
    """FP SADECE eslesmis + GT'de cocugu olan fork'lardan. Bolunmeler seyrek ->
    sadece fork'lar uzerinde donuyoruz."""
    gt_edges = gt["edges"]
    if len(gt_edges) == 0:
        return 0, 0, 0
    gt_out = np.bincount(gt_edges[:, 0], minlength=gt["n_nodes"])
    gt_div_nodes = np.where(gt_out >= 2)[0]
    n_gt_div = len(gt_div_nodes)
    if len(e) == 0:
        return 0, 0, n_gt_div

    gt_child, gt_has_child = {}, set(np.unique(gt_edges[:, 0]).tolist())
    if n_gt_div:
        sel = gt_edges[np.isin(gt_edges[:, 0], gt_div_nodes)]
        for s, t in sel.tolist():
            gt_child.setdefault(s, []).append(t)

    pred_out = np.bincount(e[:, 0], minlength=n_pred)
    forks = np.where(pred_out >= 2)[0]
    if len(forks) == 0:
        return 0, 0, n_gt_div
    pred_child = {}
    for s, t in e[np.isin(e[:, 0], forks)].tolist():
        pred_child.setdefault(s, []).append(t)

    tp, fp, hit = 0, 0, set()
    for p, ch in pred_child.items():
        mp = int(matched[p])
        if mp < 0:
            continue                          # eslesmemis fork -> BEDAVA
        if mp in gt_child:
            got = {int(matched[c]) for c in ch if matched[c] >= 0}
            if len(set(gt_child[mp]) & got) >= 2:
                tp += 1
                hit.add(mp)
            else:
                fp += 1
        elif mp in gt_has_child:
            fp += 1
    return tp, fp, n_gt_div - len(hit)


def evaluate_dataset(pred_nodes, pred_edges, gt, n_total=None, scale=SCALE):
    pred_nodes = np.asarray(pred_nodes, np.float64).reshape(-1, 4)
    matched = match_nodes(pred_nodes, gt, scale)
    e = canonical_edges(pred_nodes, pred_edges, matched)

    gt_edges, n_gt_nodes = gt["edges"], gt["n_nodes"]
    out_deg = np.zeros(n_gt_nodes, np.int64)
    in_deg = np.zeros(n_gt_nodes, np.int64)
    if len(gt_edges):
        np.add.at(out_deg, gt_edges[:, 0], 1)
        np.add.at(in_deg, gt_edges[:, 1], 1)

    tp = valid = 0
    if len(e):
        ms, mt = matched[e[:, 0]], matched[e[:, 1]]
        out_valid = (ms >= 0) & (out_deg[np.clip(ms, 0, None)] > 0)
        in_valid = (mt >= 0) & (in_deg[np.clip(mt, 0, None)] > 0)
        valid = int((out_valid | in_valid).sum())
        both = np.where((ms >= 0) & (mt >= 0))[0]
        if len(both) and len(gt_edges):
            span = int(n_gt_nodes) + 2
            gt_keys = np.sort(gt_edges[:, 0].astype(np.int64) * span + gt_edges[:, 1])
            pk = ms[both].astype(np.int64) * span + mt[both].astype(np.int64)
            pos = np.searchsorted(gt_keys, pk)
            pos = np.clip(pos, 0, len(gt_keys) - 1)
            tp = int((gt_keys[pos] == pk).sum())

    edge_tp, edge_fp, edge_fn = tp, max(0, valid - tp), len(gt_edges) - tp
    n_matched = len(np.unique(matched[matched >= 0]))
    node_recall = n_matched / n_gt_nodes if n_gt_nodes else float("nan")
    dtp, dfp, dfn = division_counts(e, matched, gt, len(pred_nodes))

    num_pred = len(pred_nodes)
    if n_total is None:
        n_total = gt.get("est_nodes", float("nan"))
    ratio = (num_pred - n_total) / n_total if (n_total and n_total > 0) else float("nan")
    denom = edge_tp + edge_fp + edge_fn
    J = edge_tp / denom if denom > 0 else float("nan")
    adj = max(0.0, J * (1 - ALPHA * ratio)) if (J == J and ratio == ratio) else float("nan")
    return dict(edge_tp=edge_tp, edge_fp=edge_fp, edge_fn=edge_fn,
                division_tp=dtp, division_fp=dfp, division_fn=dfn,
                num_pred_nodes=num_pred, n_total=n_total,
                n_ratio=(num_pred / n_total) if (n_total and n_total > 0) else float('nan'),
                node_recall=node_recall, total_node_ratio=ratio,
                edge_jaccard=J, adj_edge_jaccard=adj, n_edges_kept=len(e))


def summarise(rows):
    v = [r for r in rows if r["edge_jaccard"] == r["edge_jaccard"]]
    if not v:
        return dict(score=float("nan"))
    w = np.array([r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in v], float)
    aj = np.array([r["adj_edge_jaccard"] for r in v], float)
    ok = aj == aj
    adj = float((w[ok] * aj[ok]).sum() / w[ok].sum()) if ok.any() and w[ok].sum() > 0 else float("nan")
    ej = float(sum(r["edge_tp"] for r in v) /
               max(1, sum(r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in v)))
    dtp = sum(r["division_tp"] for r in v)
    dfp = sum(r["division_fp"] for r in v)
    dfn = sum(r["division_fn"] for r in v)
    dj = dtp / (dtp + dfp + dfn) if (dtp + dfp + dfn) else float("nan")
    score = adj + DIV_W * dj if (dtp + dfp + dfn) else adj
    return dict(n=len(v), edge_jaccard=ej, adj_edge_jaccard=adj, div_jaccard=dj,
                node_recall=float(np.mean([r["node_recall"] for r in v])),
                n_ratio=float(np.mean([r["n_ratio"] for r in v])),
                div_tp=dtp, div_fp=dfp, div_fn=dfn, score=score)


# =============================================================================
# ADAY YUKLEME + HIZLI NMS
# =============================================================================
def load_cands(name):
    d = np.load(os.path.join(CAND_DIR, f"cand_v2_{name}.npz"))
    T = int(d["T"])
    t = d["t"].astype(np.int64)
    coords = np.stack([d["z"], d["y"], d["x"]], 1).astype(np.float64)
    sc = d["score"].astype(np.float64)
    fstart = np.searchsorted(t, np.arange(T + 1))   # t artan sirada
    return dict(T=T, coords=coords, score=sc, fstart=fstart,
                est=float(d["est_nodes"]))


def frame_pairs(coords, r_max, scale=SCALE):
    """(pairs, dist) - frame icindeki r_max'tan yakin tum ciftler (fiziksel)"""
    if len(coords) < 2:
        return np.zeros((0, 2), np.int64), np.zeros(0)
    P = coords * np.asarray(scale)
    pr = cKDTree(P).query_pairs(r_max, output_type='ndarray')
    if len(pr) == 0:
        return pr.reshape(-1, 2), np.zeros(0)
    dd = np.linalg.norm(P[pr[:, 0]] - P[pr[:, 1]], axis=1)
    return pr, dd


def build_csr(pairs, dist, r, M):
    """yonsuz komsuluk -> CSR (b, starts); sadece dist<=r olan ciftler"""
    if len(pairs):
        p = pairs[dist <= r]
    else:
        p = pairs
    if len(p) == 0:
        return np.zeros(0, np.int64), np.zeros(M + 1, np.int64)
    a = np.concatenate([p[:, 0], p[:, 1]])
    b = np.concatenate([p[:, 1], p[:, 0]])
    o = np.argsort(a, kind='stable')
    a, b = a[o], b[o]
    return b, np.searchsorted(a, np.arange(M + 1))


def nms_frame(order, b, starts, active):
    """greedy NMS: skor sirasinda gez, kabul edileni tut, komsularini bastir"""
    sup = ~active
    keep = []
    for i in order:
        if sup[i]:
            continue
        keep.append(i)
        sup[b[starts[i]:starts[i + 1]]] = True
    return keep


# =============================================================================
# LINKER  (v9 mantigi; yerel hareket alani opsiyonel, LAP yerine kapili greedy)
# =============================================================================
def _knn(A, B, k=2):
    if len(B) == 0 or len(A) == 0:
        return np.full((len(A), k), np.inf), np.zeros((len(A), k), np.int64)
    ke = min(k, len(B))
    d, i = cKDTree(B).query(A, k=ke)
    if ke == 1:
        d, i = d[:, None], i[:, None]
    if ke < k:
        d = np.pad(d, ((0, 0), (0, k - ke)), constant_values=np.inf)
        i = np.pad(i, ((0, 0), (0, k - ke)), constant_values=0)
    return d, i


def _greedy_gated(A, B, gate):
    """gate icindeki ciftleri mesafeye gore greedy esle -> {i: j}"""
    if len(A) == 0 or len(B) == 0 or gate <= 0:
        return {}
    nb = cKDTree(B).query_ball_point(A, gate)
    cand = []
    for i, js in enumerate(nb):
        for j in js:
            cand.append((float(np.linalg.norm(A[i] - B[j])), i, j))
    cand.sort()
    ua, ub, out = set(), set(), {}
    for _d, i, j in cand:
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); out[i] = j
    return out


def link_frames(P0, P1, gate=8.0, mnn_ratio=0.80, fill_gate=8.0,
                local_motion=False, motion_iters=3, bw=25.0, min_support=4):
    n0, n1 = len(P0), len(P1)
    assign = np.full(n0, -1, np.int64)
    if n0 == 0 or n1 == 0:
        return assign, np.zeros((max(n0, 1), 3))

    # --- global oteleme (medyan akis) --- tree1 bir kez kurulur
    tree1 = cKDTree(P1)
    shift = np.zeros(3)
    for _ in range(4):
        d, i = tree1.query(P0 + shift, k=1)
        m = d < gate
        if m.sum() < 8:
            break
        step = np.median(P1[i[m]] - (P0[m] + shift), axis=0)
        shift += step
        if np.linalg.norm(step) < 1e-3:
            break
    disp = np.tile(shift, (n0, 1)).astype(np.float64)

    iters = motion_iters if local_motion else 1
    k1 = min(2, n1)
    for it in range(iters):
        Q = P0 + disp
        d01, i01 = tree1.query(Q, k=k1)
        if k1 == 1:
            d01, i01 = d01[:, None], i01[:, None]
            d01 = np.pad(d01, ((0, 0), (0, 1)), constant_values=np.inf)
            i01 = np.pad(i01, ((0, 0), (0, 1)), constant_values=0)
        d10, i10 = _knn(P1, Q, k=2)
        mutual = i10[i01[:, 0], 0] == np.arange(n0)
        gated = d01[:, 0] < gate
        distinct = np.isinf(d01[:, 1]) | (d01[:, 0] < mnn_ratio * d01[:, 1])
        conf = mutual & gated & distinct
        if it < iters - 1:
            if conf.sum() >= min_support:
                vecs = P1[i01[conf, 0]] - P0[conf]
                tree = cKDTree(P0[conf])
                out = np.tile(np.median(vecs, axis=0), (n0, 1)).astype(np.float64)
                for j, nb in enumerate(tree.query_ball_point(P0, 2.5 * bw)):
                    if len(nb) < min_support:
                        continue
                    nb = np.asarray(nb)
                    w = np.exp(-0.5 * ((P0[conf][nb] - P0[j]) ** 2).sum(1) / bw ** 2)
                    sw = w.sum()
                    if sw > 1e-8:
                        out[j] = (w[:, None] * vecs[nb]).sum(0) / sw
                disp = out
        else:
            assign[conf] = i01[conf, 0]

    # --- hedef basina tek kaynak (en yakini tut) ---
    best = {}
    for a in np.where(assign >= 0)[0]:
        bb = int(assign[a])
        dd = float(np.linalg.norm(P0[a] + disp[a] - P1[bb]))
        if bb not in best or dd < best[bb][1]:
            if bb in best:
                assign[best[bb][0]] = -1
            best[bb] = (a, dd)
        else:
            assign[a] = -1

    # --- artakalanlari kapili greedy ile doldur ---
    free0 = np.where(assign < 0)[0]
    used1 = np.zeros(n1, bool)
    used1[assign[assign >= 0]] = True
    free1 = np.where(~used1)[0]
    if len(free0) and len(free1):
        for i, j in _greedy_gated(P0[free0] + disp[free0], P1[free1], fill_gate).items():
            assign[free0[i]] = free1[j]
    return assign, disp


def find_divisions(P0, P1, assign, disp, radius=9.0, min_angle=100.0, max_ratio=2.0):
    import math
    n1 = len(P1)
    used1 = np.zeros(n1, bool)
    used1[assign[assign >= 0]] = True
    free1 = np.where(~used1)[0]
    parents = np.where(assign >= 0)[0]
    if len(free1) == 0 or len(parents) == 0 or radius <= 0:
        return []
    d, i = _knn(P1[free1], P0[parents] + disp[parents], k=1)
    cos_lim = math.cos(math.radians(min_angle))
    out, busy = [], set()
    for oi in np.argsort(d[:, 0]):
        if d[oi, 0] >= radius:
            break
        p = int(parents[i[oi, 0]])
        if p in busy:
            continue
        c1 = P1[assign[p]] - P0[p]
        c2 = P1[free1[oi]] - P0[p]
        n1_, n2_ = np.linalg.norm(c1), np.linalg.norm(c2)
        if min(n1_, n2_) < 1e-6 or max(n1_, n2_) / min(n1_, n2_) > max_ratio:
            continue
        if float(c1 @ c2) / (n1_ * n2_) > cos_lim:
            continue
        out.append((p, int(free1[oi])))
        busy.add(p)
    return out


# =============================================================================
# SON ISLEM: GAP CLOSING (FIZIKSEL) + KISA TRACK FILTRESI
# =============================================================================
def gap_close(nodes, edges, max_gap=4, gate_um=10.0, scale=SCALE):
    """v10'daki hata: mesafe HAM VOXEL uzayinda olculuyordu (z'de 3.5x gevsek)."""
    if len(nodes) == 0 or len(edges) == 0 or max_gap < 2 or gate_um <= 0:
        return nodes, edges
    sc = np.asarray(scale)
    N = len(nodes)
    out_deg = np.zeros(N, np.int32); in_deg = np.zeros(N, np.int32)
    np.add.at(out_deg, edges[:, 0], 1)
    np.add.at(in_deg, edges[:, 1], 1)
    tails = np.where(out_deg == 0)[0]
    heads = np.where(in_deg == 0)[0]
    if len(tails) == 0 or len(heads) == 0:
        return nodes, edges

    heads_by_t, trees = {}, {}
    for h in heads:
        heads_by_t.setdefault(int(nodes[h, 0]), []).append(int(h))
    for tv, hl in heads_by_t.items():
        trees[tv] = cKDTree(nodes[hl, 1:] * sc)

    new_nodes, new_edges, used = [], edges.tolist(), set()
    for tail in tails:
        t_tail = int(nodes[tail, 0])
        bh, bd, bg = -1, gate_um, 0
        q = nodes[tail, 1:] * sc
        for gap in range(2, max_gap + 1):
            ts = t_tail + gap
            if ts not in trees:
                continue
            d, i = trees[ts].query(q, k=1)
            if d < bd:
                cand = heads_by_t[ts][i]
                if cand not in used:
                    bd, bh, bg = float(d), cand, gap
        if bh != -1:
            used.add(bh)
            nt, nh, prev = nodes[tail], nodes[bh], int(tail)
            for step in range(1, bg):
                a = step / bg
                interp = (1.0 - a) * nt + a * nh
                interp[0] = t_tail + step
                nid = N + len(new_nodes)
                new_nodes.append(interp)
                new_edges.append([prev, nid])
                prev = nid
            new_edges.append([prev, int(bh)])
    if new_nodes:
        nodes = np.vstack([nodes, np.array(new_nodes)])
        edges = np.asarray(new_edges, np.int64)
    return nodes, edges


def filter_short(nodes, edges, min_len):
    if min_len <= 1 or len(nodes) == 0:
        return nodes, edges
    parent = np.arange(len(nodes))
    def find(i):
        r = i
        while parent[r] != r:
            r = parent[r]
        while parent[i] != r:
            parent[i], i = r, parent[i]
        return r
    for u, v in edges.tolist():
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[ru] = rv
    roots = np.array([find(i) for i in range(len(nodes))])
    size = np.bincount(roots, minlength=len(nodes))
    ok = size[roots] >= min_len
    remap = np.cumsum(ok) - 1
    nodes = nodes[ok]
    if len(edges):
        m = ok[edges[:, 0]] & ok[edges[:, 1]]
        edges = remap[edges[m]]
    return nodes, edges


def build_graph(frames, T, link_kw, div_kw, gap_kw, min_len, scale=SCALE):
    sc = np.asarray(scale)
    all_nodes, all_edges = [], []
    offset, prev_P, prev_off = 0, None, None
    for t in range(T):
        c = frames.get(t, np.zeros((0, 3)))
        P = c * sc
        all_nodes.append(np.column_stack([np.full(len(c), t, np.float64), c]))
        if prev_P is not None and len(prev_P) and len(P):
            assign, disp = link_frames(prev_P, P, **link_kw)
            src = np.where(assign >= 0)[0]
            if len(src):
                all_edges.append(np.column_stack([prev_off + src, offset + assign[src]]))
            if div_kw is not None:
                dv = find_divisions(prev_P, P, assign, disp, **div_kw)
                if dv:
                    all_edges.append(np.array([[prev_off + p, offset + ch] for p, ch in dv],
                                              np.int64))
        prev_P, prev_off = P, offset
        offset += len(c)
    nodes = np.vstack(all_nodes) if all_nodes else np.zeros((0, 4))
    edges = (np.vstack(all_edges).astype(np.int64) if all_edges
             else np.zeros((0, 2), np.int64))
    if gap_kw is not None:
        nodes, edges = gap_close(nodes, edges, **gap_kw)
    nodes, edges = filter_short(nodes, edges, min_len)
    return nodes, edges



# =============================================================================
# 0) RESMI SKORLAYICIYI BAGLA
# =============================================================================
import warnings
import polars as pl
import tracksdata as td
from tracking_cellmot.metrics import (
    evaluate as off_evaluate,
    node_recall as off_node_recall,
    per_sample_metrics as off_per_sample,
    summarise as off_summarise,
)
print(f"tracksdata {getattr(td, '__version__', '?')} | resmi metrik baglandi")

AK = td.DEFAULT_ATTR_KEYS


def to_td(nodes, edges):
    """(N,4)[t,z,y,x] + (E,2) -> tracksdata grafi (csv_to_geffs.py ile ayni yol)"""
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    assigned = g.bulk_add_nodes([
        {"t": int(r[0]), "z": float(r[1]), "y": float(r[2]), "x": float(r[3])}
        for r in np.asarray(nodes)
    ])
    if len(edges):
        g.bulk_add_edges([{"source_id": assigned[int(s)], "target_id": assigned[int(t)]}
                          for s, t in np.asarray(edges)])
    return g


_GT_CACHE = {}
def gt_td(name):
    if name not in _GT_CACHE:
        r = td.graph.IndexedRXGraph.from_geff(os.path.join(TRAIN, name + ".geff"))
        _GT_CACHE[name] = r[0] if isinstance(r, tuple) else r
    return _GT_CACHE[name]


def official_row(nodes, edges, name, est):
    g = to_td(nodes, edges)
    gg = gt_td(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        er = off_evaluate(g, gg, scale=SCALE, max_distance=MAX_MATCH_UM)
        nr = off_node_recall(g, gg)
    return off_per_sample(er, n_total=est, node_recall=nr)


# =============================================================================
# 1) DUMAN TESTI: GT'yi tahmin olarak ver -> resmi skor ne diyor?
#    Bizim metrik: adj=1.0989.  Resmi ayni mi? div_J = 1.0 cikiyor mu?
# =============================================================================
ALL = sorted(os.path.basename(p)[8:-4] for p in glob.glob(os.path.join(CAND_DIR, "cand_v2_*.npz")))
assert ALL, "cand_v2_*.npz bulunamadi"
GT = {n: read_gt(os.path.join(TRAIN, n + ".geff")) for n in ALL}
CORE_L = [n for n in ALL if n in CORE]
G44 = [n for n in CORE_L if n.startswith("44b6")]
G6B = [n for n in CORE_L if n.startswith("6bba")]

print("\n" + "=" * 100)
print("1) DUMAN TESTI - GT'nin kendisi tahmin olarak verilirse")
print("=" * 100)
print(f"  {'dataset':22s} | {'BIZIM':>28s} | {'RESMI':>40s}")
print(f"  {'':22s} | {'edgeJ':>7}{'adj':>8}{'divJ':>7}{'rec':>6} | "
      f"{'edgeJ':>7}{'adj':>8}{'divJ':>7}{'dtp':>5}{'dfp':>5}{'dfn':>5}{'rec':>6}")
smoke_o, smoke_m = [], []
for name in ALL[:6]:
    g = GT[name]
    nd = np.column_stack([g["t"], g["z"], g["y"], g["x"]]).astype(np.float64)
    ed = g["edges"].copy()
    mine = evaluate_dataset(nd, ed, g)
    off = official_row(nd, ed, name, g["est_nodes"])
    smoke_m.append(mine); smoke_o.append(off)
    dj = (off["division_tp"] / max(1, off["division_tp"] + off["division_fp"] + off["division_fn"]))
    print(f"  {name:22s} | {mine['edge_jaccard']:7.4f}{mine['adj_edge_jaccard']:8.4f}"
          f"{(mine['division_tp']/max(1,mine['division_tp']+mine['division_fp']+mine['division_fn'])):7.4f}"
          f"{mine['node_recall']:6.3f} | "
          f"{off['edge_jaccard']:7.4f}{off['adj_edge_jaccard']:8.4f}{dj:7.4f}"
          f"{off['division_tp']:5d}{off['division_fp']:5d}{off['division_fn']:5d}{off['node_recall']:6.3f}")

so = off_summarise(smoke_o)
sm = summarise(smoke_m)
print(f"\n  BIZIM  toplam: adj={sm['adj_edge_jaccard']:.4f} divJ={sm['div_jaccard']:.4f} score={sm['score']:.4f}")
print(f"  RESMI  toplam: adj={so['adj_edge_jaccard']:.4f} divJ={so['division_jaccard']:.4f} score={so['score']:.4f}")
print(f"  >>> RESMI TAVAN (mukemmel tahminle ulasilabilecek en yuksek skor): {so['score']:.4f}")
_ok = abs(so['adj_edge_jaccard'] - sm['adj_edge_jaccard']) < 0.02
print(f"  >>> kenar metrigi {'UYUSUYOR' if _ok else 'AYRIILIYOR - DIKKAT'}")


# =============================================================================
# 2) NODE URETIMI  (ADIM 3'TEN GELEN KONFIGURASYON)
#    !!! Adim 3 raporu gelince bu blogu guncelle !!!
# =============================================================================
# ADIM 3 NIHAI SONUCU (core=0.6640, 44b6=0.7263, 6bba=0.6442)
CFG_NSTAR   = 0.95      # n* taramasi: 0.3->0.379 0.7->0.612 0.95->0.664 1.3->0.639
CFG_GATE    = 10.0      # 6.5->0.6077 8->0.6099 10->0.6109 12->0.6101 (duz)
CFG_MNN     = 0.80      # 0.7->0.6619 0.8->0.6635 0.9->0.6612
CFG_FILL    = 4.0       # 0->0.6635 4->0.6640 8->0.6499
CFG_GAP     = dict(max_gap=8, gate_um=8.0)   # ic optimum: (6,8)=0.6471 (8,8)=0.6528 (12,8)=0.6299
CFG_MINLEN  = 6         # GT hat uzunlugu p10=8 -> 6-8 ilkeli sinir; 24->0.605 64->0.344

VOL_UM3 = 64 * 1.625 * 256 * 0.40625 * 256 * 0.40625
LINKF = dict(gate=CFG_GATE, mnn_ratio=CFG_MNN, fill_gate=CFG_FILL, local_motion=False)

def build_nodes(name, nstar):
    cd = load_cands(name)
    T, coords, score, fstart, est = cd["T"], cd["coords"], cd["score"], cd["fstart"], cd["est"]
    if FRAME_LIMIT:
        T = min(T, FRAME_LIMIT)
    target = nstar * est * (T / 100.0)
    fr = []
    for tv in range(T):
        sl = slice(fstart[tv], fstart[tv + 1])
        c, s = coords[sl], score[sl]
        pr, dd = frame_pairs(c, BEST_R)
        b, stt = build_csr(pr, dd, BEST_R, len(c))
        fr.append((c, s, b, stt, np.argsort(-s, kind='stable')))
    def count_at(thr):
        tot, out = 0, {}
        for tv in range(T):
            c, s, b, stt, o = fr[tv]
            if len(c) == 0:
                continue
            act = s > thr
            if not act.any():
                continue
            k = nms_frame(o, b, stt, act)
            out[tv] = c[k]; tot += len(k)
        return tot, out
    lo, hi, mid, frames = 1e-5, 0.9, 0.9, {}
    for _ in range(16):
        mid = math.sqrt(lo * hi)
        tot, frames = count_at(mid)
        if tot > target:
            lo = mid
        else:
            hi = mid
    return frames, T

print("\n  node uretimi (n*=%.2f)..." % CFG_NSTAR)
NODES = {n: build_nodes(n, CFG_NSTAR) for n in ALL}
print(f"  bitti ({elapsed():.1f} dk)")


# =============================================================================
# 3) BIZIM METRIK vs RESMI METRIK  (gercek konfigurasyonla)
# =============================================================================
print("\n" + "=" * 100)
print("3) KONFIGURASYON DOGRULAMASI - bizim metrik resmi metrikle uyusuyor mu?")
print("=" * 100)
print(f"  {'dataset':22s} | {'BIZIM':>23s} | {'RESMI':>38s}")
print(f"  {'':22s} | {'edgeJ':>7}{'adj':>8}{'rec':>7} | "
      f"{'edgeJ':>7}{'adj':>8}{'rec':>7}{'dtp':>5}{'dfp':>5}{'dfn':>5}")
mine_rows, off_rows = {}, {}
for name in ALL:
    frames, T = NODES[name]
    gt = subset_gt(GT[name], 0, T - 1) if FRAME_LIMIT else GT[name]
    nd, ed = build_graph(frames, T, LINKF, None, CFG_GAP, CFG_MINLEN)
    m = evaluate_dataset(nd, ed, gt)
    o = official_row(nd, ed, name, gt["est_nodes"])
    mine_rows[name], off_rows[name] = m, o
    print(f"  {name:22s} | {m['edge_jaccard']:7.4f}{m['adj_edge_jaccard']:8.4f}{m['node_recall']:7.3f} | "
          f"{o['edge_jaccard']:7.4f}{o['adj_edge_jaccard']:8.4f}{o['node_recall']:7.3f}"
          f"{o['division_tp']:5d}{o['division_fp']:5d}{o['division_fn']:5d}")

def both(sel):
    return summarise([mine_rows[n] for n in sel]), off_summarise([off_rows[n] for n in sel])
mc, oc = both(CORE_L)
m4, o4 = both(G44)
m6, o6 = both(G6B)
print(f"\n  BIZIM core={mc['score']:.4f} (adj={mc['adj_edge_jaccard']:.4f} divJ={mc['div_jaccard']:.4f})")
print(f"  RESMI core={oc['score']:.4f} (adj={oc['adj_edge_jaccard']:.4f} divJ={oc['division_jaccard']:.4f})")
print(f"  RESMI 44b6={o4['score']:.4f}  6bba={o6['score']:.4f}")
print(f"  >>> adj farki = {abs(mc['adj_edge_jaccard']-oc['adj_edge_jaccard']):.4f}")


# =============================================================================
# 4) DIVISION TARAMASI - RESMI METRIKLE  (skorun 0.1'i masada)
# =============================================================================
print("\n" + "=" * 100)
print("4) DIVISION TARAMASI - resmi division metrigiyle")
print("=" * 100)
DIVS = [
    ("KAPALI",            None),
    ("r9  a100 x2.0 (v9)", dict(radius=9.0,  min_angle=100.0, max_ratio=2.0)),
    ("r12 a90  x2.5",      dict(radius=12.0, min_angle=90.0,  max_ratio=2.5)),
    ("r14 a70  x3.0",      dict(radius=14.0, min_angle=70.0,  max_ratio=3.0)),
    ("r18 a60  x3.5",      dict(radius=18.0, min_angle=60.0,  max_ratio=3.5)),
    ("r22 a45  x4.0",      dict(radius=22.0, min_angle=45.0,  max_ratio=4.0)),
    ("r30 a30  x6.0",      dict(radius=30.0, min_angle=30.0,  max_ratio=6.0)),
]
print(f"  {'div konfig':22s} {'core':>7} {'44b6':>7} {'6bba':>7} | {'adj':>7} {'divJ':>7} "
      f"{'dtp':>5} {'dfp':>5} {'dfn':>5} {'edgeJ':>7}")
div_res = {}
DIV_SEL = CORE_L          # public 4 skorlanmiyor -> tarama core uzerinde
for tag, dv in DIVS:
    if elapsed() > 150:
        print(f"  !!! sure siniri ({elapsed():.0f} dk) - kalan div konfigleri atlandi")
        break
    rows = {}
    for name in DIV_SEL:
        frames, T = NODES[name]
        gt = subset_gt(GT[name], 0, T - 1) if FRAME_LIMIT else GT[name]
        nd, ed = build_graph(frames, T, LINKF, dv, CFG_GAP, CFG_MINLEN)
        rows[name] = official_row(nd, ed, name, gt["est_nodes"])
    s = off_summarise([rows[n] for n in DIV_SEL])
    a = off_summarise([rows[n] for n in G44])
    b = off_summarise([rows[n] for n in G6B])
    div_res[tag] = (s, a, b)
    print(f"  {tag:22s} {s['score']:7.4f} {a['score']:7.4f} {b['score']:7.4f} | "
          f"{s['adj_edge_jaccard']:7.4f} {s['division_jaccard']:7.4f} "
          f"{s['division_tp']:5d} {s['division_fp']:5d} {s['division_fn']:5d} "
          f"{s['edge_jaccard']:7.4f}   ({elapsed():.1f}dk)")
DBEST = max(div_res, key=lambda k: div_res[k][0]['score'])
print(f"\n  >>> EN IYI: {DBEST}  core={div_res[DBEST][0]['score']:.4f}  "
      f"(KAPALI'ya gore {div_res[DBEST][0]['score']-div_res['KAPALI'][0]['score']:+.4f})")


# =============================================================================
# RAPOR
# =============================================================================
print("\n" + "#" * 100)
print("### V11-ADIM4-RAPOR-BASLANGIC")
print(f"sure_dk={elapsed():.1f} n_dataset={len(ALL)}")
print(f"--- duman testi (GT=tahmin) ---")
print(f"bizim adj={sm['adj_edge_jaccard']:.4f} divJ={sm['div_jaccard']:.4f} score={sm['score']:.4f}")
print(f"resmi adj={so['adj_edge_jaccard']:.4f} divJ={so['division_jaccard']:.4f} score={so['score']:.4f}")
print(f"RESMI_TAVAN={so['score']:.4f} kenar_uyusuyor={_ok}")
print(f"--- konfig: n*={CFG_NSTAR} gate={CFG_GATE} mnn={CFG_MNN} fill={CFG_FILL} "
      f"gap={CFG_GAP} min_len={CFG_MINLEN} ---")
print(f"bizim core={mc['score']:.4f} adj={mc['adj_edge_jaccard']:.4f}")
print(f"resmi core={oc['score']:.4f} adj={oc['adj_edge_jaccard']:.4f} divJ={oc['division_jaccard']:.4f} "
      f"44b6={o4['score']:.4f} 6bba={o6['score']:.4f}")
print(f"adj_farki={abs(mc['adj_edge_jaccard']-oc['adj_edge_jaccard']):.4f}")
print(f"--- division taramasi (RESMI) ---")
for tag, (s, a, b) in div_res.items():
    print(f"{tag} | core={s['score']:.4f} 44b6={a['score']:.4f} 6bba={b['score']:.4f} "
          f"adj={s['adj_edge_jaccard']:.4f} divJ={s['division_jaccard']:.4f} "
          f"dtp={s['division_tp']} dfp={s['division_fp']} dfn={s['division_fn']} "
          f"edgeJ={s['edge_jaccard']:.4f}")
print(f"DIV_BEST={DBEST}")
print(f"--- dataset basina (resmi, div={DBEST}) ---")
for n in ALL:
    o = off_rows[n]
    print(f"{n} Np={o['num_pred_nodes']} ratio={o['total_node_ratio']:+.3f} "
          f"edgeJ={o['edge_jaccard']:.4f} adj={o['adj_edge_jaccard']:.4f} rec={o['node_recall']:.4f} "
          f"dtp={o['division_tp']} dfp={o['division_fp']} dfn={o['division_fn']}")
print("### V11-ADIM4-RAPOR-BITIS")
print("#" * 100)
