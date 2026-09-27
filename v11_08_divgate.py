# =============================================================================
# V11 ADIM 8 - OGRENILEN BOLUNME KAPISI  (CPU, internet ACIK, ~40-60 dk)
# =============================================================================
# ADIM 4 BULGUSU: geometrik bolunme dedektoru hassasiyeti 1/72 (dtp=1 dfp=71),
#   her yanlis fork IKI kez odetiyor (division FP + kenar FP). "Masada 0.1 var"
#   diye park edilmisti.
# ADIM 6 BULGUSU: OGRENILEN kenar siniflandiricisi calisiyor (K_NN=16, p_min=0.55,
#   core +0.0094 iki foldda). Ayni prensip bolunmeye de uygulanabilir mi?
# ADIM 7d: bolunmece-zengin dokum (39 dataset, 99 GT bolunme, val haric) artik
#   EGITIM icin yeterli veri sagliyor. Held-out (val_core+val_public) hala ince
#   (8 bolunme) -> sonuc GURULTULU olacak, TUR 2'nin dersi gibi asiri yorumlama.
#
# TASARIM (Adim 6 ile AYNI disiplin):
#   POZ: ebeveyn eslesmis VE GT'de gercek bolunme (>=2 cocuk) VE iki aday da
#        GT'nin iki kizina eslesiyor
#   NEG: ebeveyn eslesmis VE GT'de >=1 cocugu var (devam ediyor) AMA bu ikili
#        GT bolunmesiyle eslesmiyor -> promote edilirse FP olur
#   HARIC: ebeveyn eslesmemis (metrikte BEDAVA) VEYA GT'de hic cocugu yok
#          (belirsiz kenar durum, gurultu enjekte etmemek icin disari birakildi)
#
# ADAY URETIMI: Adim 6c linker'i (K_NN=16,p_min=0.55) ile birincil (parent->c1)
#   baglantiyi kur; t+1'deki SAHIPSIZ (eslesmemis) node'lar, o parent'in
#   tahmini konumuna R_SEARCH icindeyse IKINCI KIZ ADAYI olur.
#
# KARAR: sadece RESMI SKORLAYICI (tracksdata) ile - Adim 4'teki gibi. Kendi
#   yaklasik metrigimiz division'da GUVENILMEZ (Adim 4'te KANITLANDI).
# =============================================================================
import sys, subprocess
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
_pip("scikit-learn")
_pip("git+https://github.com/royerlab/tracksdata@main")
_pip("--no-deps", "git+https://github.com/royerlab/kaggle-cell-tracking-competition.git")

# numpy/scipy'yi geri sabitle (henuz hicbiri import EDILMEDI -> ABI kirilmasi yok)
_pin = [f"{n}=={v}" for n, v in (("numpy", _NPV), ("scipy", _SPV)) if v]
if _pin:
    _pip("--no-deps", *_pin)
print(f"[bootstrap] kurulum sonrasi numpy={_v('numpy')} scipy={_v('scipy')} "
      f"zarr={_v('zarr')} tracksdata={_v('tracksdata')} sklearn={_v('scikit-learn')}",
      flush=True)

import os, glob, json, time, math, itertools
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

COMP  = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
TRAIN = os.path.join(COMP, "train")
WORK  = "/kaggle/working"

# cand_v3_*.npz nerede? (notebook output'unu input olarak ekledigin klasor)
CAND_TAG  = "v3"        # TEK yerden yonetilir (glob + dosya adi ayni etiketi kullanir)
CAND_DIRS = sorted(glob.glob(f"/kaggle/input/**/cand_{CAND_TAG}_*.npz", recursive=True))
CAND_DIR  = os.path.dirname(CAND_DIRS[0]) if CAND_DIRS else "/kaggle/working"

SCALE        = (1.625, 0.40625, 0.40625)
MAX_MATCH_UM = 7.0
ALPHA        = 0.1
DIV_W        = 0.1

CORE = {'44b6_341df25f','44b6_3bb3690f','44b6_c771cb04','44b6_8f9ecab4',
        '6bba_2540cd90','6bba_3a1849c2','6bba_67ebd073','6bba_57b7cc1e'}
VAL_PUBLIC_NAMES = {'44b6_0113de3b', '44b6_0b24845f', '6bba_05b6850b', '6bba_05db0fb1'}

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
    d = np.load(os.path.join(CAND_DIR, f"cand_{CAND_TAG}_{name}.npz"))
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
# ADIM 6b - KENAR SINIFLANDIRICISI, VEKTOREL OZELLIK  (CPU, internet ACIK)
# =============================================================================
# 0.947 temiz notebook'u (reyhanksatria) kazancin ogrenilen linking'den geldigini
# gosteriyor. Kendi surumumuz, KAYIP BUTCESINE ve METRIGIN TANIMINA bagli:
#
# ETIKETLER metrigi BIREBIR yansitir:
#   POZ : iki ucu eslesmis ve (a,b) GT kenari                     -> TP
#   NEG : pred_valid (kaynak GT'de cocuklu / hedef GT'de ebeveynli)
#         AMA TP degil                                          -> metrikte FP
#   ETIKETSIZ: iki uc da eslesmemis -> metrikte BEDAVA -> EGITIME GIRMEZ
#   (TrackerBrain her seyi egitime sokuyordu; kenarlarin ~%96'si metrige gorunmez)
#
# OZELLIKLER mesafeyi degil BELIRSIZLIGI olcer (TrackerBrain'in 5 geometrik
#   ozelligi mesafe kapisinin monoton fonksiyonuydu, bilgi eklemiyordu):
#   sira, oran testi, karsilikli-en-yakin, yerel yogunluk, dedektor skoru.
#
# EGITIM: v3tr (36 dataset, val HARIC)   TEST: val_core (8)  -> SIZINTI YOK
# KARAR : greedy-geometrik (Adim 3c, core 0.7197) ile IKI FOLDDA kiyasla.
# =============================================================================
import glob as _glob
from sklearn.ensemble import HistGradientBoostingClassifier

GATE_CAND = 12.0            # aday kenar kapisi (um) - genis tut, siniflandirici elesin
DENS_R    = 15.0
SC        = np.asarray(SCALE)
N_STAR    = 0.95
CFG_GAP   = dict(max_gap=6, gate_um=8.0)
CFG_MINLEN = 3

def find_npz(tag, name):
    c = _glob.glob(f"/kaggle/input/**/cand_{tag}_{name}.npz", recursive=True)
    return c[0] if c else None

def all_names(tag):
    return sorted(os.path.basename(p)[len(f"cand_{tag}_"):-4]
                  for p in _glob.glob(f"/kaggle/input/**/cand_{tag}_*.npz", recursive=True))

TRAIN_NAMES = all_names("v3div")            # Adim 7d: 39 dataset, 99 GT bolunme
VAL_NAMES   = all_names("v3")                # val_core(8)+val_public(4) TUMU -> 8 GT bolunme
print(f"egitim: {len(TRAIN_NAMES)} dataset (v3div, bolunmece-zengin) | "
      f"degerlendirme: {len(VAL_NAMES)} dataset (v3, val_core+val_public)")
assert TRAIN_NAMES and VAL_NAMES, "aday dokumleri eksik - v11-bolunme-gpu + v11-aday-v3 ekle"
assert not (set(TRAIN_NAMES) & (CORE | set(VAL_PUBLIC_NAMES))), "SIZINTI: egitim setinde val var"


def load_nodes(tag, name):
    """Adim 3c konfigiyle node'lar: tek gecis NMS + n*=0.95 (GERCEK est)."""
    d = np.load(find_npz(tag, name))
    T = int(d["T"]); t = d["t"].astype(np.int64)
    coords = np.stack([d["z"], d["y"], d["x"]], 1).astype(np.float64)
    score = d["score"].astype(np.float64)
    fstart = np.searchsorted(t, np.arange(T + 1))
    acc = []
    for tv in range(T):
        sl = slice(fstart[tv], fstart[tv + 1])
        c, s = coords[sl], score[sl]
        if len(c) == 0:
            acc.append((np.zeros((0, 3)), np.zeros(0))); continue
        pr, dd = frame_pairs(c, R_LIST[0])
        b, st = build_csr(pr, dd, R_LIST[0], len(c))
        k = nms_frame(np.argsort(-s, kind='stable'), b, st, np.ones(len(c), bool))
        acc.append((c[k], s[k]))
    allsc = np.sort(np.concatenate([s for _, s in acc]))[::-1]
    est = float(d["est_nodes"])
    K = max(1, min(int(round(N_STAR * est)), len(allsc)))
    thr = float(allsc[K - 1])
    frames = {tv: (c[s >= thr], s[s >= thr]) for tv, (c, s) in enumerate(acc) if (s >= thr).any()}
    return T, frames


K_NN = 16   # v1'de 8 idi; 6bba (yogun) foldunda kayip supheli -> buyutup olc
TRUNC_STATS = {"n_i": 0, "n_i_full": 0}   # kesilme tanisi: k=K_NN dolu mu?

def frame_features(P0, P1, s0, s1):
    """VEKTOREL: k-NN sorgusu adaylari ZATEN SIRALI verir -> rank_ij bedava,
    geri kalan her ozellik numpy ile. Python cift dongusu YOK.
    Egitim ve test AYNI fonksiyonu ayni sekilde cagirir -> maske tutarsizligi
    YAPISAL OLARAK imkansiz (v1'de rank_ij boyle bir hataya dusmustu)."""
    n0, n1 = len(P0), len(P1)
    if n0 == 0 or n1 == 0:
        return np.zeros((0, 2), np.int64), np.zeros((0, 17))
    t1 = cKDTree(P1)
    shift = np.zeros(3)
    for _ in range(4):
        dd, ii = t1.query(P0 + shift, k=1)
        m = dd < 10.0
        if m.sum() < 8: break
        step = np.median(P1[ii[m]] - (P0[m] + shift), axis=0); shift += step
        if np.linalg.norm(step) < 1e-3: break
    Q = P0 + shift
    t0 = cKDTree(Q)
    k1, k0 = min(K_NN, n1), min(K_NN, n0)
    D01, I01 = t1.query(Q, k=k1, distance_upper_bound=GATE_CAND)
    D10, I10 = t0.query(P1, k=k0, distance_upper_bound=GATE_CAND)
    if k1 == 1: D01, I01 = D01[:, None], I01[:, None]
    if k0 == 1: D10, I10 = D10[:, None], I10[:, None]
    # KESILME TANISI: k1 slotunun TAMAMI dolu olan node sayisi -> daha da yakini
    # olabilirdi ama K_NN yuzunden goremedik. n_i_full/n_i yuksekse K_NN yetersiz.
    if k1 == K_NN:
        TRUNC_STATS["n_i"] += n0
        TRUNC_STATS["n_i_full"] += int(np.isfinite(D01[:, -1]).sum())
    ii, rr = np.nonzero(np.isfinite(D01))                 # aday ciftleri
    if len(ii) == 0:
        return np.zeros((0, 2), np.int64), np.zeros((0, 17))
    jj = I01[ii, rr]
    d = D01[ii, rr]
    v = np.abs(P1[jj] - Q[ii])
    rank_ij = rr.astype(np.float64)
    hit = (I10[jj] == ii[:, None])                        # i, j'nin k-NN listesinde mi
    rank_ji = np.where(hit.any(1), hit.argmax(1), k0).astype(np.float64)
    mutual = ((rank_ij == 0) & (rank_ji == 0)).astype(np.float64)
    d_i1 = D01[ii, 0]
    d_i2 = D01[ii, 1] if k1 > 1 else np.full(len(ii), np.inf)
    d_j1 = D10[jj, 0]
    d_j2 = D10[jj, 1] if k0 > 1 else np.full(len(ii), np.inf)
    fb = d + 1e-3 + GATE_CAND                             # 2. komsu yoksa: belirsizlik YOK
    ratio_i = d / np.where(np.isfinite(d_i2) & (d_i2 > 0), d_i2, fb)
    ratio_j = d / np.where(np.isfinite(d_j2) & (d_j2 > 0), d_j2, fb)
    dens0 = t0.query_ball_point(Q, DENS_R, return_length=True) - 1
    dens1 = t1.query_ball_point(P1, DENS_R, return_length=True) - 1
    sh = np.full(len(ii), float(np.linalg.norm(shift)))
    X = np.column_stack([d, v[:, 0], v[:, 1], v[:, 2], rank_ij, rank_ji, mutual,
                         ratio_i, ratio_j, d - d_i1, np.where(np.isfinite(d_j1), d - d_j1, 0.0),
                         s0[ii], s1[jj], np.minimum(s0[ii], s1[jj]),
                         dens0[ii], dens1[jj], sh])
    return np.column_stack([ii, jj]).astype(np.int64), X

FEAT_NAMES = ["d", "|dz|", "|dy|", "|dx|", "rank_ij", "rank_ji", "mutual", "ratio_i",
              "ratio_j", "d-d1_i", "d-d1_j", "s_i", "s_j", "s_min", "dens_i", "dens_j", "shift"]
def dataset_edges(tag, name, with_labels=True):
    """Tum aday kenarlar + ozellik (+ metrik-uyumlu etiket). Maske yok: vektorel hesap
    zaten hizli, ayni kod egitimde ve testte ayni sekilde calisir."""
    t_start = time.time()
    T, frames = load_nodes(tag, name)
    offs, nodes, o = {}, [], 0
    for tv in range(T):
        if tv in frames:
            c, s_ = frames[tv]; offs[tv] = o
            nodes.append(np.column_stack([np.full(len(c), tv, np.float64), c])); o += len(c)
    nodes = np.vstack(nodes) if nodes else np.zeros((0, 4))
    t_feat = time.time()
    E, X = [], []
    for tv in range(T - 1):
        if tv not in frames or tv + 1 not in frames: continue
        (c0, s0), (c1, s1) = frames[tv], frames[tv + 1]
        r, f = frame_features(c0 * SC, c1 * SC, s0, s1)
        if len(r):
            E.append(np.column_stack([offs[tv] + r[:, 0], offs[tv + 1] + r[:, 1]])); X.append(f)
    E = np.vstack(E) if E else np.zeros((0, 2), np.int64)
    X = np.vstack(X) if X else np.zeros((0, 17))
    feat_s = time.time() - t_feat
    y = None
    if with_labels:
        gt = read_gt(os.path.join(TRAIN, name + ".geff"))
        m = match_nodes(nodes, gt)
        ge = gt["edges"]
        outd = np.zeros(gt["n_nodes"], np.int64); ind = np.zeros(gt["n_nodes"], np.int64)
        if len(ge):
            np.add.at(outd, ge[:, 0], 1); np.add.at(ind, ge[:, 1], 1)
        span = int(gt["n_nodes"]) + 2
        gkeys = np.sort(ge[:, 0].astype(np.int64) * span + ge[:, 1]) if len(ge) else np.zeros(0, np.int64)
        ms, mt = m[E[:, 0]], m[E[:, 1]]
        both = (ms >= 0) & (mt >= 0)
        pk = np.where(both, ms.astype(np.int64) * span + mt.astype(np.int64), -1)
        pos = np.clip(np.searchsorted(gkeys, pk), 0, max(0, len(gkeys) - 1))
        tp = both & (len(gkeys) > 0) & (gkeys[pos] == pk) if len(gkeys) else np.zeros(len(E), bool)
        valid = ((ms >= 0) & (outd[np.clip(ms, 0, None)] > 0)) | ((mt >= 0) & (ind[np.clip(mt, 0, None)] > 0))
        y = np.where(tp, 1, np.where(valid, 0, -1))      # -1 = ETIKETSIZ (bedava)
    return nodes, E, X, y, T, feat_s


# RESMI SKORLAYICI import'lari (v04'ten sed ile cikarilirken KESILMISTI - duzeltildi)
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
# EDGE CLASSIFIER YUKLE  (Adim 6c - BIRINCIL linking icin, degistirilmiyor)
# =============================================================================
import pickle
_clf_paths = _glob.glob("/kaggle/input/**/v11_edge_clf.pkl", recursive=True)
assert _clf_paths, "v11_edge_clf.pkl bulunamadi - v11-kenar-clf-knn16 ciktisini ekle"
with open(_clf_paths[0], "rb") as _fh:
    _CLF_META = pickle.load(_fh)
EDGE_CLF = _CLF_META["model"]
P_MIN_LINK = 0.55
print(f"kenar siniflandirici yuklendi: k_nn={_CLF_META['k_nn']} gate={_CLF_META['gate']} "
      f"(birincil linking icin, DEGISTIRILMIYOR)")

R_SEARCH_DIV = 20.0   # ikinci kiz adayi arama yaricapi (um) - genis, siniflandirici elesin
DIV_FEAT_NAMES = ["d_c1", "d_cand", "ratio_len", "cos_angle", "s_p", "s_c1", "s_cand",
                  "dens_p", "n_competitors", "edge_prob_c1"]


def dataset_scores(tag, name):
    """dataset_edges'in bilmedigi tek sey: GLOBAL skor dizisi (node basina).
    Frame offsetleriyle AYNI sirada -> nodes[i] <-> scores[i]."""
    T, frames = load_nodes(tag, name)
    offs, scores, o = {}, [], 0
    for tv in range(T):
        if tv in frames:
            c, s_ = frames[tv]; offs[tv] = o
            scores.append(s_); o += len(c)
    return np.concatenate(scores) if scores else np.zeros(0), T


def division_candidates(nodes, scores, T, assign, edge_prob_by_pair):
    """Her BIRINCIL baglantili ebeveyn (p->c1) icin, t+1'deki SAHIPSIZ node'lardan
    R_SEARCH_DIV icinde olanlari IKINCI KIZ ADAYI olarak isaretler."""
    Pphys = nodes[:, 1:] * SC
    by_t = {}
    for i in range(len(nodes)):
        by_t.setdefault(int(nodes[i, 0]), []).append(i)
    claimed = set(assign.values())
    rows, X = [], []
    for t in range(T - 1):
        idx_t0 = by_t.get(t, [])
        idx_t1 = by_t.get(t + 1, [])
        orphans = [i for i in idx_t1 if i not in claimed]
        if not orphans or not idx_t0:
            continue
        otree = cKDTree(Pphys[orphans])
        dtree = cKDTree(Pphys[idx_t0])          # GERCEK yerel yogunluk icin (TUM t-frame node'lari)
        for p in idx_t0:
            c1 = assign.get(p)
            if c1 is None:
                continue
            nb = otree.query_ball_point(Pphys[p], R_SEARCH_DIV)
            if not nb:
                continue
            d_c1 = float(np.linalg.norm(Pphys[c1] - Pphys[p]))
            vec_c1 = Pphys[c1] - Pphys[p]
            n_comp = len(nb)                                          # aday RAKIP sayisi
            dens_p = len(dtree.query_ball_point(Pphys[p], DENS_R)) - 1  # yerel hucre yogunlugu
            ep = edge_prob_by_pair.get((p, c1), 0.5)
            for oi in nb:
                cand = orphans[oi]
                if cand == c1:
                    continue
                vec_cand = Pphys[cand] - Pphys[p]
                d_cand = float(np.linalg.norm(vec_cand))
                if d_cand < 1e-6:
                    continue
                cosang = (float(np.dot(vec_c1, vec_cand) / (d_c1 * d_cand))
                         if d_c1 > 1e-9 else 0.0)
                ratio_len = min(d_c1, d_cand) / max(d_c1, d_cand, 1e-9)
                rows.append((p, c1, cand))
                X.append([d_c1, d_cand, ratio_len, cosang, scores[p], scores[c1],
                          scores[cand], float(dens_p), float(n_comp), ep])
    return (np.array(rows, np.int64).reshape(-1, 3),
            np.array(X, np.float64).reshape(-1, len(DIV_FEAT_NAMES)))


def build_primary(nodes, E, X):
    """TUM aday kenarlari EDGE_CLF ile puanla, greedy 1-1 -> assign + edge_prob."""
    if len(E) == 0:
        return {}, {}
    P = EDGE_CLF.predict_proba(X)[:, 1]
    order = np.argsort(-P)
    used_s, used_t, assign, eprob = set(), set(), {}, {}
    for k in order:
        if P[k] < P_MIN_LINK:
            break
        s, t = int(E[k, 0]), int(E[k, 1])
        if s in used_s or t in used_t:
            continue
        used_s.add(s); used_t.add(t)
        assign[s] = t; eprob[(s, t)] = float(P[k])
    return assign, eprob


def prepare_dataset(tag, name, with_labels):
    """nodes, primer-baglanti, bolunme adaylari (+ egitimde ETIKET) - HER SEYI BIR KEZ hesaplar."""
    nodes, E, X, _, T, _ = dataset_edges(tag, name, with_labels=False)
    scores, _ = dataset_scores(tag, name)
    assign, eprob = build_primary(nodes, E, X)
    rows, DX = division_candidates(nodes, scores, T, assign, eprob)
    y = None
    if with_labels and len(rows):
        gt = read_gt(os.path.join(TRAIN, name + ".geff"))
        m = match_nodes(nodes, gt)
        child_map = {}
        for s_, t_ in gt["edges"].tolist():
            child_map.setdefault(s_, []).append(t_)
        y = np.full(len(rows), -1, np.int64)
        for i, (p, c1, cand) in enumerate(rows.tolist()):
            mp = int(m[p])
            if mp < 0:
                continue                                    # ebeveyn eslesmemis -> BEDAVA
            ch = child_map.get(mp, [])
            if len(ch) >= 2:
                mc1, mcand = int(m[c1]), int(m[cand])
                if mc1 in ch and mcand in ch and mc1 != mcand and mc1 >= 0 and mcand >= 0:
                    y[i] = 1                                 # GERCEK bolunme
                else:
                    y[i] = 0                                 # yanlis esleme -> promote FP olur
            elif len(ch) >= 1:
                y[i] = 0                                     # devam ediyor, bolunmuyor -> FP
            # len(ch)==0: belirsiz yaprak -> y kalir -1 (HARIC, gurultu enjekte etme)
    return dict(nodes=nodes, T=T, assign=assign, rows=rows, X=DX, y=y)


# =============================================================================
# 1) EGITIM VERISI  (v3div: 39 dataset, 99 GT bolunme)
# =============================================================================
print("\n[egitim verisi hazirlaniyor]")
Xs, ys = [], []
n_pos_tot = n_neg_tot = n_cand_tot = 0
for n in TRAIN_NAMES:
    d = prepare_dataset("v3div", n, with_labels=True)
    if d["y"] is None or len(d["y"]) == 0:
        continue
    keep = d["y"] >= 0
    Xs.append(d["X"][keep]); ys.append(d["y"][keep])
    n_pos_tot += int((d["y"] == 1).sum()); n_neg_tot += int((d["y"] == 0).sum())
    n_cand_tot += len(d["y"])
    print(f"  {n:22s} aday={len(d['rows']):6d} poz={int((d['y']==1).sum()):3d} "
          f"neg={int((d['y']==0).sum()):4d}  ({elapsed():.1f}dk)", flush=True)

Xtr = np.vstack(Xs) if Xs else np.zeros((0, len(DIV_FEAT_NAMES)))
ytr = np.concatenate(ys) if ys else np.zeros(0, np.int64)
print(f"\nTOPLAM: aday={n_cand_tot}  etiketli={len(ytr)}  poz={n_pos_tot}  neg={n_neg_tot}  "
      f"poz_orani=%{100*ytr.mean() if len(ytr) else 0:.1f}")
assert n_pos_tot >= 20, f"pozitif ornek COK AZ ({n_pos_tot}) - guvenilir egitim icin yetersiz"

DIV_CLF = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_leaf_nodes=15,
                                         l2_regularization=1.0, random_state=0)
DIV_CLF.fit(Xtr, ytr)

from sklearn.metrics import roc_auc_score
print("\n  tek-ozellik ayirt ediciligi (AUC, egitim uzerinde, 0.5=bilgisiz):")
for k, nm in enumerate(DIV_FEAT_NAMES):
    try:
        a = roc_auc_score(ytr, Xtr[:, k]); a = max(a, 1 - a)
    except Exception:
        a = float('nan')
    print(f"    {nm:14s} {a:.3f}")


# =============================================================================
# 2) DEGERLENDIRME  (v3: val_core+val_public, RESMI SKORLAYICI)
# =============================================================================
print("\n[degerlendirme verisi hazirlaniyor]")
VAL = {n: prepare_dataset("v3", n, with_labels=False) for n in VAL_NAMES}
for n, d in VAL.items():
    print(f"  {n:22s} node={len(d['nodes']):6d} birincil={len(d['assign']):6d} "
          f"aday_bolunme={len(d['rows']):5d}  ({elapsed():.1f}dk)", flush=True)

G44 = [n for n in VAL_NAMES if n.startswith("44b6")]
G6B = [n for n in VAL_NAMES if n.startswith("6bba")]


def graph_at_threshold(d, p_min_div):
    """Her ebeveyn icin EN YUKSEK olasilikli tek adayi (esik ustundeyse) ekle
    -> canonicalize()'in keyfi kenar-id sirasina GUVENMIYORUZ, en iyisini BIZ seciyoruz."""
    nodes, assign, rows, X = d["nodes"], d["assign"], d["rows"], d["X"]
    edges = [[p, c] for p, c in assign.items()]
    if len(rows):
        P = DIV_CLF.predict_proba(X)[:, 1]
        best = {}          # parent -> (prob, cand)
        for (p, c1, cand), pr in zip(rows.tolist(), P.tolist()):
            if pr < p_min_div:
                continue
            if p not in best or pr > best[p][0]:
                best[p] = (pr, cand)
        for p, (pr, cand) in best.items():
            edges.append([p, cand])
    edges = np.array(edges, np.int64).reshape(-1, 2) if edges else np.zeros((0, 2), np.int64)
    nd, ed = gap_close(nodes, edges, **CFG_GAP)
    return filter_short(nd, ed, CFG_MINLEN)


print("\n" + "=" * 100)
print("RESMI SKORLAYICI ILE ESIK TARAMASI  (0.1*divJ dogrudan LB'ye ekleniyor)")
print("=" * 100)
BASE_ROWS = {n: official_row(*graph_at_threshold(VAL[n], 1.01), n,
                             read_gt(os.path.join(TRAIN, n + ".geff"))["est_nodes"])
             for n in VAL_NAMES}      # p_min_div=1.01 -> HICBIR aday gecmez = division KAPALI
base = off_summarise([BASE_ROWS[n] for n in VAL_NAMES])
print(f"  BASE (division KAPALI, mevcut v40)  score={base['score']:.4f} "
      f"adj={base['adj_edge_jaccard']:.4f} divJ={base['division_jaccard']:.4f} "
      f"edgeJ={base['edge_jaccard']:.4f}")

RES = {}
for p_min_div in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
    rows = {n: official_row(*graph_at_threshold(VAL[n], p_min_div), n,
                            read_gt(os.path.join(TRAIN, n + ".geff"))["est_nodes"])
            for n in VAL_NAMES}
    s = off_summarise([rows[n] for n in VAL_NAMES])
    a = off_summarise([rows[n] for n in G44]) if G44 else dict(score=float('nan'))
    b = off_summarise([rows[n] for n in G6B]) if G6B else dict(score=float('nan'))
    RES[p_min_div] = (s, a, b)
    print(f"  p_min_div={p_min_div:.2f}  score={s['score']:.4f} ({s['score']-base['score']:+.4f})  "
          f"44b6={a['score']:.4f} 6bba={b['score']:.4f}  "
          f"adj={s['adj_edge_jaccard']:.4f} divJ={s['division_jaccard']:.4f} "
          f"dtp={s['division_tp']} dfp={s['division_fp']} dfn={s['division_fn']}  "
          f"edgeJ={s['edge_jaccard']:.4f}  ({elapsed():.1f}dk)", flush=True)

best_thr = max(RES, key=lambda k: RES[k][0]['score'])
bs, ba, bb = RES[best_thr]
print(f"\n>>> EN IYI p_min_div={best_thr}  score={bs['score']:.4f}  "
      f"kazanc={bs['score']-base['score']:+.4f}  (BASE={base['score']:.4f})")
print(f"    44b6={ba['score']:.4f}  6bba={bb['score']:.4f}   "
      f"({'IKI FOLD DA IYI' if (ba['score']==ba['score'] and bb['score']==bb['score'] and bs['score']>base['score']) else 'DIKKAT: tek fold veya gurultulu (8 bolunme - AZ ORNEK)'})")

print("\n" + "#" * 100)
print("### V11-ADIM8-RAPOR-BASLANGIC")
print(f"sure_dk={elapsed():.1f} egitim_dataset={len(TRAIN_NAMES)} egitim_bolunme={n_pos_tot} "
      f"degerlendirme_dataset={len(VAL_NAMES)}")
print(f"BASE score={base['score']:.4f} adj={base['adj_edge_jaccard']:.4f} divJ={base['division_jaccard']:.4f}")
for p, (s, a, b) in RES.items():
    print(f"p_min_div={p} score={s['score']:.4f} 44b6={a['score']:.4f} 6bba={b['score']:.4f} "
          f"adj={s['adj_edge_jaccard']:.4f} divJ={s['division_jaccard']:.4f} "
          f"dtp={s['division_tp']} dfp={s['division_fp']} dfn={s['division_fn']} edgeJ={s['edge_jaccard']:.4f}")
print(f"EN_IYI p_min_div={best_thr} score={bs['score']:.4f} kazanc={bs['score']-base['score']:+.4f}")
if bs['score'] > base['score']:
    import pickle as _pk2
    with open(os.path.join(WORK, "v11_div_clf.pkl"), "wb") as _fh2:
        _pk2.dump(dict(model=DIV_CLF, feat_names=DIV_FEAT_NAMES, r_search=R_SEARCH_DIV,
                       p_min_div=best_thr, p_min_link=P_MIN_LINK), _fh2)
    print(f"model kaydedildi: v11_div_clf.pkl (p_min_div={best_thr})")
else:
    print("KAZANC YOK -> model kaydedilmedi, division KAPALI kalmali")
print("### V11-ADIM8-RAPOR-BITIS")
print("#" * 100)
