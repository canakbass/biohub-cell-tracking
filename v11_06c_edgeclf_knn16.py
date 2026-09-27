# =============================================================================
# V11 ADIM 3c - KONFIG YENIDEN TARAMASI, ADIM 5 DETEKTORU (CPU, ~25 dk)
# =============================================================================
# ADIM 2 BULGUSU: linker, tespitin sagladiginin ~%20'sini kaybediyor
#   (edgeJ 0.606 vs recall^2 0.761).  2540cd90'da recall 0.996 iken edgeJ 0.805.
# Bu script NEDEN kaybettigini SAYIYOR: her GT kenari icin iki ucu da eslesmisse,
# bizim grafta o kenar var mi, yoksa neden yok?
#
# Girdi : yarisma verisi (GT icin) + cand_v3_*.npz  (Adim 1b notebook cikti'si)
# Cikti : sweep tablolari + RAPOR blogu
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

TRAIN_NAMES = all_names("v3tr")
VAL_NAMES   = [n for n in all_names("v3") if n in CORE]
print(f"egitim: {len(TRAIN_NAMES)} dataset (v3tr) | test: {len(VAL_NAMES)} val_core (v3)")
assert TRAIN_NAMES and VAL_NAMES, "aday dokumleri eksik - v11-aday-egitim + v11-aday-v3 ekle"
assert not (set(TRAIN_NAMES) & set(CORE)), "SIZINTI: egitim setinde val_core var"


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


# =============================================================================
# 1) EGITIM VERISI
# =============================================================================
print("\n[egitim verisi]")
Xs, ys, cols = [], [], []
for n in TRAIN_NAMES:
    _, E, X, y, _, fs = dataset_edges("v3tr", n)
    keep = y >= 0
    Xs.append(X[keep]); ys.append(y[keep]); cols += [n[:4]] * int(keep.sum())
    print(f"  {n:22s} aday={len(E):7d} etiketli={int(keep.sum()):6d} "
          f"poz={int((y==1).sum()):5d} neg={int((y==0).sum()):5d} etiketsiz={int((y==-1).sum()):7d}"
          f"  ozellik={fs:.1f}s  ({elapsed():.1f}dk)", flush=True)
Xtr, ytr = np.vstack(Xs), np.concatenate(ys)
cols = np.array(cols)
print(f"  TOPLAM etiketli {len(ytr)}  poz orani %{100*ytr.mean():.1f}")

clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31,
                                     l2_regularization=1.0, random_state=0)
clf.fit(Xtr, ytr)
import pickle, sklearn
with open(os.path.join(WORK, "v11_edge_clf.pkl"), "wb") as fh:
    pickle.dump(dict(model=clf, feat_names=FEAT_NAMES, gate=GATE_CAND, k_nn=K_NN,
                     dens_r=DENS_R, sklearn=sklearn.__version__), fh)
print(f"  model kaydedildi: v11_edge_clf.pkl (sklearn {sklearn.__version__})")
# ozellik onemi (permutation yerine hizli: tek-ozellik AUC)
from sklearn.metrics import roc_auc_score
print("\n  tek-ozellik ayirt ediciligi (AUC, 0.5=bilgisiz):")
for k, nm in enumerate(FEAT_NAMES):
    try:
        a = roc_auc_score(ytr, Xtr[:, k]); a = max(a, 1 - a)
    except Exception:
        a = float('nan')
    print(f"    {nm:10s} {a:.3f}")

# kolonlar arasi genelleme: 44b6 ile egit -> 6bba'da AUC, ve tersi
for a_col, b_col in (("44b6", "6bba"), ("6bba", "44b6")):
    m_a, m_b = cols == a_col, cols == b_col
    if m_a.sum() > 100 and m_b.sum() > 100:
        c2 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=0)
        c2.fit(Xtr[m_a], ytr[m_a])
        print(f"  {a_col} ile egit -> {b_col} AUC = {roc_auc_score(ytr[m_b], c2.predict_proba(Xtr[m_b])[:,1]):.4f}")


# =============================================================================
# 2) SINIFLANDIRICI ILE LINKING  vs  GREEDY-GEOMETRIK (Adim 3c)
# =============================================================================
def link_by_prob(nodes, E, P, p_min):
    """Olasiliga gore greedy 1-1: en yuksek P'den basla, iki uc bosta ise kabul."""
    order = np.argsort(-P)
    used_s, used_t, keep = set(), set(), []
    for k in order:
        if P[k] < p_min: break
        s, t = int(E[k, 0]), int(E[k, 1])
        if s in used_s or t in used_t: continue
        used_s.add(s); used_t.add(t); keep.append(k)
    return E[keep] if keep else np.zeros((0, 2), np.int64)

tr = TRUNC_STATS
print(f"\n[K_NN KESILME TANISI] K_NN={K_NN}: "
      f"{tr['n_i_full']}/{max(1,tr['n_i'])} node (%{100*tr['n_i_full']/max(1,tr['n_i']):.2f}) "
      f"k={K_NN} slotunun TAMAMINI doldurdu -> bunlarda gercek komsu sayisi >= {K_NN} olabilir")
print(f"  (bu oran YUKSEKSE K_NN dahada buyutulmeli; DUSUKSE mesele K_NN degil)")

print("\n[val_core degerlendirmesi]")
VAL = {}
for n in VAL_NAMES:
    nodes, E, X, _, T, fs = dataset_edges("v3", n, with_labels=False)
    print(f"  {n:22s} aday={len(E):7d} ozellik={fs:.1f}s -> 199 dataset'te ~{199*fs/60:.1f} dk", flush=True)
    VAL[n] = (nodes, E, clf.predict_proba(X)[:, 1] if len(X) else np.zeros(0), T)

G44 = [n for n in VAL_NAMES if n.startswith("44b6")]
G6B = [n for n in VAL_NAMES if n.startswith("6bba")]
def score_cfg(p_min):
    rows = {}
    for n, (nodes, E, P, T) in VAL.items():
        ed = link_by_prob(nodes, E, P, p_min)
        nd, ed2 = gap_close(nodes, ed, **CFG_GAP)
        nd, ed2 = filter_short(nd, ed2, CFG_MINLEN)
        rows[n] = evaluate_dataset(nd, ed2, read_gt(os.path.join(TRAIN, n + ".geff")))
    s = summarise([rows[n] for n in VAL_NAMES])
    a = summarise([rows[n] for n in G44]); b = summarise([rows[n] for n in G6B])
    return s, a, b

# REFERANS: greedy-geometrik linker (Adim 3c konfigi) AYNI node setleriyle.
#   Elle 0.7197 yazmak yanlis olurdu: o skor ikili-arama esigiyle uretildi,
#   burada tek-gecis NMS kullaniliyor. Tek degisken LINKING olmali.
LINK3C = dict(gate=8.0, mnn_ratio=0.80, fill_gate=0.0, local_motion=False)
_ref = {}
for n in VAL_NAMES:
    T_, fr_ = load_nodes("v3", n)
    fr_c = {t: c for t, (c, _s) in fr_.items()}
    nd, ed = build_graph(fr_c, T_, LINK3C, None, CFG_GAP, CFG_MINLEN)
    _ref[n] = evaluate_dataset(nd, ed, read_gt(os.path.join(TRAIN, n + ".geff")))
BASE = dict(core=summarise([_ref[n] for n in VAL_NAMES])['score'],
            g44=summarise([_ref[n] for n in G44])['score'],
            g6b=summarise([_ref[n] for n in G6B])['score'])
print(f"  (Adim 3c raporu 0.7197 demisti; burada ayni node'larla yeniden olculdu)")
print(f"  REFERANS greedy-geometrik (Adim 3c): core={BASE['core']:.4f} "
      f"44b6={BASE['g44']:.4f} 6bba={BASE['g6b']:.4f}")
RES = {}
for p_min in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:   # v2: coku 0.70 sonrasi -> ince tara
    s, a, b = score_cfg(p_min)
    RES[p_min] = (s, a, b)
    both = (a['score'] > BASE['g44']) and (b['score'] > BASE['g6b'])
    print(f"  p_min={p_min:.2f}  core={s['score']:.4f} ({s['score']-BASE['core']:+.4f})  "
          f"44b6={a['score']:.4f} ({a['score']-BASE['g44']:+.4f})  "
          f"6bba={b['score']:.4f} ({b['score']-BASE['g6b']:+.4f})  "
          f"edgeJ={s['edge_jaccard']:.4f} rec={s['node_recall']:.4f}  "
          f"{'IKI FOLD DA IYI' if both else ''}  ({elapsed():.1f}dk)", flush=True)

best = max(RES, key=lambda k: RES[k][0]['score'])
s, a, b = RES[best]
print("\n" + "#" * 104)
print("### V11-ADIM6-RAPOR-BASLANGIC")
print(f"egitim_dataset={len(TRAIN_NAMES)} etiketli_kenar={len(ytr)} poz_orani={ytr.mean():.4f}")
print(f"REFERANS core={BASE['core']:.4f} 44b6={BASE['g44']:.4f} 6bba={BASE['g6b']:.4f}")
for p, (s_, a_, b_) in RES.items():
    print(f"p_min={p:.2f} core={s_['score']:.4f} 44b6={a_['score']:.4f} 6bba={b_['score']:.4f} "
          f"edgeJ={s_['edge_jaccard']:.4f} rec={s_['node_recall']:.4f} n={s_['n_ratio']:.3f}")
print(f"EN IYI p_min={best} core={s['score']:.4f} kazanc={s['score']-BASE['core']:+.4f} "
      f"44b6 {a['score']-BASE['g44']:+.4f} 6bba {b['score']-BASE['g6b']:+.4f}")
print("### V11-ADIM6-RAPOR-BITIS")
print("#" * 104)
