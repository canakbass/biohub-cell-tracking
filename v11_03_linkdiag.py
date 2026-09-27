# =============================================================================
# V11 ADIM 3 - LINKER TANILAMASI + BIRLESIK SWEEP  (CPU, ~15 dk)
# =============================================================================
# ADIM 2 BULGUSU: linker, tespitin sagladiginin ~%20'sini kaybediyor
#   (edgeJ 0.606 vs recall^2 0.761).  2540cd90'da recall 0.996 iken edgeJ 0.805.
# Bu script NEDEN kaybettigini SAYIYOR: her GT kenari icin iki ucu da eslesmisse,
# bizim grafta o kenar var mi, yoksa neden yok?
#
# Girdi : yarisma verisi (GT icin) + cand_v2_*.npz  (Adim 1b notebook cikti'si)
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
# 0) OZ-TEST + YUKLEME
# =============================================================================
ALL = sorted(os.path.basename(p)[8:-4] for p in glob.glob(os.path.join(CAND_DIR, "cand_v2_*.npz")))
assert ALL, "cand_v2_*.npz bulunamadi - Adim 1b cikti'sini input olarak ekle"
GT = {n: read_gt(os.path.join(TRAIN, n + ".geff")) for n in ALL}
print(f"{len(ALL)} dataset yuklendi ({CAND_DIR})")

g0 = GT[ALL[0]]
_nd = np.column_stack([g0["t"], g0["z"], g0["y"], g0["x"]]).astype(np.float64)
_st = evaluate_dataset(_nd, g0["edges"].copy(), g0)
assert abs(_st["edge_jaccard"] - 1.0) < 1e-9, "metrik bozuk"
print(f"  metrik oz-testi: edge_J={_st['edge_jaccard']:.6f}  adj={_st['adj_edge_jaccard']:.6f}  OK")

VOL_UM3 = 64 * 1.625 * 256 * 0.40625 * 256 * 0.40625      # ~104^3 um^3
def spacing_um(est, T=100):
    return float((6.0 * VOL_UM3 / (math.pi * max(1.0, est / T))) ** (1.0 / 3.0))

CORE_L = [n for n in ALL if n in CORE]
G44 = [n for n in CORE_L if n.startswith("44b6")]
G6B = [n for n in CORE_L if n.startswith("6bba")]


# =============================================================================
# 1) GT YAPISI: kac soy hatti, ne kadar uzun, hangi t araliginda?
#    -> min_len filtresini NE KADAR agresif kurabilecegimizi belirler
# =============================================================================
print("\n" + "=" * 106)
print("1) GT YAPISI  -  hedefledigimiz sey ne?")
print("=" * 106)
print(f"  {'dataset':22s} {'est':>7} {'GTn':>5} {'GTe':>5} {'hat':>4} {'ort.uz':>7} "
      f"{'min':>4} {'med':>4} {'max':>4} {'t_ara':>9} {'GTn/est':>8}")
gt_struct = {}
for name in ALL:
    g = GT[name]
    N, E = g["n_nodes"], g["edges"]
    par = np.arange(N)
    def find(i):
        r = i
        while par[r] != r:
            r = par[r]
        while par[i] != r:
            par[i], i = r, par[i]
        return r
    for u, v in E.tolist():
        ru, rv = find(u), find(v)
        if ru != rv:
            par[ru] = rv
    roots = np.array([find(i) for i in range(N)])
    sizes = np.bincount(roots)
    sizes = sizes[sizes > 0]
    gt_struct[name] = sizes
    trng = "%d-%d" % (g["t"].min(), g["t"].max()) if N else "-"
    print(f"  {name:22s} {g['est_nodes']:7.0f} {N:5d} {len(E):5d} {len(sizes):4d} "
          f"{N/max(1,len(sizes)):7.1f} {sizes.min():4d} {int(np.median(sizes)):4d} "
          f"{sizes.max():4d} {trng:>9} {N/max(1.0,g['est_nodes']):8.4f}")
allsz = np.concatenate([gt_struct[n] for n in ALL])
print(f"\n  TUM GT HATLARI: {len(allsz)} hat  uzunluk p10={np.percentile(allsz,10):.0f} "
      f"p25={np.percentile(allsz,25):.0f} p50={np.percentile(allsz,50):.0f} "
      f"p75={np.percentile(allsz,75):.0f} p90={np.percentile(allsz,90):.0f} min={allsz.min()} max={allsz.max()}")
print(f"  >>> min_len filtresi bu dagilimin ALTINDA kalmali: p10={np.percentile(allsz,10):.0f}")


# =============================================================================
# 2) NODE URETIMI  (n* hedefine ikili arama ile oturan esik)
# =============================================================================
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
    return frames, T, mid


LINK0 = dict(gate=8.0, mnn_ratio=0.80, fill_gate=8.0, local_motion=False)
NODES, THRS = {}, {}
print("\n  node uretimi (n*=%.2f):" % BEST_N)
for name in ALL:
    frames, T, thr = build_nodes(name, BEST_N)
    NODES[name], THRS[name] = (frames, T), thr
    print(f"    {name:22s} thr={thr:.5f} Np={sum(len(v) for v in frames.values()):6d} "
          f"aralik={spacing_um(GT[name]['est_nodes']):5.1f}um")


def run_cfg(link_kw, div_kw=None, gap_kw=None, min_len=1, sel=None, nodes_src=None):
    src = nodes_src or NODES
    rows = {}
    for name in (sel or ALL):
        frames, T = src[name]
        gt = subset_gt(GT[name], 0, T - 1) if FRAME_LIMIT else GT[name]
        nd, ed = build_graph(frames, T, link_kw, div_kw, gap_kw, min_len)
        rows[name] = evaluate_dataset(nd, ed, gt)
    return rows


def show(tag, rows, quiet=False):
    s = summarise([rows[n] for n in CORE_L])
    a = summarise([rows[n] for n in G44])
    b = summarise([rows[n] for n in G6B])
    if not quiet:
        print(f"  {tag:38s} core={s['score']:.4f} 44b6={a['score']:.4f} 6bba={b['score']:.4f} "
              f"edgeJ={s['edge_jaccard']:.4f} rec={s['node_recall']:.4f} n={s['n_ratio']:.3f}")
    return s['score'], a['score'], b['score'], s


# =============================================================================
# 3) LINKER TANILAMASI: kayip GT kenarlarini SINIFLANDIR
# =============================================================================
print("\n" + "=" * 106)
print("3) LINKER TANILAMASI - iki ucu da tespit edilmis GT kenarina ne oldu?")
print("=" * 106)
print(f"  {'dataset':22s} {'aralik':>7} {'iki-uc':>6} {'TP':>5} {'kapi':>5} "
      f"{'yanlisHdf':>9} {'calindi':>7} {'atanmadi':>8} | {'d50':>5} {'d90':>5} {'d99':>5} {'dmax':>6}")
diag = dict(both=0, tp=0, gate=0, wrong=0, stolen=0, unass=0, nodet=0)
disp_all = []
for name in ALL:
    frames, T = NODES[name]
    gt = subset_gt(GT[name], 0, T - 1) if FRAME_LIMIT else GT[name]
    nd, ed = build_graph(frames, T, LINK0, None, None, 1)
    matched = match_nodes(nd, gt)
    e_can = canonical_edges(nd, ed, matched)
    inv = {int(gi): pi for pi, gi in enumerate(matched) if gi >= 0}
    eset = set(map(tuple, e_can.tolist()))
    out_t, in_s = set(), set()
    for s_, t_ in e_can.tolist():
        out_t.add(s_); in_s.add(t_)
    P = nd[:, 1:] * np.asarray(SCALE)
    c = dict(both=0, tp=0, gate=0, wrong=0, stolen=0, unass=0, nodet=0)
    disp = []
    for gs, gtt in gt["edges"].tolist():
        ps, pt = inv.get(int(gs)), inv.get(int(gtt))
        if ps is None or pt is None:
            c["nodet"] += 1
            continue
        c["both"] += 1
        d = float(np.linalg.norm(P[pt] - P[ps]))
        disp.append(d)
        if (ps, pt) in eset:
            c["tp"] += 1
        elif d > LINK0["gate"]:
            c["gate"] += 1
        elif ps in out_t:
            c["wrong"] += 1
        elif pt in in_s:
            c["stolen"] += 1
        else:
            c["unass"] += 1
    for k in diag:
        diag[k] += c[k]
    disp_all += disp
    q = np.percentile(disp, [50, 90, 99]) if disp else [0, 0, 0]
    print(f"  {name:22s} {spacing_um(GT[name]['est_nodes']):7.1f} {c['both']:6d} {c['tp']:5d} "
          f"{c['gate']:5d} {c['wrong']:9d} {c['stolen']:7d} {c['unass']:8d} | "
          f"{q[0]:5.1f} {q[1]:5.1f} {q[2]:5.1f} {max(disp) if disp else 0:6.1f}")

B = max(1, diag["both"])
print(f"\n  tespit-yok={diag['nodet']}  iki-ucu-var={B}")
for k, lbl in [("tp", "DOGRU baglandi"), ("gate", "KAPI DISI kaldi"),
               ("wrong", "kaynak YANLIS hedefe bagli"), ("stolen", "hedefi BASKASI aldi"),
               ("unass", "hic ATANMADI")]:
    print(f"    {lbl:28s} {diag[k]:6d}  %{100*diag[k]/B:5.1f}")
qa = np.percentile(disp_all, [50, 75, 90, 95, 99]) if disp_all else [0]*5
print(f"\n  GERCEK yer degistirme (pred-pred, um): p50={qa[0]:.2f} p75={qa[1]:.2f} "
      f"p90={qa[2]:.2f} p95={qa[3]:.2f} p99={qa[4]:.2f} max={max(disp_all) if disp_all else 0:.1f}")


# =============================================================================
# 4) DATASET BASINA EN IYI KAPI -> aralikla olceklenmeli mi?
# =============================================================================
print("\n" + "=" * 106)
print("4) DATASET BASINA EN IYI KAPI")
print("=" * 106)
GATES = [5.0, 6.5, 8.0, 10.0, 12.0, 15.0, 20.0]
print(f"  {'dataset':22s} {'aralik':>7} " + " ".join(f"{g:>6.1f}" for g in GATES) + "  en_iyi   oran")
gfit = []
for name in ALL:
    sp = spacing_um(GT[name]["est_nodes"])
    vals = [run_cfg(dict(LINK0, gate=g, fill_gate=g), sel=[name])[name]["adj_edge_jaccard"]
            for g in GATES]
    bg = GATES[int(np.argmax(vals))]
    gfit.append((sp, bg))
    print(f"  {name:22s} {sp:7.1f} " + " ".join(f"{v:6.3f}" for v in vals) +
          f"  {bg:5.1f}  {bg/sp:5.2f}")
sps = np.array([a for a, _ in gfit]); bgs = np.array([b for _, b in gfit])
ADAPT = float(np.median(bgs / sps))
print(f"\n  en_iyi_kapi/aralik: medyan={ADAPT:.3f} min={np.min(bgs/sps):.3f} max={np.max(bgs/sps):.3f}")


# =============================================================================
# 5) BIRLESIK SWEEP  (division KAPALI sabit)
# =============================================================================
print("\n" + "=" * 106)
print("5) BIRLESIK SWEEP - sirayla, her adimda iki fold kontrollu")
print("=" * 106)
R5 = {}

print("\n  -- 5a) kapi --")
cands = [(f"sabit{g}", dict(LINK0, gate=g, fill_gate=0.0)) for g in [6.5, 8.0, 10.0, 12.0]]
for tag, lk in cands:
    R5[("gate", tag)] = show(tag, run_cfg(lk))
rows = {n: run_cfg(dict(LINK0, gate=float(np.clip(ADAPT * spacing_um(GT[n]['est_nodes']), 5, 20)),
                        fill_gate=0.0), sel=[n])[n] for n in ALL}
R5[("gate", "uyarlanabilir")] = show(f"uyarlanabilir {ADAPT:.2f}*aralik", rows)
_fx = max(R5[("gate", t)][0] for t, _ in cands)
ADAPT_WINS = R5[("gate", "uyarlanabilir")][0] > _fx + 1e-6
GBEST = float(max(cands, key=lambda kv: R5[("gate", kv[0])][0])[0].replace("sabit", ""))
print(f"  >>> sabit kapi={GBEST}" + ("   !!! UYARLANABILIR DAHA IYI" if ADAPT_WINS else ""))

def gate_of(name):
    return (float(np.clip(ADAPT * spacing_um(GT[name]['est_nodes']), 5, 20))
            if ADAPT_WINS else GBEST)

def run2(gap_kw=None, min_len=1, mnn=0.80, fill=0.0, sel=None, nodes_src=None):
    src = nodes_src or NODES
    rows = {}
    for name in (sel or ALL):
        frames, T = src[name]
        gt = subset_gt(GT[name], 0, T - 1) if FRAME_LIMIT else GT[name]
        lk = dict(gate=gate_of(name), mnn_ratio=mnn, fill_gate=fill, local_motion=False)
        nd, ed = build_graph(frames, T, lk, None, gap_kw, min_len)
        rows[name] = evaluate_dataset(nd, ed, gt)
    return rows

print("\n  -- 5b) gap closing (fiziksel, AGRESIF) --")
for mg in [2, 3, 4, 6, 8, 12]:
    for gu in [4.0, 6.0, 8.0, 10.0]:
        R5[("gap", (mg, gu))] = show(f"gap max={mg} kapi={gu}um", run2(gap_kw=dict(max_gap=mg, gate_um=gu)))
GK = max([k for k in R5 if k[0] == "gap"], key=lambda k: R5[k][0])[1]
GAP1 = dict(max_gap=GK[0], gate_um=GK[1])
print(f"  >>> gap: max_gap={GK[0]} gate={GK[1]}um")

print("\n  -- 5c) kisa track filtresi (AGRESIF: GT hatlari 19-100 frame) --")
for ml in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]:
    R5[("len", ml)] = show(f"min_len={ml}", run2(gap_kw=GAP1, min_len=ml))
ML = max([k for k in R5 if k[0] == "len"], key=lambda k: R5[k][0])[1]
print(f"  >>> min_len={ML}")

print("\n  -- 5d) mnn_ratio --")
for mr in [0.70, 0.80, 0.90, 0.99]:
    R5[("mnn", mr)] = show(f"mnn={mr}", run2(gap_kw=GAP1, min_len=ML, mnn=mr))
MR = max([k for k in R5 if k[0] == "mnn"], key=lambda k: R5[k][0])[1]

print("\n  -- 5e) fill --")
for fg in [0.0, 4.0, 8.0]:
    R5[("fill", fg)] = show(f"fill={fg}", run2(gap_kw=GAP1, min_len=ML, mnn=MR, fill=fg))
FG = max([k for k in R5 if k[0] == "fill"], key=lambda k: R5[k][0])[1]

print("\n  -- 5f) TESPIT ESIGINI (n*) yeni filtreyle YENIDEN optimize et --")
NSW = {}
for ns in [0.3, 0.5, 0.7, 0.95, 1.3, 1.8]:
    src = {}
    for name in ALL:
        f_, T_, th_ = build_nodes(name, ns)
        src[name] = (f_, T_)
    R5[("nstar", ns)] = show(f"n*={ns}", run2(gap_kw=GAP1, min_len=ML, mnn=MR, fill=FG, nodes_src=src))
    NSW[ns] = src
NS = max([k for k in R5 if k[0] == "nstar"], key=lambda k: R5[k][0])[1]
print(f"  >>> n*={NS}")

print("\n  -- 5g) en iyi n* ile min_len'i BIR DAHA tara (etkilesim) --")
for ml in [1, 3, 8, 16, 24, 32, 48, 64, 96]:
    R5[("len2", ml)] = show(f"n*={NS} min_len={ml}",
                            run2(gap_kw=GAP1, min_len=ml, mnn=MR, fill=FG, nodes_src=NSW[NS]))
ML2 = max([k for k in R5 if k[0] == "len2"], key=lambda k: R5[k][0])[1]

print("\n  -- 5h) NIHAI --")
fin = run2(gap_kw=GAP1, min_len=ML2, mnn=MR, fill=FG, nodes_src=NSW[NS])
fs, f44, f6b, fsum = show("NIHAI", fin)


# =============================================================================
# RAPOR
# =============================================================================
print("\n" + "#" * 106)
print("### V11-ADIM3-RAPOR-BASLANGIC")
print(f"sure_dk={elapsed():.1f} n_dataset={len(ALL)}")
print(f"gt_hat_uzunluk p10={np.percentile(allsz,10):.0f} p50={np.percentile(allsz,50):.0f} "
      f"p90={np.percentile(allsz,90):.0f} n_hat={len(allsz)}")
print(f"--- tanilama (iki ucu var={B}) ---")
for k in ("tp", "gate", "wrong", "stolen", "unass"):
    print(f"{k}={diag[k]} ({100*diag[k]/B:.1f}%)")
print(f"nodet={diag['nodet']}")
print(f"disp p50={qa[0]:.2f} p75={qa[1]:.2f} p90={qa[2]:.2f} p95={qa[3]:.2f} p99={qa[4]:.2f}")
print(f"kapi_oran medyan={ADAPT:.3f} adaptif_kazandi={ADAPT_WINS} sabit_kapi={GBEST}")
print("--- sweep ---")
for k, v in R5.items():
    print(f"{k[0]}:{k[1]} core={v[0]:.4f} 44b6={v[1]:.4f} 6bba={v[2]:.4f} "
          f"edgeJ={v[3]['edge_jaccard']:.4f} rec={v[3]['node_recall']:.4f} n={v[3]['n_ratio']:.3f}")
print("--- NIHAI KONFIG ---")
print(f"r_nms={BEST_R} n*={NS} gate={'adaptif '+str(round(ADAPT,3))+'*aralik' if ADAPT_WINS else GBEST} "
      f"mnn={MR} fill={FG} gap={GAP1} min_len={ML2} division=KAPALI")
print(f"NIHAI core={fs:.4f} 44b6={f44:.4f} 6bba={f6b:.4f} edgeJ={fsum['edge_jaccard']:.4f} "
      f"rec={fsum['node_recall']:.4f} n={fsum['n_ratio']:.3f}")
print("--- dataset basina NIHAI ---")
for n in ALL:
    r = fin[n]
    print(f"{n} Np={r['num_pred_nodes']} n={r['n_ratio']:.3f} edgeJ={r['edge_jaccard']:.4f} "
          f"adj={r['adj_edge_jaccard']:.4f} rec={r['node_recall']:.4f}")
print("### V11-ADIM3-RAPOR-BITIS")
print("#" * 106)
