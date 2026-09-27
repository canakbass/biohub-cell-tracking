# =============================================================================
# V11 ADIM 3b - TRACK-GUDUMLU TESPIT KURTARMA  (CPU, ~25 dk)
# =============================================================================
# FIKIR: Adim 3'te gap closing recall'u 0.8687 -> 0.9216 cikardi (BEDAVA),
#   cunku interpolasyonla eklenen ara node'lar kacirilan GT hucrelerine 7 um
#   icinde denk geliyordu. Adim 2.5 ise ESIK ALTINDA gercek hucreler oldugunu
#   gosterdi (recall k=1.3'te 0.9295, k=2.0'da 0.9451) ama esigi dusurmek
#   node sayisi carpanini yiyor.
#
#   -> DUSUK ESIGIN RECALL'UNU, YUKSEK ESIGIN NODE SAYISIYLA al:
#      track'leri kur, her track'in HIZINDAN sonraki konumu ongor, ve SADECE
#      o konuma yakin bir ESIK-ALTI aday varsa onu ekle.
#      Gap closing'den farki: o mevcut kuyruk<->bas ciftlerini BAGLIYOR,
#      bu ise alt havuzdan YENI NODE ekliyor.
#
# Metrik tam bunu odullendiriyor: node sayisi AZ artar (sadece track'in
# ongordugu yerlerde), recall BELIRGIN artar.
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
# =============================================================================
# 0) YUKLEME + OZ-TEST
# =============================================================================
ALL = sorted(os.path.basename(p)[8:-4] for p in glob.glob(os.path.join(CAND_DIR, "cand_v2_*.npz")))
assert ALL, "cand_v2_*.npz bulunamadi"
GT = {n: read_gt(os.path.join(TRAIN, n + ".geff")) for n in ALL}
CORE_L = [n for n in ALL if n in CORE]
G44 = [n for n in CORE_L if n.startswith("44b6")]
G6B = [n for n in CORE_L if n.startswith("6bba")]
print(f"{len(ALL)} dataset  (core {len(CORE_L)}: 44b6 {len(G44)} / 6bba {len(G6B)})")

_g = GT[ALL[0]]
_nd = np.column_stack([_g["t"], _g["z"], _g["y"], _g["x"]]).astype(np.float64)
_st = evaluate_dataset(_nd, _g["edges"].copy(), _g)
assert abs(_st["edge_jaccard"] - 1.0) < 1e-9, "metrik bozuk"
print(f"metrik oz-testi: edge_J={_st['edge_jaccard']:.6f} OK")

# ADIM 3 NIHAI KONFIGURASYONU
CFG = dict(n_star=0.95, gate=10.0, mnn=0.80, fill=4.0,
           gap=dict(max_gap=8, gate_um=8.0), min_len=6)
SC = np.asarray(SCALE)


# =============================================================================
# 1) ADAY HAVUZU:  tek gecis NMS -> her frame icin (koord, skor) skor sirali
# =============================================================================
def build_pool(name):
    cd = load_cands(name)
    T, coords, score, fstart = cd["T"], cd["coords"], cd["score"], cd["fstart"]
    pool, allsc = [], []
    for tv in range(T):
        sl = slice(fstart[tv], fstart[tv + 1])
        c, s = coords[sl], score[sl]
        if len(c) == 0:
            pool.append((np.zeros((0, 3)), np.zeros(0))); continue
        pr, dd = frame_pairs(c, BEST_R)
        b, stt = build_csr(pr, dd, BEST_R, len(c))
        k = nms_frame(np.argsort(-s, kind='stable'), b, stt, np.ones(len(c), bool))
        pool.append((c[k].astype(np.float64), s[k]))
        allsc.append(s[k])
    allsc = np.concatenate(allsc) if allsc else np.zeros(0)
    return dict(T=T, pool=pool, srt=np.sort(allsc)[::-1], est=cd["est"])


def select(p, n_star):
    """n_star*est hedefine gore esik ve secilen node'lar"""
    K = max(1, min(int(round(n_star * p["est"])), len(p["srt"])))
    thr = float(p["srt"][K - 1]) if len(p["srt"]) else 1.0
    return thr, {t: c[s >= thr] for t, (c, s) in enumerate(p["pool"]) if (s >= thr).any()}


# =============================================================================
# 2) TRACK-GUDUMLU KURTARMA
# =============================================================================
def recover(nodes, edges, p, thr_main, radius_um=4.0, max_ext=3,
            floor_ratio=0.10, min_track=3):
    """Track uclarindan hizla ongoru yapip ESIK-ALTI havuzdan node ekle.

    radius_um   : ongorulen konuma bu mesafede aday ara
    max_ext     : bir uctan en fazla kac adim uzat
    floor_ratio : adayin skoru >= floor_ratio * thr_main olmali (cop eklemeyi onler)
    min_track   : sadece bu uzunluktan buyuk track'lerin ucunu uzat (hiz guvenilir olsun)
    """
    if len(nodes) == 0 or len(edges) == 0 or max_ext < 1 or radius_um <= 0:
        return nodes, edges, 0
    T = p["T"]
    floor = floor_ratio * thr_main

    # secilmis node'lari frame basina KD-tree'ye koy (tekrar eklemeyi onlemek icin)
    sel_by_t = {}
    for i, nd in enumerate(nodes):
        sel_by_t.setdefault(int(nd[0]), []).append(i)
    sel_trees = {t: cKDTree(nodes[np.asarray(ix), 1:] * SC) for t, ix in sel_by_t.items()}

    # esik-alti havuz: frame -> (koord, skor)
    sub = {}
    for t, (c, s) in enumerate(p["pool"]):
        m = (s < thr_main) & (s >= floor)
        if m.any():
            sub[t] = (c[m], s[m], cKDTree(c[m] * SC))

    N0 = len(nodes)
    out_deg = np.zeros(N0, np.int32); in_deg = np.zeros(N0, np.int32)
    np.add.at(out_deg, edges[:, 0], 1)
    np.add.at(in_deg, edges[:, 1], 1)
    pred_of = {}   # node -> onceki node (hiz icin)
    next_of = {}
    for a, b in edges.tolist():
        pred_of[b] = a
        next_of[a] = b

    def chain_len(i, back=True):
        n, cur = 1, i
        mp = pred_of if back else next_of
        while cur in mp and n < 12:
            cur = mp[cur]; n += 1
        return n

    new_nodes, new_edges, used = [], [], set()

    def extend(start, forward):
        """Track ucundan hizla ongoru yapip esik-alti havuzdan node ekle.

        HIZ TURETIMI (iki hata duzeltildi):
          ileri : prev = pred_of[cur] (t-1'deki node) -> vel = cur - prev = ileri hareket
                  ongoru(t+1) = cur + vel
          geri  : prev = next_of[cur] (t+1'deki node) -> vel = cur - prev = GERI hareket
                  ongoru(t-1) = cur + vel
          Yani IKI durumda da `vel = cur - prev` ve `+vel` dogru; isaret cevirmek
          geri durumda ongoruyu t+1'e goturuyordu (1. hata).
          Ayrica hiz her adimda GUNCELLENIYOR; yoksa 2. adimda sifirlaniyordu (2. hata).
        """
        cn = nodes[start].copy()
        prev = (pred_of if forward else next_of).get(start)
        vel = (cn[1:] - nodes[prev][1:]) if (prev is not None and prev < N0) else np.zeros(3)
        cur = start
        for _ in range(max_ext):
            t_next = int(cn[0]) + (1 if forward else -1)
            if not (0 <= t_next < T) or t_next not in sub:
                return
            predicted = cn[1:] + vel
            c_sub, s_sub, tree = sub[t_next]
            idx = tree.query_ball_point(predicted * SC, radius_um)
            if not idx:
                return
            j = max(idx, key=lambda q: s_sub[q])
            key = (t_next, j)
            if key in used:
                return
            cand = c_sub[j]
            # zaten secilmis bir node'a cok yakinsa ekleme (tekrar olur)
            if t_next in sel_trees:
                d, _ = sel_trees[t_next].query(cand * SC, k=1)
                if d < BEST_R:
                    return
            used.add(key)
            nid = N0 + len(new_nodes)
            new_nodes.append(np.concatenate([[t_next], cand]))
            new_edges.append([cur, nid] if forward else [nid, cur])
            vel = cand - cn[1:]                      # hizi GUNCELLE
            cn = np.concatenate([[t_next], cand])
            cur = nid

    tails = [i for i in range(N0) if out_deg[i] == 0 and chain_len(i, True) >= min_track]
    heads = [i for i in range(N0) if in_deg[i] == 0 and chain_len(i, False) >= min_track]
    for i in tails:
        extend(i, True)
    for i in heads:
        extend(i, False)

    if not new_nodes:
        return nodes, edges, 0
    nodes2 = np.vstack([nodes, np.array(new_nodes)])
    edges2 = np.vstack([edges, np.array(new_edges, np.int64)])
    return nodes2, edges2, len(new_nodes)


def pipeline(p, n_star, rec_kw=None):
    thr, frames = select(p, n_star)
    nodes, edges = build_graph(frames, p["T"],
                               dict(gate=CFG["gate"], mnn_ratio=CFG["mnn"],
                                    fill_gate=CFG["fill"], local_motion=False),
                               None, CFG["gap"], 1)
    n_rec = 0
    if rec_kw:
        nodes, edges, n_rec = recover(nodes, edges, p, thr, **rec_kw)
    nodes, edges = filter_short(nodes, edges, CFG["min_len"])
    # kanoniklestirme GEREKMIYOR: evaluate_dataset zaten canonical_edges cagiriyor
    return nodes, edges, n_rec, thr


# =============================================================================
# 3) SWEEP
# =============================================================================
print("\n[aday havuzlari hazirlaniyor]")
POOL = {}
for n in ALL:
    POOL[n] = build_pool(n)
    print(f"  {n:22s} T={POOL[n]['T']} aday={len(POOL[n]['srt'])} est={POOL[n]['est']:.0f}"
          f"  ({elapsed():.1f}dk)", flush=True)

def run(rec_kw, n_star=None):
    rows = {}
    for n in ALL:
        nd, ed, nrec, thr = pipeline(POOL[n], n_star or CFG["n_star"], rec_kw)
        r = evaluate_dataset(nd, ed, GT[n]); r["n_rec"] = nrec
        rows[n] = r
    return rows

def show(tag, rows):
    s = summarise([rows[n] for n in CORE_L])
    a = summarise([rows[n] for n in G44]); b = summarise([rows[n] for n in G6B])
    rec_tot = sum(rows[n]["n_rec"] for n in CORE_L)
    print(f"  {tag:40s} core={s['score']:.4f} 44b6={a['score']:.4f} 6bba={b['score']:.4f} "
          f"edgeJ={s['edge_jaccard']:.4f} rec={s['node_recall']:.4f} n={s['n_ratio']:.3f} "
          f"+node={rec_tot}", flush=True)
    return s['score'], a['score'], b['score'], s

print("\n" + "=" * 110)
print("TRACK-GUDUMLU KURTARMA SWEEP")
print("=" * 110)
R = {}
R["BASE"] = show("BASE (kurtarma YOK)", run(None))

print("\n  -- yaricap --")
for r_um in [2.5, 4.0, 6.0]:
    R[("r", r_um)] = show(f"radius={r_um}um ext=3 floor=0.10",
                          run(dict(radius_um=r_um, max_ext=3, floor_ratio=0.10, min_track=3)))
rb = max([k for k in R if isinstance(k, tuple) and k[0] == "r"], key=lambda k: R[k][0])[1]

print(f"\n  -- uzatma adimi (radius={rb}) --")
for ext in [1, 2, 3, 5]:
    R[("e", ext)] = show(f"radius={rb} ext={ext} floor=0.10",
                         run(dict(radius_um=rb, max_ext=ext, floor_ratio=0.10, min_track=3)))
eb = max([k for k in R if isinstance(k, tuple) and k[0] == "e"], key=lambda k: R[k][0])[1]

print(f"\n  -- skor tabani (radius={rb} ext={eb}) --")
for fl in [0.0, 0.05, 0.10, 0.30, 0.60]:
    R[("f", fl)] = show(f"floor={fl}",
                        run(dict(radius_um=rb, max_ext=eb, floor_ratio=fl, min_track=3)))
fb = max([k for k in R if isinstance(k, tuple) and k[0] == "f"], key=lambda k: R[k][0])[1]

print(f"\n  -- min track uzunlugu --")
for mt in [2, 3, 5]:
    R[("m", mt)] = show(f"min_track={mt}",
                        run(dict(radius_um=rb, max_ext=eb, floor_ratio=fb, min_track=mt)))
mb = max([k for k in R if isinstance(k, tuple) and k[0] == "m"], key=lambda k: R[k][0])[1]

BEST = dict(radius_um=rb, max_ext=eb, floor_ratio=fb, min_track=mb)
print(f"\n  -- kurtarma ACIK iken n* yeniden --")
for ns in [0.80, 0.90, 0.95, 1.05]:
    R[("n", ns)] = show(f"n*={ns} + kurtarma", run(BEST, n_star=ns))
nb = max([k for k in R if isinstance(k, tuple) and k[0] == "n"], key=lambda k: R[k][0])[1]

print("\n  -- NIHAI --")
fin = run(BEST, n_star=nb)
fs, f44, f6b, fsum = show("NIHAI", fin)

print("\n" + "#" * 110)
print("### V11-ADIM3B-RAPOR-BASLANGIC")
print(f"sure_dk={elapsed():.1f} n_dataset={len(ALL)}")
print(f"BASE core={R['BASE'][0]:.4f} 44b6={R['BASE'][1]:.4f} 6bba={R['BASE'][2]:.4f} "
      f"edgeJ={R['BASE'][3]['edge_jaccard']:.4f} rec={R['BASE'][3]['node_recall']:.4f}")
for k, v in R.items():
    if k == "BASE": continue
    print(f"{k[0]}:{k[1]} core={v[0]:.4f} 44b6={v[1]:.4f} 6bba={v[2]:.4f} "
          f"edgeJ={v[3]['edge_jaccard']:.4f} rec={v[3]['node_recall']:.4f} n={v[3]['n_ratio']:.3f}")
print(f"BEST={BEST} n*={nb}")
print(f"NIHAI core={fs:.4f} 44b6={f44:.4f} 6bba={f6b:.4f} edgeJ={fsum['edge_jaccard']:.4f} "
      f"rec={fsum['node_recall']:.4f} n={fsum['n_ratio']:.3f} "
      f"kazanc={fs-R['BASE'][0]:+.4f}")
for n in ALL:
    r = fin[n]
    print(f"{n} Np={r['num_pred_nodes']} +rec={r['n_rec']} n={r['n_ratio']:.3f} "
          f"edgeJ={r['edge_jaccard']:.4f} adj={r['adj_edge_jaccard']:.4f} rec={r['node_recall']:.4f}")
print("### V11-ADIM3B-RAPOR-BITIS")
print("#" * 110)
