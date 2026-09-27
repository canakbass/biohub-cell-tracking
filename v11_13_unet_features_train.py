# =============================================================================
# V11 ADIM 1c - ADAY DOKUMU, ADIM 5 DETEKTORU ILE  (GPU, ~18 dk)
# =============================================================================
# v9 modelini 12 dataset uzerinde BIR KEZ cesirip her frame'in yogun aday
# tepe noktalarini (z,y,x,score) diske yaziyoruz.
#
# Bundan sonra esik / NMS / linking / division sweep'lerinin TAMAMI saf CPU
# isine donuyor -> GPU kotasi harcamadan dakikalar icinde yuzlerce kombinasyon.
#
# Cikti: /kaggle/working/cand_<dataset>.npz  (+ ekrana RAPOR blogu)
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

import os, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

COMP  = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
TRAIN = os.path.join(COMP, "train")
WORK  = "/kaggle/working"
# ADIM 5'in egittigi detektor (v11-egitim kernel cikti'sindan)
import glob as _g
_c = _g.glob("/kaggle/input/**/v11_detector_best.pt", recursive=True)
MODEL_PATH = _c[0] if _c else "/kaggle/input/models/hcanakbas/biohub-v9-03-sota/pytorch/default/1/biohub_v9_sota_f1_868.pt"
print(f"MODEL: {MODEL_PATH}")

SCALE   = (1.625, 0.40625, 0.40625)

# --- ADIM 0 raporundan gelen val seti ---------------------------------------
VAL_CORE = ['44b6_341df25f', '44b6_3bb3690f', '44b6_c771cb04', '44b6_8f9ecab4',
            '6bba_2540cd90', '6bba_3a1849c2', '6bba_67ebd073', '6bba_57b7cc1e']
VAL_PUBLIC = ['44b6_0113de3b', '44b6_0b24845f', '6bba_05b6850b', '6bba_05db0fb1']
# ADIM 6 EGITIM HAVUZU (val HARIC, v11_06a_cand_train.py ile AYNI secim):
_all = sorted(d[:-5] for d in os.listdir(TRAIN) if d.endswith(".zarr"))
_pool = [n for n in _all if n not in VAL_CORE and n not in VAL_PUBLIC]
PER_COLONY = 18
DATASETS = []
for _col in ("44b6", "6bba"):
    _c = [n for n in _pool if n.startswith(_col)]
    _idx = np.linspace(0, len(_c) - 1, min(PER_COLONY, len(_c))).astype(int)
    DATASETS += [_c[i] for i in sorted(set(_idx.tolist()))]
print(f"ADIM 6 egitim havuzu (feature): {len(DATASETS)} dataset "
      f"(44b6 {sum(n.startswith('44b6') for n in DATASETS)} / "
      f"6bba {sum(n.startswith('6bba') for n in DATASETS)})")
assert not (set(DATASETS) & set(VAL_CORE + VAL_PUBLIC)), "SIZINTI: val seti egitim havuzunda"

# --- aday dokumu ayarlari ---------------------------------------------------
TAG           = "v3trfeat"  # ADIM 6 egitim havuzu + UNet ara-katman feature'i (32 kanal)
CAND_FLOOR    = 1e-4       # ADIM1-v1: 0.005 BAGLAYICIYDI -> tabani indir
CAND_NMS_VOX  = (3, 11, 11)  # ~2 um minimal NMS -> plato tekrarlarini kirp, yogunlugu koru
CAND_CAP_MULT = 5          # frame basina sinir = 5 x (est_nodes/T) -> yoguna gore uyarlanir
CAND_CAP_MIN  = 1500
MAX_MINUTES   = 480        # guvenlik: oturum bitmeden kaydet
SKIP_DONE     = True       # yeniden calistirirsan tamamlananlari atla

# tanilama icin bakilacak esikler
DIAG_THR = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.010, 0.025, 0.050, 0.100, 0.200]

t_start = time.time()
def elapsed(): return (time.time() - t_start) / 60.0

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class ConvNeXtBlock3D(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=(3, 5, 5), padding=(1, 2, 2), groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 4, 1, 2, 3) 
        return input + self.drop_path(x)

class ASPP3D(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[1, 2, 4]):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False),
                    LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
                    nn.GELU()
                )
            )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1, bias=False),
            LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            nn.GELU()
        )
        self.project = nn.Sequential(
            nn.Conv3d(out_channels * (len(dilations) + 1), out_channels, 1, bias=False),
            LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
            nn.GELU()
        )
        
    def forward(self, x):
        res = [conv(x) for conv in self.convs]
        global_feat = self.global_pool(x)
        global_feat = torch.nn.functional.interpolate(global_feat, size=x.shape[2:], mode='trilinear', align_corners=False)
        res.append(global_feat)
        return self.project(torch.cat(res, dim=1))

class UNet4D_SOTA_V9(nn.Module):
    def __init__(self, in_channels=3, channels=(32, 64, 128, 256), drop_path_rate=0.2):
        super().__init__()
        self.pool_kernels = [(1, 2, 2), (2, 2, 2), (2, 2, 2)]
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, channels[0], kernel_size=3, padding=1),
            LayerNorm(channels[0], eps=1e-6, data_format="channels_first")
        )
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum([1, 1, 1, 1]))]
        self.enc = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(len(channels)-1):
            pk = self.pool_kernels[i]
            self.pools.append(nn.MaxPool3d(pk))
            self.enc.append(nn.Sequential(
                ConvNeXtBlock3D(channels[i], drop_path=dpr[i]),
                nn.Conv3d(channels[i], channels[i+1], kernel_size=1) 
            ))
        self.bottleneck = nn.Sequential(
            ConvNeXtBlock3D(channels[-1], drop_path=dpr[-1]),
            ASPP3D(channels[-1], channels[-1])
        )
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.ds_heads = nn.ModuleList()
        for i in range(len(channels)-2, -1, -1):
            pk = self.pool_kernels[i]
            self.up.append(nn.Sequential(
                nn.Upsample(scale_factor=pk, mode='trilinear', align_corners=False),
                nn.Conv3d(channels[i+1], channels[i], kernel_size=1)
            ))
            self.dec.append(ConvNeXtBlock3D(channels[i], drop_path=0.0))
            self.ds_heads.append(nn.Conv3d(channels[i], 1, 1))

    def forward(self, x, return_features=False):
        """return_features=True (sadece eval modunda): son (tam cozunurluk,
        channels[0]=32 kanal) decoder katmaninin HAM feature haritasini da
        dondurur (1x1 head'den ONCE) -> (heatmap, feat). Checkpoint AYNI,
        sadece forward'tan ek bir ara-katman ciktisi aliniyor (2026-09-21,
        Adim 3: UNet-feature-tabanli kenar/bolunme siniflandiricisi)."""
        z_div, y_div, x_div = 1, 1, 1
        for pk in self.pool_kernels:
            z_div *= pk[0]; y_div *= pk[1]; x_div *= pk[2]

        pad = []
        divs = (x_div, y_div, z_div)
        for d, div in zip(reversed(x.shape[2:]), divs):
            r = d % div
            p = (div - r) % div
            pad.extend([0, p])

        orig_shape = x.shape[2:]

        if any(p > 0 for p in pad):
            x = F.pad(x, pad, mode='reflect')
        x = self.stem(x)
        skips = [x]

        for pool, enc in zip(self.pools, self.enc):
            skips.append(enc(pool(skips[-1])))

        x = skips.pop()
        x = self.bottleneck(x)

        ds_outputs = []
        feat_last = None
        for i, (up, dec, sk) in enumerate(zip(self.up, self.dec, reversed(skips))):
            x = up(x)
            if x.shape[2:] != sk.shape[2:]:
                x = torch.nn.functional.pad(x, [0, sk.shape[4]-x.shape[4], 0, sk.shape[3]-x.shape[3], 0, sk.shape[2]-x.shape[2]])
            x = x + sk
            x = dec(x)

            out = self.ds_heads[i](x)
            out = out[:, :, :orig_shape[0], :orig_shape[1], :orig_shape[2]]
            ds_outputs.append(out)
            feat_last = x[:, :, :orig_shape[0], :orig_shape[1], :orig_shape[2]]

        if self.training:
            return ds_outputs
        if return_features:
            return ds_outputs[-1], feat_last
        return ds_outputs[-1]


# =============================================================================
# VERI OKUMA
# =============================================================================
def read_zarr_meta(zarr_dir):
    import zarr
    grp = zarr.open_group(zarr_dir, mode='r')
    q = {float(k): float(v)
         for k, v in (grp.attrs.get("image_statistics", {}).get("quantiles", {}) or {}).items()}
    arr = zarr.open(os.path.join(zarr_dir, "0"), mode='r')
    return q, tuple(arr.shape)


def normalize(vol_u16, q):
    """v9 egitimi/submit'i ile BIREBIR ayni normalizasyon."""
    lo = q.get(0.001, 100.0)
    hi = q.get(0.999, 1000.0)
    if 0.001 not in q and vol_u16.size > 0:
        lo = float(np.percentile(vol_u16[::4], 0.1))
        hi = float(np.percentile(vol_u16[::4], 99.9))
    img = (vol_u16.astype(np.float32) - lo) / max(1e-6, hi - lo)
    return np.clip(img, 0.0, 4.0, out=img)


def read_gt(geff_path):
    """GT voxel cinsinden (ADIM 0: 199/199 dataset voxel yaziyor)."""
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
        ids = np.arange(len(t), np.int64)
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


# =============================================================================
# GREEDY NMS  (fiziksel uzayda, skor sirasina gore)
#   Adim 2 sweep'i bunu yeniden kullanacak -> pairs onceden verilebilir
# =============================================================================
def greedy_nms(coords_vox, scores, r_um, scale=SCALE, pairs=None, pdist=None):
    n = len(coords_vox)
    if n == 0:
        return np.zeros(0, np.int64)
    if pairs is None:
        P = coords_vox.astype(np.float64) * np.asarray(scale)
        pairs = cKDTree(P).query_pairs(r_um, output_type='ndarray')
        pdist = None
    elif pdist is not None and len(pairs):
        pairs = pairs[pdist <= r_um]

    if len(pairs):
        a = np.concatenate([pairs[:, 0], pairs[:, 1]])
        b = np.concatenate([pairs[:, 1], pairs[:, 0]])
        o = np.argsort(a, kind='stable')
        a, b = a[o], b[o]
        starts = np.searchsorted(a, np.arange(n + 1))
    else:
        b = np.zeros(0, np.int64)
        starts = np.zeros(n + 1, np.int64)

    sup = np.zeros(n, bool)
    keep = []
    for i in np.argsort(-scores, kind='stable'):
        if sup[i]:
            continue
        keep.append(i)
        sup[b[starts[i]:starts[i + 1]]] = True
    return np.asarray(keep, np.int64)


# =============================================================================
# NODE RECALL  (frame ici optimal bipartite, 7 um) - metrikteki "r"
# =============================================================================
def node_recall_at(nodes, gt, scale=SCALE, max_dist=7.0):
    """nodes: (N,4) [t,z,y,x] voxel -> eslesen GT node orani"""
    from scipy.optimize import linear_sum_assignment
    if len(nodes) == 0 or gt["n_nodes"] == 0:
        return 0.0
    sc = np.asarray(scale)
    gt_P = np.stack([gt["z"], gt["y"], gt["x"]], 1) * sc
    p_t = nodes[:, 0].astype(np.int64)
    p_P = nodes[:, 1:] * sc
    hit = 0
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
        hit += int((C[r, c] < 1e6).sum())
    return hit / gt["n_nodes"]


# =============================================================================
# MODEL
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"donanim: {device}")
assert os.path.exists(MODEL_PATH), f"model yok: {MODEL_PATH}"

model = UNet4D_SOTA_V9().to(device)
_ck = torch.load(MODEL_PATH, map_location=device, weights_only=False)
_sd = _ck.get('model_state_dict', _ck)
_missing, _unexp = model.load_state_dict(_sd, strict=False)
model.eval()
print(f"model yuklendi | eksik={len(_missing)} beklenmeyen={len(_unexp)} "
      f"| ckpt epoch={_ck.get('epoch','?')} f1={_ck.get('f1','?')}")
if _missing or _unexp:
    print(f"  !!! AGIRLIK UYUSMAZLIGI  missing={_missing[:5]} unexpected={_unexp[:5]}")


# =============================================================================
# ANA DONGU
# =============================================================================
report = []
for di, name in enumerate(DATASETS):
    out_npz = os.path.join(WORK, f"cand_{TAG}_{name}.npz")
    if SKIP_DONE and os.path.exists(out_npz):
        print(f"[{di+1}/{len(DATASETS)}] {name} -> zaten var, atlandi")
        continue
    if elapsed() > MAX_MINUTES:
        print(f"\n!!! sure doldu ({elapsed():.0f} dk), kalanlar atlandi")
        break

    import zarr
    zdir = os.path.join(TRAIN, name + ".zarr")
    q, shape = read_zarr_meta(zdir)
    T, Z, Y, X = shape
    gt = read_gt(os.path.join(TRAIN, name + ".geff"))
    arr = zarr.open(os.path.join(zdir, "0"), mode='r')

    print(f"\n[{di+1}/{len(DATASETS)}] {name}  T={T} est={gt['est_nodes']:.0f} "
          f"({gt['est_nodes']/T:.0f} hucre/frame)  GT {gt['n_nodes']}n/{len(gt['edges'])}e")

    CAND_MAX = max(CAND_CAP_MIN, int(CAND_CAP_MULT * gt['est_nodes'] / T))
    print(f"    frame basina aday siniri = {CAND_MAX}")

    ct, cz, cy, cx, cs, cf = [], [], [], [], [], []
    hm_max, n_above, n_capped = [], [], 0
    cache, t0 = {}, time.time()

    with torch.no_grad():
        for t in range(T):
            for tt in (t - 1, t, t + 1):
                tc = min(max(tt, 0), T - 1)
                if tc not in cache:
                    cache[tc] = normalize(np.asarray(arr[tc]), q)
            for old in [k for k in cache if k < t - 1]:
                del cache[old]

            inp = np.stack([cache[max(t - 1, 0)], cache[t], cache[min(t + 1, T - 1)]], 0)
            x_t = torch.from_numpy(inp).unsqueeze(0).to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16,
                                enabled=(device.type == 'cuda')):
                logits, feat = model(x_t, return_features=True)
            hm = torch.sigmoid(logits.float()).squeeze().cpu().numpy()

            hm_max.append(float(hm.max()))
            pooled = maximum_filter(hm, size=CAND_NMS_VOX, mode="nearest")
            mask = (hm >= pooled) & (hm > CAND_FLOOR)
            idx = np.argwhere(mask)
            sc = hm[mask]
            n_above.append(len(idx))
            if len(idx) > CAND_MAX:
                sel = np.argpartition(-sc, CAND_MAX)[:CAND_MAX]
                idx, sc = idx[sel], sc[sel]
                n_capped += 1

            if len(idx):
                idx_t = torch.from_numpy(idx).to(device)
                fvec = feat[0, :, idx_t[:, 0], idx_t[:, 1], idx_t[:, 2]].t().float().cpu().numpy()
            else:
                fvec = np.zeros((0, feat.shape[1]), np.float32)

            ct.append(np.full(len(idx), t, np.int16))
            cz.append(idx[:, 0].astype(np.int16))
            cy.append(idx[:, 1].astype(np.int16))
            cx.append(idx[:, 2].astype(np.int16))
            cs.append(sc.astype(np.float32))
            cf.append(fvec.astype(np.float32))

            if t % 20 == 0:
                print(f"    t={t:3d} aday={len(idx):5d} hm_max={hm.max():.3f} "
                      f"({(time.time()-t0)/max(1,t+1):.2f} s/frame, {elapsed():.1f} dk)")

    ct = np.concatenate(ct); cz = np.concatenate(cz)
    cy = np.concatenate(cy); cx = np.concatenate(cx); cs = np.concatenate(cs)
    cf = np.concatenate(cf, axis=0) if cf else np.zeros((0, 32), np.float32)

    np.savez_compressed(out_npz, t=ct, z=cz, y=cy, x=cx, score=cs, feat=cf,
                        est_nodes=gt['est_nodes'], T=T, shape=np.asarray(shape),
                        hm_max=np.asarray(hm_max, np.float32))
    print(f"  kaydedildi -> cand_{TAG}_{name}.npz  ({len(ct)} aday, feat={cf.shape}, "
          f"{os.path.getsize(out_npz)/1e6:.1f} MB, {elapsed():.1f} dk)")

    # --- TANILAMA: esik -> recall / node sayisi / proxy skor -----------------
    print(f"  {'esik':>6} {'ham/fr':>8} {'NMS7/fr':>8} {'N_pred':>8} "
          f"{'n':>6} {'carpan':>7} {'recall':>7} {'r^2*c':>7}")
    diag = {}
    # ct zaten artan t sirasinda -> frame dilimleri searchsorted ile (hizli)
    fstart = np.searchsorted(ct, np.arange(T + 1))
    for thr in DIAG_THR:
        m = cs > thr
        kept = []
        for tv in range(T):
            sl = slice(fstart[tv], fstart[tv + 1])
            fm = m[sl]
            if not fm.any():
                continue
            cv = np.stack([cz[sl][fm], cy[sl][fm], cx[sl][fm]], 1).astype(np.float64)
            k = greedy_nms(cv, cs[sl][fm], 7.0)
            kept.append(np.column_stack([np.full(len(k), tv, np.float64), cv[k]]))
        nodes = np.vstack(kept) if kept else np.zeros((0, 4))
        Np = len(nodes)
        rec = node_recall_at(nodes, gt)
        n_ratio = Np / gt['est_nodes'] if gt['est_nodes'] == gt['est_nodes'] else float('nan')
        factor = 1 - 0.1 * (n_ratio - 1)
        proxy = rec * rec * max(0.0, factor)
        diag[thr] = (int(m.sum()), Np, n_ratio, factor, rec, proxy)
        print(f"  {thr:6.3f} {m.sum()/T:8.0f} {Np/T:8.0f} {Np:8d} "
              f"{n_ratio:6.2f} {factor:7.3f} {rec:7.3f} {proxy:7.3f}")
    best = max(DIAG_THR, key=lambda t: diag[t][5])
    print(f"  >>> bu dataset icin en iyi proxy: esik={best:.3f} r^2*c={diag[best][5]:.3f}")

    report.append(dict(name=name, est=gt['est_nodes'], T=T,
                       gt_n=gt['n_nodes'], gt_e=len(gt['edges']),
                       hm_max_med=float(np.median(hm_max)),
                       hm_max_min=float(np.min(hm_max)),
                       capped=n_capped, diag=diag))

# =============================================================================
# RAPOR
# =============================================================================
print("\n" + "#" * 78)
print("### V11-ADIM1-RAPOR-BASLANGIC")
print(f"sure_dk={elapsed():.1f} dataset_islenen={len(report)}")
print(f"ckpt_missing={len(_missing)} ckpt_unexpected={len(_unexp)}")
for r in report:
    d = r['diag']
    print(f"DS {r['name']} est={r['est']:.0f} gt={r['gt_n']}n/{r['gt_e']}e "
          f"hm_max_med={r['hm_max_med']:.3f} hm_max_min={r['hm_max_min']:.3f} "
          f"capped={r['capped']}")
    print("   " + " ".join(f"{t:.3f}/Np{d[t][1]}/n{d[t][2]:.2f}/r{d[t][4]:.3f}/p{d[t][5]:.3f}"
                           for t in DIAG_THR))
    bb = max(DIAG_THR, key=lambda t: d[t][5])
    print(f"   BEST thr={bb:.3f} proxy={d[bb][5]:.4f} recall={d[bb][4]:.4f} n={d[bb][2]:.2f}")
print("### V11-ADIM1-RAPOR-BITIS")
print("#" * 78)
print("\n>>> /kaggle/working icindeki cand_*.npz dosyalarini 'Save Version' ile")
print(f">>> output olarak kaydet (cand_{TAG}_*.npz), Adim 2 onlari dataset olarak okuyacak.")
