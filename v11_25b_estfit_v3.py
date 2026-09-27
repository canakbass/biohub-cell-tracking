# =============================================================================
# V11 ADIM 2.5b - est_nodes KESTIRICISI, ADIM 5 DETEKTORU  (GPU, ~40 dk)
# ZORUNLU: yeni detektorun skor dagilimi TAMAMEN KAYDI (hm_max_med 0.955->0.836,
#   3bb3690f'te 0.931->0.514). v9 icin fit edilen est_hat ARTIK GECERSIZ.
# =============================================================================
# NEDEN: gizli test setinde est_nodes YOK (geff yok) -> goruntuden tahmin sart.
#   Mevcut fit sadece 12 noktayla yapildi (medyan hata %3.5, ama max %38).
#   Bu script TUM 199 train dataset'inde ozellik toplar, yeniden fit eder ve
#   leave-one-COLONY-out dogrular (test embriyo-ayrik!).
#
# AYRICA S1'DEN CIKAN HIPOTEZI TEST EDER:
#   0b24845f'te kestiricinin %38 FAZLA tahmini PUANI ARTIRDI
#   (gercek est ile rec=0.628/edgeJ=0.400 ; tahminle rec=0.902/edgeJ=0.661)
#   -> Hipotez: detektorun O DATASET'te zayif oldugu yerlerde n* BUYUK olmali.
#   Olcum: her dataset icin recall@(N = k*est) egrisi, k in {0.7..2.0}
#          ve "en iyi k" ile olculebilir bir dataset ozelligi arasindaki iliski.
#
# Frame basina TAM tarama yerine esit arali T_SAMPLE frame -> 199 dataset ~25 dk.
# =============================================================================
# =============================================================================
import sys, subprocess
from importlib.metadata import version as _pkgver
def _v(n):
    try: return _pkgver(n)
    except Exception: return None
if _v("zarr") is None:
    print("[bootstrap] zarr kuruluyor...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zarr"], check=False)
print(f"[bootstrap] numpy={_v('numpy')} scipy={_v('scipy')} zarr={_v('zarr')}", flush=True)

import os, gzip, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

COMP = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
TRAIN = os.path.join(COMP, "train")
TEST = os.path.join(COMP, "test")
WORK = "/kaggle/working"
import glob as _g
_c = _g.glob("/kaggle/input/**/v11_detector_best.pt", recursive=True)
assert _c, "v11_detector_best.pt bulunamadi - v11-egitim cikti'sini ekle"
MODEL_PATH = _c[0]
print(f"MODEL: {MODEL_PATH}")

SCALE = (1.625, 0.40625, 0.40625)   # varsayilan; her dataset kendi attrs'inden okunur

# --- ADIM 3 KONFIGURASYONU ---------------------------------------------------
R_NMS      = 7.0
N_STAR     = 0.95
GATE       = 10.0
MNN_RATIO  = 0.80
FILL_GATE  = 4.0
GAP_MAX    = 8
GAP_UM     = 8.0
MIN_LEN    = 6
DIVISIONS  = False

# --- est_nodes kestiricisi (12 dataset log-log fit; medyan hata %3.5) --------
# est_hat = EST_A * Np100^EST_B   (Np100 = thr=EST_THR ustu, 7um NMS sonrasi,
#                                  T=100'e normalize edilmis film geneli sayim)
EST_THR = 0.200
EST_A   = 0.315171
EST_B   = 1.141545

# --- aday dokumu -------------------------------------------------------------
CAND_FLOOR = 1e-4
PEAK_KERNEL = (3, 11, 11)     # ~2 um: z +-1, y/x +-5 voxel
CAND_MAX    = 8000            # frame basina, skora gore ilk N

MAX_MINUTES = 500             # kernel limiti 9 saat; guvenlik payi

t_start = time.time()
def elapsed(): return (time.time() - t_start) / 60.0

T_SAMPLE  = 12        # dataset basina esit arali frame sayisi
MAX_DS    = 199       # islenecek dataset sayisi (sure yetmezse azalt)
MAX_MIN   = 400
K_GRID    = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0]   # n* adaylari
THR_FEAT  = [0.025, 0.05, 0.10, 0.20, 0.35]               # ozellik esikleri

ALL_NAMES = sorted(d[:-5] for d in os.listdir(TRAIN) if d.endswith(".zarr"))
print(f"train dataset sayisi: {len(ALL_NAMES)}")
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

    def forward(self, x):
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
        for i, (up, dec, sk) in enumerate(zip(self.up, self.dec, reversed(skips))):
            x = up(x)
            if x.shape[2:] != sk.shape[2:]:
                x = torch.nn.functional.pad(x, [0, sk.shape[4]-x.shape[4], 0, sk.shape[3]-x.shape[3], 0, sk.shape[2]-x.shape[2]])
            x = x + sk 
            x = dec(x)
            
            out = self.ds_heads[i](x)
            out = out[:, :, :orig_shape[0], :orig_shape[1], :orig_shape[2]]
            ds_outputs.append(out)
            
        return ds_outputs if self.training else ds_outputs[-1]
def _decompress(buf, spec):
    if spec is None: return buf
    cid = (spec.get("id") or spec.get("name") or "").lower()
    if cid == "blosc":
        for imp in ("blosc", "numcodecs.blosc", "blosc2"):
            try: return __import__(imp, fromlist=["x"]).decompress(buf)
            except: pass
    if cid in ("zstd", "zstandard"):
        try: return __import__("zstandard").ZstdDecompressor().decompress(buf, max_output_size=64*1024*1024)
        except: pass
        try: return __import__("numcodecs").Zstd().decode(buf)
        except: pass
    if cid == "gzip": return gzip.decompress(buf)
    if cid in ("bytes", "transpose"): return buf
    return buf

def read_timepoint(zarr_dir, t_idx):
    apath = os.path.join(zarr_dir, "0")
    v2 = os.path.join(apath, ".zarray")
    if os.path.exists(v2):
        m = json.load(open(v2))
        shape, chunks = tuple(m["shape"]), tuple(m["chunks"])
        dt, comp, sep = np.dtype(m["dtype"]), m.get("compressor"), m.get("dimension_separator", ".")
    else:
        m = json.load(open(os.path.join(apath, "zarr.json")))
        shape, chunks = tuple(m["shape"]), tuple(m["chunk_grid"]["configuration"]["chunk_shape"])
        dt, comp = np.dtype(m.get("data_type", "uint16")), None
        sep = m.get("chunk_key_encoding", {}).get("configuration", {}).get("separator", "/")
    
    t_chunk, t_local = t_idx // chunks[0], t_idx % chunks[0]
    out = np.zeros(shape[1:], dtype=dt)
    
    for zi in range(-(-shape[1]//chunks[1])):
        for yi in range(-(-shape[2]//chunks[2])):
            for xi in range(-(-shape[3]//chunks[3])):
                if os.path.exists(v2):
                    key = sep.join(str(i) for i in (t_chunk, zi, yi, xi))
                else:
                    cke = m.get("chunk_key_encoding", {"name": "default"})
                    key = ("c"+sep if cke.get("name")=="default" else "") + sep.join(str(i) for i in (t_chunk, zi, yi, xi))
                fp = os.path.join(apath, key)
                if not os.path.exists(fp): continue
                buf = open(fp, "rb").read()
                if comp: buf = _decompress(buf, comp)
                elif not os.path.exists(v2):
                    for c in reversed(m.get("codecs", [])): buf = _decompress(buf, c)
                arr = np.frombuffer(buf, dtype=dt).reshape(chunks)
                z0, z1 = zi*chunks[1], min((zi+1)*chunks[1], shape[1])
                y0, y1 = yi*chunks[2], min((yi+1)*chunks[2], shape[2])
                x0, x1 = xi*chunks[3], min((xi+1)*chunks[3], shape[3])
                out[z0:z1, y0:y1, x0:x1] = arr[t_local, :z1-z0, :y1-y0, :x1-x0]
    return out

def read_zarr_attrs(zarr_dir):
    for f in (".zattrs", "zarr.json"):
        p = os.path.join(zarr_dir, f)
        if os.path.exists(p):
            a = json.load(open(p))
            if "attributes" in a: a = a["attributes"]
            break
    else: a = {}
    
    sc = SCALE
    try:
        tr = a["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
        if tr["type"] == "scale": sc = tuple(float(v) for v in tr["scale"][-3:])
    except: pass
    q = {float(k): float(v) for k, v in (a.get("image_statistics", {}).get("quantiles", {}) or {}).items()}
    v2 = os.path.join(zarr_dir, "0", ".zarray")
    shape = tuple(json.load(open(v2))["shape"]) if os.path.exists(v2) else tuple(json.load(open(os.path.join(zarr_dir, "0", "zarr.json")))["shape"])
    return sc, q, shape

def normalize(vol_u16, q):
    lo = q.get(0.001, 100.0)
    hi = q.get(0.999, 1000.0)
    if 0.001 not in q and vol_u16.size > 0:
        lo = float(np.percentile(vol_u16[::4], 0.1))
        hi = float(np.percentile(vol_u16[::4], 99.9))
    img = (vol_u16.astype(np.float32) - lo) / max(1e-6, hi - lo)
    return np.clip(img, 0.0, 4.0, out=img)



# =============================================================================
# GPU TEPE BULMA  (CPU maximum_filter yerine max_pool3d -> ~10x hiz)
# =============================================================================
def peaks_gpu(hm, floor=CAND_FLOOR, cap=CAND_MAX):
    """hm: (1,1,Z,Y,X) GPU tensor -> (K,3) int16 voxel koord + (K,) float32 skor"""
    kz, ky, kx = PEAK_KERNEL
    pooled = F.max_pool3d(hm, kernel_size=PEAK_KERNEL, stride=1,
                          padding=(kz // 2, ky // 2, kx // 2))
    mask = (hm >= pooled) & (hm > floor)
    idx = mask[0, 0].nonzero()                       # (K,3) GPU
    if idx.numel() == 0:
        return np.zeros((0, 3), np.int16), np.zeros(0, np.float32)
    sc = hm[0, 0][idx[:, 0], idx[:, 1], idx[:, 2]]
    if idx.shape[0] > cap:
        sc, sel = torch.topk(sc, cap)
        idx = idx[sel]
    return idx.to(torch.int16).cpu().numpy(), sc.float().cpu().numpy()


# =============================================================================
# GREEDY NMS  (fiziksel uzayda, skor sirasinda)
#   KRITIK OZELLIK: kabul(esik t) = kabul(taban) & (skor > t)
#   Esigi yukseltmek sadece daha DUSUK skorlulari siler; onlar zaten sonra
#   islendigi icin onceki kararlari etkilemez. -> tek gecis, tum esikler icin.
# =============================================================================
def nms_accept(coords, scores, r_um, scale):
    """-> kabul edilen adaylarin indeksleri (skor sirasinda)"""
    n = len(coords)
    if n == 0:
        return np.zeros(0, np.int64)
    P = coords.astype(np.float64) * np.asarray(scale)
    pr = cKDTree(P).query_pairs(r_um, output_type='ndarray')
    if len(pr):
        a = np.concatenate([pr[:, 0], pr[:, 1]])
        b = np.concatenate([pr[:, 1], pr[:, 0]])
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
# GT OKUMA (voxel; Adim 0'da 199/199 dogrulandi)
# =============================================================================
def read_gt(geff_path):
    import zarr
    g = zarr.open_group(str(geff_path), mode="r")
    est = ((dict(g.attrs).get("geff", {}) or {}).get("extra") or {}).get("estimated_number_of_nodes")
    def arr(p):
        n = g
        for k in p.split("/"):
            n = n[k]
        return np.asarray(n[:])
    def get(k):
        for p in (f"nodes/props/{k}/values", f"nodes/{k}"):
            try: return arr(p)
            except Exception: pass
        raise KeyError(k)
    t = get("t").astype(np.int64)
    z, y, x = (get(k).astype(np.float64) for k in ("z", "y", "x"))
    return dict(t=t, z=z, y=y, x=x, n_nodes=len(t),
                est_nodes=float(est) if est is not None else float("nan"))


def recall_on_frames(nodes_by_t, gt, scale, max_dist=7.0):
    """sadece verilen frame'lerde: eslesen GT node / o frame'lerdeki GT node"""
    from scipy.optimize import linear_sum_assignment
    sc = np.asarray(scale)
    gt_P = np.stack([gt["z"], gt["y"], gt["x"]], 1) * sc
    hit = tot = 0
    for tv, cc in nodes_by_t.items():
        gi = np.where(gt["t"] == tv)[0]
        if len(gi) == 0:
            continue
        tot += len(gi)
        if len(cc) == 0:
            continue
        D = np.linalg.norm((cc * sc)[:, None, :] - gt_P[gi][None, :, :], axis=2)
        C = np.where(D <= max_dist, D, 1e6)
        if not np.any(C < 1e6):
            continue
        r, c = linear_sum_assignment(C)
        hit += int((C[r, c] < 1e6).sum())
    return hit, tot


# =============================================================================
# ANA: her dataset icin ozellik + recall@k egrisi
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"donanim={device}")
    model = UNet4D_SOTA_V9().to(device)
    ck = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    miss, unexp = model.load_state_dict(ck.get('model_state_dict', ck), strict=False)
    model.eval()
    assert not miss and not unexp, "AGIRLIK UYUSMAZLIGI"
    print(f"model OK epoch={ck.get('epoch','?')}")

    rows = []
    for di, name in enumerate(ALL_NAMES[:MAX_DS]):
        if elapsed() > MAX_MIN:
            print(f"!!! sure siniri ({elapsed():.0f} dk), {di}/{MAX_DS} islendi"); break
        zdir = os.path.join(TRAIN, name + ".zarr")
        try:
            scale, q, shape = read_zarr_attrs(zdir)
            gt = read_gt(os.path.join(TRAIN, name + ".geff"))
        except Exception as ex:
            print(f"  ATLA {name}: {type(ex).__name__} {ex}"); continue
        T = int(shape[0])
        if gt["n_nodes"] == 0 or not (gt["est_nodes"] == gt["est_nodes"]):
            print(f"  ATLA {name}: GT/est yok"); continue

        # GT'si olan frame'leri tercih et (recall olculebilsin)
        gt_ts = np.unique(gt["t"])
        cand_ts = gt_ts if len(gt_ts) >= T_SAMPLE else np.arange(T)
        ts = sorted(set(np.asarray(cand_ts)[np.linspace(
            0, len(cand_ts) - 1, min(T_SAMPLE, len(cand_ts))).astype(int)].tolist()))

        acc, pooled, cache = {}, [], {}
        with torch.no_grad():
            for t in ts:
                for tt in (t - 1, t, t + 1):
                    tc = min(max(int(tt), 0), T - 1)
                    if tc not in cache:
                        cache[tc] = normalize(read_timepoint(zdir, tc), q)
                inp = np.stack([cache[max(t - 1, 0)], cache[t], cache[min(t + 1, T - 1)]], 0)
                xt = torch.from_numpy(inp).unsqueeze(0).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=(device.type == 'cuda')):
                    lg = model(xt)
                hm = torch.sigmoid(lg.float())
                c, s = peaks_gpu(hm)
                k = nms_accept(c, s, R_NMS, scale)
                acc[int(t)] = (c[k], s[k]); pooled.append(s[k])
                del xt, lg, hm
                cache = {kk: vv for kk, vv in cache.items() if kk >= t - 1}
        pooled = np.concatenate(pooled) if pooled else np.zeros(0, np.float32)
        srt = np.sort(pooled)[::-1]

        # --- ozellikler: esik basina, T=100'e normalize edilmis film geneli sayim
        frac = len(ts) / max(1, T)          # ornekledigimiz frame orani
        feats = {}
        for th in THR_FEAT:
            n_th = int((pooled > th).sum())
            feats[th] = n_th / max(1e-9, frac)        # tum filme olcekle

        # --- recall@(k*est): tek gecis NMS ozelligi sayesinde bedava
        est = gt["est_nodes"]
        rec_k = {}
        for kk in K_GRID:
            target = int(round(kk * est * frac))       # ornekledigimiz frame'ler icin
            target = max(1, min(target, len(srt)))
            thr = float(srt[target - 1]) if len(srt) else 1.0
            sel = {t: cc[ss >= thr] for t, (cc, ss) in acc.items()}
            hit, tot = recall_on_frames(sel, gt, scale)
            rec_k[kk] = hit / max(1, tot)
        best_k = max(K_GRID, key=lambda kk: rec_k[kk] ** 2 * max(0.0, 1 - 0.1 * (kk - 1)))

        rows.append(dict(name=name, colony=name[:4], est=est, T=T, n_frames=len(ts),
                         gt_n=gt["n_nodes"], feats=feats, rec_k=rec_k, best_k=best_k,
                         n_acc=len(pooled)))
        if di % 10 == 0 or di < 5:
            print(f"  [{di+1}/{min(MAX_DS,len(ALL_NAMES))}] {name} est={est:.0f} "
                  f"fr={len(ts)} Np@.2={feats[0.20]:.0f} rec@1.0={rec_k[1.0]:.3f} "
                  f"rec@2.0={rec_k[2.0]:.3f} best_k={best_k}  ({elapsed():.1f}dk)", flush=True)

    print(f"\ntoplandi: {len(rows)} dataset  ({elapsed():.1f} dk)")
    np.save(os.path.join(WORK, "v11_est_feats.npy"),
            np.array([{k: v for k, v in r.items()} for r in rows], dtype=object),
            allow_pickle=True)

    # =====================================================================
    # FIT + leave-one-COLONY-out
    # =====================================================================
    def lstsq(X, y):
        k = len(X[0])
        A = [[sum(X[i][a] * X[i][b] for i in range(len(X))) for b in range(k)] +
             [sum(X[i][a] * y[i] for i in range(len(X)))] for a in range(k)]
        for c in range(k):
            p = max(range(c, k), key=lambda r: abs(A[r][c])); A[c], A[p] = A[p], A[c]
            for r in range(k):
                if r == c: continue
                f = A[r][c] / A[c][c]
                for j in range(c, k + 1): A[r][j] -= f * A[c][j]
        return [A[c][k] / A[c][c] for c in range(k)]

    def fit_report(th, train_col=None):
        tr = [r for r in rows if (train_col is None or r["colony"] == train_col)
              and r["feats"][th] > 0]
        if len(tr) < 5: return None
        X = [[1.0, math.log(r["feats"][th])] for r in tr]
        y = [math.log(r["est"]) for r in tr]
        w = lstsq(X, y)
        def pred(r): return math.exp(w[0] + w[1] * math.log(max(1.0, r["feats"][th])))
        def errs(sel): return sorted(abs(math.log(pred(r) / r["est"])) for r in sel if r["feats"][th] > 0)
        ea = errs(rows)
        held = [r for r in rows if train_col is not None and r["colony"] != train_col]
        eh = errs(held) if held else None
        def pc(e, q): return 100 * (math.exp(e[int(q * (len(e) - 1))]) - 1)
        line = (f"  thr={th:<6.3f} n={len(tr):3d} a={w[0]:+8.4f} b={w[1]:.4f} | "
                f"TUM med%{pc(ea,.5):5.1f} p90%{pc(ea,.9):6.1f} max%{pc(ea,1):6.1f}")
        if eh: line += f" | TUTULAN({train_col}) med%{pc(eh,.5):5.1f} p90%{pc(eh,.9):6.1f}"
        print(line)
        return w, pc(ea, .5), pc(ea, .9)

    print("\n" + "=" * 104)
    print("A) est_nodes FIT  (tum veri)")
    print("=" * 104)
    best = None
    for th in THR_FEAT:
        r = fit_report(th)
        # HATA DUZELTMESI: best[1] esik degeri, medyan hata best[0]
        if r and (best is None or r[1] < best[0]):
            best = (r[1], th, r[0])
    print("\nB) leave-one-COLONY-out (test EMBRIYO-AYRIK -> gercek genelleme testi)")
    for th in THR_FEAT:
        fit_report(th, "44b6"); fit_report(th, "6bba")
    if best:
        med, th, w = best
        print(f"\n  >>> SECILEN: thr={th}  est_hat = {math.exp(w[0]):.6f} * Np^{w[1]:.6f}"
              f"   (medyan %{med:.1f})")
        print(f"      MEVCUT (12 nokta): 0.315171 * Np^1.141545 @ thr=0.200")

    # =====================================================================
    # C) n* HIPOTEZI:  en iyi k dataset'e gore degisiyor mu?
    # =====================================================================
    print("\n" + "=" * 104)
    print("C) n* SABIT OLMALI MI?  -  dataset basina en iyi k")
    print("=" * 104)
    from collections import Counter
    print(f"  best_k dagilimi: {dict(sorted(Counter(r['best_k'] for r in rows).items()))}")
    print(f"  {'k':>6} " + " ".join(f"{'rec@k':>8}" for _ in [0]) + f" {'proxy(ort)':>11}")
    for kk in K_GRID:
        rr = np.array([r["rec_k"][kk] for r in rows])
        px = np.array([r["rec_k"][kk] ** 2 * max(0.0, 1 - 0.1 * (kk - 1)) for r in rows])
        print(f"  {kk:6.2f} {rr.mean():8.4f} {px.mean():11.4f}")
    kbest_global = max(K_GRID, key=lambda kk: np.mean(
        [r["rec_k"][kk] ** 2 * max(0.0, 1 - 0.1 * (kk - 1)) for r in rows]))
    px_global = np.mean([r["rec_k"][kbest_global] ** 2 *
                         max(0.0, 1 - 0.1 * (kbest_global - 1)) for r in rows])
    px_oracle = np.mean([max(r["rec_k"][kk] ** 2 * max(0.0, 1 - 0.1 * (kk - 1))
                             for kk in K_GRID) for r in rows])
    print(f"\n  SABIT en iyi k={kbest_global}: proxy={px_global:.4f}")
    print(f"  DATASET BASINA oracle k  : proxy={px_oracle:.4f}   "
          f"kazanc={px_oracle-px_global:+.4f}")
    print(f"  >>> kazanc kucukse (<0.01) n* SABIT kalsin; buyukse k'yi ongoren"
          f" bir ozellik ara.")
    # k ile iliskili olabilecek ozellikler
    print(f"\n  best_k ile iliski (Spearman-benzeri sira korelasyonu):")
    for lbl, val in [("est_nodes", [r["est"] for r in rows]),
                     ("rec@1.0", [r["rec_k"][1.0] for r in rows]),
                     ("Np@0.2/est", [r["feats"][0.20] / r["est"] for r in rows]),
                     ("Np@0.35/Np@0.025", [r["feats"][0.35] / max(1, r["feats"][0.025]) for r in rows])]:
        a = np.argsort(np.argsort(np.array(val, float)))
        b = np.argsort(np.argsort(np.array([r["best_k"] for r in rows], float)))
        rho = np.corrcoef(a, b)[0, 1] if len(a) > 2 else float('nan')
        print(f"    {lbl:22s} rho={rho:+.3f}")

    print("\n" + "#" * 104)
    print("### V11-ADIM25-RAPOR-BASLANGIC")
    print(f"sure_dk={elapsed():.1f} n_dataset={len(rows)} "
          f"kolonı={dict(sorted(Counter(r['colony'] for r in rows).items()))}")
    if best:
        print(f"FIT thr={best[1]} A={math.exp(best[2][0]):.6f} B={best[2][1]:.6f} med%{best[0]:.1f}")
    print(f"kbest_global={kbest_global} proxy_sabit={px_global:.4f} proxy_oracle={px_oracle:.4f} "
          f"kazanc={px_oracle-px_global:+.4f}")
    print(f"best_k_dagilim={dict(sorted(Counter(r['best_k'] for r in rows).items()))}")
    for kk in K_GRID:
        print(f"k={kk} rec_ort={np.mean([r['rec_k'][kk] for r in rows]):.4f} "
              f"proxy_ort={np.mean([r['rec_k'][kk]**2*max(0.0,1-0.1*(kk-1)) for r in rows]):.4f}")
    print("### V11-ADIM25-RAPOR-BITIS")
    print("#" * 104)


if __name__ == "__main__":
    main()
