# =============================================================================
# V11 SUBMIT  (Adim S1)  -  v9 agirliklari + Adim 3'te olculen konfigurasyon
# =============================================================================
# Gizli test seti ~199 dataset, embriyo-ayrik, est_nodes YOK -> goruntuden tahmin.
#
# ADIM 3 OLCUMU (val_core, gercek resmi metrik): 0.6081 -> 0.6640
#   r_nms=7.0  n*=0.95  gate=10.0  mnn=0.80  fill=4.0
#   gap=(max_gap=8, gate_um=8.0) FIZIKSEL  min_len=6  division=KAPALI
#
# v9/v10'a gore duzeltilenler:
#   * gap closing artik FIZIKSEL uzayda (v10'da ham voxel -> z'de kapi 3.5x gevsek)
#   * esik dataset basina, N_pred ~ 0.95*est_hat olacak sekilde secilir (v10: sabit 0.025)
#   * TrackerBrain kaldirildi (mesafe kapisinin monoton fonksiyonu, bilgi eklemiyor)
#   * yerel hareket alani kaldirildi (olculdu: fark yok, hiz kazanci)
#   * division kapatildi (yaklasik metrikte dtp=0/dfp=57; resmi olcum Adim 4'te)
#   * tepe bulma GPU'da max_pool3d ile (CPU maximum_filter darbogazi kalkti)
#   * greedy NMS tek gecis: kabul(esik) = kabul(taban) & (skor>esik)  -> ikili arama yok
#
# INTERNET KAPALI CALISIR (zarr okuyucu manuel, pip yok).
# =============================================================================
import os, gzip, json, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree

COMP = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
TEST = os.path.join(COMP, "train")   # DOGRULAMA SCRIPT'I: TRAIN klasorune bakar (GT var)
WORK = "/kaggle/working"
# ADIM 5 DETEKTORU (v11-egitim kernel cikti'si). Internet KAPALI calisir:
# kernel veri kaynagi olarak baglanir, pip gerekmez.
import glob as _glob
_cand = _glob.glob("/kaggle/input/**/v11_detector_best.pt", recursive=True)
MODEL_PATH = _cand[0] if _cand else None
assert MODEL_PATH, ("v11_detector_best.pt BULUNAMADI - submit notebook'una "
                    "v11-egitim cikti'sini veri kaynagi olarak ekle")

SCALE = (1.625, 0.40625, 0.40625)   # varsayilan; her dataset kendi attrs'inden okunur

# --- ADIM 3 KONFIGURASYONU ---------------------------------------------------
# ADIM 3c (Adim 5 detektoru icin YENIDEN ayarlandi; 12 dataset, iki fold)
#   core 0.7197  44b6 0.7833  6bba 0.6997   (v9 konfig: 0.6640 / 0.7263 / 0.6442)
R_NMS      = 7.0
N_STAR     = 0.95      # 0.70->0.689 0.85->0.701 0.95->0.720 1.10->0.712 (proxy'ye GUVENME)
GATE       = 8.0       # duz: 6.5/8/10/12 hepsi 0.681
MNN_RATIO  = 0.80
FILL_GATE  = 0.0       # 0->0.7197 4->0.7193 8->0.7101
GAP_MAX    = 6         # ic optimum: (4,8)=0.710 (6,8)=0.717 (8,8)=0.712
GAP_UM     = 8.0
MIN_LEN    = 3         # 1->0.717 3->0.720 6->0.716 24->0.666
DIVISIONS  = False     # resmi metrik: geometrik dedektor ZARARLI (Adim 4)
TTA_FLIP_X = True       # 2026-09-19: orijinal+X-flip ortalamasi. N=2 -> ~7.3sa/199ds
                        # (9sa limitinin altinda). Ogrenilen degil, simetrik
                        # donusum ortalamasi -> embriyoya ozgu ogrenme riski YOK.

# --- est_nodes kestiricisi (199 dataset, Adim 2.5; embriyo-ayrik dogrulandi) -
# est_hat = EST_A * Np100^EST_B   (Np100 = thr=EST_THR ustu, 7um NMS sonrasi,
#                                  T=100'e normalize edilmis film geneli sayim)
# !!! KESTIRICI DETEKTORE BAGLI - model degisince YENIDEN FIT SART !!!
# Adim 5 detektorunun skorlari cok daha dusuk (hm_max_med 0.955->0.836);
# v9'un 0.350 esigi burada COKUYOR (med %73, max %1434).
# ADIM 2.5b (199 dataset, Adim 5 detektoru):
#   thr=0.025 -> tum med %6.1 | 44b6-fit->6bba %7.1 | 6bba-fit->44b6 %11.8 | max %72
EST_THR = 0.025
EST_A   = 0.617044
EST_B   = 1.064065

# --- aday dokumu -------------------------------------------------------------
CAND_FLOOR = 1e-4
PEAK_KERNEL = (3, 11, 11)     # ~2 um: z +-1, y/x +-5 voxel
CAND_MAX    = 8000            # frame basina, skora gore ilk N

MAX_MINUTES = 500             # kernel limiti 9 saat; guvenlik payi

t_start = time.time()
def elapsed(): return (time.time() - t_start) / 60.0

# DOGRULAMA: sadece val_core+val_public (12 dataset, gercek GT var, egitime hic
# girmedi) - TUM 199'u degil (gereksiz pahali).
TEST_NAMES = ['44b6_341df25f', '44b6_3bb3690f', '44b6_c771cb04', '44b6_8f9ecab4',
             '6bba_2540cd90', '6bba_3a1849c2', '6bba_67ebd073', '6bba_57b7cc1e',
             '44b6_0113de3b', '44b6_0b24845f', '6bba_05b6850b', '6bba_05db0fb1']
print(f"DOGRULAMA dataset sayisi: {len(TEST_NAMES)} (val_core+val_public)")
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
# LINKER
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
    if len(A) == 0 or len(B) == 0 or gate <= 0:
        return {}
    cand = []
    for i, js in enumerate(cKDTree(B).query_ball_point(A, gate)):
        for j in js:
            cand.append((float(np.linalg.norm(A[i] - B[j])), i, j))
    cand.sort()
    ua, ub, out = set(), set(), {}
    for _d, i, j in cand:
        if i in ua or j in ub:
            continue
        ua.add(i); ub.add(j); out[i] = j
    return out


def link_frames(P0, P1, gate=GATE, mnn_ratio=MNN_RATIO, fill_gate=FILL_GATE):
    n0, n1 = len(P0), len(P1)
    assign = np.full(n0, -1, np.int64)
    if n0 == 0 or n1 == 0:
        return assign, np.zeros((max(n0, 1), 3))

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

    Q = P0 + disp
    k1 = min(2, n1)
    d01, i01 = tree1.query(Q, k=k1)
    if k1 == 1:
        d01, i01 = d01[:, None], i01[:, None]
        d01 = np.pad(d01, ((0, 0), (0, 1)), constant_values=np.inf)
        i01 = np.pad(i01, ((0, 0), (0, 1)), constant_values=0)
    d10, i10 = _knn(P1, Q, k=2)
    mutual = i10[i01[:, 0], 0] == np.arange(n0)
    gated = d01[:, 0] < gate
    distinct = np.isinf(d01[:, 1]) | (d01[:, 0] < mnn_ratio * d01[:, 1])
    assign[mutual & gated & distinct] = i01[mutual & gated & distinct, 0]

    # hedef basina tek kaynak
    best = {}
    for a in np.where(assign >= 0)[0]:
        bb = int(assign[a])
        dd = float(np.linalg.norm(Q[a] - P1[bb]))
        if bb not in best or dd < best[bb][1]:
            if bb in best:
                assign[best[bb][0]] = -1
            best[bb] = (a, dd)
        else:
            assign[a] = -1

    free0 = np.where(assign < 0)[0]
    used1 = np.zeros(n1, bool)
    used1[assign[assign >= 0]] = True
    free1 = np.where(~used1)[0]
    if len(free0) and len(free1):
        for i, j in _greedy_gated(Q[free0], P1[free1], fill_gate).items():
            assign[free0[i]] = free1[j]
    return assign, disp


def find_divisions(P0, P1, assign, disp, radius=9.0, min_angle=100.0, max_ratio=2.0):
    n1 = len(P1)
    used1 = np.zeros(n1, bool)
    used1[assign[assign >= 0]] = True
    free1 = np.where(~used1)[0]
    parents = np.where(assign >= 0)[0]
    if len(free1) == 0 or len(parents) == 0:
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
# SON ISLEM
# =============================================================================
def gap_close(nodes, edges, scale, max_gap=GAP_MAX, gate_um=GAP_UM):
    """FIZIKSEL uzayda. v10 hatasi: ham voxel -> z'de kapi 3.5x gevsekti."""
    if len(nodes) == 0 or len(edges) == 0 or max_gap < 2 or gate_um <= 0:
        return nodes, edges
    sc = np.asarray(scale)
    N = len(nodes)
    out_deg = np.zeros(N, np.int32); in_deg = np.zeros(N, np.int32)
    np.add.at(out_deg, edges[:, 0], 1)
    np.add.at(in_deg, edges[:, 1], 1)
    tails, heads = np.where(out_deg == 0)[0], np.where(in_deg == 0)[0]
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


def filter_short(nodes, edges, min_len=MIN_LEN):
    if min_len <= 1 or len(nodes) == 0:
        return nodes, edges
    par = np.arange(len(nodes))
    def find(i):
        r = i
        while par[r] != r:
            r = par[r]
        while par[i] != r:
            par[i], i = r, par[i]
        return r
    for u, v in edges.tolist():
        ru, rv = find(u), find(v)
        if ru != rv:
            par[ru] = rv
    roots = np.array([find(i) for i in range(len(nodes))])
    ok = np.bincount(roots, minlength=len(nodes))[roots] >= min_len
    remap = np.cumsum(ok) - 1
    nodes = nodes[ok]
    if len(edges):
        m = ok[edges[:, 0]] & ok[edges[:, 1]]
        edges = remap[edges[m]]
    return nodes, edges


def canonicalize(nodes, edges):
    """Resmi filtreler: dt==1 / tekille / out-degree<=2. Skorlayici zaten yapiyor
    ama CSV'yi kucultur ve davranisi ongorulebilir kilar."""
    if len(edges) == 0:
        return edges.reshape(-1, 2)
    e = np.asarray(edges, np.int64).reshape(-1, 2)
    e = e[(nodes[e[:, 1], 0] - nodes[e[:, 0], 0]) == 1]
    if len(e) == 0:
        return e
    _, first = np.unique(e, axis=0, return_index=True)
    e = e[np.sort(first)]
    e = e[np.lexsort((np.arange(len(e)), e[:, 0]))]
    src = e[:, 0]
    grp = np.searchsorted(src, src, side="left")
    return e[(np.arange(len(e)) - grp) <= 1]


# =============================================================================
# OGRENILEN KENAR SINIFLANDIRICISI  (Adim 6, K_NN=16 surumu, 2026-09-18)
#   core 0.7197 (geometrik) -> 0.7291 (siniflandirici) İKİ FOLDDA da olculdu:
#     44b6 +0.0109  6bba +0.0090  (val_core, gercek resmi metrik)
#   Pickle bulunamazsa/yuklenemezse GEOMETRIK LINKER'A duser -> submission
#   HICBIR ZAMAN bu yuzden COKMEZ.
# =============================================================================
# !!! KAPALI (2026-09-19): v40 GERCEK hidden test'te v39'dan KOTU cikti (0.859
#     vs 0.863), val_core/val_public'te (İKİ bilinen embriyodan) net kazanc
#     gostermesine RAGMEN. Sebep: TUM doğrulama setlerimiz (val_core, val_public,
#     egitim havuzu) yarismanin TEK 2 bilinen embriyosundan (44b6/6bba);
#     yarisma "embryo-disjoint" -> hidden test TAMAMEN FARKLI embriyolardan.
#     Ogrenilen siniflandirici bilmeden bu 2 embriyoya ozgu oruntu ogrenmis
#     olabilir; within-domain capraz dogrulama bunu ONCEDEN GOSTEREMEZDI.
#     -> GEOMETRIK linker'a (embriyo-bagimsiz fiziksel ilkeler) GERI DONULDU.
USE_EDGE_CLF = False     # tekrar denenecekse: daha basit/regularize model VEYA
                         # geometrik+ogrenilen HIBRIT (sadece tie-breaker) dene

EDGE_CLF, EDGE_CLF_META = None, None
try:
    import pickle as _pickle
    _clf_paths = _glob.glob("/kaggle/input/**/v11_edge_clf.pkl", recursive=True) if USE_EDGE_CLF else []
    if _clf_paths:
        with open(_clf_paths[0], "rb") as _fh:
            EDGE_CLF_META = _pickle.load(_fh)
        EDGE_CLF = EDGE_CLF_META["model"]
        print(f"kenar siniflandirici yuklendi: {_clf_paths[0]} "
              f"(k_nn={EDGE_CLF_META['k_nn']} gate={EDGE_CLF_META['gate']} "
              f"sklearn={EDGE_CLF_META['sklearn']})")
    else:
        print("kenar siniflandirici KAPALI (USE_EDGE_CLF=False) veya bulunamadi "
              "-> GEOMETRIK linker kullanilacak (v39 davranisi)")
except Exception as _ex:
    print(f"UYARI: kenar siniflandirici yuklenemedi ({type(_ex).__name__}: {_ex}) "
          f"-> GEOMETRIK linker kullanilacak")
    EDGE_CLF, EDGE_CLF_META = None, None

CLF_P_MIN = 0.55         # Adim 6c: val_core'da en iyi nokta (iki fold da iyilesti)


def frame_features(P0, P1, s0, s1, k_nn, gate_cand, dens_r):
    """VEKTOREL kenar ozellikleri (Adim 6c ile BIREBIR ayni tanim).
    P0,P1 FIZIKSEL (um) koordinat; s0,s1 dedektor guven skoru."""
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
    k1, k0 = min(k_nn, n1), min(k_nn, n0)
    D01, I01 = t1.query(Q, k=k1, distance_upper_bound=gate_cand)
    D10, I10 = t0.query(P1, k=k0, distance_upper_bound=gate_cand)
    if k1 == 1: D01, I01 = D01[:, None], I01[:, None]
    if k0 == 1: D10, I10 = D10[:, None], I10[:, None]
    ii, rr = np.nonzero(np.isfinite(D01))
    if len(ii) == 0:
        return np.zeros((0, 2), np.int64), np.zeros((0, 17))
    jj = I01[ii, rr]
    d = D01[ii, rr]
    v = np.abs(P1[jj] - Q[ii])
    rank_ij = rr.astype(np.float64)
    hit = (I10[jj] == ii[:, None])
    rank_ji = np.where(hit.any(1), hit.argmax(1), k0).astype(np.float64)
    mutual = ((rank_ij == 0) & (rank_ji == 0)).astype(np.float64)
    d_i1 = D01[ii, 0]
    d_i2 = D01[ii, 1] if k1 > 1 else np.full(len(ii), np.inf)
    d_j1 = D10[jj, 0]
    d_j2 = D10[jj, 1] if k0 > 1 else np.full(len(ii), np.inf)
    fb = d + 1e-3 + gate_cand
    ratio_i = d / np.where(np.isfinite(d_i2) & (d_i2 > 0), d_i2, fb)
    ratio_j = d / np.where(np.isfinite(d_j2) & (d_j2 > 0), d_j2, fb)
    dens0 = t0.query_ball_point(Q, dens_r, return_length=True) - 1
    dens1 = t1.query_ball_point(P1, dens_r, return_length=True) - 1
    sh = np.full(len(ii), float(np.linalg.norm(shift)))
    X = np.column_stack([d, v[:, 0], v[:, 1], v[:, 2], rank_ij, rank_ji, mutual,
                         ratio_i, ratio_j, d - d_i1, np.where(np.isfinite(d_j1), d - d_j1, 0.0),
                         s0[ii], s1[jj], np.minimum(s0[ii], s1[jj]),
                         dens0[ii], dens1[jj], sh])
    return np.column_stack([ii, jj]).astype(np.int64), X


def link_by_prob(E, P, p_min):
    """Olasiliga gore greedy 1-1: en yuksek P'den basla, iki uc bosta ise kabul.
    -> KEPT edge indeksleri (E uzerinde), Adim 6c ile BIREBIR ayni algoritma."""
    if len(E) == 0:
        return np.zeros(0, np.int64)
    order = np.argsort(-P)
    used_s, used_t, keep = set(), set(), []
    for k in order:
        if P[k] < p_min: break
        s, t = int(E[k, 0]), int(E[k, 1])
        if s in used_s or t in used_t: continue
        used_s.add(s); used_t.add(t); keep.append(k)
    return np.asarray(keep, np.int64)


def build_graph(frames, T, scale, min_len=MIN_LEN, gap=True,
                edge_clf=None, scores=None, p_min=CLF_P_MIN):
    """frames: {t: (M,3) voxel} -> (nodes (N,4)[t,z,y,x], edges (E,2)).

    edge_clf verilirse (ve scores doluysa) OGRENILEN linker kullanilir
    (Adim 6c); aksi halde GEOMETRIK linker (link_frames) -> ayni fonksiyon
    hem submit hem selftest'te calisir, geriye donuk UYUMLU."""
    sc = np.asarray(scale)
    use_clf = edge_clf is not None and scores is not None
    all_nodes, all_edges = [], []
    offset, prev_P, prev_off, prev_s = 0, None, None, None
    for t in range(T):
        cc = np.asarray(frames.get(t, np.zeros((0, 3))), np.float64).reshape(-1, 3)
        P = cc * sc
        s_cur = np.asarray(scores.get(t, np.zeros(0)), np.float64) if use_clf else None
        all_nodes.append(np.column_stack([np.full(len(cc), t, np.float64), cc]))
        if prev_P is not None and len(prev_P) and len(P):
            if use_clf:
                E, X = frame_features(prev_P, P, prev_s, s_cur,
                                      EDGE_CLF_META["k_nn"], EDGE_CLF_META["gate"],
                                      EDGE_CLF_META["dens_r"])
                if len(E):
                    Pprob = edge_clf.predict_proba(X)[:, 1]
                    keep = link_by_prob(E, Pprob, p_min)
                    if len(keep):
                        e = E[keep]
                        all_edges.append(np.column_stack([prev_off + e[:, 0], offset + e[:, 1]]))
            else:
                assign, disp = link_frames(prev_P, P)
                src = np.where(assign >= 0)[0]
                if len(src):
                    all_edges.append(np.column_stack([prev_off + src, offset + assign[src]]))
                if DIVISIONS:
                    dv = find_divisions(prev_P, P, assign, disp)
                    if dv:
                        all_edges.append(np.array([[prev_off + p, offset + ch] for p, ch in dv], np.int64))
        prev_P, prev_off, prev_s = P, offset, s_cur
        offset += len(cc)
    nodes = np.vstack(all_nodes) if all_nodes else np.zeros((0, 4))
    edges = np.vstack(all_edges).astype(np.int64) if all_edges else np.zeros((0, 2), np.int64)
    if gap:
        nodes, edges = gap_close(nodes, edges, scale)
    nodes, edges = filter_short(nodes, edges, min_len)
    return nodes, canonicalize(nodes, edges)


# =============================================================================
# OZ-TEST  -  5 SAATLIK GPU ISINDEN ONCE boru hattini dogrula
# =============================================================================
def selftest():
    A = lambda t: [10.0, 50.0 + 2 * t, 50.0 + 2 * t]
    B = lambda t: [10.0, 150.0 + 2 * t, 150.0 + 2 * t]

    # T1: kesintisiz 3 frame, 2 hucre -> 6 node / 4 kenar
    f = {t: np.array([A(t), B(t)]) for t in (0, 1, 2)}
    n, e = build_graph(f, 3, SCALE, min_len=1, gap=False)
    assert len(n) == 6 and len(e) == 4, f"T1 basarisiz: {len(n)} node {len(e)} kenar"

    # T2: min_len=6 -> 3'luk bilesenler silinir
    n2, e2 = build_graph(f, 3, SCALE, min_len=6, gap=False)
    assert len(n2) == 0 and len(e2) == 0, f"T2 basarisiz: {len(n2)} node"

    # T3: t=2 eksik -> gap closing ara node ekleyip baglar
    fg = {t: np.array([A(t), B(t)]) for t in (0, 1, 3, 4)}
    n3, e3 = build_graph(fg, 5, SCALE, min_len=1, gap=True)
    assert len(n3) == 10 and len(e3) == 8, f"T3 basarisiz: {len(n3)} node {len(e3)} kenar"
    assert sorted(np.unique(n3[:, 0]).tolist()) == [0, 1, 2, 3, 4], "T3: t=2 eklenmedi"

    # T4: gap KAPALI iken baglanmamali
    n4, e4 = build_graph(fg, 5, SCALE, min_len=1, gap=False)
    assert len(e4) == 4, f"T4 basarisiz: {len(e4)} kenar"

    # T5: NMS -> 7um icindeki iki aday tek adaya inmeli
    c = np.array([[10, 50, 50], [10, 50, 55], [10, 150, 150]], np.int16)
    k = nms_accept(c, np.array([0.9, 0.5, 0.8], np.float32), R_NMS, SCALE)
    assert len(k) == 2 and 0 in k and 2 in k, f"T5 basarisiz: {k}"

    # T6: yazici -> gecerli CSV
    import pandas as pd
    tmp = os.path.join(WORK, "_selftest.csv")
    write_submission({"ds1": (n, e), "ds2": (n3, e3)}, tmp)
    df = pd.read_csv(tmp)
    assert list(df.columns) == ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
                               "source_id", "target_id"], f"T6 kolonlar: {list(df.columns)}"
    assert (df.row_type == "node").sum() == 16 and (df.row_type == "edge").sum() == 12, \
        f"T6 satir sayisi: {(df.row_type=='node').sum()}/{(df.row_type=='edge').sum()}"
    os.remove(tmp)

    # T7: link_by_prob -> saf algoritma, GERCEK sklearn modeline gerek YOK.
    #   3 aday kenar: (0,0) yuksek P, (0,1) dusuk P (0 zaten meskul), (1,1) orta P
    #   -> greedy: (0,0) once secilir (en yuksek P), (0,1) CAKISIR (0 mesgul) atlanir,
    #      (1,1) secilir. p_min altindaki (varsayimsal) kenar ELENIR.
    E7 = np.array([[0, 0], [0, 1], [1, 1], [2, 2]], np.int64)
    P7 = np.array([0.90, 0.80, 0.70, 0.10])          # son kenar p_min=0.55 altinda
    k7 = link_by_prob(E7, P7, 0.55)
    kept = set(map(tuple, E7[k7].tolist()))
    assert kept == {(0, 0), (1, 1)}, f"T7 basarisiz: {kept}"

    # T8: frame_features -> BOS girdide COKMEMELI (0 node kalan bir frame olabilir)
    r8, x8 = frame_features(np.zeros((0, 3)), np.array([[0., 0., 0.]]),
                            np.zeros(0), np.array([0.5]), 16, 12.0, 15.0)
    assert len(r8) == 0 and x8.shape == (0, 17), f"T8 basarisiz: {r8.shape} {x8.shape}"

    print("OZ-TEST: T1-T8 GECTI (linking, min_len, gap closing, NMS, CSV, "
          "ogrenilen-linker greedy, bos-girdi guvenligi)")


# =============================================================================
# SUBMISSION YAZICI  (v9/v10'un kabul edilmis kolon duzeni birebir)
# =============================================================================
def write_submission(datasets, out_path):
    import pandas as pd
    frames = []
    for name, (nodes, edges) in datasets.items():
        N = len(nodes)
        if N == 0:
            continue      # bos dataset -> hic satir yazma
        frames.append(pd.DataFrame({
            "dataset": name, "row_type": "node", "node_id": np.arange(N, dtype=np.int64),
            "t": nodes[:, 0].astype(np.int64), "z": np.rint(nodes[:, 1]).astype(np.int64),
            "y": np.rint(nodes[:, 2]).astype(np.int64), "x": np.rint(nodes[:, 3]).astype(np.int64),
            "source_id": -1, "target_id": -1}))
        if len(edges):
            frames.append(pd.DataFrame({
                "dataset": name, "row_type": "edge", "node_id": -1,
                "t": -1, "z": -1, "y": -1, "x": -1,
                "source_id": edges[:, 0].astype(np.int64),
                "target_id": edges[:, 1].astype(np.int64)}))
    cols = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]
    if frames:
        df = pd.concat(frames, ignore_index=True)
        df.insert(0, "id", np.arange(len(df), dtype=np.int64))
        df.to_csv(out_path, index=False)
    else:
        pd.DataFrame(columns=cols).to_csv(out_path, index=False)
    return out_path


# =============================================================================
# TEK DATASET
# =============================================================================
def process(zarr_dir, name, model, device):
    scale, q, shape = read_zarr_attrs(zarr_dir)
    T = int(shape[0])
    t_ds = time.time()

    # --- 1) model gecisi + aday dokumu -------------------------------------
    #   TTA (2026-09-19): orijinal + X-ekseninde flip, TEK forward cagrisinda
    #   (batch=2, ayri iki cagridan daha verimli). Ogrenilen linker'daki gibi
    #   embriyoya ozgu bir sey OGRENMIYOR -> ayni girdinin simetrik donusumunu
    #   ortalayarak varyans azaltiyor, kategorik olarak daha dusuk riskli.
    #   BUTCE: N=2 -> ~7.3 saat/199 dataset (9 saatlik limitin altinda, AZ pay).
    #   N=3+ limiti asar, YAPILMADI.
    cands = []                      # frame -> (coords int16 (K,3), scores (K,))
    cache = {}
    with torch.no_grad():
        for t in range(T):
            for tt in (t - 1, t, t + 1):
                tc = min(max(tt, 0), T - 1)
                if tc not in cache:
                    cache[tc] = normalize(read_timepoint(zarr_dir, tc), q)
            for old_t in [k for k in cache if k < t - 1]:
                del cache[old_t]
            inp = np.stack([cache[max(t - 1, 0)], cache[t], cache[min(t + 1, T - 1)]], 0)
            x = torch.from_numpy(inp).unsqueeze(0)                    # (1,3,Z,Y,X)
            if TTA_FLIP_X:
                xb = torch.cat([x, torch.flip(x, dims=[-1])], dim=0)  # (2,3,Z,Y,X)
            else:
                xb = x
            xb = xb.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == 'cuda')):
                logits = model(xb)
            hm = torch.sigmoid(logits.float())
            if TTA_FLIP_X:
                hm = (hm[0:1] + torch.flip(hm[1:2], dims=[-1])) / 2.0
            cands.append(peaks_gpu(hm))
            del xb, logits, hm
    t_fwd = time.time() - t_ds

    # --- 2) frame basina greedy NMS (TABANDA, tek gecis) -------------------
    acc = []                        # frame -> (coords (M,3), scores (M,))
    pooled = []
    for c, s in cands:
        k = nms_accept(c, s, R_NMS, scale)
        acc.append((c[k], s[k]))
        pooled.append(s[k])
    pooled = np.concatenate(pooled) if pooled else np.zeros(0, np.float32)
    n_acc = len(pooled)

    # --- 3) est_nodes tahmini + hedef sayim -> esik -------------------------
    np_hi = int((pooled > EST_THR).sum())
    if np_hi > 0 and T > 0:
        np100 = np_hi * (100.0 / T)
        est_hat = EST_A * (np100 ** EST_B) * (T / 100.0)
    else:
        est_hat = float(n_acc)                       # geri donus: hepsini al
    K = int(round(N_STAR * est_hat))
    K = max(1, min(K, n_acc))
    order = np.sort(pooled)[::-1]
    thr = float(order[K - 1]) if n_acc else 1.0

    frames, fscores = {}, {}
    n_sel = 0
    for t, (c, s) in enumerate(acc):
        m = s >= thr
        if m.any():
            frames[t] = c[m].astype(np.float64)
            fscores[t] = s[m].astype(np.float64)
            n_sel += int(m.sum())

    # --- 4/5) linking + son islem --------------------------------------------
    #   EDGE_CLF varsa OGRENILEN linker (Adim 6c, core +0.0094 iki foldda);
    #   yoksa/patlarsa GEOMETRIK linker'a duser -> submission asla cokmez.
    try:
        nodes, edges = build_graph(frames, T, scale, edge_clf=EDGE_CLF, scores=fscores)
    except Exception as _ex:
        print(f"  !!! OGRENILEN LINKER HATASI ({type(_ex).__name__}: {_ex}) "
              f"-> GEOMETRIK linker'a duruluyor", flush=True)
        nodes, edges = build_graph(frames, T, scale)

    print(f"  {name}: T={T} aday={sum(len(c) for c,_ in cands)} NMS={n_acc} "
          f"np@{EST_THR}={np_hi} est_hat={est_hat:.0f} thr={thr:.5f} "
          f"secilen={n_sel} -> node={len(nodes)} kenar={len(edges)} "
          f"(fwd {t_fwd:.0f}s, toplam {time.time()-t_ds:.0f}s)", flush=True)
    return nodes, edges




# =============================================================================
# HACK DENETCISI  (kullanici kurali 2026-09-18: "hackli bir sey kullanmayalim")
#   Submission'i metrik exploit imzalarina karsi tarar: goruntu-disi koordinat,
#   negatif zaman, cross-clip, dt!=1, cok ebeveyn, out-degree>2, dev bilesen.
#   Kirliyse submission.csv SILINIR ve kernel COKER -> Kaggle'a kirli dosya gitmez.
#   Test edildi: hack notebook'larindaki birebir hub+fork yapisini 3 ayri
#   imzadan yakaliyor; kendi v39 ciktimiz TEMIZ.
# =============================================================================
import csv
from collections import defaultdict
def audit(path, T=None, Z=None, Y=None, X=None, giant_frac=0.5, raise_on_fail=True,
          verbose=True):
    nodes = defaultdict(dict)          # dataset -> node_id -> (t,z,y,x)
    edges = defaultdict(list)          # dataset -> [(s,t)]
    bad = []
    n_node = n_edge = 0
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            ds, rt = row["dataset"], row["row_type"]
            if rt == "node":
                n_node += 1
                t, z, y, x = (float(row[k]) for k in ("t", "z", "y", "x"))
                nodes[ds][int(float(row["node_id"]))] = (t, z, y, x)
            elif rt == "edge":
                n_edge += 1
                edges[ds].append((int(float(row["source_id"])), int(float(row["target_id"]))))
            else:
                bad.append(f"bilinmeyen row_type={rt!r}")

    for ds in sorted(set(nodes) | set(edges)):
        nd, ed = nodes.get(ds, {}), edges.get(ds, [])
        # 1-2. koordinat ve zaman sınırları
        for nid, (t, z, y, x) in nd.items():
            if t < 0:
                bad.append(f"[{ds}] NEGATİF ZAMAN node {nid}: t={t}"); break
            if min(z, y, x) < 0:
                bad.append(f"[{ds}] GÖRÜNTÜ-DIŞI koordinat node {nid}: ({z},{y},{x})"); break
            for v, lim, nm in ((t, T, "t"), (z, Z, "z"), (y, Y, "y"), (x, X, "x")):
                if lim is not None and v >= lim + 1:
                    bad.append(f"[{ds}] SINIR DIŞI {nm}={v} >= {lim} (node {nid})"); break
        # 3,4,8. kenarlar
        indeg, outdeg = defaultdict(int), defaultdict(int)
        for s, tg in ed:
            if s not in nd or tg not in nd:
                bad.append(f"[{ds}] TANIMSIZ/CROSS-CLIP kenar {s}->{tg} (node bu dataset'te yok)"); break
            dt = nd[tg][0] - nd[s][0]
            if dt != 1:
                bad.append(f"[{ds}] dt={dt:g} kenar {s}->{tg} (hub/atlama imzası)"); break
            indeg[tg] += 1; outdeg[s] += 1
        # 5-6. derece
        mi = max(indeg.values(), default=0)
        mo = max(outdeg.values(), default=0)
        if mi > 1:
            bad.append(f"[{ds}] ÇOK EBEVEYNLİ node (in-degree={mi}) — hub birleştirme imzası")
        if mo > 2:
            bad.append(f"[{ds}] out-degree={mo} > 2")
        # 7. dev bileşen (union-find)
        par = {n: n for n in nd}
        def find(a):
            while par[a] != a:
                par[a] = par[par[a]]; a = par[a]
            return a
        for s, tg in ed:
            if s in par and tg in par:
                ra, rb = find(s), find(tg)
                if ra != rb:
                    par[ra] = rb
        comp = defaultdict(int)
        for n in nd:
            comp[find(n)] += 1
        if nd:
            big = max(comp.values())
            if big > giant_frac * len(nd) and len(nd) > 50:
                bad.append(f"[{ds}] DEV BİLEŞEN: {big}/{len(nd)} node tek bileşende "
                           f"(>%{100*giant_frac:.0f}) — hub imzası")

    ok = not bad
    if verbose:
        print(f"[DENETÇİ] {path}: {len(nodes)} dataset, {n_node} node, {n_edge} kenar -> "
              f"{'TEMİZ ✓' if ok else f'{len(bad)} İHLAL ✗'}")
        for b in bad[:20]:
            print(f"   ✗ {b}")
    if not ok and raise_on_fail:
        raise ValueError(f"HACK DENETÇİSİ BAŞARISIZ ({len(bad)} ihlal) — submission GÖNDERİLMEZ")
    return ok, bad


# =============================================================================
# ANA
# =============================================================================
def main():
    selftest()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"donanim={device}  torch={torch.__version__}")
    assert os.path.exists(MODEL_PATH), f"model yok: {MODEL_PATH}"

    model = UNet4D_SOTA_V9().to(device)
    ck = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    sd = ck.get('model_state_dict', ck)
    miss, unexp = model.load_state_dict(sd, strict=False)
    model.eval()
    print(f"model OK eksik={len(miss)} beklenmeyen={len(unexp)} epoch={ck.get('epoch','?')}")
    assert not miss and not unexp, "AGIRLIK UYUSMAZLIGI"

    print(f"konfig: r_nms={R_NMS} n*={N_STAR} gate={GATE} mnn={MNN_RATIO} fill={FILL_GATE} "
          f"gap=({GAP_MAX},{GAP_UM}um) min_len={MIN_LEN} div={DIVISIONS}")
    print(f"linker: {'OGRENILEN (p_min=' + str(CLF_P_MIN) + ')' if EDGE_CLF is not None else 'GEOMETRIK (yedek)'}")

    datasets = {}
    for i, name in enumerate(TEST_NAMES):
        if elapsed() > MAX_MINUTES:
            print(f"!!! SURE SINIRI ({elapsed():.0f} dk) - kalan {len(TEST_NAMES)-i} dataset BOS")
            break
        print(f"[{i+1}/{len(TEST_NAMES)}] {name}  ({elapsed():.1f} dk)", flush=True)
        try:
            datasets[name] = process(os.path.join(TEST, name + ".zarr"), name, model, device)
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"  HATA {name}: {type(ex).__name__} {ex} -> bos graf")
            datasets[name] = (np.zeros((0, 4)), np.zeros((0, 2), np.int64))
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if i == 0:
            per = elapsed()
            print(f"  >>> ilk dataset {per:.2f} dk -> {len(TEST_NAMES)} dataset icin "
                  f"tahmini {per*len(TEST_NAMES):.0f} dk ({per*len(TEST_NAMES)/60:.1f} saat)")

    # islenmeyenler icin bos graf (gecerli CSV garantisi)
    for name in TEST_NAMES:
        datasets.setdefault(name, (np.zeros((0, 4)), np.zeros((0, 2), np.int64)))

    out = write_submission(datasets, os.path.join(WORK, "submission.csv"))
    # --- HACK DENETIMI: kirliyse dosyayi sil ve cok ---
    try:
        audit(out, T=None, Z=None, Y=None, X=None, raise_on_fail=True, verbose=True)
    except ValueError:
        os.remove(out)
        print("!!! submission.csv SILINDI - hack denetcisi gecmedi")
        raise
    tn = sum(len(n) for n, _ in datasets.values())
    te = sum(len(e) for _, e in datasets.values())
    print(f"\nTAMAMLANDI  {len(datasets)} dataset  {tn} node  {te} kenar  "
          f"({elapsed():.1f} dk)  -> {out}")
    print(f"dosya boyutu: {os.path.getsize(out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
