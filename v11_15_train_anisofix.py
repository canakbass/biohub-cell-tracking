# =============================================================================
# V11 ADIM 5 - DETEKTORU GERCEK GT ILE YENIDEN EGIT  (GPU, internet ACIK)
# =============================================================================
# HEDEF TEK SAYI:  recall @ (N = k*est_nodes)
#   v9 su an: 0.915 @ n=1.10   (tavani 0.957 ama n=2.1 gerekiyor -> carpan yiyor)
#   hedef:    0.97  @ n~1.0
#
# NEDEN v10 BASARISIZ OLURDU (F1): seyrek GT (%3.6) yogun supervision gibi
#   kullanilinca, patch'teki onlarca GERCEK hucre "arka plan" olarak ogretiliyor.
#   CenterNet focal loss'un negatif terimi onlari AKTIF OLARAK bastiriyor.
#
# BU SCRIPT'IN DEGISTIRDIGI TEK SEY: ETIKETLER VE LOSS.  Mimari v9 ile BIREBIR
#   ayni -> agirliklar sicak baslatilabiliyor VE tek degisken izole ediliyor.
#
#   (F1) YOKSAY MASKESI: parlak (hucre benzeri) ama GT'de olmayan bolgelerde
#        negatif loss = 0. Floresan cekirdek goruntusunde hucreler parlak
#        bolgeler; oralara "arka plan" demek yanlis. Sadece GERCEKTEN karanlik
#        arka plani negatif sayiyoruz.
#   (F2) DEEP SUPERVISION KAPALI: coarse seviyelerde target.eq(1.0) BOS kaliyor
#        -> num_pos=0 -> normalize edilmemis sadece-negatif loss, agirligin
#        %50'si "her yerde 0 bas" diyordu. Artik sadece en ince seviye.
#   (F3) CHECKPOINT: state_dict() REFERANS dondurur + ModelEMA in-place copy_()
#        -> v9'da best_1/2/3.pt hepsi EN SON agirliklari iceriyordu.
#        Artik copy.deepcopy ile kaydediliyor.
#   (ek) ORNEKLEME: v9/v10 %80 hucre-merkezli patch aliyordu -> inference
#        dagilimindan sapma. Artik %50 GT-merkezli / %50 duzgun rastgele.
#
# Dogrulama: val_core uzerinde recall@(k*est) EGRISI (k in 0.7..2.0), yani
#   dogrudan optimize ettigimiz sey. En iyi k'ye gore checkpoint kaydediliyor.
# =============================================================================
import sys, subprocess
from importlib.metadata import version as _pkgver

def _v(n):
    try: return _pkgver(n)
    except Exception: return None

if _v("zarr") is None:
    print("[bootstrap] zarr kuruluyor...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zarr"], check=False)
print(f"[bootstrap] numpy={_v('numpy')} scipy={_v('scipy')} zarr={_v('zarr')} "
      f"torch={_v('torch')}", flush=True)

import os, json, math, time, copy, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

COMP  = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
TRAIN = os.path.join(COMP, "train")
WORK  = "/kaggle/working"
# SICAK BASLATMA: v9 DEGIL, mevcut EN IYI detektor (v11-egitim ep91, hidden
# testte 0.863 dogrulanmis). Tur 2 (r2) AYNI mimariyle devam edip KAZANC
# bulamamisti ("DETEKTOR BU TARIFLE YAKINSAMIS... farkli tarif lazim,
# izotropik pooling" - STATE.md 4c) -> simdi mimari GERCEKTEN degisiyor,
# en iyi bilinen agirliklardan baslamak mantikli (bkz asagida pool_kernels).
import glob as _glob
_prev_cand = _glob.glob("/kaggle/input/**/v11_detector_best.pt", recursive=True)
PREV = (_prev_cand[0] if _prev_cand else
        "/kaggle/input/models/hcanakbas/biohub-v9-03-sota/pytorch/default/1/biohub_v9_sota_f1_868.pt")
print(f"SICAK BASLATMA KAYNAGI: {PREV}")

SCALE        = (1.625, 0.40625, 0.40625)
MAX_MATCH_UM = 7.0

# --- val seti (ADIM 0'da secildi; EGITIME GIRMEZ) ---------------------------
VAL_CORE = ['44b6_341df25f', '44b6_3bb3690f', '44b6_c771cb04', '44b6_8f9ecab4',
            '6bba_2540cd90', '6bba_3a1849c2', '6bba_67ebd073', '6bba_57b7cc1e']
VAL_PUBLIC = ['44b6_0113de3b', '44b6_0b24845f', '6bba_05b6850b', '6bba_05db0fb1']

# --- egitim ------------------------------------------------------------------
# zarr chunk = TAM timepoint (1,64,256,256) -> kucuk patch okumak IO tasarrufu
# SAGLAMIYOR (tum chunk zaten aciliyor). Patch buyutmek bedava; sinir BELLEK.
SMOKE        = bool(int(os.environ.get("V11_SMOKE", "0")))
PATCH        = tuple(int(v) for v in os.environ.get("V11_PATCH", "48,256,256").split(","))
BATCH        = 1
ACCUM        = 4
LR           = 1.2e-4          # sicak baslatma -> dusuk
WD           = 5e-4
EMA_DECAY    = 0.999
STEPS_PER_EP = 40 if SMOKE else 200
MAX_EPOCH    = 3 if SMOKE else 92    # 92x4dk + recall 25dk + adj 18dk = 411dk (butce 455)
MAX_MINUTES  = 14 if SMOKE else 455    # 9 sa GPU limiti - guvenlik payi
WORKERS      = 4

GT_SIGMA_UM  = 3.0             # v9 2.0 kullaniyordu; tolerans 7 um -> daha genis kolay
POS_FRAC     = 0.50            # patch'lerin %50'si GT merkezli (v9/v10: %80)
# YOKSAY ESIGI: SABIT DEGIL.  Beklenen hucre hacim orani yogunlukla %0.9 - %37
# arasi degisiyor (38 vs 786 hucre/frame) -> tek esik iki rejime hizmet edemez.
# Bunun yerine: est_nodes'tan beklenen hucre hacim oranini hesapla, patch'in
# o yuzdelik dilimindeki EN PARLAK vokselleri yoksay bolgesi yap.
CELL_R_UM    = 4.5             # hucre yaricapi (parlak hacim tahmini icin)
IGN_FRAC_MIN = 0.02            # yoksay oraninin alt/ust siniri
IGN_FRAC_MAX = 0.45
IGNORE_TAU   = 0.10            # geri donus (est_nodes yoksa)
IGNORE_KEEP_UM = 5.0           # GT'ye bu kadar yakin parlaklik yoksayilmaz (pozitif bolge)
FOCAL_A, FOCAL_B = 2.0, 4.0

# --- dogrulama ---------------------------------------------------------------
# Duman testi 19 GT node ile dogruluyordu -> granularite %5.3, GURULTU.
# Anlamli olcum icin >=1000 GT node: 8 dataset x 20 frame ~ 1000-1500 node.
VAL_EVERY    = 1 if SMOKE else 10      # epoch (val ~3 dk, %8 ek yuk)
VAL_DS       = 2 if SMOKE else 8       # val_core'un TAMAMI
VAL_FRAMES   = 3 if SMOKE else 20      # dataset basina frame
R_NMS        = 7.0
K_GRID       = [0.8, 1.0, 1.2, 1.5, 2.0]
# GERCEK metrik dogrulamasi (pahali: dataset basina TAM 100 frame ~ 75 s)
ADJ_EVERY    = 1 if SMOKE else 25      # epoch
ADJ_DS       = 1 if SMOKE else 4       # dataset
CAND_FLOOR   = 1e-4
CAND_MAX     = 8000
PEAK_KERNEL  = (3, 11, 11)

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
        # IZOTROPIK DUZELTME (2026-09-22): voxel olcegi (1.625, 0.40625, 0.40625)
        # -> Z fiziksel olarak Y/X'ten 4x KABA. Eski semada toplam havuzlama
        # Z=4x Y/X=8x idi -> etkin fiziksel cozunurluk Z'de 6.5um, Y/X'te 3.25um
        # (2x kaba). Bu SABIT bir fiziksel gercek, 2 bilinen embriyoda kalibre
        # edilmis bir HIPERPARAMETRE DEGIL -> LAP/kenar-siniflandirici gibi
        # embriyo-ozel ezberleme riski tasimiyor (bkz STATE.md LAP regresyonu).
        # Duzeltme: orta katmanda Z havuzlamasini kaldir -> toplam Z=2x, Y/X=8x
        # -> etkin cozunurluk ikisinde de 3.25um (izotropik).
        self.pool_kernels = [(1, 2, 2), (1, 2, 2), (2, 2, 2)]
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


# =============================================================================
# VERI
# =============================================================================
def read_zarr_meta(zarr_dir):
    import zarr
    grp = zarr.open_group(zarr_dir, mode='r')
    q = {float(k): float(v)
         for k, v in (grp.attrs.get("image_statistics", {}).get("quantiles", {}) or {}).items()}
    arr = zarr.open(os.path.join(zarr_dir, "0"), mode='r')
    return q, tuple(arr.shape)


def normalize(vol_u16, q):
    lo, hi = q.get(0.001, 100.0), q.get(0.999, 1000.0)
    if 0.001 not in q and vol_u16.size > 0:
        lo = float(np.percentile(vol_u16[::4], 0.1))
        hi = float(np.percentile(vol_u16[::4], 99.9))
    img = (vol_u16.astype(np.float32) - lo) / max(1e-6, hi - lo)
    return np.clip(img, 0.0, 4.0, out=img)


def read_gt(geff_path):
    """GT VOXEL cinsinden (Adim 0: 199/199 dogrulandi) -> frame -> [[z,y,x],...]"""
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
    byf = {}
    for i in range(len(t)):
        byf.setdefault(int(t[i]), []).append((z[i], y[i], x[i]))
    return dict(byframe={k: np.asarray(v) for k, v in byf.items()},
                t=t, z=z, y=y, x=x, n_nodes=len(t),
                est_nodes=float(est) if est is not None else float("nan"))


def gauss_blob(sigma_um, scale):
    """GT noktalarina konacak Gauss cekirdegi (fiziksel sigma -> voxel)"""
    sv = tuple(max(0.5, sigma_um / s) for s in scale)
    r = [int(np.ceil(3 * s)) for s in sv]
    zz, yy, xx = np.ogrid[-r[0]:r[0]+1, -r[1]:r[1]+1, -r[2]:r[2]+1]
    b = np.exp(-((zz**2)/(2*sv[0]**2) + (yy**2)/(2*sv[1]**2) + (xx**2)/(2*sv[2]**2)))
    return b.astype(np.float32), r


BLOB, BR = gauss_blob(GT_SIGMA_UM, SCALE)
# yoksay maskesinde GT cevresinde KORUNACAK yaricap (voxel)
KEEP_R = [max(1, int(round(IGNORE_KEEP_UM / s))) for s in SCALE]


def splat(dst, center, kern, rad, op="max"):
    """kern'i dst icine center'a yerlestir (sinir kirpmali)"""
    lo = [max(0, center[i] - rad[i]) for i in range(3)]
    hi = [min(dst.shape[i], center[i] + rad[i] + 1) for i in range(3)]
    if any(lo[i] >= hi[i] for i in range(3)):
        return
    ko = [lo[i] - (center[i] - rad[i]) for i in range(3)]
    kh = [ko[i] + (hi[i] - lo[i]) for i in range(3)]
    sub = dst[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    ksub = kern[ko[0]:kh[0], ko[1]:kh[1], ko[2]:kh[2]]
    if op == "max":
        np.maximum(sub, ksub, out=sub)
    else:
        sub[...] = np.minimum(sub, ksub) if op == "min" else ksub


class GTPatchDataset(Dataset):
    """Patch + (heatmap, yoksay maskesi).

    ETIKET MANTIGI (F1 duzeltmesi):
      pozitif : GT node'larina Gauss blob
      yoksay  : PARLAK (img > IGNORE_TAU) ama GT'ye IGNORE_KEEP_UM'den uzak
                -> negatif loss BURADA SIFIR. Cunku floresan cekirdek
                   goruntusunde parlak bolge = hucre; annotate edilmemis
                   olmasi "arka plan" demek DEGIL.
      negatif : geri kalan (gercekten karanlik) her sey
    """
    def __init__(self, names, n_samples):
        self.names = names
        self.n = n_samples
        self.meta = {}
        for nm in names:
            try:
                q, shape = read_zarr_meta(os.path.join(TRAIN, nm + ".zarr"))
                gt = read_gt(os.path.join(TRAIN, nm + ".geff"))
                if gt["n_nodes"] == 0:
                    continue
                self.meta[nm] = (q, shape, gt)
            except Exception as ex:
                print(f"  [ds] ATLA {nm}: {type(ex).__name__} {ex}")
        self.last_stats = None
        self.names = [n for n in names if n in self.meta]
        print(f"  [ds] {len(self.names)} dataset, "
              f"{sum(m[2]['n_nodes'] for m in self.meta.values())} GT node")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        import zarr
        pz, py, px = PATCH
        for _try in range(8):
            nm = random.choice(self.names)
            q, shape, gt = self.meta[nm]
            T, Z, Y, X = shape
            frames = sorted(gt["byframe"].keys())
            if not frames:
                continue
            t = random.choice(frames)
            coords = gt["byframe"][t]

            if random.random() < POS_FRAC and len(coords):
                c = coords[random.randrange(len(coords))]
                z0 = int(np.clip(c[0] - pz // 2 + random.randint(-4, 4), 0, max(0, Z - pz)))
                y0 = int(np.clip(c[1] - py // 2 + random.randint(-24, 24), 0, max(0, Y - py)))
                x0 = int(np.clip(c[2] - px // 2 + random.randint(-24, 24), 0, max(0, X - px)))
            else:
                z0 = random.randint(0, max(0, Z - pz))
                y0 = random.randint(0, max(0, Y - py))
                x0 = random.randint(0, max(0, X - px))

            arr = zarr.open(os.path.join(TRAIN, nm + ".zarr", "0"), mode='r')
            tp, tn = max(0, t - 1), min(T - 1, t + 1)
            try:
                raw = [np.asarray(arr[tt, z0:z0+pz, y0:y0+py, x0:x0+px]) for tt in (tp, t, tn)]
            except Exception:
                continue
            if raw[1].shape != PATCH:
                pad = [np.zeros(PATCH, np.float32) for _ in range(3)]
                for i, r in enumerate(raw):
                    pad[i][:r.shape[0], :r.shape[1], :r.shape[2]] = normalize(r, q)
                stack = np.stack(pad, 0)
            else:
                stack = np.stack([normalize(r, q) for r in raw], 0)

            cur = stack[1]
            # --- beklenen parlak (hucre) hacim orani -> uyarlanabilir esik ---
            vox_um3 = SCALE[0] * SCALE[1] * SCALE[2]
            patch_um3 = pz * py * px * vox_um3
            full_um3 = Z * Y * X * vox_um3
            cell_um3 = 4.0 / 3.0 * math.pi * CELL_R_UM ** 3
            est = gt["est_nodes"]
            if est == est and est > 0 and T > 0:
                cells_here = (est / T) * (patch_um3 / max(1e-9, full_um3))
                ign_frac = float(np.clip(cells_here * cell_um3 / patch_um3,
                                         IGN_FRAC_MIN, IGN_FRAC_MAX))
                tau = float(np.quantile(cur[::2, ::2, ::2], 1.0 - ign_frac))
            else:
                ign_frac, tau = float('nan'), IGNORE_TAU
            hm = np.zeros(PATCH, np.float32)
            keep = np.zeros(PATCH, np.float32)      # GT cevresi (yoksanmaz)
            ones = np.ones([2 * r + 1 for r in KEEP_R], np.float32)
            n_pos = 0
            for c in coords:
                cz, cy, cx = int(round(c[0])) - z0, int(round(c[1])) - y0, int(round(c[2])) - x0
                if not (0 <= cz < pz and 0 <= cy < py and 0 <= cx < px):
                    continue
                splat(hm, (cz, cy, cx), BLOB, BR, "max")
                splat(keep, (cz, cy, cx), ones, KEEP_R, "max")
                n_pos += 1

            # F1: parlak ama GT'den uzak -> yoksay
            ignore = ((cur > tau) & (keep < 0.5)).astype(np.float32)
            neg_w = 1.0 - ignore                    # negatif loss agirligi

            if n_pos == 0 and random.random() < 0.85:
                continue                            # cogunlukla pozitifli patch istiyoruz
            self.last_stats = (float(ignore.mean()), tau, ign_frac, n_pos)
            return (torch.from_numpy(stack), torch.from_numpy(hm).unsqueeze(0),
                    torch.from_numpy(neg_w).unsqueeze(0))
        # 8 denemede olmadi -> bos negatif patch
        return (torch.zeros((3,) + PATCH), torch.zeros((1,) + PATCH),
                torch.ones((1,) + PATCH))


# =============================================================================
# LOSS  (F2 duzeltmesi: SADECE en ince seviye + yoksay maskeli negatif)
# =============================================================================
class MaskedCenterNetLoss(nn.Module):
    def __init__(self, alpha=FOCAL_A, beta=FOCAL_B):
        super().__init__()
        self.a, self.b = alpha, beta

    def forward(self, logits, target, neg_w):
        if isinstance(logits, (list, tuple)):
            logits = logits[-1]                     # F2: sadece tam cozunurluk
        p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
        pos = target.eq(1.0).float()
        neg = target.lt(1.0).float() * neg_w        # <-- yoksay maskesi
        pos_loss = torch.log(p) * torch.pow(1 - p, self.a) * pos
        neg_loss = (torch.log(1 - p) * torch.pow(p, self.a)
                    * torch.pow(1 - target, self.b) * neg)
        n = pos.sum()
        if n < 1:
            return -(neg_loss.sum()) / max(1.0, float(neg.sum().item()) / 1e4)
        return -(pos_loss.sum() + neg_loss.sum()) / n


# =============================================================================
# RESMI METRIK  (Adim 4'te resmi skorlayiciyla adj_farki=0.0000 dogrulandi)
# =============================================================================
ALPHA, DIV_W = 0.1, 0.1

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
# LINKER + SON ISLEM  (submit ile AYNI kod -> dogrulama gercek boru hattini olcer)
# =============================================================================
GATE, MNN_RATIO, FILL_GATE = 10.0, 0.80, 4.0
GAP_MAX, GAP_UM, MIN_LEN, DIVISIONS = 8, 8.0, 6, False

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


def build_graph(frames, T, scale, min_len=MIN_LEN, gap=True):
    """frames: {t: (M,3) voxel} -> (nodes (N,4)[t,z,y,x], edges (E,2))"""
    sc = np.asarray(scale)
    all_nodes, all_edges = [], []
    offset, prev_P, prev_off = 0, None, None
    for t in range(T):
        cc = np.asarray(frames.get(t, np.zeros((0, 3))), np.float64).reshape(-1, 3)
        P = cc * sc
        all_nodes.append(np.column_stack([np.full(len(cc), t, np.float64), cc]))
        if prev_P is not None and len(prev_P) and len(P):
            assign, disp = link_frames(prev_P, P)
            src = np.where(assign >= 0)[0]
            if len(src):
                all_edges.append(np.column_stack([prev_off + src, offset + assign[src]]))
            if DIVISIONS:
                dv = find_divisions(prev_P, P, assign, disp)
                if dv:
                    all_edges.append(np.array([[prev_off + p, offset + ch] for p, ch in dv], np.int64))
        prev_P, prev_off = P, offset
        offset += len(cc)
    nodes = np.vstack(all_nodes) if all_nodes else np.zeros((0, 4))
    edges = np.vstack(all_edges).astype(np.int64) if all_edges else np.zeros((0, 2), np.int64)
    if gap:
        nodes, edges = gap_close(nodes, edges, scale)
    nodes, edges = filter_short(nodes, edges, min_len)
    return nodes, canonicalize(nodes, edges)



# =============================================================================
# TEPE BULMA + NMS  (submit ile AYNI kod -> egitim/inference tutarli)
# =============================================================================
def peaks_gpu(hm, floor=CAND_FLOOR, cap=CAND_MAX):
    kz, ky, kx = PEAK_KERNEL
    pooled = F.max_pool3d(hm, PEAK_KERNEL, stride=1, padding=(kz//2, ky//2, kx//2))
    mask = (hm >= pooled) & (hm > floor)
    idx = mask[0, 0].nonzero()
    if idx.numel() == 0:
        return np.zeros((0, 3), np.int16), np.zeros(0, np.float32)
    sc = hm[0, 0][idx[:, 0], idx[:, 1], idx[:, 2]]
    if idx.shape[0] > cap:
        sc, sel = torch.topk(sc, cap); idx = idx[sel]
    return idx.to(torch.int16).cpu().numpy(), sc.float().cpu().numpy()


def nms_accept(coords, scores, r_um, scale):
    """kabul(esik t) = kabul(taban) & (skor>t)  -> tek gecis, tum esikler icin"""
    n = len(coords)
    if n == 0:
        return np.zeros(0, np.int64)
    P = coords.astype(np.float64) * np.asarray(scale)
    pr = cKDTree(P).query_pairs(r_um, output_type='ndarray')
    if len(pr):
        a = np.concatenate([pr[:, 0], pr[:, 1]]); b = np.concatenate([pr[:, 1], pr[:, 0]])
        o = np.argsort(a, kind='stable'); a, b = a[o], b[o]
        starts = np.searchsorted(a, np.arange(n + 1))
    else:
        b = np.zeros(0, np.int64); starts = np.zeros(n + 1, np.int64)
    sup = np.zeros(n, bool); keep = []
    for i in np.argsort(-scores, kind='stable'):
        if sup[i]:
            continue
        keep.append(i); sup[b[starts[i]:starts[i+1]]] = True
    return np.asarray(keep, np.int64)


# =============================================================================
# DOGRULAMA:  recall @ (k * est_nodes)  -  optimize ettigimiz seyin KENDISI
# =============================================================================
class ValSet:
    """Hacimleri ONDEN YUKLEMEZ (8 dataset x 20 frame x 3 tp = ~8 GB RAM olurdu).
    Sadece metadata tutar, degerlendirme aninda okur."""
    def __init__(self, names, n_ds=VAL_DS, n_fr=VAL_FRAMES):
        self.items = []
        for nm in names[:n_ds]:
            try:
                q, shape = read_zarr_meta(os.path.join(TRAIN, nm + ".zarr"))
                gt = read_gt(os.path.join(TRAIN, nm + ".geff"))
            except Exception as ex:
                print(f"  [val] ATLA {nm}: {ex}"); continue
            T = int(shape[0])
            fr = sorted(gt["byframe"].keys())
            if not fr:
                continue
            sel = sorted({int(fr[i]) for i in
                          np.linspace(0, len(fr) - 1, min(n_fr, len(fr))).astype(int)})
            self.items.append(dict(name=nm, T=T, q=q, est=gt["est_nodes"], frames=sel,
                                   gts={t: gt["byframe"][t] for t in sel},
                                   n_gt=sum(len(gt["byframe"][t]) for t in sel)))
            print(f"  [val] {nm}: {len(sel)} frame, {self.items[-1]['n_gt']} GT node, "
                  f"est={gt['est_nodes']:.0f}")
        tot = sum(i["n_gt"] for i in self.items)
        print(f"  [val] TOPLAM {tot} GT node -> recall granularitesi {100.0/max(1,tot):.2f}%")

    @torch.no_grad()
    def evaluate(self, model, device):
        model.eval()
        sc_arr = np.asarray(SCALE)
        per_k = {k: [0, 0] for k in K_GRID}      # [hit, total]
        import zarr
        for it in self.items:
            arr = zarr.open(os.path.join(TRAIN, it["name"] + ".zarr", "0"), mode='r')
            acc, pooled = {}, []
            for t in it["frames"]:
                tp, tn = max(0, t - 1), min(it["T"] - 1, t + 1)
                vol = np.stack([normalize(np.asarray(arr[tt]), it["q"]) for tt in (tp, t, tn)], 0)
                x = torch.from_numpy(vol).unsqueeze(0).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=(device.type == 'cuda')):
                    lg = model(x)
                hm = torch.sigmoid(lg.float())
                c, s = peaks_gpu(hm)
                k = nms_accept(c, s, R_NMS, SCALE)
                acc[t] = (c[k], s[k]); pooled.append(s[k])
                del x, lg, hm, vol
            pooled = np.concatenate(pooled) if pooled else np.zeros(0, np.float32)
            srt = np.sort(pooled)[::-1]
            frac = len(it["frames"]) / max(1, it["T"])
            for kk in K_GRID:
                tgt = max(1, min(int(round(kk * it["est"] * frac)), len(srt)))
                thr = float(srt[tgt-1]) if len(srt) else 1.0
                hit = tot = 0
                for t in it["frames"]:
                    g = it["gts"][t]
                    tot += len(g)
                    cc, ss = acc[t]
                    cc = cc[ss >= thr]
                    if len(cc) == 0 or len(g) == 0:
                        continue
                    D = np.linalg.norm((cc * sc_arr)[:, None, :] - (g * sc_arr)[None, :, :], axis=2)
                    C = np.where(D <= MAX_MATCH_UM, D, 1e6)
                    if not np.any(C < 1e6):
                        continue
                    r, cidx = linear_sum_assignment(C)
                    hit += int((C[r, cidx] < 1e6).sum())
                per_k[kk][0] += hit; per_k[kk][1] += tot
        rec = {k: (v[0] / max(1, v[1])) for k, v in per_k.items()}
        proxy = {k: rec[k]**2 * max(0.0, 1 - 0.1*(k-1)) for k in K_GRID}
        best_k = max(K_GRID, key=lambda k: proxy[k])
        return rec, proxy, best_k


# =============================================================================
# GERCEK METRIK DOGRULAMASI  (adj_edge_jaccard)
#   recall proxy'si linking'i GORMUYOR. Checkpoint'i yanlis metrige gore secmek
#   egitim koşusu harcamanin klasik yolu -> tam boru hattini olcuyoruz:
#   tepe -> NMS -> n-hedefli esik -> link -> gap -> filtre -> resmi metrik
# =============================================================================
class AdjVal:
    def __init__(self, names, n_ds=ADJ_DS):
        self.items = []
        for nm in names[:n_ds]:
            try:
                q, shape = read_zarr_meta(os.path.join(TRAIN, nm + ".zarr"))
                gt = read_gt(os.path.join(TRAIN, nm + ".geff"))
            except Exception as ex:
                print(f"  [adj] ATLA {nm}: {ex}"); continue
            gt2 = dict(t=gt["t"], z=gt["z"], y=gt["y"], x=gt["x"], n_nodes=gt["n_nodes"],
                       est_nodes=gt["est_nodes"],
                       edges=np.zeros((0, 2), np.int64))
            self.items.append(dict(name=nm, T=int(shape[0]), q=q, gt=gt, gt2=gt2))
            print(f"  [adj] {nm}: T={shape[0]} est={gt['est_nodes']:.0f}")
        # GT kenarlarini ekle (read_gt vermiyordu -> geff'ten tekrar oku)
        for it in self.items:
            it["gt2"]["edges"] = _gt_edges(os.path.join(TRAIN, it["name"] + ".geff"),
                                           it["gt"]["n_nodes"])

    @torch.no_grad()
    def evaluate(self, model, device, n_star=0.95):
        import zarr
        model.eval()
        rows = []
        for it in self.items:
            arr = zarr.open(os.path.join(TRAIN, it["name"] + ".zarr", "0"), mode='r')
            T, q = it["T"], it["q"]
            acc, pooled = [], []
            for t in range(T):
                tp, tn = max(0, t - 1), min(T - 1, t + 1)
                vol = np.stack([normalize(np.asarray(arr[tt]), q) for tt in (tp, t, tn)], 0)
                x = torch.from_numpy(vol).unsqueeze(0).to(device)
                with torch.autocast(device_type=device.type, dtype=torch.float16,
                                    enabled=(device.type == 'cuda')):
                    lg = model(x)
                hm = torch.sigmoid(lg.float())
                c, sc = peaks_gpu(hm)
                k = nms_accept(c, sc, R_NMS, SCALE)
                acc.append((c[k].astype(np.float64), sc[k])); pooled.append(sc[k])
                del x, lg, hm, vol
            pooled = np.concatenate(pooled) if pooled else np.zeros(0, np.float32)
            srt = np.sort(pooled)[::-1]
            est = it["gt"]["est_nodes"]
            K = max(1, min(int(round(n_star * est)), len(srt)))
            thr = float(srt[K - 1]) if len(srt) else 1.0
            frames = {t: cc[ss >= thr] for t, (cc, ss) in enumerate(acc) if (ss >= thr).any()}
            nodes, edges = build_graph(frames, T, SCALE)
            rows.append(evaluate_dataset(nodes, edges, it["gt2"]))
        return summarise(rows)


def _gt_edges(geff_path, n_nodes):
    import zarr
    g = zarr.open_group(str(geff_path), mode="r")
    def arr(p):
        n = g
        for k in p.split("/"):
            n = n[k]
        return np.asarray(n[:])
    try:
        ids = arr("nodes/ids").astype(np.int64)
    except Exception:
        ids = np.arange(n_nodes, dtype=np.int64)
    try:
        e = arr("edges/ids").astype(np.int64).reshape(-1, 2)
    except Exception:
        return np.zeros((0, 2), np.int64)
    if not len(e):
        return np.zeros((0, 2), np.int64)
    o = np.argsort(ids)
    return o[np.searchsorted(ids[o], e.ravel())].reshape(-1, 2)


# =============================================================================
# EMA  (F3: state_dict REFERANS dondurur -> kayitta deepcopy SART)
# =============================================================================
class ModelEMA:
    def __init__(self, model, decay, device):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = self.decay * (1 - math.exp(-self.n / 1500.0))
        for e, m in zip(self.module.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.mul_(d).add_(m.detach(), alpha=1 - d)
            else:
                e.copy_(m)


def save_ckpt(path, model, epoch, rec, proxy, best_k, extra=None):
    """F3: deepcopy -> canlı tensorlere referans DEGIL, gercek kopya"""
    sd = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    torch.save(dict(model_state_dict=sd, epoch=epoch, recall=rec, proxy=proxy,
                    best_k=best_k, config=dict(patch=PATCH, sigma=GT_SIGMA_UM,
                    ignore_tau=IGNORE_TAU, pos_frac=POS_FRAC, lr=LR),
                    **(extra or {})), path)


# =============================================================================
# EGITIM
# =============================================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True        # sabit girdi sekli -> %10-30
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"donanim={device} torch={torch.__version__} cudnn.benchmark=True tf32=True")
    all_names = sorted(d[:-5] for d in os.listdir(TRAIN) if d.endswith(".zarr"))
    # 2026-09-20: TUR 1 SIZINTISI DUZELTILDI - VAL_PUBLIC de HARIC
    train_names = [n for n in all_names if n not in VAL_CORE and n not in VAL_PUBLIC]
    print(f"egitim {len(train_names)} / val {len(VAL_CORE)} dataset")

    print("\n[val seti hazirlaniyor - recall (hizli)]")
    val = ValSet(VAL_CORE)
    print("\n[adj seti hazirlaniyor - GERCEK metrik (pahali)]")
    try:
        # TUR 1 KOR NOKTASI DUZELTILDI: VAL_CORE[:4] hepsi 44b6 idi -> dengeli sec
        _adj_names = ([n for n in VAL_CORE if n.startswith("44b6")][:ADJ_DS // 2] +
                     [n for n in VAL_CORE if n.startswith("6bba")][:ADJ_DS - ADJ_DS // 2])
        print(f"  [adj] dengeli set: {_adj_names}")
        adjval = AdjVal(_adj_names)
    except Exception as ex:
        import traceback; traceback.print_exc()
        print(f"  !!! adj dogrulamasi kurulamadi ({type(ex).__name__}) -> recall'a dusuluyor")
        adjval = None
    print("\n[egitim seti hazirlaniyor]")
    ds = GTPatchDataset(train_names, STEPS_PER_EP * BATCH)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=WORKERS,
                    pin_memory=True, persistent_workers=WORKERS > 0,
                    prefetch_factor=2 if WORKERS > 0 else None)

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"PATCH={PATCH} SMOKE={SMOKE} STEPS_PER_EP={STEPS_PER_EP} MAX_MIN={MAX_MINUTES}")

    model = UNet4D_SOTA_V9().to(device)
    if os.path.exists(PREV):
        ck = torch.load(PREV, map_location='cpu', weights_only=False)
        miss, unexp = model.load_state_dict(ck.get('model_state_dict', ck), strict=False)
        print(f"v9 sicak baslatma: eksik={len(miss)} beklenmeyen={len(unexp)} "
              f"epoch={ck.get('epoch','?')}")
        assert not miss and not unexp, "AGIRLIK UYUSMAZLIGI"
    else:
        print("!!! v9 checkpoint YOK - sifirdan basliyor")
    ema = ModelEMA(model, EMA_DECAY, device)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    total_steps = MAX_EPOCH * (STEPS_PER_EP // ACCUM)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    crit = MaskedCenterNetLoss()

    # === YOKSAY MASKESI TANILAMASI ===================================
    # Bu koşunun TUM temeli bu maske. Ne kapsadigini OLCMEDEN 7.5 saat harcamam.
    print("\n[yoksay maskesi tanilamasi - 24 ornek patch]")
    ig, taus, igf, npos = [], [], [], []
    for _ in range(24):
        _x, _hm, _nw = ds[0]
        if ds.last_stats:
            a, b, c, d = ds.last_stats
            ig.append(a); taus.append(b); igf.append(c); npos.append(d)
    if ig:
        ig, taus, igf, npos = map(np.asarray, (ig, taus, igf, npos))
        print(f"  yoksay orani : ort %{100*ig.mean():.1f}  min %{100*ig.min():.1f}  "
              f"max %{100*ig.max():.1f}")
        print(f"  hedef oran   : ort %{100*np.nanmean(igf):.1f} (est_nodes'tan hesaplanan)")
        print(f"  esik (tau)   : ort {taus.mean():.3f}  min {taus.min():.3f}  max {taus.max():.3f}")
        print(f"  patch basi GT: ort {npos.mean():.1f}  (0 olan: {int((npos==0).sum())}/{len(npos)})")
        if ig.mean() > 0.50:
            print("  !!! UYARI: yoksay orani %50'yi asiyor -> negatif sinyal cok zayif")
        if ig.mean() < 0.005:
            print("  !!! UYARI: yoksay orani ~0 -> maske ETKISIZ, v10 hatasi surer")

    print("\n[BASLANGIC dogrulamasi - v9 agirliklariyla]")
    rec0, px0, bk0 = val.evaluate(ema.module, device)
    print("  " + "  ".join(f"k={k}: rec={rec0[k]:.4f} px={px0[k]:.4f}" for k in K_GRID))
    print(f"  >>> BASLANGIC en iyi k={bk0} proxy={px0[bk0]:.4f} recall={rec0[bk0]:.4f}")

    def run_adj(model):
        """GERCEK metrik. Hata verirse egitimi OLDURMEZ, None doner."""
        if adjval is None:
            return None
        try:
            t0 = time.time()
            r = adjval.evaluate(model, device)
            print(f"   [ADJ] adj={r['adj_edge_jaccard']:.4f} edgeJ={r['edge_jaccard']:.4f} "
                  f"rec={r['node_recall']:.4f} n={r['n_ratio']:.3f} "
                  f"({(time.time()-t0)/60:.1f}dk)", flush=True)
            return r
        except Exception as ex:
            import traceback; traceback.print_exc()
            print(f"   [ADJ] HATA {type(ex).__name__} -> atlandi", flush=True)
            return None

    print("\n[BASLANGIC gercek metrik - v9 agirliklariyla]")
    adj0 = run_adj(ema.module)
    base_adj = adj0['adj_edge_jaccard'] if adj0 else None

    # secim olcutu: adj varsa ADJ, yoksa recall proxy'si
    best_px, best_ep = (base_adj if base_adj is not None else px0[bk0]), -1
    use_adj = base_adj is not None
    print(f"  >>> SECIM OLCUTU: {'GERCEK adj' if use_adj else 'recall proxy'}  "
          f"baslangic={best_px:.4f}")
    hist = [dict(epoch=-1, rec=rec0, proxy=px0, best_k=bk0, loss=float('nan'),
                 adj=base_adj)]

    step = 0
    for ep in range(MAX_EPOCH):
        if elapsed() > MAX_MINUTES:
            print(f"\n[sure siniri {elapsed():.0f} dk] egitim sonlandiriliyor"); break
        model.train()
        tot, nb = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for x, hm, nw in dl:
            x = x.to(device, non_blocking=True)
            hm = hm.to(device, non_blocking=True)
            nw = nw.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == 'cuda')):
                loss = crit(model(x), hm, nw) / ACCUM
            scaler.scale(loss).backward()
            tot += float(loss.item()) * ACCUM; nb += 1
            if nb == 1 and ep == 0 and device.type == 'cuda':
                torch.cuda.synchronize()
                print(f"  [bellek] ilk adim sonrasi tepe={torch.cuda.max_memory_allocated()/1e9:.2f} GB "
                      f"/ {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB", flush=True)
            if nb % ACCUM == 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                ema.update(model); sched.step(); step += 1
        avg = tot / max(1, nb)
        mem = (torch.cuda.max_memory_allocated()/1e9) if device.type == 'cuda' else 0.0
        print(f"ep{ep:3d} loss={avg:.5f} lr={opt.param_groups[0]['lr']:.2e} "
              f"tepe_bellek={mem:.2f}GB ({elapsed():.1f}dk)", flush=True)

        if ep % VAL_EVERY == 0 or ep == MAX_EPOCH - 1:
            rec, px, bk = val.evaluate(ema.module, device)
            print("   " + "  ".join(f"k={k}:{rec[k]:.4f}/{px[k]:.4f}" for k in K_GRID))
            print(f"   en iyi k={bk} proxy={px[bk]:.4f} recall={rec[bk]:.4f}"
                  f"   (baslangic {px0[bk0]:.4f})", flush=True)
            adj_r = run_adj(ema.module) if (use_adj and (ep % ADJ_EVERY == 0
                                            or ep == MAX_EPOCH - 1)) else None
            cur = adj_r['adj_edge_jaccard'] if adj_r else (None if use_adj else px[bk])
            hist.append(dict(epoch=ep, rec=rec, proxy=px, best_k=bk, loss=avg,
                             adj=(adj_r['adj_edge_jaccard'] if adj_r else None)))
            if cur is not None and cur > best_px:
                best_px, best_ep = cur, ep
                save_ckpt(os.path.join(WORK, "v11_detector_best.pt"),
                          ema.module, ep, rec, px, bk,
                          extra=dict(hist=hist, adj=cur, criterion="adj" if use_adj else "proxy"))
                print(f"   *** YENI EN IYI ({'adj' if use_adj else 'proxy'})={best_px:.4f}"
                      f"  (baslangic {hist[0]['adj'] if use_adj else px0[bk0]:.4f})", flush=True)
            save_ckpt(os.path.join(WORK, "v11_detector_last.pt"), ema.module, ep, rec, px, bk,
                      extra=dict(hist=hist))

    print("\n" + "#" * 100)
    print("### V11-ADIM5-RAPOR-BASLANGIC")
    print(f"sure_dk={elapsed():.1f} epoch={len(hist)-1} step={step}")
    print(f"BASLANGIC(v9) k={bk0} proxy={px0[bk0]:.4f} recall={rec0[bk0]:.4f}")
    print(f"  " + " ".join(f"k{k}={rec0[k]:.4f}" for k in K_GRID))
    print(f"OLCUT={'adj' if use_adj else 'proxy'} baslangic={hist[0]['adj'] if use_adj else px0[bk0]:.4f}")
    print(f"EN IYI epoch={best_ep} deger={best_px:.4f}  "
          f"kazanc={best_px-(hist[0]['adj'] if use_adj else px0[bk0]):+.4f}")
    for h in hist:
        a = h.get('adj')
        print(f"ep{h['epoch']} loss={h['loss']:.5f} bestk={h['best_k']} "
              f"proxy={h['proxy'][h['best_k']]:.4f} adj={'-' if a is None else f'{a:.4f}'} " +
              " ".join(f"k{k}={h['rec'][k]:.4f}" for k in K_GRID))
    print("### V11-ADIM5-RAPOR-BITIS")
    print("#" * 100)


if __name__ == "__main__":
    main()
