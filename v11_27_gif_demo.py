# =============================================================================
# V11 GORSELLESTIRME - ham goruntu vs GT vs model tahmini GIF'i (README icin)
# =============================================================================
# Uretim detektorunu (v11-egitim, hidden testte 0.863 dogrulanmis) TRAIN'den
# secilen tek bir dataset uzerinde calistirir; her frame icin Z-ekseninde
# maksimum-yogunluk izdusumu (MIP) uzerine GT (yesil) ve tahmin (kirmizi)
# noktalarini bindirir, animasyonlu GIF olarak kaydeder.
# =============================================================================
import sys, subprocess

def _ensure(mod, spec=None):
    try:
        __import__(mod); return True
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", spec or mod], check=False)
        try:
            __import__(mod); return True
        except ImportError:
            print(f"[bootstrap] {mod} KURULAMADI"); return False

_ensure("zarr")

import os, glob, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from PIL import Image, ImageDraw, ImageFont

COMP  = "/kaggle/input/biohub-cell-tracking-during-development"
TRAIN = os.path.join(COMP, "train")
WORK  = "/kaggle/working"

DATASET_NAME = "6bba_67ebd073"   # orta yogunluk (~160 hucre/frame), defalarca basariyla okunmus
N_FRAMES     = 60                # GIF boyutu icin sinirli
OUT_SIZE     = 320               # panel basina piksel (kare)
SCALE = (1.625, 0.40625, 0.40625)

_cand = glob.glob("/kaggle/input/**/v11_detector_best.pt", recursive=True)
_cand = [p for p in _cand if "seed2" not in p and "anisofix" not in p]
MODEL_PATH = _cand[0] if _cand else None
assert MODEL_PATH, "v11_detector_best.pt bulunamadi - v11-egitim ciktisini ekle"
print(f"MODEL: {MODEL_PATH}")


# =============================================================================
# MIMARI (v11_S1_submit.py ile BIREBIR ayni, degistirilmedi)
# =============================================================================
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
            self.convs.append(nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, padding=d, dilation=d, bias=False),
                LayerNorm(out_channels, eps=1e-6, data_format="channels_first"),
                nn.GELU()
            ))
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


def normalize(vol_u16, q=None):
    lo = float(np.percentile(vol_u16[::4], 0.1))
    hi = float(np.percentile(vol_u16[::4], 99.9))
    img = (vol_u16.astype(np.float32) - lo) / max(1e-6, hi - lo)
    return np.clip(img, 0.0, 4.0, out=img)


def read_gt_points(geff_path):
    g = zarr.open_group(str(geff_path), mode="r")

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
    by_t = {}
    for i in range(len(t)):
        by_t.setdefault(int(t[i]), []).append((z[i], y[i], x[i]))
    return by_t


# =============================================================================
# MODEL YUKLE + VERI OKU
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet4D_SOTA_V9().to(device)
ck = torch.load(MODEL_PATH, map_location=device, weights_only=False)
sd = ck.get('model_state_dict', ck)
missing, unexpected = model.load_state_dict(sd, strict=False)
assert not missing and not unexpected, f"AGIRLIK UYUSMAZLIGI eksik={missing} beklenmeyen={unexpected}"
model.eval()
print(f"model yuklendi: epoch={ck.get('epoch','?')}")

zdir = os.path.join(TRAIN, DATASET_NAME + ".zarr")
gdir = os.path.join(TRAIN, DATASET_NAME + ".geff")

print(f"COMP var mi: {os.path.isdir(COMP)}")
print(f"COMP icerik: {sorted(os.listdir(COMP)) if os.path.isdir(COMP) else 'YOK'}")
print(f"/kaggle/input icerik: {sorted(os.listdir('/kaggle/input'))}")
print(f"/kaggle/input/competitions var mi: {os.path.isdir('/kaggle/input/competitions')}")
if os.path.isdir('/kaggle/input/competitions'):
    print(f"/kaggle/input/competitions icerik: {sorted(os.listdir('/kaggle/input/competitions'))}")
print(f"TRAIN var mi: {os.path.isdir(TRAIN)}")
print(f"TRAIN icerik (ilk 10): {sorted(os.listdir(TRAIN))[:10] if os.path.isdir(TRAIN) else 'YOK'}")
print(f"zdir={zdir} var mi: {os.path.isdir(zdir)}")
if os.path.isdir(zdir):
    print(f"zdir icerik: {sorted(os.listdir(zdir))}")
    sub0 = os.path.join(zdir, "0")
    print(f"zdir/0 var mi: {os.path.isdir(sub0)}")
    if os.path.isdir(sub0):
        print(f"zdir/0 icerik: {sorted(os.listdir(sub0))}")

arr = zarr.open(os.path.join(zdir, "0"), mode='r')
T, Z, Y, X = arr.shape
N_FRAMES = min(N_FRAMES, T)
gt_by_t = read_gt_points(gdir)
print(f"dataset={DATASET_NAME} T={T} Z={Z} Y={Y} X={X}  gosterilecek frame={N_FRAMES}")


def mip_to_rgb(vol_zyx, size):
    """(Z,Y,X) float -> gri MIP -> RGB, [size,size]'a olcekli PIL.Image"""
    mip = vol_zyx.max(axis=0)
    mip = np.clip(mip / max(1e-6, np.percentile(mip, 99.5)), 0, 1)
    img8 = (mip * 255).astype(np.uint8)
    im = Image.fromarray(img8, mode="L").convert("RGB")
    return im.resize((size, size), Image.BILINEAR)


def draw_points(im, pts_yx, color, r=4):
    d = ImageDraw.Draw(im)
    for (y, x) in pts_yx:
        d.ellipse([x - r, y - r, x + r, y + r], outline=color, width=2)
    return im


# =============================================================================
# FRAME BASINA: forward pass -> tahmin noktalari, GT noktalari, GIF karesi
# =============================================================================
frames_out = []
cache = {}
with torch.no_grad():
    for t in range(N_FRAMES):
        for tt in (t - 1, t, t + 1):
            tc = min(max(tt, 0), T - 1)
            if tc not in cache:
                cache[tc] = normalize(np.asarray(arr[tc]))
        for old_t in [k for k in cache if k < t - 1]:
            del cache[old_t]
        inp = np.stack([cache[max(t - 1, 0)], cache[t], cache[min(t + 1, T - 1)]], 0)
        x_t = torch.from_numpy(inp).unsqueeze(0).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == 'cuda')):
            logits = model(x_t)
        hm = torch.sigmoid(logits.float()).squeeze().cpu().numpy()

        from scipy.ndimage import maximum_filter
        pooled = maximum_filter(hm, size=(3, 11, 11), mode="nearest")
        mask = (hm >= pooled) & (hm > 0.05)
        idx = np.argwhere(mask)
        sc = hm[mask]
        n_gt_here = len(gt_by_t.get(t, []))
        if len(idx) > max(5, int(1.2 * n_gt_here)):
            keep = np.argpartition(-sc, max(5, int(1.2 * n_gt_here)))[:max(5, int(1.2 * n_gt_here))]
            idx = idx[keep]
        pred_yx = [(p[1] * OUT_SIZE / Y, p[2] * OUT_SIZE / X) for p in idx]

        gt_yx = [(gy * OUT_SIZE / Y, gx * OUT_SIZE / X) for (_, gy, gx) in gt_by_t.get(t, [])]

        raw = cache[t]
        im = mip_to_rgb(raw, OUT_SIZE)
        im = draw_points(im, gt_yx, (60, 220, 90), r=5)     # GT: yesil
        im = draw_points(im, pred_yx, (235, 60, 60), r=3)   # tahmin: kirmizi

        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, OUT_SIZE, 18], fill=(0, 0, 0))
        d.text((4, 2), f"t={t:03d}  yesil=GT  kirmizi=tahmin", fill=(255, 255, 255))
        frames_out.append(im)
        if t % 10 == 0:
            print(f"  t={t:3d}  GT={n_gt_here}  tahmin={len(pred_yx)}")

out_path = os.path.join(WORK, "demo.gif")
frames_out[0].save(out_path, save_all=True, append_images=frames_out[1:],
                   duration=140, loop=0, optimize=True)
print(f"\nGIF kaydedildi: {out_path}  ({os.path.getsize(out_path)/1e6:.2f} MB, {len(frames_out)} kare)")
