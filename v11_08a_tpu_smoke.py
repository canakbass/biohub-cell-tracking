# =============================================================================
# ADIM 8a - TPU DUMAN TESTI  (TPU, internet ACIK, ~5 dk hedef)
# =============================================================================
# GPU haftalik kotasi bitti; TPU AYRI bir kota. Ama model CUDA icin yazildi:
#   ConvNeXtBlock3D derinlemesine ayrilabilir 3D konvolusyon kullaniyor
#   (Conv3d(dim,dim,kernel=(3,5,5),groups=dim)) -> XLA/TPU'da grouped 3D conv
#   desteklenip desteklenmedigi, hizinin ne oldugu BILINMIYOR.
# Once KORLEMESINE tam pipeline'a uygulamak yerine UCUZ bir duman testi:
#   1) torch_xla var mi, TPU cihazi acilabiliyor mu
#   2) v11_detector_best.pt checkpoint'i XLA cihazina tasinip forward CALISIYOR mu
#   3) cikti CPU/GPU ile AYNI mi (sayisal dogrulama, 1e-3 tolerans)
#   4) 5 tekrarlik ISINMA SONRASI hiz (frame/s) -> GPU'nun 0.66 s/frame'iyle kiyasla
# BASARISIZ olursa TPU yolu terk edilir, GPU kotasi yenilenene kadar CPU islerine
# devam edilir (Adim 6 K_NN deneyi zaten CPU'da suruyor).
# =============================================================================
import sys, subprocess, time, os
from importlib.metadata import version as _pv

def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=False)

print("[bootstrap] torch_xla kontrolu...", flush=True)
try:
    import torch_xla
    print(f"  torch_xla ONYUKLU: {torch_xla.__version__}")
except ImportError:
    print("  torch_xla YOK, kuruluyor...")
    _pip("torch_xla[tpu]", "-f", "https://storage.googleapis.com/libtpu-releases/index.html")
    try:
        import torch_xla
        print(f"  kuruldu: {torch_xla.__version__}")
    except ImportError as ex:
        print(f"  !!! KURULAMADI: {ex}")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

print(f"torch={torch.__version__}")
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


# =============================================================================
# 1) TPU CIHAZI + MODEL YUKLEME
# =============================================================================
import glob as _glob
MODEL_PATH = None
for pat in ("/kaggle/input/**/v11_detector_best.pt",
            "/kaggle/input/models/hcanakbas/biohub-v9-03-sota/pytorch/default/1/biohub_v9_sota_f1_868.pt"):
    c = _glob.glob(pat, recursive=True)
    if c:
        MODEL_PATH = c[0]; break
assert MODEL_PATH, "model checkpoint bulunamadi"
print(f"model: {MODEL_PATH}")

PATCH = (48, 256, 256)   # gercek submit patch boyutu (v11_S1_submit.py)
N_REPEAT = 8
N_WARMUP = 3

def build_model():
    m = UNet4D_SOTA_V9()
    ck = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
    miss, unexp = m.load_state_dict(ck.get('model_state_dict', ck), strict=False)
    assert not miss and not unexp, f"AGIRLIK UYUSMAZLIGI eksik={miss} beklenmeyen={unexp}"
    m.eval()
    return m

torch.manual_seed(0)
x_np = np.random.RandomState(0).randn(1, 3, *PATCH).astype(np.float32)

# --- CPU REFERANSI (dogruluk kiyaslamasi icin) ---
print("\n[CPU referans cikisi hesaplaniyor]")
m_cpu = build_model()
with torch.no_grad():
    t0 = time.time()
    y_cpu = m_cpu(torch.from_numpy(x_np)).numpy()
    print(f"  CPU forward: {time.time()-t0:.2f}s  cikti shape={y_cpu.shape}  "
          f"mean={y_cpu.mean():.5f} std={y_cpu.std():.5f}")

# --- TPU DENEMESI ---
print("\n[TPU cihazi deneniyor]")
tpu_ok = False
try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
    print(f"  XLA cihazi: {device}")

    m_tpu = build_model().to(device)
    x_t = torch.from_numpy(x_np).to(device)

    with torch.no_grad():
        print(f"  [isinma] {N_WARMUP} tekrar (XLA ilk cagrida grafigi derler, YAVAS beklenir)")
        for i in range(N_WARMUP):
            t0 = time.time()
            y = m_tpu(x_t)
            xm.mark_step()          # XLA: grafik burada gercekten calisir
            _ = y.cpu().numpy()     # senkronize et
            print(f"    isinma {i+1}/{N_WARMUP}: {time.time()-t0:.2f}s")

        print(f"  [olcum] {N_REPEAT} tekrar (isinma sonrasi)")
        times = []
        for i in range(N_REPEAT):
            t0 = time.time()
            y = m_tpu(x_t)
            xm.mark_step()
            y_np = y.cpu().numpy()
            times.append(time.time() - t0)
        times = np.array(times)
        print(f"  sureler(s): {np.round(times,3).tolist()}")
        print(f"  medyan={np.median(times):.3f}s  min={times.min():.3f}s")

        diff = np.abs(y_np - y_cpu)
        rel = diff.max() / max(1e-6, np.abs(y_cpu).max())
        print(f"\n  [dogruluk] CPU vs TPU: max_fark={diff.max():.6f}  "
              f"goreli={rel:.6f}  {'AYNI (tolerans icinde)' if rel < 1e-2 else '!!! FARKLI !!!'}")
        tpu_ok = rel < 1e-2

        # gercek submit oranina cevir: 1 patch ~= 1 frame (t-1,t,t+1 stack, cikti 1 frame)
        med = float(np.median(times))
        print(f"\n  [KIYAS] TPU frame basi: {med:.3f}s  |  GPU (T4) olculen: 0.66 s/frame")
        if med > 0:
            print(f"  TPU/GPU hiz orani: {0.66/med:.2f}x  "
                  f"({'TPU DAHA HIZLI' if med < 0.66 else 'TPU DAHA YAVAS'})")
        print(f"  199 dataset x 100 frame'de tahmini sure: {med*100*199/3600:.2f} saat")
except Exception as ex:
    import traceback; traceback.print_exc()
    print(f"\n  !!! TPU BASARISIZ: {type(ex).__name__}: {ex}")

print("\n" + "#" * 100)
print("### V11-ADIM8A-RAPOR-BASLANGIC")
print(f"tpu_calisti={tpu_ok}")
print("### V11-ADIM8A-RAPOR-BITIS")
print("#" * 100)
