# =============================================================================
# V11 HATA ARAMASI 1 - VOXEL OLCEGI SESSIZ-VARSAYILAN DENETIMI (CPU, ~1 dk)
# =============================================================================
# v11_S1_submit.py::read_zarr_attrs() OME-Zarr metadata'sindan scale okumaya
# calisir, HERHANGI bir hata olursa (bare except: pass) SESSIZCE sabit
# SCALE=(1.625,0.40625,0.40625)'e duser -> hicbir uyari YOK. LAP/gap-close/
# NMS gibi TUM fiziksel-mesafe hesaplari bu deger UZERINDEN calisiyor.
# Bu script: 199 TRAIN dataset'inin TUMUNDE bu okuma GERCEKTEN basariyor mu,
# ve deger GERCEKTEN sabit mi -> ham veriye hic dokunmadan (SADECE metadata,
# internet/pip GEREKMEZ).
# =============================================================================
import os, json, glob

COMP  = "/kaggle/input/competitions/biohub-cell-tracking-during-development"
TRAIN = os.path.join(COMP, "train")
SCALE_DEFAULT = (1.625, 0.40625, 0.40625)

def read_scale_raw(zarr_dir):
    """v11_S1_submit.py::read_zarr_attrs ile BIREBIR ayni mantik, ama basarim/
    basarisizlik durumunu da rapor ediyor (orijinalde bu bilgi kayboluyordu)."""
    a = {}
    used_file = None
    for f in (".zattrs", "zarr.json"):
        p = os.path.join(zarr_dir, f)
        if os.path.exists(p):
            a = json.load(open(p))
            if "attributes" in a:
                a = a["attributes"]
            used_file = f
            break
    if used_file is None:
        return None, "METADATA_DOSYASI_YOK", None

    try:
        tr = a["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
        if tr["type"] == "scale":
            sc = tuple(float(v) for v in tr["scale"][-3:])
            return sc, "OK", used_file
        else:
            return None, f"TRANSFORM_TIPI_SCALE_DEGIL({tr.get('type')})", used_file
    except Exception as ex:
        return None, f"PARSE_HATASI({type(ex).__name__}:{ex})", used_file


all_ds = sorted(d[:-5] for d in os.listdir(TRAIN) if d.endswith(".zarr"))
print(f"{len(all_ds)} TRAIN dataset taraniyor...")

results = {}
fail_count = 0
values_seen = {}
for name in all_ds:
    zdir = os.path.join(TRAIN, name + ".zarr")
    sc, status, used_file = read_scale_raw(zdir)
    results[name] = (sc, status, used_file)
    if sc is None:
        fail_count += 1
        print(f"  !!! {name}: {status} (dosya={used_file}) -> SESSIZCE varsayilana duser: {SCALE_DEFAULT}")
    else:
        values_seen.setdefault(sc, []).append(name)

print("\n" + "#" * 90)
print("### V11-SCALE-AUDIT-RAPOR-BASLANGIC")
print(f"toplam_dataset={len(all_ds)} basarisiz_okuma={fail_count}")
print(f"benzersiz_deger_sayisi={len(values_seen)}")
for sc, names in sorted(values_seen.items(), key=lambda kv: -len(kv[1])):
    eslesme = "VARSAYILANLA_AYNI" if sc == SCALE_DEFAULT else "!!! VARSAYILANDAN FARKLI !!!"
    print(f"  deger={sc}  n={len(names)}  {eslesme}")
    if sc != SCALE_DEFAULT:
        print(f"    ornek dataset'ler: {names[:10]}")
if fail_count == 0 and len(values_seen) == 1 and SCALE_DEFAULT in values_seen:
    print(">>> SONUC: 199/199 basarili okundu, TUMU sabit degere esit. Varsayilan"
          " ASLA tetiklenmedi, TRAIN'de risk YOK.")
    print(">>> UYARI: bu, HIDDEN TEST'te de ayni olacagi ANLAMINA GELMEZ - ayri"
          " bir risk kalemi olarak kalir (train/test embriyo-ayrik).")
else:
    print(">>> SONUC: RISK DOGRULANDI - ya okuma basarisiz oluyor ya da deger"
          " sabit degil. read_zarr_attrs() ACIL duzeltme gerektiriyor.")
print("### V11-SCALE-AUDIT-RAPOR-BITIS")
print("#" * 90)
