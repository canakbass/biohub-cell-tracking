#!/usr/bin/env python3
"""Kaggle surucusu - saf stdlib (bagimlilik yok).

Kimlik: ~/.kaggle/kaggle.json  {"username": "...", "key": "..."}
        veya KAGGLE_USERNAME / KAGGLE_KEY ortam degiskenleri.
API anahtari HICBIR ZAMAN yazdirilmaz.

Kullanim:
  kdrive.py whoami
  kdrive.py list [--n 20]
  kdrive.py push <script.py> --slug <slug> [--title T] [--gpu] [--internet]
                 [--comp SLUG]... [--kernel KSLUG]... [--model MPATH]... [--dataset DSLUG]...
  kdrive.py status <slug>
  kdrive.py wait <slug> [--timeout-min 720] [--poll 60]
  kdrive.py output <slug> [--dest DIR]
  kdrive.py log <slug> [--tail 200]
"""
from __future__ import annotations
import argparse, base64, json, os, re, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

API = "https://www.kaggle.com/api/v1"
UA = "kdrive/1.0"


def creds():
    u, k = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    if not (u and k):
        p = Path.home() / ".kaggle" / "kaggle.json"
        if not p.exists():
            sys.exit("HATA: ~/.kaggle/kaggle.json yok ve KAGGLE_USERNAME/KAGGLE_KEY bos.\n"
                     "  kaggle.com/settings -> API -> Create New Token")
        d = json.loads(p.read_text())
        u, k = d.get("username"), d.get("key")
    if not (u and k):
        sys.exit("HATA: kimlik bilgisi eksik (username/key).")
    return u, k


def call(path, method="GET", params=None, body=None, raw=False, timeout=300):
    u, k = creds()
    url = f"{API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{u}:{k}".encode()).decode())
    req.add_header("User-Agent", UA)
    if data:
        req.add_header("Content-Type", "application/json")
    # GECICI AG HATALARI (DNS, baglanti kopmasi, 5xx) OLUMCUL DEGIL: tekrar dene.
    # Onceki surum ilk DNS hatasinda cikiyordu -> uzun beklemeler olup gidiyordu.
    last = None
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                blob = r.read()
                return blob if raw else json.loads(blob or b"{}")
        except urllib.error.HTTPError as e:
            if e.code >= 500 or e.code == 429:
                last = f"HTTP {e.code}"
            else:
                detail = e.read().decode(errors="replace")[:600]
                sys.exit(f"HTTP {e.code} {method} {path}\n{detail}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            last = f"{type(e).__name__}: {getattr(e, 'reason', e)}"
        wait_s = min(300, 10 * 2 ** attempt)
        print(f"  [ag] {method} {path} gecici hata ({last}) -> {wait_s}s sonra tekrar "
              f"({attempt+1}/8)", file=sys.stderr, flush=True)
        time.sleep(wait_s)
    sys.exit(f"AG HATASI (8 deneme sonrasi) {method} {path}: {last}")


# ---------------------------------------------------------------- komutlar
def cmd_whoami(a):
    u, _ = creds()
    r = call("/kernels/list", params={"user": u, "pageSize": 1})
    print(f"kimlik OK  kullanici={u}  (kernel listesi erisilebilir: {len(r)} kayit dondu)")


def cmd_list(a):
    u, _ = creds()
    r = call("/kernels/list", params={"user": u, "pageSize": a.n, "sortBy": "dateRun"})
    if not r:
        print("kernel bulunamadi")
        return
    for x in r:
        print(f"  {str(x.get('ref','?')):55s} {str(x.get('lastRunTime','?'))[:19]:20s} "
              f"{x.get('totalVotes','')}")


def cmd_push(a):
    u, _ = creds()
    src = Path(a.script)
    if not src.exists():
        sys.exit(f"HATA: {src} yok")
    text = src.read_text()
    slug = a.slug.lower()
    if not re.fullmatch(r"[a-z0-9-]{5,}", slug):
        sys.exit("HATA: slug sadece kucuk harf/rakam/tire, en az 5 karakter")
    body = {
        "id": None,
        "slug": f"{u}/{slug}",
        "newTitle": a.title or slug,
        "text": text,
        "language": "python",
        "kernelType": "script",
        "isPrivate": True,
        "enableGpu": bool(a.gpu),
        "enableTpu": bool(a.tpu),
        "enableInternet": bool(a.internet),
        "datasetDataSources": a.dataset or [],
        "competitionDataSources": a.comp or [],
        "kernelDataSources": a.kernel or [],
        "modelDataSources": a.model or [],
        "categoryIds": [],
        "dockerImagePinningType": "original",
    }
    r = call("/kernels/push", method="POST", body=body)
    real = str(r.get("ref", "")).rstrip("/").split("/")[-1]
    print(f"push OK  ref={r.get('ref')}  versionNumber={r.get('versionNumber')}")
    if real and real != slug:
        print(f"  !!! DIKKAT: Kaggle slug'i BASLIKTAN uretti -> gercek slug = '{real}'")
        print(f"      (istenen '{slug}'). Bundan sonra --title'i slug ile AYNI ver.")
        slug = real
    if r.get("error"):
        print(f"  UYARI: {r['error']}")
    print(f"  url=https://www.kaggle.com/code/{u}/{slug}")


def _status(u, slug):
    return call("/kernels/status", params={"userName": u, "kernelSlug": slug})


def cmd_status(a):
    u, _ = creds()
    s = _status(u, a.slug.lower())
    print(f"status={s.get('status')}  {s.get('failureMessage') or ''}")


def cmd_wait(a):
    u, slug = creds()[0], a.slug.lower()
    t0 = time.time()
    last = None
    while True:
        s = _status(u, slug)
        st = s.get("status")
        if st != last:
            print(f"[{(time.time()-t0)/60:6.1f} dk] status={st}", flush=True)
            last = st
        if st in ("complete", "error", "cancelAcknowledged", "cancelRequested"):
            if s.get("failureMessage"):
                print(f"  failureMessage: {s['failureMessage']}")
            print(f"SONUC={st}")
            return 0 if st == "complete" else 1
        if (time.time() - t0) / 60 > a.timeout_min:
            print(f"ZAMAN ASIMI ({a.timeout_min} dk) - status={st}")
            return 2
        time.sleep(a.poll)


def _output(u, slug):
    return call("/kernels/output", params={"userName": u, "kernelSlug": slug})


def cmd_output(a):
    u, slug = creds()[0], a.slug.lower()
    r = _output(u, slug)
    dest = Path(a.dest or f"./out_{slug}")
    dest.mkdir(parents=True, exist_ok=True)
    log = r.get("log")
    if log:
        (dest / f"{slug}.log").write_text(log if isinstance(log, str) else json.dumps(log))
        print(f"  log -> {dest/(slug+'.log')}")
    files = r.get("files") or []
    for f in files:
        name, url = f.get("fileName"), f.get("url")
        if not (name and url):
            continue
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=600) as resp, open(dest / name, "wb") as fh:
                fh.write(resp.read())
            print(f"  {name} -> {dest/name} ({(dest/name).stat().st_size/1e6:.2f} MB)")
        except Exception as e:
            print(f"  {name}: INDIRILEMEDI {type(e).__name__} {e}")
    print(f"toplam {len(files)} dosya, dizin={dest}")


def cmd_subs(a):
    r = call("/competitions/submissions/list/" + a.comp, params={"page": 1})
    if not r:
        print("submission bulunamadi"); return
    for x in r[:a.n]:
        print(f"  {str(x.get('date',''))[:19]:20s} {str(x.get('status','')):12s} "
              f"pub={x.get('publicScore')}  priv={x.get('privateScore')}  "
              f"{str(x.get('description',''))[:40]}")


def cmd_log(a):
    u, slug = creds()[0], a.slug.lower()
    r = _output(u, slug)
    log = r.get("log") or ""
    if not isinstance(log, str):
        log = json.dumps(log, indent=1)
    lines = log.splitlines()
    print("\n".join(lines[-a.tail:]) if lines else "(log bos)")


def main():
    p = argparse.ArgumentParser(prog="kdrive.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("whoami").set_defaults(fn=cmd_whoami)
    q = sub.add_parser("list"); q.add_argument("--n", type=int, default=20); q.set_defaults(fn=cmd_list)

    q = sub.add_parser("push")
    q.add_argument("script"); q.add_argument("--slug", required=True)
    q.add_argument("--title", help="verilmezse slug kullanilir; Kaggle slug'i BASLIKTAN turetir")
    q.add_argument("--gpu", action="store_true"); q.add_argument("--tpu", action="store_true")
    q.add_argument("--internet", action="store_true")
    q.add_argument("--comp", action="append"); q.add_argument("--kernel", action="append")
    q.add_argument("--model", action="append"); q.add_argument("--dataset", action="append")
    q.set_defaults(fn=cmd_push)

    q = sub.add_parser("status"); q.add_argument("slug"); q.set_defaults(fn=cmd_status)
    q = sub.add_parser("wait"); q.add_argument("slug")
    q.add_argument("--timeout-min", type=float, default=720); q.add_argument("--poll", type=int, default=60)
    q.set_defaults(fn=cmd_wait)
    q = sub.add_parser("output"); q.add_argument("slug"); q.add_argument("--dest"); q.set_defaults(fn=cmd_output)
    q = sub.add_parser("log"); q.add_argument("slug"); q.add_argument("--tail", type=int, default=200)
    q.set_defaults(fn=cmd_log)
    q = sub.add_parser("subs"); q.add_argument("--comp", default="biohub-cell-tracking-during-development")
    q.add_argument("--n", type=int, default=15); q.set_defaults(fn=cmd_subs)

    a = p.parse_args()
    sys.exit(a.fn(a) or 0)


if __name__ == "__main__":
    main()
