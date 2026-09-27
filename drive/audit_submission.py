#!/usr/bin/env python3
"""HACK DENETÇİSİ — submission.csv'yi metrik exploit imzalarına karşı tarar.

KULLANICI KURALI (2026-09-18): "hackli bir şey kullanmayalım aman dikkat".
Topluluğun 'no-hack' tanımındaki TÜM kategorileri kontrol eder:
  1. görüntü-dışı koordinat  (ör. -10000; hub/sahte-fork node'ları böyle yerleşir)
  2. negatif zamanlı node     (hub t=-1000, sahte fork'lar t=-999...)
  3. dataset'ler arası kenar  (cross-clip)
  4. dt != 1 kenar            (hub'dan köklere t=-1000 -> t>=0 atlayan kenarlar)
  5. çok ebeveynli node       (in-degree > 1: bir hub'ın tüm track'leri birleştirmesi)
  6. out-degree > 2           (biyolojik olarak geçersiz)
  7. dev bağlı bileşen        (hub tüm track'leri tek bileşende toplar)
  8. tanımsız node'a kenar

Saf stdlib — aynı dosya lokalde de Kaggle kernel'inde de çalışır.
İhlal varsa ValueError fırlatır: submit kernel'i submission'ı YAZMADAN çöker.

Kullanım:
  python3 audit_submission.py submission.csv [--T 100 --Z 64 --Y 256 --X 256]
  (kernel içinde)  from audit_submission import audit ; audit("submission.csv")
"""
from __future__ import annotations
import csv, sys, argparse
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    for k in ("T", "Z", "Y", "X"):
        ap.add_argument(f"--{k}", type=int, default=None)
    a = ap.parse_args()
    ok, _ = audit(a.csv, a.T, a.Z, a.Y, a.X, raise_on_fail=False)
    sys.exit(0 if ok else 1)
