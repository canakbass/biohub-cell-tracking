# Biohub — Cell Tracking During Development

My solution to Kaggle's [Biohub Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) competition: detecting and tracking cell nuclei through 3D+time light-sheet microscopy volumes of developing embryos, including cell divisions.

**Final validated result: LB 0.863** (public leaderboard, embryo-disjoint hidden test set). Full engineering journal with every measurement, failure, and root-cause analysis is in [`STATE.md`](STATE.md).

## The task

- 3D+time volumes (`(T, Z, Y, X)`, physical voxel scale `1.625 × 0.40625 × 0.40625 µm`), sparse ground truth (~3.6% of true cells annotated).
- Train/test split is **embryo-disjoint**: all 199 training datasets come from 2 known embryo colonies; the hidden test set is entirely different embryos. This single fact drove most of the interesting engineering decisions below.
- Official metric (`adj_edge_jaccard + 0.1·division_jaccard`, verified byte-for-byte against the [official scorer](https://github.com/royerlab/kaggle-cell-tracking-competition)) is **not F1** — optimizing node F1 can actively lower the score.

## Pipeline

1. **Detector**: a 3D ConvNeXt-style U-Net (`UNet4D_SOTA_V9`) trained with a masked CenterNet-style loss that explicitly ignores unannotated-but-bright voxels instead of treating them as background — the single fix that produced the biggest real, hidden-test-validated gain this project made (adj +0.0449 → LB 0.786 → 0.863 after further tuning).
2. **Candidate generation**: GPU-side peak detection (`max_pool3d`) + single-pass greedy NMS, density-adaptive per-frame node count targeting (since the hidden test has no ground truth to calibrate against directly).
3. **Linking**: greedy mutual-nearest-neighbor with a physically-scaled distance gate, plus physical-space gap closing for short detection dropouts.
4. **Hack auditor**: every submission is scanned (`drive/audit_submission.py`) for the known exploit patterns (hub nodes, out-of-volume coordinates, cross-clip edges) before it's ever written to disk.
5. **Kaggle driver**: a dependency-free (`drive/kdrive.py`) push/run/monitor/score CLI, so a full experiment cycle (push kernel → poll status → pull logs → parse metric) never requires touching the Kaggle web UI.

## What worked, and — more usefully — what didn't

The theoretical ceiling of the metric is ~1.20; the score plateaued around 0.863 despite several methods that looked like clear wins on local validation. The most important finding of this project is *why* they didn't generalize:

| Attempt | Local signal | Real hidden-test result |
|---|---|---|
| Detector loss fix (ignore-mask for sparse GT) | adj +0.0449 | **held** — 0.786 → 0.863 |
| Learned edge classifier (geometric features, K-NN=16) | +0.0094 core, both folds up | **regressed** −0.004 |
| LAP (Hungarian) linking, tight gate | +0.0097 core, both folds up | **regressed** −0.004, same magnitude |
| LAP re-tested across a full gate sweep (5.5→12µm) | monotonically worse as gate widens; no safe middle ground | never shipped — ruled out analytically |
| Isotropic-pooling architecture fix (physically motivated, not tuned on local data) | training-internal proxy +0.06 | **regressed** on the real end-to-end pipeline (root cause: Z-axis peak-splitting) |
| Division detection, 2 independent redesigns | went from 0/99 → 63/99 recoverable candidates after fixing a linker-dependency bug | still 0 true positives — severe class imbalance, not yet solved |

The pattern across the first three rows is the real lesson: **anything calibrated only against the 2 known training embryos — whether it's learned weights or just a hand-picked hyperparameter — has no guarantee of surviving contact with a genuinely new embryo.** A wide, physically-loose linking gate generalized; a narrow one that scored better locally did not. Every non-detector improvement attempted after the initial fix is written up in detail — including the ones that failed — in `STATE.md`, because the failures were more informative than the one success.

## Repo layout

```
STATE.md                       full engineering journal (numbered "Adım" steps, every measurement)
v11_00_probe.py                metric verification against the official scorer
v11_01*.py, v11_02*.py         candidate generation + post-processing sweeps
v11_03*.py                     linker diagnostics, official-scorer cross-check
v11_05*.py                     detector training (incl. the sparse-GT loss fix, seed2 for ensembling)
v11_06*.py                     learned edge classifier experiments
v11_07*.py, v11_08*.py         division detection (geometric, then two learned redesigns)
v11_09*.py                     LAP/Hungarian linking + gate sweeps
v11_10*.py, v11_11*.py         ensemble (2-seed) experiments
v11_12*.py–v11_14*.py          UNet-embedding-based edge features
v11_15*.py–v11_21*.py          isotropic-pooling architecture fix + diagnosis
v11_18*.py, v11_19*.py, v11_23*.py   targeted "is this actually a bug" audits
v11_S1_submit.py               production submission script (self-contained, offline zarr reader, hack-audited)
drive/kdrive.py                stdlib-only Kaggle API driver (push/status/wait/log/subs)
drive/audit_submission.py      standalone hack/exploit auditor
```

Scripts are numbered in the order the ideas were tried, not in a "clean pipeline" order — this is a research log, not a package. `v11_S1_submit.py` is the one file that matters for reproducing the final score.

## Stack

PyTorch (3D ConvNeXt U-Net, mixed precision), NumPy/SciPy (cKDTree-based linking and NMS), scikit-learn (`HistGradientBoostingClassifier` for the learned-linking experiments), [tracksdata](https://github.com/royerlab/kaggle-cell-tracking-competition) for official-metric cross-validation.
