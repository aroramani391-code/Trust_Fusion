# TrustFusion reproducibility repository — implementation specification

This file is the single source of truth that the manuscript (revised version) describes. Every value below
appears in the manuscript; code, configs and docs must match it exactly.

## 0. Ground rules
* No experimental results may be generated or hard-coded. Result folders (`runs/`, `results/`) ship empty.
  `paper_figures/data/*.csv` already exists and holds the values plotted in the manuscript figures; leave it
  untouched and describe it as such.
* Everything must run end-to-end on a tiny synthetic dataset on CPU (`tests/` + `scripts/smoke_test.sh`).
* Python >= 3.10; PyTorch 2.4 (CPU install is fine for tests), torchvision, LightGBM 4.3, scikit-learn, scipy,
  numpy, pandas, pyyaml, networkx. Optional: `lief` / EMBER2024 `thrember` package for PE feature extraction,
  `androguard` for DEX method offsets, `timm` for ViT/ConvNeXt baselines (fall back to torchvision where possible).
* Repository: `https://github.com/mani-arora/TrustFusion` (authors: Mani Arora, Anuj Kumar Gupta). Checkpoints and
  per-sample predictions: release assets, DOI `10.5281/zenodo.17106842` (as quoted in the response to reviewers).

## 1. Data, subsets and splits (split seed 2024; identical for all methods and seeds)
### BODMAS [Yang et al. 2021]
* 57,293 malicious PE (581 families, Aug 2019 – Sep 2020), disarmed binaries + 2,381-d EMBER-v2 vectors;
  metadata gives sha256, timestamp (first seen), family.
* Family label space (P-R): families with >= 25 samples → 96 named families; all others → `residual` (97 classes).
* Benign pool: 24,180 PE files with a `source` column: `win10` 11,368 (clean Windows 10), `win11` 9,527 (clean
  Windows 11), `repo` 3,285 (open-source repositories).
* **P-R family split** (stratified by class), exact two-stage procedure with sklearn `train_test_split`:
  1. test = 15% (`test_size=0.15`, stratify, `random_state=2024`) → 8,594
  2. from the rest, val+cal = 8,594 absolute (stratified) → train 40,105
  3. val+cal halved (stratified) → val 4,297, cal 4,297
* **P-R detection subset**: 24,180 malicious sampled stratified by family (`random_state=2024`) + 24,180 benign;
  strata = class × benign source; same two-stage procedure → 33,852 / 3,627 / 3,627 / 7,254.
* **P-T temporal protocol** (first-seen timestamp):
  * fit: 2019-08-01 <= t < 2019-12-01, with a stratified random 10% held out as validation (early stopping)
  * initial calibration pool D1: 2019-12-01 <= t < 2020-01-01 (never used for fitting or early stopping)
  * test windows W1..W7 = calendar months Jan 2020 .. Jul 2020; Aug–Sep 2020 unused
  * label space: families with >= 25 samples in the development period (Aug–Dec 2019) + residual
    (68 named families + residual = 69 classes; `splits/summary.json` stores `n_named_families` = 68 and `K` = 69)
  * D_k for k >= 2 = labelled samples first seen in the calendar month before W_k (i.e. W_{k-1});
    labels of W_k are released only after W_k closes. Optional `label_delay_months` (default 0; 1 → use W_{k-2}).
  * the network, gate and head are frozen after training; no retraining.

### EMBER2024 [Joyce et al. 2025]
* Official split: train 2,626,000 files (2023-09-24 .. 2024-09-21), test 606,000 (2024-09-22 .. 2024-12-14);
  2,568-d feature vectors, six formats (Win32, Win64, .NET, APK, ELF, PDF). No binaries → single-stream mode
  (image view permanently masked).
* **Family subset** (Table 4 Acc/Macro-F1, Table 6, coverage, Fig. 5): malicious files of all six formats whose
  ClarAVy family is among the families with >= 10 occurrences in the training set (2,358 families; recomputed and
  asserted from metadata). Calibration = family-subset training files first seen in the last four training weeks
  (2024-08-25 .. 2024-09-21); validation = stratified 10% of the remaining training files; test = official test
  files of the subset. Table 2 sizes: 1,101,898 official training files (train + val + cal) and 254,284 test files.
  Only malicious files carry a family, so these sizes are bounded by the malicious files of the official split;
  `scripts/make_splits.py` writes both (`train_official`, `test_official`, `malicious_train_official`,
  `malicious_test_official`) to `splits/summary.json` and asserts the bound and the Table 2 sizes.
* **Detection subset** (Table 4 AUC, Fig. 3): Win32 + Win64 + .NET only; official train (2,340,000; 10% stratified
  validation) and test (540,000), balanced benign/malicious.

### MalNet-Image Tiny [Freitas et al. 2022]
* Released split: 61,201 train / 8,743 val / 17,486 test, 43 types (4 largest types removed), 256×256 DEX images.
* Released val is halved (stratified, seed 2024) into val 4,371 and cal 4,372.
* Second view: 48-d function-call-graph descriptor from MalNet-Graph matched by sha256; 4 groups × 12 features
  (size/connectivity, degree statistics, degree histogram, local structure). Unmatched samples (1,286): graph view
  masked.
* Detection AUC requires a benign class that the Tiny partition does not contain; the loader must take an
  explicit `--benign-source` argument and refuse to compute detection metrics without it. Manuscript benign source:
  `googleplay`, 5,000 benign Android APKs collected from the Google Play Store with zero VirusTotal detections.

## 2. TrustFusion model
* Byte image: width table (Nataraj et al. 2011): <10 KB: 32; 10–30 KB: 64; 30–60 KB: 128; 60–100 KB: 256;
  100–200 KB: 384; 200–500 KB: 512; 500 KB–1 MB: 768; >1 MB: 1024. h = ceil(|b|/w); zero padding.
  Channels: (1) grey = b/255; (2) entropy: window L = 256 bytes, stride 128, Shannon entropy/8 averaged over
  overlapping windows per byte; (3) bigram: log(1+count[b_{t-1}, b_t]) / log(1+max count). Resize to 224×224
  (bilinear, antialias); per-channel normalisation with training-split statistics. Augmentation: entropy jitter
  ×U(0.95, 1.05); random overlay truncation (p = 0.3, 0–50% of bytes after the last section) for PE only.
  No flips/crops.
* Image encoder: torchvision ResNet-50 (IMAGENET1K_V2 when `pretrained: true`), 3-channel stem, layer4 maps
  (7×7×2048) kept as A^k for Grad-CAM++, projected to d = 512 tokens (49 tokens).
* Static encoder: 8 field groups (EMBER v2 layout for BODMAS: byte histogram 256, byte-entropy histogram 256,
  strings 104, general 10, header 62, sections 255, imports+exports 1280+128, data directories 30 = 2,381);
  EMBER2024 layout read from `configs/feature_layouts/ember2024.yaml` (or derived from `thrember` if installed)
  and mapped onto the same 8 group names; MalNet: 4 groups. Each group → Linear → 512-d token + group-type
  embedding; residual trunk of 2 blocks (LayerNorm → Linear → GELU → Linear) of width 512.
* Co-attention: modality-type embeddings added, 2 layers, 4 heads, width 512; each layer: image tokens attend
  to static tokens and static tokens attend to image tokens (bidirectional), residual + LayerNorm + FFN.
  Mean-pool → z̃_v, z̃_s.
* Gate: g = sigmoid(W_g [z̃_v; z̃_s] + b_g) (scalar per sample); z_f = g z̃_v + (1 − g) z̃_s.
* Modality dropout: each view replaced by a learned mask token with p = 0.15 during training (never both).
* Head: Dropout(0.2) → Linear(512, C). **Only this dropout is stochastic at inference.** `mc_predict(x, T=12)`
  computes z_f once and draws T head passes; returns probs [T, B, C], p̄, entropy H[p̄], mutual information I.
* Loss: class-balanced focal (effective number, β = 0.999, γ = 2) + λ1·L_cal + λ2·L_align with λ1 = 0.35,
  λ2 = 0.10. L_cal = Σ_m |B_m|/n · |acc(B_m) − conf(B_m)| over M = 15 equal-width bins, acc detached.
  L_align = mean(1 − cos(proj(z_v), proj(z_s))).
* Optimisation: AdamW lr 3e-4, weight decay 0.05, cosine schedule, batch 128 (64 on EMBER2024), max 80 epochs,
  early stopping on validation macro-F1 with patience 12; seeds {0,1,2,3,4}.
* Attribution: Grad-CAM++ on A^k with the gradient of the class logit computed from z_f (i.e. through
  co-attention and gate); offsets recovered by inverting the resize and raster (Eq. 3) and intersected with
  the PE section table (header, .text, .rdata, .data, .rsrc, imports, overlay).

## 3. Conformal layer
* Score s_i = 1 − p̄(y_i | x_i); prediction set C(x) = {y : 1 − p̄(y|x) <= q̂}; predict iff |C(x)| = 1.
* Split (P-R, EMBER2024, MalNet): q̂ = k-th smallest score with k = ceil((n + 1)(1 − α)); q̂ = +inf if k > n.
  α = 0.05. (BODMAS: n = 4,297 → k = 4,084.)
* Weighted (P-T): w_i = 2^(−(τ_k − t_i)/h), h = 15 days, t in days; p̃_i = w_i / (Σ_j w_j + 1); the test point
  mass 1/(Σ_j w_j + 1) sits at +inf; q̂_k = inf{q : Σ_i p̃_i 1[s_i <= q] >= 1 − α} (+inf if unattainable).
  n_eff = (Σ w_i)^2 / Σ w_i^2 (≈ 0.87·|D_k| for uniform arrivals over a 30-day month).
  Threshold computed once at τ_k and frozen during W_k.

## 4. Baselines (same splits/seeds/budget; selected values below; search spaces in configs)
Shared: 20 random-search trials per corpus with seed 0 scored by validation macro-F1 (EMBER2024: on a stratified
10% training subsample); best-validation-macro-F1 checkpoint, patience 12, max 80 epochs; no TTA, no snapshot
ensembles, no per-class thresholds.
| Baseline | Selected configuration | Search space |
|---|---|---|
| Byte-image CNN | ResNet-50 (ImageNet), grey image ×3 channels, 224²; AdamW lr 3e-4, wd 0.05, batch 128, CE | lr {1e-4,3e-4,1e-3}, wd {0.01,0.05} |
| Multi-scale ViT | ViT-S/16 (ImageNet-21k) at 112 and 224 px + scale attention; AdamW lr 1e-4, wd 0.05, batch 64; consistency 0.1 | lr {5e-5,1e-4,3e-4}, consistency {0.05,0.1,0.2} |
| Gradient-boosted static | LightGBM, 1,024 leaves, lr 0.05, feature_fraction 0.5, bagging_fraction 0.5 (freq 1), ≤ 4,000 rounds, early stop 100 on validation loss, balanced class weights for families | leaves {256,512,1024,2048}, lr {0.03,0.05,0.1} |
| Feature-graph network | 9 nodes (8 groups + global), edges where |Pearson r| >= 0.3 on training data, 3-layer GCN width 256, mean–max readout; AdamW lr 1e-3, wd 1e-4, batch 256 | width {128,256}, lr {3e-4,1e-3} |
| Structural GIN | 5 GIN layers width 128, sum readout, local-degree-profile node features; Adam lr 1e-3, batch 64 | width {64,128}, lr {3e-4,1e-3} |
| Late-fusion ensemble | byte-image CNN + LightGBM (MalNet: + GIN) probabilities, multinomial logistic stacker C = 1 fitted on validation | C {0.1,1,10} |
| Cross-attention fusion | ConvNeXt-T (ImageNet-1k) + 2×512 MLP, 1 cross-attention layer (image→static), 4 heads, concat head; AdamW lr 1e-4, wd 0.05, batch 64 | lr {5e-5,1e-4,3e-4} |
| MC-dropout image CNN | byte-image CNN, dropout 0.3 before the head, T = 12 | dropout {0.1,0.2,0.3,0.5} |
| Conformal-gated static | LightGBM as above + split conformal (α = 0.05) on the calibration split; predict on singleton sets | as LightGBM |

## 5. Metrics and statistics
* Accuracy, macro-F1 (all classes incl. residual), ROC-AUC, TPR at FPR ∈ {1e-3, 1e-2} by linear interpolation of
  the test ROC, ECE (15 equal-width bins on max probability), selective error at coverage c (top ceil(cN) by
  max p̄), empirical conformal coverage, mean set size, acceptance rate (fraction of singleton sets).
* Seed statistics: mean ± SD over seeds (sample SD); paired differences over seeds; 95% t-interval (df = n−1); paired
  t-test (`scipy.stats.ttest_rel`); Holm correction over the six comparisons of Table 5; paired bootstrap over test
  samples within each seed (10,000 resamples) for the metric difference. Table 5 reports the interval to two decimals
  and the Holm-corrected p as "< 0.001" below 0.001. The t-test p and the t-interval share one mean and standard
  error, so they must agree (`stats.p_from_t_interval`); with five seeds, all six manuscript intervals imply p < 0.001.
* Table 5 comparisons: BODMAS macro-F1 TF vs cross-attention; EMBER2024 macro-F1 TF vs conformal-gated;
  MalNet macro-F1 TF vs cross-attention; BODMAS ECE TF vs cross-attention; P-T loss (W1 − W7) TF vs cross-attention;
  P-T loss TF vs gradient-boosted.

## 6. Control analyses
* Faithfulness (Table 9): deletion / insertion AUC of the target-class probability (deterministic head) over 21
  points (0%, 5%, …, 100% of file length), bytes ranked by projected attribution. Insertion starts from an all-zero
  file. Protocols: `image_view` (bytes perturbed → image re-rendered; static vector fixed) and `coupled` (bytes
  zeroed in the binary → image re-rendered and descriptor re-extracted via an extractor callback; extraction failure
  → static mask token; MalNet: call-graph descriptor recomputed after removing methods whose code lies in masked
  ranges, via a callback). Variants: Grad-CAM image branch, Grad-CAM++ image branch, Grad-CAM++ on fused logits
  with image-branch-only gradients (fusion detached), Grad-CAM++ through fusion.
* Provenance (Sec. 5.2): FPR per benign source at the global threshold giving 1% FPR on all benign test files;
  TPR@1%FPR with benign test files restricted to each source; source probe = LightGBM predicting benign source
  from the descriptor, 5-fold CV macro one-vs-rest AUC; cross-source run: train with benign sources {win10, win11},
  test benign = repo. Consistency: every source shares the malicious test files, so a source with FPR < 1% at the
  global threshold has TPR@1%FPR >= the global-threshold TPR, and a source with FPR > 1% has TPR@1%FPR <= it
  (`provenance.provenance_consistency`, `results/provenance/consistency.csv`).
* Reliability factorial (Table 8): 2×2 over {L_cal on/off} × {MC head on/off (off = dropout disabled at test)}, all
  with the same split-conformal layer; metrics ECE, error at 80% coverage, coverage, set size on P-R; W7 coverage on
  P-T with static D1 threshold and with the weighted sliding window.
* Component ablation (Table 7): no co-attention (concatenation), no image stream, no static stream, no L_cal,
  no conformal gate, no class-balanced objective (plain CE), no modality dropout.

## 7. Outputs
* Split index files: `splits/<corpus>_<protocol>.csv` with columns row, sample_id (sha256), split, window, label,
  timestamp (first seen; empty for MalNet).
* Per-sample predictions: `runs/<corpus>/<protocol>/<method>/seed<k>/predictions.parquet` (or `.csv.gz` fallback)
  with columns: sample_id, corpus, task, protocol, split, window, seed, method, y_true, y_pred, p_max, p_true,
  entropy, mutual_info, gate, set_size, in_set, accepted, benign_source, timestamp; full probability matrices in
  `probs.npz`.
* Checkpoints: `runs/.../best.pt`; release instructions in `docs/RELEASE.md`.
* `scripts/aggregate_results.py` builds `results/tables/table{4..9}.csv` and rewrites `paper_figures/data/*.csv`.
* `docs/MANUSCRIPT_MAP.md`: every table, figure and data-dependent number of the revised manuscript → script,
  command and output file.
