# TrustFusion — reproducibility code

Code, configurations and protocols for the article **"TrustFusion: An Explainable, Uncertainty-Calibrated Efficient
Multimodal Deep Learning Framework for Malware Detection and Classification"** by Mani Arora and Anuj Kumar Gupta
(revised version, submitted to the International Journal of Intelligent Engineering and Systems, IJIES).

TrustFusion fuses two views of a file:

* a **byte-image stream**: the raw bytes rendered as a 3-channel image (grey level, windowed entropy, bigram
  frequency) with the width table of Nataraj et al. (2011), encoded by a ResNet-50 whose layer-4 maps are kept as
  49 tokens;
* a **static-feature stream**: the EMBER descriptor (or a 48-d call-graph descriptor for MalNet) split into field
  groups, one token per group, encoded by a residual set encoder.

The streams interact through **bidirectional co-attention** and a **scalar confidence gate**; **modality dropout**
with learned mask tokens makes the model robust to a missing view. Uncertainty comes from **head-only Monte-Carlo
dropout** (the fused representation is computed once, T = 12 head passes), the training objective adds a
**differentiable calibration loss** to a class-balanced focal loss, a **split / time-weighted conformal layer** decides
whether to predict or abstain, and **Grad-CAM++ through the fusion** is projected back to byte offsets and PE regions.

> This repository contains no experimental results. `runs/` and `results/` ship empty; every number in the
> manuscript is recomputed by the scripts below. `SPEC.md` lists every protocol value the code implements.

## Contents

```
README.md  SPEC.md  LICENSE  CITATION.cff  pyproject.toml  requirements.txt  environment.yml
configs/
  base.yaml                     seeds, split seed, tuning budget, early stopping, alpha, T, Table 5 comparisons, figure layout
  data/{bodmas,ember2024,malnet_tiny}.yaml        corpus paths, protocols, split rules, P-T windows, image settings
  trustfusion/{bodmas_pr,bodmas_pt,ember2024,malnet_tiny}.yaml
  baselines/{byte_cnn,msvit,lgbm_static,feature_graph,gin_callgraph,late_fusion,cross_attention,mcdropout_cnn,conformal_lgbm}.yaml
  ablation/{component,factorial}.yaml  faithfulness.yaml  provenance.yaml
  feature_layouts/{bodmas_ember_v2,ember2024,malnet_callgraph}.yaml
trustfusion/                    the Python package
  data/{byte_image,static_features,datasets,splits}.py
  models/{image_encoder,set_encoder,coattention,trustfusion,base}.py  models/baselines/*.py
  losses.py conformal.py temporal.py attribution.py faithfulness.py metrics.py stats.py provenance.py
  training.py evaluation.py aggregate.py io.py utils.py cli.py
scripts/                        prepare_*, make_splits, tune, train, train_gbdt, calibrate, evaluate, run_temporal,
                                run_component_ablation, run_factorial_ablation, run_faithfulness, run_provenance,
                                seed_stats, aggregate_results, collect_benign, make_synthetic, reproduce_all.sh, smoke_test.sh
docs/                           PROTOCOL.md  BASELINES.md  RESULTS_SCHEMA.md  RELEASE.md  MANUSCRIPT_MAP.md
paper_figures/                  make_figures.py and data/*.csv (values plotted in the manuscript)
tests/                          pytest suite on synthetic data
runs/  results/  splits/        empty output folders
```

## Installation

Python >= 3.10. The manuscript setting uses PyTorch 2.4, torchvision 0.19 and LightGBM 4.3.

```bash
git clone https://github.com/mani-arora/TrustFusion && cd TrustFusion
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or: conda env create -f environment.yml
pip install -e .
```

For a CPU-only machine install the CPU wheels first, e.g.
`pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu`.

Optional packages (never required by the tests):

| package | used for |
|---|---|
| `timm` | ImageNet-21k ViT-S/16 weights of the multi-scale ViT baseline (otherwise a randomly initialised ViT-S/16 built from torchvision blocks) |
| `ember` + `lief` | EMBER v2 extraction for the benign pool and the BODMAS *coupled* faithfulness protocol |
| `thrember` | EMBER2024 vectorisation (`pip install git+https://github.com/FutureComputing4AI/EMBER2024`) |
| `androguard` | DEX method offsets for the MalNet *coupled* faithfulness protocol |
| `pyarrow` | parquet prediction files (`.csv.gz` fallback otherwise) |

Check the installation (about one minute of tests plus one to two minutes of smoke test on two CPU cores):

```bash
pytest -q
bash scripts/smoke_test.sh     # synthetic end-to-end run in a temporary directory
```

## Data

No dataset is downloaded by this code. Obtain the data yourself, respecting each licence, and point the prepare
scripts at the local copies (`RAW=raw` below):

| corpus | source | what the scripts read |
|---|---|---|
| BODMAS (Yang et al., 2021) | https://whyisyoung.github.io/BODMAS/ — `bodmas.npz`, `bodmas_metadata.csv`; the disarmed binaries are available **on request from the dataset authors** | `raw/bodmas/{bodmas.npz,bodmas_metadata.csv,binaries/<sha256>}` |
| benign PE pool | 24,180 Windows PE files with a `source` column: clean Windows 10 (`win10`, 11,368), Windows 11 (`win11`, 9,527) and open-source repositories (`repo`, 3,285); collected with `scripts/collect_benign.py`, not redistributed (see `docs/RELEASE.md`) | `raw/benign/manifest.csv` (`sha256,source,path`), `raw/benign/ember_v2_features.npy` |
| EMBER2024 (Joyce et al., 2025) | https://github.com/FutureComputing4AI/EMBER2024 (`thrember.download_dataset`, `thrember.create_vectorized_features`) | `raw/ember2024/*.jsonl`, `X_train.dat`, `X_test.dat` |
| MalNet (Freitas et al., 2022) | https://mal-net.org — MalNet-Image Tiny and MalNet-Graph | `raw/malnet/malnet-images-tiny/{train,val,test}/<type>/<family>/<sha256>.png`, `raw/malnet/malnet-graphs/**/<sha256>.edgelist` |
| MalNet detection benign source | 5,000 benign Android APKs collected from the Google Play Store with zero VirusTotal detections (source name `googleplay`), rendered as MalNet-style DEX images; not redistributed, listed by sha256 in the release index | `raw/malnet/benign/<sha256>.png` (`prepare_malnet.py --benign-images-dir raw/malnet/benign --benign-source googleplay`) |

```bash
python scripts/prepare_bodmas.py --bodmas-npz raw/bodmas/bodmas.npz --bodmas-metadata raw/bodmas/bodmas_metadata.csv \
    --binaries-dir raw/bodmas/binaries --benign-manifest raw/benign/manifest.csv \
    --benign-features raw/benign/ember_v2_features.npy --out data/processed/bodmas --check-counts
python scripts/prepare_ember2024.py --data-dir raw/ember2024 --vectorized-dir raw/ember2024 --out data/processed/ember2024 --check-counts
python scripts/prepare_malnet.py --images-dir raw/malnet/malnet-images-tiny --graphs-dir raw/malnet/malnet-graphs \
    --out data/processed/malnet_tiny --workers 16 --check-counts
```

## Splits

One split per corpus and protocol, built with split seed 2024 and identical for every method and seed
(`docs/PROTOCOL.md` has the exact procedures and counts):

```bash
python scripts/make_splits.py --corpus bodmas --check-counts         # P-R family, P-R detection, P-T, cross-source
python scripts/make_splits.py --corpus ember2024 --check-counts      # family subset, detection subset
python scripts/make_splits.py --corpus malnet_tiny --check-counts    # add --benign-source googleplay for detection
```

`--check-counts` asserts every count reported in the manuscript (values in `configs/data/*.yaml`): BODMAS P-R
40,105 / 4,297 / 4,297 / 8,594 and 33,852 / 3,627 / 3,627 / 7,254; the benign pool of 11,368 Windows 10, 9,527
Windows 11 and 3,285 repository files; the P-T label space of 68 families + residual; 2,358 EMBER2024 families with a
family subset of 1,101,898 training and 254,284 test files (detection subset 2,340,000 / 540,000); MalNet-Tiny
61,201 / 4,371 / 4,372 / 17,486 with 1,286 files lacking a call graph and, for detection, 5,000 `googleplay` benign
files. Every count is also written to `splits/summary.json`; the EMBER2024 family subset is recorded next to the
malicious files of the official split it is drawn from, which bound it. The split files
`splits/<corpus>_<protocol>.csv` are sha256 lists with the first-seen timestamp of every file.

## Reproducing the manuscript

`scripts/reproduce_all.sh` runs every stage in order; each stage can also be run alone
(`bash scripts/reproduce_all.sh train calibrate evaluate`). `docs/MANUSCRIPT_MAP.md` maps each table, figure and
number of the manuscript to its command and output file. The main commands are:

```bash
# (optional) 20-trial random search per baseline and corpus; configs already hold the selected values
python scripts/tune.py --config configs/baselines/cross_attention.yaml --corpus bodmas

# TrustFusion, 5 seeds (P-R family task; --protocol pr_detection for the detection subset)
python scripts/train.py --config configs/trustfusion/bodmas_pr.yaml
python scripts/calibrate.py --run runs/bodmas/pr_family/trustfusion
python scripts/evaluate.py  --run runs/bodmas/pr_family/trustfusion

# baselines (torch methods via train.py, LightGBM and the late-fusion stacker via train_gbdt.py)
python scripts/train.py      --config configs/baselines/cross_attention.yaml --corpus bodmas
python scripts/train_gbdt.py --config configs/baselines/conformal_lgbm.yaml --corpus ember2024

# P-T chronological protocol: frozen model, weighted sliding-window conformal over W1..W7
python scripts/train.py --config configs/trustfusion/bodmas_pt.yaml
python scripts/run_temporal.py --run runs/bodmas/pt_family/trustfusion --mode both

# ablations, controls, statistics, tables and figures
python scripts/run_component_ablation.py
python scripts/run_factorial_ablation.py --train-missing
python scripts/run_faithfulness.py --run runs/bodmas/pr_family/trustfusion
python scripts/run_provenance.py      # also checks per-source TPR@1% FPR against per-source FPR (consistency.csv)
python scripts/seed_stats.py --paired --out results/tables/table5.csv   # t_ci and p_holm_formatted as in Table 5
python scripts/aggregate_results.py
python paper_figures/make_figures.py --data paper_figures/data --out paper_figures/out
```

Outputs follow `runs/<corpus>/<protocol>/<method>/seed<k>/` with `best.pt`, `predictions.parquet` and `probs.npz`;
the column schema is in `docs/RESULTS_SCHEMA.md`.

### Figure data

`paper_figures/data/*.csv` currently holds the values plotted in the manuscript figures. `scripts/aggregate_results.py`
regenerates these files from the per-sample prediction files (seed-averaged) and overwrites a file only when all of
its inputs exist; `paper_figures/make_figures.py` then redraws the figures.

## Seeds and determinism

* Model seeds {0, 1, 2, 3, 4}; tuning uses seed 0; every split uses seed 2024.
* `trustfusion.utils.seed_everything` seeds `random`, NumPy and PyTorch, sets `cudnn.deterministic = True`,
  `cudnn.benchmark = False` and `torch.use_deterministic_algorithms(True, warn_only=True)`; DataLoader workers are
  seeded from the torch generator. LightGBM runs with `deterministic=True` and the model seed.
* Bit-identical results across different GPUs, driver versions or thread counts are not guaranteed by PyTorch; the
  released per-sample predictions (`docs/RELEASE.md`) allow every table to be recomputed without retraining.

## Hardware

The unit tests and the smoke test run on a CPU. Full reproduction trains ResNet-50, ConvNeXt-T and ViT-S models on
224 x 224 inputs for five seeds and needs a CUDA GPU; memory use scales with the batch sizes of the configs (128 for
TrustFusion and the byte-image CNN, 64 on EMBER2024 and for the ViT / cross-attention baselines). EMBER2024 needs
disk space for the vectorised features (about 33 GB as float32 for 3.2 M x 2,568) and enough RAM for LightGBM on the
2.3 M-file training subsets. The manuscript's experiments ran on one NVIDIA RTX 4090 with PyTorch 2.4 and LightGBM 4.3.

## Licence

Code: MIT (`LICENSE`). Datasets keep their own licences; BODMAS binaries and the Windows benign files are not
redistributed (`docs/RELEASE.md`).

## Citation

See `CITATION.cff`. Please cite the article (M. Arora and A. K. Gupta, "TrustFusion: An Explainable,
Uncertainty-Calibrated Efficient Multimodal Deep Learning Framework for Malware Detection and Classification",
International Journal of Intelligent Engineering and Systems, submitted) when using this code.

## Code availability

The source code, configuration files, split indices and evaluation scripts required to reproduce all experiments are
publicly available at https://github.com/mani-arora/TrustFusion. Trained checkpoints and per-sample predictions are
distributed as release assets (DOI: 10.5281/zenodo.17106842); see `docs/RELEASE.md`.
