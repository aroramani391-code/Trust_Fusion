====================================================================
  ILLUSTRATIVE TEMPLATE — NOT THE REAL ARCHIVE
  All data files below contain SMALL SYNTHETIC EXAMPLES so you can
  see the exact format. Before you upload anything to Zenodo/GitHub,
  REGENERATE every file from YOUR OWN data (see "What to do").
====================================================================

# TrustFusion — Reproducibility Archive (release v1.0-r3)

Backs the experiments in:
> M. Arora and A. K. Gupta, "TrustFusion: An Explainable, Uncertainty-Calibrated
> Efficient Multimodal Deep Learning Framework for Malware Detection and
> Classification," *Int. J. of Intelligent Engineering and Systems*.

Code: https://github.com/aroramani391-code/Trust_Fusion
DOI:  10.5281/zenodo.XXXXXXXX   <-- fill in after you deposit on Zenodo

This archive exists because `splits/`, `runs/` and `results/` in the code repo are
*regenerable output folders* (they ship empty). The files here are the files
themselves, so the results can be checked without re-running anything.

--------------------------------------------------------------------
## What each file answers (reviewer map)
--------------------------------------------------------------------
Reviewer 1 (split indices):  splits/*.idx + splits/manifest.sha256
   -> the actual index files + their checksums, so "no repartitioning" is verifiable.
Reviewer 2 (temporal log):   logs/pt_training.log + config/bodmas.yaml
   -> per-batch timestamps proving Dec-2019 gave NO gradient update.
Reviewer 3 (parameters):     config/bodmas.yaml (model.freeze_backbone: true)
   -> shows the ResNet-50 is frozen, so 18.4M = trainable, 41.9M = total.

--------------------------------------------------------------------
## Contents
--------------------------------------------------------------------
scripts/
  make_splits.py              build the P-R index files from your metadata
  build_release_index.py      write manifest.sha256 (checksums + counts)
  verify_temporal_windows.py  check the Aug-Nov-2019-only chronology
splits/
  bodmas_family_PR_train.idx  (real: 40,105 rows)   | columns: sha256,first_seen_utc,label
  bodmas_family_PR_val.idx    (real:  4,297 rows)
  bodmas_family_PR_cal.idx    (real:  4,297 rows)
  bodmas_family_PR_test.idx   (real:  8,594 rows)
  manifest.sha256             count + SHA-256 of each .idx above
logs/
  pt_training.log             per-batch step, timestamp, first-seen month
config/
  bodmas.yaml                 splits + pt chronology + frozen-backbone flag

--------------------------------------------------------------------
## Verify (what a reviewer runs)
--------------------------------------------------------------------
    cd splits && sha256sum -c manifest.sha256          # all must say OK
    python ../scripts/verify_temporal_windows.py --log ../logs/pt_training.log

On the REAL data the four checksums become the paper's values:
    ca84c013…  db3d8342…  c84c812d…  3beb59cf…

--------------------------------------------------------------------
## What to do (turn this template into the real archive)
--------------------------------------------------------------------
1. Build your per-sample table metadata.csv  (sha256,first_seen_utc,family,benign_source).
2. python scripts/make_splits.py --meta metadata.csv --config config/bodmas.yaml --out splits
3. python scripts/build_release_index.py --splits splits
4. Put your REAL pt_training.log in logs/ and your REAL bodmas.yaml in config/.
5. cd splits && sha256sum -c manifest.sha256     (confirm OK; confirm hashes match the paper)
6. Delete this ILLUSTRATIVE banner, then zip and upload (see HOW_TO_UPLOAD.md).

Try the scripts right now with no data:
    cd scripts && python make_splits.py --demo --out ../splits && python build_release_index.py --splits ../splits
