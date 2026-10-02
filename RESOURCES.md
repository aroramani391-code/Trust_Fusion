# Resource estimate — TrustFusion reproducibility archive

(FULL-SIZE ILLUSTRATIVE build: synthetic rows at the REAL sample counts, so the sizes
below are realistic for the archive you will actually upload.)

## A. The archive you upload (measured here)

| item                         | rows / notes                | size      |
|------------------------------|-----------------------------|-----------|
| train .idx                   | 40,105 rows                 | ~3.9 MB   |
| val .idx                     | 4,297 rows                  | ~0.4 MB   |
| cal .idx                     | 4,297 rows                  | ~0.4 MB   |
| test .idx                    | 8,594 rows                  | ~0.8 MB   |
| manifest.sha256              | 4 lines + counts            | <1 KB     |
| pt_training.log (1 seed)     | ~6,960 steps (60 epochs)    | ~0.5 MB   |
| bodmas.yaml + scripts + docs |                             | ~15 KB    |
| **Uncompressed total**       |                             | **~6 MB** |
| **Zipped**                   |                             | **~3 MB** |

Scaling notes:
* ~97 bytes per index row; add a column (e.g. benign_source) -> ~+0.6 MB total.
* Log x5 if you store all five seeds -> ~2.5 MB.
* Add EMBER2024 + MalNet-Image index files the same way -> still well under ~50 MB.

=> The archive is TINY. Zenodo's per-record limit is 50 GB, GitHub release assets
   up to 2 GB each — both are far above what you need. No special storage required.

## B. What is NOT in the archive (and why)

The archive stores only pointers (SHA-256 + timestamp) to samples, never the malware
itself. The actual data and compute you need to RE-RUN the experiments is separate and
much larger. Rough, approximate figures (confirm against your own setup):

| resource                          | approx size / spec        |
|-----------------------------------|---------------------------|
| BODMAS feature file (2,381-d)     | ~1–2 GB                   |
| BODMAS disarmed PE binaries       | ~tens of GB               |
| Your benign collection (24,180)   | ~tens of GB               |
| EMBER2024 features (2,568-d)      | ~100+ GB (3.2M files)     |
| MalNet-Image Tiny + graph view    | ~tens of GB               |
| GPU                               | 1x RTX 4090 (24 GB) — as in the paper |
| Working disk for all corpora      | plan for ~250–400 GB      |

These numbers are approximate order-of-magnitude estimates to help you plan; they are
not measured here. The reviewers only need Part A (the small archive) to verify the
splits and chronology — they do not need Part B.
