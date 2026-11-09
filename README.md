# ASR Fluency Stability: Kaggle research code

This repository packages the source code and a curated subset of the archived
artifacts for the exploratory study of *within-utterance ASR transcription
variability and human-aligned read-aloud L2 English fluency scoring*.

## What was evaluated

- Public dataset: SpeechOcean762, official **training** split only.
- Pilot: **20 speakers**, **400 recordings** (20 per speaker).
- Five Whisper small.en hypotheses per recording: H1 greedy, H2 beam search,
  H3-H5 temperature-0.3 sampling (seeds 101/202/303).
- 2,000 total ASR transcript hypotheses from ONE ASR architecture.
- Acoustic (A), transcript (B1), prompt-aware transcript (B2), acoustic +
  transcript (C1), full prompt-aware fusion (C2), and mean-label (M0) baseline.
- Fixed five-fold speaker-disjoint cross-validation; Ridge alpha=1;
  imputer/scaler fitted on training folds.
- Post-hoc speaker-bootstrap, clipping, exact speaker-level sign-flip/Holm,
  and exploratory feature augmentation audits in `reproducibility/`.

The official test split has NOT been evaluated. This is NOT a cross-corpus
validation or a new ASR model. All statistical claims are exploratory.

## Folder guide

- `notebooks/01_*` through `08_*`: original Kaggle Python cells grouped by phase,
  with outputs and execution counts removed.
- `notebooks/archive/00_*`: setup, restoration and ZIP snapshot operations.
- `notebooks/archive/99_*`: all 153 original non-empty code cells in original
  sequence with outputs cleared. Two empty cells were omitted.
- `scripts/train_models.py`: exact source extract of training stages
  (original notebook cells 106, 107, 109, 111, 112).
- `scripts/evaluate_models.py`: speaker-bootstrap and robustness extracts
  (cells 113, 115, 117, 118).
- `scripts/statistical_robustness.py`: STEP 12.4 deterministic robustness
  recomputation (one path adaptation only).
- `scripts/restore_kaggle_project.py`: restore frozen source archive safely.
- `scripts/verify_package.py`: verify every packaged file hash and notebooks.
- `configs/`: original configuration snapshots. Data, results and original
  source archive are available in the separate full private research ZIP.

## Important: executable status

**Source cleanup/partitioning is not the same as validating a new modular
pipeline.** Python source text was retained verbatim; some cells are repeated
experiments, intermediate versions or append-only historical steps. The files
in `scripts/` were syntax checked, not end-to-end re-run during repackaging.
Some notebooks depend on files and files written by previous notebooks.
Their original hard-coded project path is:

`/kaggle/working/acoustic_linguistic_research`

To verify this code-only package, run:

```bash
python scripts/verify_package.py
```

Before running the Kaggle stage scripts, first recover the original project
workspace from the separately supplied full private research archive. The
restore utility is present but its default source ZIP is intentionally omitted
from this public code-only release.

To rebuild original ASR outputs or waveform-derived acoustic features, obtain
the original SpeechOcean762 **train Parquet / audio** from its authoritative
source; the source checkpoint intentionally contains NO WAV/audio or Parquet.
Do NOT run old inference or training over frozen results unless working on a
copy of the Kaggle project. The restore script skips existing files by default.

The source notebook contained 155 code cells, of which 153 had non-empty code.
The complete ordered source and every phase notebook have ALL outputs removed.
See `CELL_INDEX.csv` for mapping of original cell indices to notebooks.

## Recorded checkpoint identifiers

- SpeechOcean762 revision:
  `06385584fad212b26134c656fdd3ccf9f093f33e`
- Whisper small.en revision:
  `e8727524f962ee844a7319d92be39ac1bd25655a`
- Recorded ASR package: Transformers 5.16.1.
- Saved random seed for pilot: 42; H3-H5 decoding seeds: 101, 202, 303.

## Re-run speaker-level Holm analysis

To run this script, you must first copy frozen OOF predictions from the
full private research package into `reproducibility/step12_4/`, then run
`python scripts/statistical_robustness.py`. The script
writes derived tables there. It takes time because it enumerates all
1,048,576 speaker-level label sign combinations and performs 50,000 bootstrap
resamples. Its hypothesis family was selected post hoc.

## Data, privacy and rights

The private full archive includes labels, ASR transcripts and metadata
derived from a third-party corpus. No such derived data are bundled in this
code-only package. **Review redistribution permissions before making any
full-data archive public.** Do not add API tokens, raw WAV/Parquet data, private
Kaggle credentials, or unpublished participant information. The uploaded
notebook's code was scanned for common token patterns; this does NOT prove
there are no secrets in every original file or output.

No code license is claimed without the author's decision. See
`LICENSE_DECISION.md`. Citation information is deferred until publication
metadata and author details are confirmed. See `CITATION_TEMPLATE.md`.


## Note for the code-only release

This variant intentionally EXCLUDES the copyrighted corpus-derived tables,
original source ZIP and STEP 12 input files; do not expect complete model
training or statistical reproduction to run without obtaining the separate
authorized research dataset and copying it to the documented project paths.
The `verify_package.py` script checks only the shipped code-only files.
For full private local reproducibility, use the full research package ZIP.
