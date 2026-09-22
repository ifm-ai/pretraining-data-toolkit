# pretraining-data-toolkit

Tools for annotating pretraining corpora, assigning quality buckets, shuffling
Parquet data into JSONL shards, and inspecting records. Runs locally or as SLURM
array jobs. The original `annotate/`, `shuffle/`, and `tools/` organization is retained.

**Release candidate:** source and dependency declarations are included. Third-party
Python distributions and model weights are downloaded separately. This environment
could not install the data-processing dependencies or run CUDA inference; see
[validation](docs/VALIDATION.md) for the exact checks and remaining gates.

## Layout

| Folder | Purpose |
| --- | --- |
| [annotate](annotate/README.md) | JSONL annotation, five-way quality bucketing, Parquet row counts |
| [shuffle](shuffle/README.md) | Distributed two-pass shuffle and Parquet diagnostics/recovery |
| [tools](tools/README.md) | `party` dataset inspector and model prefetch |
| [pretraining_data](pretraining_data/README.md) | Shared runtime helpers and model manifest |

## Installation

Use Python 3.10–3.12. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

This installs the CPU tools. For model annotation, install a CUDA-enabled PyTorch
build suitable for your hardware, then:

```bash
python -m pip install -e '.[annotate,inspect]'
python -m tools.download_models --lock-output models.lock.json
export PRETRAINING_MODEL_MANIFEST="$PWD/models.lock.json"
```

The downloader fetches model code, tokenizers and classifier weights and records
immutable Hub revisions. It does **not** fetch the JAIS 13B weights: only its
tokenizer is used. `HF_HOME` controls the cache location. See
[dependencies](docs/DEPENDENCIES.md) for optional xFormers and offline preparation.
The version ranges are declarations, not an environment lock validated here.

## End-to-end usage

Run commands from this directory, or install the project to use `python -m` from
any directory. Outputs must be outside their input tree.

```bash
# 1. GPU annotation: start with a small batch; increase after measuring memory.
python -m annotate.annotate_v2 \
  --input-dir examples/input --output-dir work/annotated \
  --glob '**/*.jsonl' --batch-size 8

# 2. Assign quality categories using the supplied score thresholds.
python -m annotate.classify work/annotated work/quality
python -m annotate.count_docs work/quality

# 3. Shuffle one category; choose M so each complete pile fits in RAM.
python -m shuffle.shuffle_1stpass_large_strings work/shuffle-high work/quality/High 16 \
  --batch-size 65536 --buffer-mb 32
python -m shuffle.shuffle_2ndpass --base work/shuffle-high --num-piles 16

# 4. Inspect a record.
party work/shuffle-high/_jsonl
```

Repeat shuffle for each nonempty quality directory. Pass 1's `M` must equal pass
2's `--num-piles`. Small input samples may produce no `High` records; choose a
category that contains Parquet files. Annotation keeps source columns (flattening
one dictionary level); bucketing intentionally projects a documented training
schema. Pass 2 emits JSONL, not Parquet.

## Parallel runs and recovery

Use `RANK` and `WORLD_SIZE`, or a contiguous SLURM array. See
[deployment](docs/DEPLOYMENT.md). Pass 2 checks every pass-1 rank's success manifest
and expected pile parts. All workers in one pass must see the same input files.
Do not change input files or rank assignment during a run.

Annotation resumes by skipping existing final Parquet files. It publishes a file
only after successful writing; incomplete `.tmp` files are not completion markers.
Use a new output directory after changing inputs, models or settings. Bucketing
and shuffling reject existing output for a rank; use fresh directories when
restarting either stage. Partial outputs from a failed shuffle are not a dataset.

## Provenance and licensing

All original source scripts and three threshold files are preserved or updated;
`tools/party.txt` is now executable Python in `tools/party.py`. macOS archive metadata
and organization-specific paths, accounts, queues, and environment activation were
removed. No source-code license was supplied, so none has been invented. The
repository owner must select the source license before public distribution.
Model artifacts retain their upstream terms; see [third-party sources](THIRD_PARTY.md).
