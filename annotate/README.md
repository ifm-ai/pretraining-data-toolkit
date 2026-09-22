# Annotation and quality bucketing

## Programs

| File | Function |
| --- | --- |
| `annotate_v2.py` | GPU classifiers, fastText quality scores, and JAIS token counts |
| `classify.py` | Convert scores to 20 buckets, then five quality categories |
| `count_docs.py` | Metadata row counts per immediate Parquet subdirectory |
| `jsonl_rw.py` | Plain/gzip JSONL readers, writers, prefetch, and optional Torch reader |
| `resumable_array.py` | Included checkpoint/task helper |
| `breaks_*.json` | Three original threshold lists, 19 boundaries each |
| `annotate_v2.sh` | Portable wrapper forwarding annotation CLI arguments |
| `annotate_v2_all.sh` | Annotate each immediate dataset subdirectory |

Install `.[annotate]` for annotation, or the base package for classification/counting.
Annotation requires a CUDA GPU; it runs TopicClassifier, FormatClassifier and
FineWeb-Edu in model-worker threads. The two fastText models score on the CPU.
JAIS is used only for tokenization. BF16 is used when supported, otherwise FP32.
Memory-efficient attention is off by default; `--memory-efficient-attention`
requires compatible xFormers. Model failures propagate instead of hanging on a
result queue. Full GPU behavior remains unverified in this release environment.

## Inputs and outputs

Input: JSONL objects with a string `text` and optional `meta.url`. Use
`--text-field messages` to read the first message's `content`, or name another
string field. For gzip use both `--glob '**/*.jsonl.gz' --compression gzip`.
One level of input dictionaries is flattened (`meta.url`, etc.). Dataset adapters
for HPLT, S2ORC, MegaMath, PhilPapers and USPTO are selected by input path names;
retain these names when their special schemas are needed.

Output: one Parquet file per input, preserving its relative parent directory.
Added columns: `TopicClassifier`, `FormatClassifier`,
`fineweb_edu_classifier_score`, `fasttext_eli5_score`,
`fasttext_preselect_score`, and `jais_token_count`.
The token count retains the supplied code's `len(tokenizer.encode(text)) + 1`
convention. It is not a tokenizer-independent token count.

The first nonempty batch establishes the file schema. Later schema drift fails
that file; normalize heterogeneous metadata upstream. Empty input files produce
no Parquet file and will be reconsidered on a later run. Malformed JSONL causes a
visible failure. Completed files are skipped by existence, not a content hash.

```bash
bash annotate/annotate_v2.sh --input-dir data/raw --output-dir work/annotated --glob '**/*.jsonl'
bash annotate/annotate_v2_all.sh data/datasets work/annotated --glob '**/*.jsonl'
python -m annotate.classify work/annotated work/quality
python -m annotate.count_docs work/quality
```

## Quality rules (preserved)

Each of the three quality scores is cut into buckets 0–19 with the supplied
boundaries; equality stays in the lower interval. The final bucket is the maximum
of the three. Missing scores become bucket 0; a row with all scores missing is
therefore `Low`. These are fixed supplied thresholds, not percentiles recalculated
for the current corpus. Their calibration dataset was not provided.

| Final bucket | Quality category |
| --- | --- |
| 19 | High |
| 18 | Medium-High |
| 12–17 | Medium |
| 7–11 | Medium-Low |
| 0–6 | Low |

Classification retains `text`, `subset`, both classifier IDs, all three scores,
`jais_token_count`, `meta.token_count`, `meta.lang`, and `meta.lang_score`; all other
source fields are dropped. It adds bucket and quality columns. Missing subset is
`unknown`; missing metadata fields become typed nulls. Threshold files resolve
relative to this package, or override them with `--breaks-dir`.

Classification processes ten source files at a time and materializes each quality
category before splitting into 50,000-row files. It is not strictly memory-bounded
streaming. Count failures propagate rather than being recorded as zero documents.
