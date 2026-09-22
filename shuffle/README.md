# Two-pass shuffle and Parquet recovery

```bash
python -m shuffle.shuffle_1stpass_large_strings work/shuffle data/parquet 32 --buffer-mb 32
python -m shuffle.shuffle_2ndpass --base work/shuffle --num-piles 32 --lines-per-shard 100000
```

The first three arguments are **output base, input directory, number of piles**.
`shuffle_1stpass.sh` launches the large-string implementation and forwards arguments;
`shuffle_2ndpass.sh` forwards pass-2 arguments. Both wrappers work from any directory.

1. Pass 1 partitions sorted input files across ranks, randomly scatters their rows
   across `M` piles, and writes `_piles/pile_NNNNN_rank_RRRRR.parquet` plus a rank
   completion manifest. Any data/schema/write error fails the process.
2. After **all** pass-1 ranks finish, pass 2 verifies manifests and parts, assigns
   piles across its ranks, shuffles each complete pile in memory, and writes
   `_jsonl/part-RRRRR-SSSSSS.jsonl`. Its world size can differ from pass 1.

`shuffle_1stpass_large_strings.py` normalizes top-level UTF-8 strings to Arrow
`large_string`. Nested text types are not recursively converted. The original
`shuffle_1stpass.py` remains available when retaining string types is preferred.
All files must share a compatible schema. Each rank selects its own reference
schema; incompatible schemas across ranks fail during pass 2.

Use the same `M` in both passes. Each pass-1 rank opens `M` output files. The
logical buffering budget can approach `M × buffer_mb`; zero-copy slices retain
backing buffers, so RSS can be higher. Pass 2 needs one entire pile, its shuffled
copy, and serialization overhead in RAM. A larger pile count reduces pass-2 memory
but increases file handles and pass-1 buffering. Start small and measure.

The default seed is 42. With fixed input contents, ordering, batch size, software
versions and pass-1 rank count, the shuffle is deterministic. Changing the rank
count in pass 1 changes the permutation. JSONL requires serializable data types:
Arrow binary/decimal values may need preprocessing. Nulls/nested JSON-compatible
values are retained. Shards never exceed `--lines-per-shard`, but a pile boundary
may make a smaller shard.

Output directories are not append targets. A failed run can leave partial pile
files or JSONL shards; use a fresh base after correcting the error. Do not run two
jobs with the same rank and output base. Pass-2 output checks also reject stale
files from an earlier run of that rank. Manifests attest successful writing, not
cryptographic content identity. Never alter pile parts after completion.

## Diagnose and salvage

```bash
python -m shuffle.salvage_parquet data/suspect.parquet
python -m shuffle.salvage_parquet data/suspect.parquet --salvage-output work/recovered.parquet
```

Salvage is explicitly lossy: unreadable columns can become null, and unreadable
row groups can be omitted. `--no-drop-bad-columns` skips a failed row group rather
than attempting individual columns. Review its report and recovered row counts.
The destination must be new and distinct from the source. This utility cannot
promise recovery of a missing/corrupt footer.
