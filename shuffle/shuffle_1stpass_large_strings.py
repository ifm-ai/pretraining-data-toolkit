# Distributed two-pass shuffle for Parquet datasets (pathlib-only).
# Stage 1 (this file):
# - Run on all ranks.
# - Each rank reads its shard of files and writes to M piles, but with per-rank pile files:
#     {base}/_piles/pile_{pile_id:05d}_rank_{rank:05d}.parquet
# - Piles are compressed to save space.
#
# IMPORTANT: We normalize all UTF-8 text columns to large_string so that
# any mixture of `string` and `large_string` source files becomes schema-compatible.
# Per-file batches are cast to the canonical (large_string) schema before writing.
#
# Stage 2 (separate script, after all pass-1 ranks finish):
# - For each pile, read all rank parts, shuffle in-memory, and write one final shard:
#     {base}/_jsonl/part-{rank:05d}-{seq:06d}.jsonl
#
# Requirements: pip install pyarrow numpy
from pathlib import Path
from typing import List, Optional
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.types as pat
import argparse
import json
import sys
import traceback


from pretraining_data.runtime import get_rank_world, separate_output


def get_shard(
    input_dir: Path, rank: int, world_size: int, glob_pattern: str = "**/*.parquet"
) -> List[Path]:
    if not input_dir.is_dir():
        raise ValueError(
            f"Input directory {input_dir} does not exist or is not a directory."
        )
    all_files = sorted(p for p in input_dir.glob(glob_pattern) if p.is_file())
    return [f for i, f in enumerate(all_files) if i % world_size == rank]


def prepare_dirs(
    base: Path, make_piles: bool = True, make_shuffled: bool = False
) -> tuple[Optional[Path], Optional[Path]]:
    piles_dir = base / "_piles" if make_piles else None
    shuffled_dir = base / "_shuffled" if make_shuffled else None
    if piles_dir:
        piles_dir.mkdir(parents=True, exist_ok=True)
    if shuffled_dir:
        shuffled_dir.mkdir(parents=True, exist_ok=True)
    return piles_dir, shuffled_dir


def estimate_batch_size_bytes(rb: pa.RecordBatch) -> int:
    total = 0
    for i in range(rb.num_columns):
        col = rb.column(i)
        try:
            total += col.nbytes
        except Exception:
            pass
    return int(total)


def _make_parquet_writer(path: Path, schema: pa.Schema, compression):
    # Compatible with multiple pyarrow versions
    try:
        return pq.ParquetWriter(
            str(path),
            schema=schema,
            compression=compression,
            use_dictionary=False,
            write_statistics=False,
        )
    except TypeError:
        try:
            return pq.ParquetWriter(
                str(path),
                schema=schema,
                compression=compression,
            )
        except TypeError:
            # Fallback without compression kw (very old versions)
            return pq.ParquetWriter(str(path), schema=schema)


def normalize_schema_to_large_string(schema: pa.Schema) -> pa.Schema:
    """
    Return a new schema where all UTF-8 text fields use large_string().
    This makes string vs large_string differences go away and provides
    a canonical schema for all ranks/files.
    """
    fields = []
    for f in schema:
        t = f.type
        if pat.is_string(t):  # 32-bit offsets → upcast to large_string
            t = pa.large_string()
        # If already large_string, leave as-is
        fields.append(pa.field(f.name, t, f.nullable))
    # Drop schema metadata; we don't need it for the shuffle.
    return pa.schema(fields)


def first_pass_distributed(
    base: Path,
    input_dir: Path,
    M: int,
    seed: int = 42,
    batch_size: int = 65536,
    buffer_mb_per_pile: int = 128,
    pile_compression: str | None = "zstd",
) -> None:
    """
    Run on every rank. Processes files one-by-one (only one file open at a time).
    Writes compressed per-rank pile files:
      {base}/_piles/pile_{pile_id:05d}_rank_{rank:05d}.parquet

    - We choose a reference schema from the first readable file in this rank's shard,
      normalize all text columns to large_string, and drop metadata.
    - Every other file's schema is normalized the same way and must match structurally
      (names/types/nullability), otherwise the pass fails.
    - Each batch is cast to the canonical large_string schema before shuffling/writing.
    - Any file-level read error aborts the pass with a nonzero exit status.
    """
    if M <= 0 or batch_size <= 0 or buffer_mb_per_pile <= 0:
        raise ValueError("M, batch size and buffer size must be positive")
    separate_output(input_dir, base)
    rank, world_size = get_rank_world()
    piles_dir, _ = prepare_dirs(base, make_piles=True, make_shuffled=False)

    if (piles_dir / f"rank_{rank:05d}.done.json").exists():
        raise FileExistsError("This rank is already complete; use a fresh base directory")
    shard_files = get_shard(input_dir, rank=rank, world_size=world_size)
    if not shard_files:
        print(f"[rank {rank}] No files to process in shard.", file=sys.stderr)
        all_inputs = list(input_dir.rglob("*.parquet"))
        if not all_inputs:
            raise ValueError("Input contains no Parquet files")
        marker = piles_dir / f"rank_{rank:05d}.done.json"
        tmp_marker = marker.with_suffix(".tmp")
        tmp_marker.write_text(json.dumps({"rank": rank, "world_size": world_size, "num_piles": M, "empty": True}))
        tmp_marker.replace(marker)
        return

    # Discover a reference schema from the first readable file (and normalize it)
    schema: Optional[pa.Schema] = None
    schema_file: Optional[Path] = None
    for f in shard_files:
        try:
            pf = pq.ParquetFile(str(f))
            raw_schema = pf.schema_arrow
            schema = normalize_schema_to_large_string(raw_schema)
            schema_file = f
            print(
                f"[rank {rank}] Using reference schema from {schema_file}",
                file=sys.stderr,
            )
            break
        except Exception as e:
            print(f"[rank {rank}] ERROR: failed to open {f}: {e}", file=sys.stderr)
            traceback.print_exc()
            continue
    if schema is None:
        raise RuntimeError("No readable Parquet files in shard")

    # Open one writer per pile (compressed to save space), using the canonical schema.
    writer_paths = [
        piles_dir / f"pile_{j:05d}_rank_{rank:05d}.parquet" for j in range(M)
    ]
    if any(p.exists() for p in writer_paths):
        raise FileExistsError("Pile files already exist for this rank; use a fresh base directory")
    writers = []
    try:
        for path in writer_paths:
            writers.append(_make_parquet_writer(path, schema, pile_compression))
    except Exception:
        for writer in writers:
            writer.close()
        raise

    # Per-pile buffered batches and approximate size counters
    pile_batches: list[list[pa.RecordBatch]] = [[] for _ in range(M)]
    pile_est_bytes: list[int] = [0] * M
    flush_target = int(buffer_mb_per_pile) * 1024 * 1024

    # Deterministic RNG per-rank
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(rank)]))

    try:
        for f in shard_files:
            # Open this file; on error, fail the pass
            try:
                pf = pq.ParquetFile(str(f))
            except Exception as e:
                print(f"[rank {rank}] ERROR: cannot open {f}: {e}", file=sys.stderr)
                traceback.print_exc()
                raise

            # Schema check (structural, normalized to large_string)
            try:
                raw_file_schema = pf.schema_arrow
                file_schema = normalize_schema_to_large_string(raw_file_schema)

                if not file_schema.equals(schema, check_metadata=False):
                    print(
                        f"[rank {rank}] ERROR: schema mismatch in {f}.",
                        file=sys.stderr,
                    )
                    raise ValueError(f"Schema mismatch: {f}")
            except Exception as e:
                print(
                    f"[rank {rank}] ERROR: cannot read/normalize schema from {f}: {e}",
                    file=sys.stderr,
                )
                traceback.print_exc()
                raise

            # Iterate batches from this file only
            try:
                for batch in pf.iter_batches(batch_size=batch_size):
                    n = batch.num_rows
                    if n == 0:
                        continue

                    # Convert to Table and cast to canonical schema if needed
                    tbl = pa.Table.from_batches([batch])
                    if not tbl.schema.equals(schema, check_metadata=False):
                        try:
                            tbl = tbl.cast(schema)
                        except Exception as e:
                            print(
                                f"[rank {rank}] ERROR: cast to canonical schema failed for {f}: {e}",
                                file=sys.stderr,
                            )
                            traceback.print_exc()
                            raise

                    # Random pile assignment and stable sort once
                    pile_ids = rng.integers(0, M, size=n, endpoint=False)
                    order = np.argsort(pile_ids, kind="stable")
                    pile_sorted = pile_ids[order]
                    boundaries = np.concatenate(
                        ([0], np.flatnonzero(np.diff(pile_sorted)) + 1, [n])
                    )

                    sorted_tbl = tbl.take(pa.array(order))

                    # Estimate bytes-per-row from original batch for flush thresholding
                    b_est = max(estimate_batch_size_bytes(batch), 1)
                    bytes_per_row = max(1, b_est // max(1, n))

                    for s, e in zip(boundaries[:-1], boundaries[1:]):
                        if e <= s:
                            continue
                        pid = int(pile_sorted[s])
                        sub_tbl = sorted_tbl.slice(s, e - s)
                        sub_rb = sub_tbl.to_batches()[0]

                        pile_batches[pid].append(sub_rb)
                        pile_est_bytes[pid] += sub_rb.num_rows * bytes_per_row

                    # Flush any pile exceeding target
                    for pid in range(M):
                        if pile_est_bytes[pid] >= flush_target and pile_batches[pid]:
                            writers[pid].write_table(
                                pa.Table.from_batches(pile_batches[pid])
                            )
                            pile_batches[pid].clear()
                            pile_est_bytes[pid] = 0

                    # Free some temps
                    del (
                        tbl,
                        sorted_tbl,
                        sub_tbl,
                        sub_rb,
                        pile_ids,
                        order,
                        pile_sorted,
                        boundaries,
                    )

            except Exception as e:
                print(
                    f"[rank {rank}] ERROR: reading batches from {f}: {e}",
                    file=sys.stderr,
                )
                traceback.print_exc()
                raise

        # Final flush of all piles
        for pid in range(M):
            if pile_batches[pid]:
                writers[pid].write_table(pa.Table.from_batches(pile_batches[pid]))
                pile_batches[pid].clear()
                pile_est_bytes[pid] = 0

    finally:
        # Always close writers
        close_errors = []
        for w in writers:
            try:
                w.close()
            except Exception as exc:
                close_errors.append(exc)
        if close_errors:
            raise RuntimeError("Failed to close pile writers") from close_errors[0]
    marker = piles_dir / f"rank_{rank:05d}.done.json"
    tmp_marker = marker.with_suffix(".tmp")
    tmp_marker.write_text(json.dumps({"rank": rank, "world_size": world_size, "num_piles": M, "empty": False}))
    tmp_marker.replace(marker)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "First pass of distributed two-pass shuffle: "
            "scatter rows into compressed per-rank pile files."
        )
    )
    parser.add_argument(
        "base", type=Path, help="Base directory where _piles/ will be written"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input directory to recursively read Parquet files from",
    )
    parser.add_argument(
        "M",
        type=int,
        help="Number of piles (choose so a full pile fits in RAM for pass 2)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=65536,
        help="Record batch size (default: 65536)",
    )
    parser.add_argument(
        "--buffer-mb",
        type=int,
        default=32,
        help=(
            "Approx. uncompressed buffer size per pile before flushing as a row group "
            "(default: 512 MB)"
        ),
    )
    parser.add_argument(
        "--pile-compression",
        type=str,
        default="zstd",
        choices=["zstd", "snappy", "gzip", "brotli", "lz4", "none"],
        help="Compression for pile files (default: zstd)",
    )
    args = parser.parse_args()
    base_dir = args.base.resolve()
    input_dir = args.input.resolve()
    if not input_dir.is_dir():
        raise ValueError(
            f"Input directory {input_dir} does not exist or is not a directory."
        )
    if args.M <= 0:
        raise ValueError("M must be a positive integer.")
    pile_comp = (
        None if args.pile_compression.lower() == "none" else args.pile_compression
    )
    first_pass_distributed(
        base=base_dir,
        input_dir=input_dir,
        M=args.M,
        seed=args.seed,
        batch_size=args.batch_size,
        buffer_mb_per_pile=args.buffer_mb,
        pile_compression=pile_comp,
    )
