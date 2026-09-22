# Second-stage in-memory shuffle over piles, then write JSONL shards with buffering.
# - Each rank owns pile IDs where (pile_id % world_size == rank).
# - For a pile j, read ALL rank parts: _piles/pile_{j:05d}_rank_*.parquet
# - Shuffle in memory, then write JSONL shards named: part-{rank}-{seq}.jsonl
#   with up to --lines-per-shard lines each (default 100k).
# - Keep only one pile in memory at a time; free between piles to stay lean.
#
# Requirements:
#   pip install pyarrow numpy orjson
#
# Usage:
#   python shuffle/shuffle_2ndpass.py --base BASE_DIR [--lines-per-shard 100000] [--seed 42]

import argparse
import json
import gc
import sys
from pathlib import Path
from typing import List

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq


from pretraining_data.runtime import get_rank_world, separate_output


def seed_for_pile(seed: int, pile_id: int) -> np.random.Generator:
    ss = np.random.SeedSequence([int(seed), int(pile_id)])
    return np.random.default_rng(ss)


def flush_jsonl_buffer(
    buf: List[bytes],
    out_dir: Path,
    rank: int,
    seq: int,
) -> int:
    if not buf:
        return seq
    out_dir.mkdir(parents=True, exist_ok=True)  # optional
    out_path = out_dir / f"part-{rank:05d}-{seq:06d}.jsonl"
    tmp_path = out_path.with_suffix(".jsonl.tmp")
    with tmp_path.open("wb") as f:
        f.writelines(buf)  # each entry already has a trailing '\n'
    tmp_path.replace(out_path)
    buf.clear()
    return seq + 1


def write_jsonl_buffered(
    table: pa.Table,
    out_dir: Path,
    rank: int,
    seq_start: int,
    lines_per_shard: int = 100_000,
) -> int:
    """
    Serialize table rows to JSONL shards with buffering using orjson.
    Returns next sequence number.
    """
    n = table.num_rows
    if n == 0:
        return seq_start

    buf: List[bytes] = []
    seq = seq_start

    # Iterate by batches to control peak Python object creation
    for batch in table.to_batches():
        # Convert columns to Python lists once
        pycols = batch.to_pydict()  # dict: name -> list of Python scalars
        names = list(pycols.keys())
        cols_lists = [pycols[name] for name in names]
        m = len(cols_lists[0]) if cols_lists else 0

        for r in range(m):
            row = {names[i]: cols_lists[i][r] for i in range(len(names))}
            buf.append(orjson.dumps(row, option=orjson.OPT_APPEND_NEWLINE))
            if len(buf) >= lines_per_shard:
                seq = flush_jsonl_buffer(buf, out_dir, rank, seq)

    # Flush any remainder
    seq = flush_jsonl_buffer(buf, out_dir, rank, seq)
    return seq


def process_pile_to_jsonl(
    pile_id: int,
    parts: List[Path],
    out_dir: Path,
    rank: int,
    seed: int,
    seq_start: int,
    lines_per_shard: int,
) -> int:
    # Read all parts for this pile
    try:
        tbl = pq.read_table([str(p) for p in parts])
    except Exception as e:
        print(f"[rank {rank}] ERROR reading pile {pile_id:05d}: {e}", file=sys.stderr)
        raise

    n = tbl.num_rows
    if n > 1:
        rng = seed_for_pile(seed, pile_id)
        try:
            perm = rng.permutation(n)
            tbl = tbl.take(pa.array(perm))
        except Exception as e:
            print(
                f"[rank {rank}] ERROR shuffling pile {pile_id:05d}: {e}",
                file=sys.stderr,
            )
            raise

    # Write JSONL shards with buffering (part-{rank}-{seq}.jsonl)
    try:
        next_seq = write_jsonl_buffered(
            tbl, out_dir, rank, seq_start, lines_per_shard=lines_per_shard
        )
    except Exception as e:
        print(
            f"[rank {rank}] ERROR writing JSONL for pile {pile_id:05d}: {e}",
            file=sys.stderr,
        )
        raise

    # Free memory
    del tbl
    gc.collect()
    return next_seq


def get_assigned_piles(
    num_piles: int, rank: int, world_size: int, strategy: str = "modulo"
) -> List[int]:
    if num_piles <= 0:
        return []
    if strategy == "modulo":
        # Round-robin: 0->rank, rank+world_size, ...
        return list(range(rank, num_piles, world_size))
    elif strategy in ("contiguous", "block", "range"):
        # Contiguous block per rank
        start = (num_piles * rank) // world_size
        end = (num_piles * (rank + 1)) // world_size
        return list(range(start, end))
    else:
        raise ValueError(f"Unknown assignment strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser(
        description="Second-stage: in-memory shuffle per pile and write buffered JSONL shards."
    )
    parser.add_argument(
        "--base", type=Path, required=True, help="Base directory containing _piles/"
    )
    parser.add_argument(
        "--lines-per-shard",
        type=int,
        default=100_000,
        help="Max lines per JSONL file (default: 100000)",
    )
    parser.add_argument(
        "--num-piles",
        type=int,
        default=4096,
        help="num piles",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    args = parser.parse_args()

    if args.lines_per_shard <= 0 or args.num_piles <= 0:
        parser.error("lines-per-shard and num-piles must be positive")
    rank, world_size = get_rank_world()
    piles_dir = args.base / "_piles"
    out_dir = args.base / "_jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob(f"part-{rank:05d}-*")):
        raise FileExistsError("Output already exists for this rank; use a fresh _jsonl directory")
    markers = [json.loads(p.read_text()) for p in sorted(piles_dir.glob("rank_*.done.json"))]
    if not markers:
        raise ValueError("No successful first-pass manifests; run pass 1 from this release")
    pass1_world = markers[0]["world_size"]
    if len(markers) != pass1_world or {m["rank"] for m in markers} != set(range(pass1_world)):
        raise ValueError("First pass is incomplete: not all rank manifests are present")
    if any(m["world_size"] != pass1_world or m["num_piles"] != args.num_piles for m in markers):
        raise ValueError("First-pass manifests disagree with the requested pile count/world size")
    expected_parts = {
        f"pile_{pid:05d}_rank_{m['rank']:05d}.parquet"
        for m in markers if not m["empty"] for pid in range(args.num_piles)
    }
    if {p.name for p in piles_dir.glob("pile_*_rank_*.parquet")} != expected_parts:
        raise ValueError("Missing or unexpected first-pass pile parts")
    actual_ids = {int(p.name.split("_")[1]) for p in piles_dir.glob("pile_*_rank_*.parquet")}
    if actual_ids != set(range(args.num_piles)):
        raise ValueError("Pile IDs do not match --num-piles; finish pass 1 and supply its M")

    if not piles_dir.is_dir():
        print(
            f"ERROR: {piles_dir} does not exist or is not a directory.", file=sys.stderr
        )
        sys.exit(1)

    # num_piles should be provided (e.g., CLI arg --num-piles)
    if args.num_piles <= 0:
        print(f"[rank {rank}] args.num_piles must be > 0", file=sys.stderr)
        return

    assigned_piles = get_assigned_piles(
        args.num_piles, rank, world_size, strategy="modulo"
    )  # or "contiguous"
    if not assigned_piles:
        print(
            f"[rank {rank}] No assigned piles (world_size={world_size}, num_piles={args.num_piles}).",
            file=sys.stderr,
        )
        return

    print(
        f"[rank {rank}] Processing {len(assigned_piles)} piles out of {args.num_piles} total."
    )
    # for pid in assigned_piles: process pile pid

    seq = 0  # per-rank JSONL shard sequence
    for pid in assigned_piles:
        part_files = sorted((piles_dir).glob(f"pile_{pid:05d}_rank_*.parquet"))
        if not part_files:
            continue
        seq = process_pile_to_jsonl(
            pile_id=pid,
            parts=part_files,
            out_dir=out_dir,
            rank=rank,
            seed=args.seed,
            seq_start=seq,
            lines_per_shard=args.lines_per_shard,
        )
        print(f"[rank {rank}] Finished pile {pid:05d}, next seq={seq}")

    print(f"[rank {rank}] All assigned piles complete. Total shards written: {seq}")


if __name__ == "__main__":
    main()
