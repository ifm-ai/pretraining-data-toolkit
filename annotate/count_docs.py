import sys
import argparse
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count

try:
    import pyarrow.parquet as pq  # metadata-only row counts
except Exception as e:
    pq = None
    _IMPORT_ERR = e


def _num_rows_parquet(path: str) -> int:
    return int(pq.read_metadata(path).num_rows)


def _count_docs_in_dir(dpath: str) -> tuple[str, int]:
    d = Path(dpath)
    total = 0
    for f in d.rglob("*.parquet"):
        total += _num_rows_parquet(str(f))
    return d.name, total


def main():
    if pq is None:
        print(
            "pyarrow is required for metadata-only row counts. Install with: pip install pyarrow",
            file=sys.stderr,
        )
        print(f"Import error: {_IMPORT_ERR}", file=sys.stderr)
        sys.exit(2)

    parser = argparse.ArgumentParser(description="Count Parquet rows in immediate subdirectories")
    parser.add_argument("base", type=Path)
    args = parser.parse_args()
    base = args.base
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        print("No subdirectories found.", file=sys.stderr)
        sys.exit(1)

    with Pool(processes=min(cpu_count(), len(subdirs))) as pool:
        results = pool.map(_count_docs_in_dir, [str(d) for d in subdirs], chunksize=1)

    counts = {k: v for k, v in sorted(results)}
    total = int(sum(counts.values()))

    for name in sorted(counts):
        print(f"{name}\t{counts[name]}")
    print(f"TOTAL\t{total}")

    out_path = base / "document_counts.json"
    with open(out_path, "w") as f:
        json.dump({"per_subdir": counts, "TOTAL": total}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
