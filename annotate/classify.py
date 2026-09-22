import os
import argparse
import polars as pl
import json
from pathlib import Path
from typing import List
from itertools import islice

from pretraining_data.runtime import get_rank_world, separate_output


def get_shard(
    input_dir: Path,
    rank: int,
    world_size: int,
    glob_pattern: str = "**/*.parquet",
) -> List[Path]:
    if not input_dir.is_dir():
        raise ValueError(
            f"Input directory {input_dir} does not exist or is not a directory."
        )
    # Collect only actual .parquet files (case-insensitive), skip dirs/hidden artifacts
    all_files = sorted(
        p
        for p in input_dir.glob(glob_pattern)
        if p.is_file()
        and p.suffix.lower() == ".parquet"
        and not p.name.startswith((".", "_"))
    )
    shard_files = [f for i, f in enumerate(all_files) if i % world_size == rank]
    return shard_files


def load_parquet_files_pf(files: List[Path]) -> pl.LazyFrame:
    if not files:
        print("Not enough files to process.")
        raise SystemExit(1)
    print(f"Loading {len(files)} Parquet files")

    required = [
        "text",
        "subset",
        "FormatClassifier",
        "TopicClassifier",
        "fineweb_edu_classifier_score",
        "fasttext_eli5_score",
        "fasttext_preselect_score",
        "jais_token_count",
        "meta.token_count",
        "meta.lang",
        "meta.lang_score",
    ]

    expected = {
        "text": pl.Utf8,
        "subset": pl.Utf8,
        "FormatClassifier": pl.Int64,
        "TopicClassifier": pl.Int64,
        "fineweb_edu_classifier_score": pl.Float64,
        "fasttext_eli5_score": pl.Float64,
        "fasttext_preselect_score": pl.Float64,
        "jais_token_count": pl.Int64,
        "meta.token_count": pl.Int64,
        "meta.lang": pl.Utf8,
        "meta.lang_score": pl.Float64,
    }

    def expr_for_col(
        file_schema: dict[str, pl.DataType], col: str, dtype: pl.DataType
    ) -> pl.Expr:
        # Handle nested fields like "meta.token_count"
        if "." in col:
            if col in file_schema:
                e = pl.col(col)
            else:
                base, field = col.split(".", 1)
                if isinstance(file_schema.get(base), pl.Struct) and field in {f.name for f in file_schema[base].fields}:
                    # Extract from struct if present
                    e = pl.col(base).struct.field(field)
                else:
                    # Completely missing: synthesize typed NULLs
                    return pl.lit(None, dtype=dtype).alias(col)
            return e.cast(dtype).alias(col)
        else:
            if col in file_schema:
                return pl.col(col).cast(dtype).alias(col)
            else:
                return pl.lit(None, dtype=dtype).alias(col)

    lfs: List[pl.LazyFrame] = []
    for f in files:
        # Scan each file independently so we can stabilize per-file dtypes
        lf = pl.scan_parquet(
            f,
            cast_options=pl.ScanCastOptions(
                missing_struct_fields="insert",
                extra_struct_fields="ignore",
            ),
        )
        # Discover columns available in this file (flattened names if applicable)
        file_schema = pl.read_parquet_schema(f)

        # Project exactly the required columns, creating typed NULLs where missing
        exprs = [expr_for_col(file_schema, col, expected[col]) for col in required]

        # Special-case: ensure 'subset' is Utf8 in every file to avoid Null-vs-String mismatches
        # (expr_for_col already casts to Utf8 or creates Utf8-typed NULLs)

        lfs.append(lf.select(exprs))

    # Concatenate all per-file projections (schemas now align on required columns)
    ldf = pl.concat(lfs, how="vertical")

    # Normalize subset values (dtype already Utf8)
    ldf = ldf.with_columns(pl.col("subset").fill_null("unknown"))

    return ldf


# Define a batched function using itertools.islice
def batched(iterable, batch_size):
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def main():
    # --- Usage: Load data using the function (replace with your directory) ---
    parser = argparse.ArgumentParser(
        description="Run queries on Parquet files in a directory using Polars"
    )
    parser.add_argument(
        "input_base", type=Path, help="Root directory containing domain subdirectories"
    )
    parser.add_argument("output_base", type=Path, help="Output root directory")
    parser.add_argument("--breaks-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    RANK, WORLD_SIZE = get_rank_world()
    separate_output(args.input_base, args.output_base)
    if any(args.output_base.glob(f"*/part-{RANK:05d}-*")):
        raise FileExistsError("Output exists for this rank; use a fresh output directory")

    # USE STREAMING
    pl.Config.set_engine_affinity("streaming")
    pl.Config.set_streaming_chunk_size(500_000)

    shard_files = get_shard(args.input_base, RANK, WORLD_SIZE)

    if not shard_files:
        print("No files assigned to this rank")
        return

    file_batch_size = 10  # Adjust as needed

    # --- Load breaks from JSON ---
    score_configs = {
        "fineweb_edu_classifier_score": "breaks_fineweb_edu_classifier.json",
        "fasttext_eli5_score": "breaks_fasttext_eli5.json",
        "fasttext_preselect_score": "breaks_fasttext_preselect.json",
    }
    breaks_dict = {}
    for col, json_file in score_configs.items():
        with (args.breaks_dir / json_file).open() as f:
            breaks_dict[col] = json.load(f)
        values = breaks_dict[col]
        if len(values) != 19 or any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError(f"{json_file} must contain 19 strictly increasing breaks")

    for chunk_idx, files in enumerate(batched(shard_files, file_batch_size)):
        df = load_parquet_files_pf(files)

        # --- Add buckets in first with_columns ---
        df_with_buckets = df.with_columns(
            pl.col("fineweb_edu_classifier_score")
            .cut(
                breaks=breaks_dict["fineweb_edu_classifier_score"],
                labels=[str(i) for i in range(20)],
                left_closed=False,
            )
            .cast(pl.Utf8)  # turn categorical labels into strings
            .cast(pl.Int32)  # parse "0".."19" into integers
            .fill_null(0)  # choose how to handle nulls
            .alias("bucket_fineweb_edu_classifier"),
            pl.col("fasttext_eli5_score")
            .cut(
                breaks=breaks_dict["fasttext_eli5_score"],
                labels=[str(i) for i in range(20)],
                left_closed=False,
            )
            .cast(pl.Utf8)  # turn categorical labels into strings
            .cast(pl.Int32)  # parse "0".."19" into integers
            .fill_null(0)  # choose how to handle nulls
            .alias("bucket_fasttext_eli5"),
            pl.col("fasttext_preselect_score")
            .cut(
                breaks=breaks_dict["fasttext_preselect_score"],
                labels=[str(i) for i in range(20)],
                left_closed=False,
            )
            .cast(pl.Utf8)  # turn categorical labels into strings
            .cast(pl.Int32)  # parse "0".."19" into integers
            .fill_null(0)  # choose how to handle nulls
            .alias("bucket_fasttext_preselect"),
        )

        # --- Add final_bucket in separate with_columns ---
        df_with_final = df_with_buckets.with_columns(
            pl.max_horizontal(
                [
                    "bucket_fineweb_edu_classifier",
                    "bucket_fasttext_eli5",
                    "bucket_fasttext_preselect",
                ]
            ).alias("final_bucket")
        )

        # --- Add quality_category in another with_columns ---
        df_final = df_with_final.with_columns(
            pl.when(pl.col("final_bucket") == 19)
            .then(pl.lit("High"))
            .when(pl.col("final_bucket") == 18)
            .then(pl.lit("Medium-High"))
            .when((pl.col("final_bucket") >= 12) & (pl.col("final_bucket") <= 17))
            .then(pl.lit("Medium"))
            .when((pl.col("final_bucket") >= 7) & (pl.col("final_bucket") <= 11))
            .then(pl.lit("Medium-Low"))
            .when(pl.col("final_bucket") <= 6)
            .then(pl.lit("Low"))
            .otherwise(pl.lit("Unknown"))
            .alias("quality_category")
        )

        # --- Static unique categories from the when-then logic ---
        unique_categories = [
            "High",
            "Medium-High",
            "Medium",
            "Medium-Low",
            "Low",
            "Unknown",
        ]

        # --- Stream write each partition to its own Parquet file ---
        args.output_base.mkdir(parents=True, exist_ok=True)

        batch_size = 50_000  # Adjust for roughly equal file sizes

        for category in unique_categories:
            cat_dir = args.output_base / category
            cat_dir.mkdir(parents=True, exist_ok=True)

            filtered_df = df_final.filter(pl.col("quality_category") == category)
            category_data = filtered_df.collect()

            batch_num = 0
            for batch in category_data.iter_slices(n_rows=batch_size):
                if len(batch) == 0:
                    continue  # Skip empty batches
                file_path = os.path.join(
                    cat_dir, f"part-{RANK:05d}-{chunk_idx:04d}-{batch_num:04d}.parquet"
                )
                tmp = Path(file_path + ".tmp")
                batch.write_parquet(tmp, compression="zstd", statistics=True)
                tmp.replace(file_path)
                print(f"Wrote batch {batch_num} for '{category}' to {file_path}")
                batch_num += 1


if __name__ == "__main__":
    main()
