#!/usr/bin/env python3
import argparse
import os
import struct
import sys
import traceback

import pyarrow as pa
import pyarrow.parquet as pq


def read_magic_and_footer(path):
    """
    Low-level check for magic bytes and footer length sanity.
    Returns dict with information and potential warnings.
    """
    info = {
        "file_size": None,
        "header_magic": None,
        "footer_magic": None,
        "metadata_len": None,
        "footer_start": None,
        "magic_ok": False,
        "warnings": [],
        "errors": [],
    }
    try:
        size = os.path.getsize(path)
        info["file_size"] = size
        with open(path, "rb") as f:
            head = f.read(4)
            info["header_magic"] = head
            if size >= 8:
                f.seek(size - 8)
                meta_len_bytes = f.read(4)
                foot_magic = f.read(4)
            else:
                info["errors"].append("File too small to be Parquet (size < 8)")
                return info
        info["footer_magic"] = foot_magic
        if head != b"PAR1":
            info["warnings"].append(
                "Header magic bytes not found at start (expected b'PAR1')."
            )
        if foot_magic != b"PAR1":
            info["warnings"].append(
                "Footer magic bytes not found at end (expected b'PAR1')."
            )

        if len(meta_len_bytes) == 4:
            # little-endian 32-bit int
            meta_len = struct.unpack("<i", meta_len_bytes)[0]
            info["metadata_len"] = meta_len
            footer_start = size - 8 - meta_len
            info["footer_start"] = footer_start
            if footer_start < 4 or footer_start > size - 8:
                info["warnings"].append(
                    f"Footer start {footer_start} is out of expected bounds for file size {size}."
                )
            if meta_len < 0:
                info["warnings"].append(
                    f"Metadata length negative ({meta_len}); likely corruption."
                )
        else:
            info["errors"].append("Could not read metadata length.")
        info["magic_ok"] = (head == b"PAR1") and (foot_magic == b"PAR1")
    except Exception as e:
        info["errors"].append(f"Exception checking magic/footer: {e}")
    return info


def schema_without_dictionaries(schema: pa.Schema) -> pa.Schema:
    fields = []
    for f in schema:
        t = f.type
        if pa.types.is_dictionary(t):
            fields.append(pa.field(f.name, t.value_type, f.nullable, f.metadata))
        else:
            fields.append(f)
    return pa.schema(fields, schema.metadata)


def cast_batch_to_schema_full(
    batch: pa.RecordBatch, target_schema: pa.Schema
) -> pa.RecordBatch:
    # Ensure the batch has all columns from target_schema (order and type).
    arrays = []
    by_name = {batch.schema.names[i]: batch.column(i) for i in range(batch.num_columns)}
    for f in target_schema:
        col = by_name.get(f.name, None)
        if col is None:
            # Fill with typed nulls
            arrays.append(pa.nulls(batch.num_rows, type=f.type))
        else:
            if col.type != f.type:
                try:
                    col = col.cast(f.type)
                except Exception:
                    # As a last resort, try to convert dictionary arrays to their value type explicitly
                    if pa.types.is_dictionary(col.type):
                        col = col.cast(col.type.value_type)
                        if col.type != f.type:
                            col = col.cast(f.type)
                    else:
                        raise
            arrays.append(col)
    return pa.record_batch(arrays, schema=target_schema)


def describe_parquet(pf: pq.ParquetFile):
    print("=== File-level metadata ===")
    try:
        md = pf.metadata
        print(f"Created by: {md.created_by}")
        print(f"Format version: {md.format_version}")
        print(f"Num row groups: {md.num_row_groups}")
        print(f"Num rows: {md.num_rows}")
        print(f"Num columns: {md.num_columns}")
        if md.schema is not None:
            print("Original (physical) schema (Thrift):")
            for i in range(md.num_columns):
                try:
                    col = md.schema.column(i)
                    print(f"  - {col.name} (path={col.path_in_schema})")
                except Exception:
                    pass
    except Exception as e:
        print(f"Could not read high-level metadata: {e}")

    print("\n=== Arrow schema (logical) ===")
    try:
        print(pf.schema_arrow)
    except Exception as e:
        print(f"Could not read Arrow schema: {e}")

    print("\n=== Row group and column chunk overview ===")
    try:
        md = pf.metadata
        for rg_idx in range(md.num_row_groups):
            rg = md.row_group(rg_idx)
            print(
                f"Row group {rg_idx}: num_rows={rg.num_rows}, total_byte_size={rg.total_byte_size}"
            )
            for col_idx in range(rg.num_columns):
                cc = rg.column(col_idx)
                try:
                    stats = cc.statistics
                except Exception:
                    stats = None
                try:
                    encs = list(cc.encodings) if cc.encodings is not None else None
                except Exception:
                    encs = None
                print(
                    f"  - Column {col_idx}: path={cc.path_in_schema}, type={cc.physical_type}, "
                    f"compressed_size={cc.total_compressed_size}, encodings={encs}, "
                    f"compression={cc.compression}"
                )
                if stats:
                    try:
                        print(
                            f"      stats: has_nulls={stats.has_null_count}, nulls={stats.null_count}, "
                            f"has_min_max={stats.has_min_max}"
                        )
                    except Exception:
                        pass
    except Exception as e:
        print(f"Could not iterate row groups: {e}")


def try_iter_batches(pf: pq.ParquetFile, rg_index: int, columns=None, batch_size=65536):
    """
    Try reading a row group as batches. Returns (ok, batches, rows_read, error).
    If an error occurs mid-stream, the returned batches contain what was read before failure.
    """
    batches = []
    rows = 0
    try:
        for b in pf.iter_batches(
            batch_size=batch_size,
            row_groups=[rg_index],
            columns=columns,
            use_threads=False,
        ):
            batches.append(b)
            rows += b.num_rows
        return True, batches, rows, None
    except Exception as e:
        return False, batches, rows, e


def detect_good_columns_for_rg(
    pf: pq.ParquetFile, rg_index: int, col_names, batch_size=65536
):
    """
    For a row group that failed to read across all columns, find which columns can be read
    end-to-end. Returns (good_columns, bad_columns_info)
    bad_columns_info is a list of tuples (name, error_str, rows_read_before_failure)
    """
    good = []
    bad = []
    for name in col_names:
        ok, batches, rows, err = try_iter_batches(
            pf, rg_index, columns=[name], batch_size=batch_size
        )
        if ok:
            good.append(name)
        else:
            bad.append((name, str(err), rows))
    return good, bad


def salvage_file(
    input_path,
    output_path,
    batch_size=65536,
    drop_bad_columns=True,
    verbose=False,
):
    """
    Salvage strategy:
      - Attempt to read every row group fully using all columns.
      - If a RG fails, identify per-column good/bad columns by reading each column independently.
      - Re-read the RG with only good columns and write them out.
      - Output file schema is the original Arrow schema but with dictionary types decoded to their value type,
        to simplify filling missing columns with typed nulls and avoid dictionary-casting issues.
      - Missing (bad) columns for a RG are filled with nulls of the target type.
    """
    print(f"Opening {input_path}")
    pf = pq.ParquetFile(input_path, memory_map=False)

    original_schema = pf.schema_arrow
    target_schema = schema_without_dictionaries(original_schema)

    # Collect column names once
    all_columns = list(original_schema.names)

    # We'll write in streaming mode
    writer = None
    try:
        writer = pq.ParquetWriter(output_path, target_schema)
        total_rows_written = 0
        total_rows_available = pf.metadata.num_rows if pf.metadata is not None else None
        num_row_groups = pf.metadata.num_row_groups if pf.metadata is not None else None

        print(f"Salvaging to {output_path} ...")
        print(f"Target schema (dictionaries decoded):\n{target_schema}")

        summary = {
            "row_groups_total": num_row_groups,
            "row_groups_fully_ok": 0,
            "row_groups_salvaged_partial": 0,
            "row_groups_skipped": 0,
            "bad_columns_per_rg": {},
        }

        for rg_idx in range(num_row_groups):
            if verbose:
                print(f"Reading row group {rg_idx} with all columns ...")
            ok, batches, rows, err = try_iter_batches(
                pf, rg_idx, columns=all_columns, batch_size=batch_size
            )
            if ok:
                if verbose:
                    print(f"Row group {rg_idx} OK. Rows: {rows}")
                for b in batches:
                    casted = cast_batch_to_schema_full(b, target_schema)
                    writer.write_table(pa.Table.from_batches([casted]))
                    total_rows_written += casted.num_rows
                summary["row_groups_fully_ok"] += 1
                continue

            # If failed:
            print(f"Row group {rg_idx} read failed: {err}")
            if not drop_bad_columns:
                print(f"Skipping row group {rg_idx} (drop_bad_columns=False).")
                summary["row_groups_skipped"] += 1
                continue

            # Identify good vs bad columns
            good_cols, bad_cols = detect_good_columns_for_rg(
                pf, rg_idx, all_columns, batch_size=batch_size
            )
            summary["bad_columns_per_rg"][rg_idx] = bad_cols
            if verbose:
                print(f"Row group {rg_idx}: good columns: {good_cols}")
                print(f"Row group {rg_idx}: bad columns: {bad_cols}")

            if not good_cols:
                print(f"No readable columns in row group {rg_idx}. Skipping.")
                summary["row_groups_skipped"] += 1
                continue

            # Re-read RG with only good columns
            ok2, batches2, rows2, err2 = try_iter_batches(
                pf, rg_idx, columns=good_cols, batch_size=batch_size
            )
            if not ok2 and rows2 == 0:
                print(
                    f"Row group {rg_idx} could not be salvaged with good columns: {err2}"
                )
                summary["row_groups_skipped"] += 1
                continue

            if not ok2 and rows2 > 0:
                print(
                    f"Row group {rg_idx} salvaged partially with good columns. Rows read: {rows2}. Error: {err2}"
                )
            else:
                if verbose:
                    print(
                        f"Row group {rg_idx} salvaged with good columns. Rows: {rows2}"
                    )

            # For each batch, fill missing columns with typed nulls and cast dictionaries out
            for b in batches2:
                casted = cast_batch_to_schema_full(b, target_schema)
                writer.write_table(pa.Table.from_batches([casted]))
                total_rows_written += casted.num_rows
            summary["row_groups_salvaged_partial"] += 1

        print("Finalize writer ...")
    finally:
        if writer is not None:
            writer.close()

    print("\n=== Salvage summary ===")
    print(
        f"Row groups total: {pf.metadata.num_row_groups if pf.metadata else 'unknown'}"
    )
    print(f"Row groups OK: {summary['row_groups_fully_ok']}")
    print(
        f"Row groups salvaged (partial or column-dropped): {summary['row_groups_salvaged_partial']}"
    )
    print(f"Row groups skipped: {summary['row_groups_skipped']}")
    print(f"Rows written: {total_rows_written} (source rows: {total_rows_available})")
    if summary["bad_columns_per_rg"]:
        print("Bad columns per row group (name, error, rows_read_before_failure):")
        for rg_idx, items in summary["bad_columns_per_rg"].items():
            print(f"  RG {rg_idx}: {items}")


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose and salvage Parquet files with potential corruption."
    )
    parser.add_argument("path", help="Path to the Parquet file")
    parser.add_argument(
        "--batch-size", type=int, default=65536, help="Batch size for iter_batches"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--salvage-output", default=None, help="Output path for salvaged Parquet"
    )
    parser.add_argument(
        "--no-drop-bad-columns",
        dest="drop_bad_columns",
        action="store_false",
        help="If set, skip entire row group on error instead of dropping bad columns",
    )
    args = parser.parse_args()

    if args.salvage_output and os.path.abspath(args.path) == os.path.abspath(args.salvage_output):
        parser.error("salvage-output must differ from the source")
    if args.salvage_output and os.path.exists(args.salvage_output):
        parser.error("salvage-output already exists")
    path = args.path
    if not os.path.exists(path):
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(2)

    print("=== Low-level footer/magic check ===")
    info = read_magic_and_footer(path)
    for k, v in info.items():
        if k in ("warnings", "errors"):
            continue
        print(f"{k}: {v}")
    if info["warnings"]:
        print("Warnings:")
        for w in info["warnings"]:
            print(f"  - {w}")
    if info["errors"]:
        print("Errors:")
        for e in info["errors"]:
            print(f"  - {e}")

    print("\n=== High-level metadata and sanity checks ===")
    # Try to open as Parquet
    try:
        pf = pq.ParquetFile(path, memory_map=False)
    except Exception as e:
        print(f"Failed to open ParquetFile: {e}")
        print(
            "If the footer is corrupted or truncated, salvage is difficult. Consider these options:"
        )
        print(
            "- Try Java parquet-tools 'meta'/'dump' to see if any metadata is readable."
        )
        print(
            "- If the file is truncated from the end, sometimes a previous valid footer exists earlier; "
            "you can search for the trailing 'PAR1' magic and try truncating to that position, "
            "then reopen. This requires careful manual work."
        )
        sys.exit(1)

    try:
        describe_parquet(pf)
    except Exception:
        print("Metadata inspection threw an exception:")
        traceback.print_exc()

    # Probe all row groups quickly to see if any fail
    print("\n=== Quick probe of row group readability ===")
    md = pf.metadata
    failed_rgs = []
    for rg_idx in range(md.num_row_groups):
        ok, batches, rows, err = try_iter_batches(
            pf, rg_idx, columns=None, batch_size=args.batch_size
        )
        if ok:
            print(f"Row group {rg_idx}: OK (rows={rows})")
        else:
            print(f"Row group {rg_idx}: FAILED after rows_read={rows}; error: {err}")
            failed_rgs.append((rg_idx, str(err), rows))

    if args.salvage_output:
        print("\n=== Salvage attempt ===")
        try:
            salvage_file(
                input_path=path,
                output_path=args.salvage_output,
                batch_size=args.batch_size,
                drop_bad_columns=args.drop_bad_columns,
                verbose=args.verbose,
            )
            print(f"Salvage complete. Output: {args.salvage_output}")
        except Exception:
            print("Salvage attempt failed with exception:")
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
