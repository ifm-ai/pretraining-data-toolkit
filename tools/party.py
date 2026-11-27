#!/usr/bin/env python3
# =============================================================================
# Party Tool (simplified) - random sampling + optional nested selector querying
#
# Supported inputs:
#   - .json, .jsonl, .jsonl.gz, .parquet, .ds (requires matching .ds.metadata)
#   - a directory containing those files (picked at random)
#
# Core modes:
#   - Sample one record (default) or interactive sampling (-I)
#   - Optional record index: -i/--index N (file only; not compatible with -I)
#   - Optional selector to print only a subfield:
#         party FILE <selector>
#         party -i 0 FILE <selector>
#
# Selector syntax (JSONPath-ish, no leading '$'):
#   - dict keys:      info.title
#   - list indices:   messages.[0].parts.[0].text   (also messages[0] works)
#   - wildcard:       messages.[].parts.[].text     ([] is shorthand for [*])
#   - quoted keys:    info["weird.key"]
#
# Notes:
# - Selector output is printed using the same dotted-path walker as full records.
# - No substring filtering in this version.
# =============================================================================

import argparse
import glob
import gzip
import json
import os
import pprint
import random
import struct
import sys
from typing import Any, List, Optional, Tuple, Union
import textwrap


# ---------- Truncation ----------
TRUNCATE_TEXT_MAX_LINES = 12
TRUNCATE_TEXT_MAX_CHARS = 4000
TRUNCATE_ARRAY_MAX_ITEMS = 6

try:
    from colorist import green
except ImportError:

    def green(x):
        print(x)


try:
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

SUPPORTED_EXTS = (".json", ".traj", ".jsonl", ".jsonl.gz", ".parquet", ".ds")


# ---------- Verbose ----------
def vdump(verbose: bool, **items):
    if not verbose:
        return
    print("\n[party verbose dump]", file=sys.stderr)
    for k, v in items.items():
        print(f"{k} =", file=sys.stderr)
        pprint.pprint(v, stream=sys.stderr, width=120, sort_dicts=True)
    print("", file=sys.stderr)


# ---------- Truncation helpers ----------
def truncate_text(s: str):
    if not isinstance(s, str):
        return s

    lines = s.splitlines()
    if len(s) <= TRUNCATE_TEXT_MAX_CHARS and len(lines) <= TRUNCATE_TEXT_MAX_LINES:
        return s

    truncated = "\n".join(lines[:TRUNCATE_TEXT_MAX_LINES])
    if len(truncated) > TRUNCATE_TEXT_MAX_CHARS:
        truncated = truncated[:TRUNCATE_TEXT_MAX_CHARS]

    suffix = (
        f"\n\n… [truncated: {len(lines)} lines, {len(s)} chars; "
        f"showing up to {TRUNCATE_TEXT_MAX_LINES} lines / {TRUNCATE_TEXT_MAX_CHARS} chars]"
    )
    return truncated + suffix


def should_truncate_list(lst):
    return isinstance(lst, list) and len(lst) > TRUNCATE_ARRAY_MAX_ITEMS


# ---------- Printing (original dotted-path style) ----------
def print_texts(obj, path=None, use_markdown=False, truncate=False):
    if path is None:
        path = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            print_texts(v, path + [str(k)], use_markdown, truncate)

    elif isinstance(obj, list):
        if truncate and should_truncate_list(obj):
            head = obj[:TRUNCATE_ARRAY_MAX_ITEMS]
            for i, item in enumerate(head):
                print_texts(item, path + [f"[{i}]"], use_markdown, truncate)
            remaining = len(obj) - len(head)
            green(".".join(path + ["[...]"]))
            print(
                f"… [truncated array: {len(obj)} items; showing first {len(head)}, omitted {remaining}]"
            )
        else:
            for i, item in enumerate(obj):
                print_texts(item, path + [f"[{i}]"], use_markdown, truncate)

    else:
        green(".".join(path))
        if isinstance(obj, str) and truncate:
            obj = truncate_text(obj)

        if isinstance(obj, str) and use_markdown and RICH_AVAILABLE:
            try:
                console.print(Markdown(obj))
            except Exception:
                print(obj)
        else:
            print(obj)


def print_record(entry, filename, line_number=None, use_markdown=False, truncate=False):
    if isinstance(entry, dict):
        out = dict(entry)
        out["filename"] = filename
        if line_number is not None:
            out["line_number"] = line_number
    else:
        out = {"filename": filename, "data": entry}
        if line_number is not None:
            out["line_number"] = line_number
    print_texts(out, use_markdown=use_markdown, truncate=truncate)


def print_selected_matches(
    matches: List[Tuple[str, Any]],
    filename: str,
    line_number: Optional[int],
    use_markdown: bool,
    truncate: bool,
):
    # Print with the exact same dotted-path walker.
    out: dict[str, Any] = {"filename": filename}
    if line_number is not None:
        out["line_number"] = line_number
    for concrete_path, value in matches:
        out[concrete_path] = value
    print_texts(out, use_markdown=use_markdown, truncate=truncate)


# ---------- File selection ----------
def pick_file(path):
    if os.path.isdir(path):
        files = []
        for f in glob.glob(os.path.join(path, "**/*"), recursive=True):
            if f.endswith("_tmp.jsonl.gz"):
                continue
            if f.endswith(SUPPORTED_EXTS) and os.path.isfile(f):
                files.append(f)
        if not files:
            raise RuntimeError("No suitable files found in the directory.")
        return random.choice(files)

    if os.path.isfile(path):
        if path.endswith("_tmp.jsonl.gz"):
            raise RuntimeError(
                "This file is marked as a temp gzip file; it will be ignored."
            )
        if not path.endswith(SUPPORTED_EXTS):
            raise RuntimeError("Unsupported file extension.")
        return path

    raise RuntimeError("The provided path is neither a file nor a directory.")


# ---------- Readers ----------
def read_json_file(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl_file(p):
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=0):  # zero-based
            line = line.strip()
            if line:
                out.append((i, json.loads(line)))
    return out


def read_jsonl_gz_file(p):
    out = []
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, start=0):  # zero-based
            line = line.strip()
            if line:
                out.append((i, json.loads(line)))
    return out


def read_parquet_sample(p, max_rows_per_sample=2000):
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError(
            "Parquet support requires pyarrow. Install with: pip install -U pyarrow"
        ) from e

    pf = pq.ParquetFile(p)

    if pf.num_row_groups and pf.num_row_groups > 0:
        rg = random.randrange(pf.num_row_groups)

        start_row = 0
        md = pf.metadata
        if md is not None:
            for i in range(rg):
                start_row += md.row_group(i).num_rows

        table = pf.read_row_group(rg)
        try:
            table = table.combine_chunks()
        except Exception:
            pass

        rows = table.to_pylist()
        if max_rows_per_sample and len(rows) > max_rows_per_sample:
            rows = rows[:max_rows_per_sample]

        return [(start_row + i, row) for i, row in enumerate(rows)]

    table = pf.read()
    try:
        table = table.combine_chunks()
    except Exception:
        pass

    rows = table.to_pylist()
    if max_rows_per_sample and len(rows) > max_rows_per_sample:
        rows = rows[:max_rows_per_sample]
    return [(i, row) for i, row in enumerate(rows)]


# ---------- Direct index readers ----------
def read_jsonl_at_index(p, index: int):
    with open(p, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=0):
            if i == index:
                line = line.strip()
                if not line:
                    raise RuntimeError(f"Line {index} is empty.")
                return i, json.loads(line)
    raise RuntimeError(f"Index {index} out of range.")


def read_jsonl_gz_at_index(p, index: int):
    with gzip.open(p, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, start=0):
            if i == index:
                line = line.strip()
                if not line:
                    raise RuntimeError(f"Line {index} is empty.")
                return i, json.loads(line)
    raise RuntimeError(f"Index {index} out of range.")


def read_json_at_index(p, index: int):
    data = read_json_file(p)
    if not isinstance(data, list):
        raise RuntimeError("Index mode for .json requires top-level JSON to be a list.")
    if index < 0 or index >= len(data):
        raise RuntimeError(f"Index {index} out of range (0..{len(data) - 1}).")
    return index, data[index]


def read_parquet_at_index(p, index: int):
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError(
            "Parquet support requires pyarrow. Install with: pip install -U pyarrow"
        ) from e

    pf = pq.ParquetFile(p)
    md = pf.metadata

    if md is None:
        table = pf.read()
        try:
            table = table.combine_chunks()
        except Exception:
            pass
        if index < 0 or index >= table.num_rows:
            raise RuntimeError(f"Index {index} out of range (0..{table.num_rows - 1}).")
        return index, table.slice(index, 1).to_pylist()[0]

    if index < 0:
        raise RuntimeError("Index must be >= 0 (zero-based).")

    start = 0
    for rg in range(pf.num_row_groups):
        n = md.row_group(rg).num_rows
        if index < start + n:
            local = index - start
            table = pf.read_row_group(rg)
            try:
                table = table.combine_chunks()
            except Exception:
                pass
            return index, table.slice(local, 1).to_pylist()[0]
        start += n

    raise RuntimeError(f"Index {index} out of range (0..{start - 1}).")


# ---------- .ds support ----------
def try_import_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer
    except ImportError:
        return None


def get_ds_metadata(metadata_file):
    if not os.path.exists(metadata_file):
        return None, 2, None
    with open(metadata_file, "r") as f:
        lines = f.readlines()
    if len(lines) < 2:
        return None, 2, None
    tokenizer_info = lines[0].strip().split("|")
    tokenizer_name = tokenizer_info[0] if tokenizer_info else None
    token_size = int(tokenizer_info[1]) if len(tokenizer_info) > 1 else 2
    token_count = int(lines[1].strip()) if lines[1].strip().isdigit() else None
    return tokenizer_name, token_size, token_count


def read_all_ds_tokens(file_path, token_size):
    token_format = "I" if token_size == 4 else "H"
    with open(file_path, "rb") as f:
        token_bytes = f.read()
    n = len(token_bytes) // token_size
    return list(struct.unpack(f"<{n}{token_format}", token_bytes[: n * token_size]))


def split_on_eos(tokens, eos_token_id, n_docs):
    docs, cur = [], []
    for t in tokens:
        cur.append(t)
        if t == eos_token_id:
            docs.append(cur)
            cur = []
            if len(docs) >= n_docs:
                break
    if cur and len(docs) < n_docs:
        docs.append(cur)
    return docs


def read_ds_file_docs(file_path, n_docs=10):
    base_path = os.path.splitext(file_path)[0]
    metadata_file = f"{base_path}.ds.metadata"
    tokenizer_name, token_size, token_count = get_ds_metadata(metadata_file)

    if not tokenizer_name:
        raise RuntimeError(
            "No tokenizer info found in metadata; cannot split documents."
        )

    AutoTokenizer = try_import_tokenizer()
    if not AutoTokenizer:
        raise RuntimeError("transformers not installed; cannot decode .ds files.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None and hasattr(tokenizer, "eos_token"):
        eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
    if eos_token_id is None:
        raise RuntimeError("Could not determine EOS token ID from tokenizer.")

    tokens = read_all_ds_tokens(file_path, token_size)
    docs = split_on_eos(tokens, eos_token_id, n_docs)

    results = []
    for i, doc_tokens in enumerate(docs):
        results.append(
            {
                "filename": file_path,
                "tokenizer": tokenizer_name,
                "token_size": token_size,
                "token_count": token_count,
                "doc_index": i,
                "tokens": doc_tokens,
                "decoded": tokenizer.decode(doc_tokens),
            }
        )
    return results


def print_ds_docs(docs):
    for i, entry in enumerate(docs, start=1):
        print("=" * 40)
        print(f"Document {i} from file: {entry['filename']}")
        print(f"Tokenizer: {entry['tokenizer']}")
        print(f"Token size: {entry['token_size']} bytes")
        print(f"Token count (from metadata): {entry['token_count']}")
        print("Decoded text:")
        print(entry["decoded"])
        print("=" * 40)


def read_data_for_sampling(filename):
    if filename.endswith(".json") or filename.endswith(".traj"):
        return read_json_file(filename)
    if filename.endswith(".jsonl"):
        return read_jsonl_file(filename)
    if filename.endswith(".jsonl.gz"):
        return read_jsonl_gz_file(filename)
    if filename.endswith(".parquet"):
        return read_parquet_sample(filename)
    if filename.endswith(".ds"):
        docs = read_ds_file_docs(filename, n_docs=10)
        print_ds_docs(docs)
        return None
    raise RuntimeError("Unsupported file extension.")


# ---------- Selector engine (keys / indices / array wildcard) ----------
class _WILDCARD:
    pass


WILDCARD = _WILDCARD()
SelTok = Union[str, int, _WILDCARD]


def parse_selector(selector: str) -> List[SelTok]:
    """
    Supports:
      a.b
      a[0].b  or a.[0].b
      a[].b   or a[*].b   (array wildcard)
      a["x.y"]
    """
    s = selector.strip()
    if not s:
        raise ValueError("Empty selector.")

    toks: List[SelTok] = []
    i = 0
    buf = ""

    def flush_buf():
        nonlocal buf
        if buf:
            toks.append(buf)
            buf = ""

    while i < len(s):
        ch = s[i]

        if ch == ".":
            flush_buf()
            i += 1
            continue

        if ch == "[":
            flush_buf()
            i += 1
            if i >= len(s):
                raise ValueError(f"Unclosed '[' in selector: {selector}")

            # quoted key
            if s[i] in ("'", '"'):
                quote = s[i]
                i += 1
                start = i
                while i < len(s) and s[i] != quote:
                    i += 1
                if i >= len(s):
                    raise ValueError(f"Unclosed quote in selector: {selector}")
                key = s[start:i]
                i += 1
                if i >= len(s) or s[i] != "]":
                    raise ValueError(
                        f"Expected ']' after quoted key in selector: {selector}"
                    )
                i += 1
                toks.append(key)
                continue

            # unquoted: index, wildcard, or bare key
            start = i
            while i < len(s) and s[i] != "]":
                i += 1
            if i >= len(s):
                raise ValueError(f"Unclosed '[' in selector: {selector}")
            inside = s[start:i].strip()
            i += 1  # consume ]

            if inside == "" or inside == "*":
                toks.append(WILDCARD)
            elif inside.lstrip("-").isdigit():
                idx = int(inside)
                if idx < 0:
                    raise ValueError("Negative indices are not supported.")
                toks.append(idx)
            else:
                toks.append(inside)
            continue

        buf += ch
        i += 1

    flush_buf()
    return toks


def _render_path(parts: List[Union[str, int]]) -> str:
    s = ""
    for p in parts:
        seg = f"[{p}]" if isinstance(p, int) else str(p)
        s += f".{seg}" if s else seg
    return s


def find_matches(obj: Any, selector: str) -> List[Tuple[str, Any]]:
    toks = parse_selector(selector)
    states: List[Tuple[List[Union[str, int]], Any]] = [([], obj)]

    for t in toks:
        nxt: List[Tuple[List[Union[str, int]], Any]] = []

        if t is WILDCARD:
            for path, cur in states:
                if isinstance(cur, list):
                    for idx, item in enumerate(cur):
                        nxt.append((path + [idx], item))
            states = nxt
            continue

        if isinstance(t, int):
            for path, cur in states:
                if isinstance(cur, list) and 0 <= t < len(cur):
                    nxt.append((path + [t], cur[t]))
            states = nxt
            continue

        # string key
        for path, cur in states:
            if isinstance(cur, dict) and t in cur:
                nxt.append((path + [t], cur[t]))
        states = nxt

    return [(_render_path(path), val) for path, val in states]


# ---------- Sampling helpers ----------
def sample_one(data):
    """
    Return (line_number, entry) where line_number may be None.
    """
    if isinstance(data, list):
        if (
            data
            and isinstance(data[0], tuple)
            and len(data[0]) == 2
            and isinstance(data[0][0], int)
        ):
            return random.choice(data)
        return None, random.choice(data)
    return None, data


def read_and_get_at_index(filename, index):
    if filename.endswith(".ds"):
        raise RuntimeError("Index mode is not supported for .ds files.")
    if filename.endswith(".parquet"):
        return read_parquet_at_index(filename, index)
    if filename.endswith(".jsonl"):
        return read_jsonl_at_index(filename, index)
    if filename.endswith(".jsonl.gz"):
        return read_jsonl_gz_at_index(filename, index)
    if filename.endswith(".json") or filename.endswith(".traj"):
        return read_json_at_index(filename, index)
    raise RuntimeError("Unsupported file extension.")


def main():
    help_text = textwrap.dedent(
        """\
        Randomly sample and print dataset records, or query nested subfields.

        Supported inputs:
          - Files: .json, .jsonl, .jsonl.gz, .parquet, .ds
          - Directories: recursively searches for supported files and picks one at random
            (skips files ending with "_tmp.jsonl.gz")

        Selector syntax (JSONPath-ish; no leading '$'):
          - dict keys:        info.title
          - list indices:     messages.[0].parts.[0].text   (messages[0] also works)
          - array wildcard:   messages.[].parts.[].text     ([] is shorthand for [*])
          - quoted keys:      info["weird.key"]

        Notes:
          - All indices are zero-based.
          - If a selector expands to multiple matches (because of []), all matches are printed.
          - Queried output is printed using the same dotted-path walker as full records.

        Examples:
          # Sample one record from a file:
          party opencode.traj.json

          # Sample one record from a directory (picks a random supported file):
          party /path/to/datasets

          # Interactive sampling loop:
          party -I opencode.traj.jsonl

          # Print a specific nested field from a sampled record:
          party opencode.traj.json messages.[1].info.cost

          # Print all matching fields using wildcard []:
          party opencode.traj.json messages.[].parts.[].type
          party opencode.traj.json messages.[].parts.[].text

          # Jump to a specific record index (file only):
          party -i 0 opencode.traj.jsonl messages.[0].info.role
          party -i 123 opencode.traj.parquet info.title

          # Truncate big text/arrays:
          party -t opencode.traj.json messages.[0].parts.[0].text

          # Debug parsing/choices:
          party -v opencode.traj.json messages.[0].parts.[].text
        """
    )

    parser = argparse.ArgumentParser(
        description=help_text,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("path", help="input file or directory")

    parser.add_argument(
        "-i",
        "--index",
        type=int,
        default=None,
        help="optional zero-based record index (file only; not compatible with -I/--interactive)",
    )
    parser.add_argument(
        "-I",
        "--interactive",
        action="store_true",
        help="interactive sampling loop (also enables markdown rendering if rich is installed)",
    )
    parser.add_argument(
        "-t",
        "--truncate",
        action="store_true",
        help="truncate very large text fields and very large arrays in printed output",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="disable markdown rendering (plain text output)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="dump parsed args and derived options to stderr",
    )
    parser.add_argument(
        "selector",
        nargs="?",
        help="optional selector to print only a nested subfield (supports [] wildcard for arrays)",
    )
    args = parser.parse_args()

    vdump(args.verbose, argv=sys.argv, parsed_args=vars(args))

    if args.index is not None and args.interactive:
        print("Error: --index is not compatible with --interactive.")
        sys.exit(1)
    if args.index is not None and args.index < 0:
        print("Error: --index must be >= 0 (zero-based).")
        sys.exit(1)

    truncate = bool(args.truncate)
    use_markdown = bool(args.interactive) and (not args.no_markdown)

    # -------- Index mode --------
    if args.index is not None:
        if os.path.isdir(args.path):
            print(
                "Error: --index is only valid when the path is a file, not a directory."
            )
            sys.exit(1)
        if not os.path.isfile(args.path):
            print("Error: the provided path is not a file.")
            sys.exit(1)
        if args.path.endswith("_tmp.jsonl.gz"):
            print("Error: this file is marked as a temp gzip file; it will be ignored.")
            sys.exit(1)
        if not args.path.endswith(SUPPORTED_EXTS):
            print("Error: unsupported file extension.")
            sys.exit(1)

        try:
            line_number, entry = read_and_get_at_index(args.path, args.index)

            if not args.selector:
                print_record(
                    entry,
                    args.path,
                    line_number=line_number,
                    use_markdown=use_markdown,
                    truncate=truncate,
                )
                return

            matches = find_matches(entry, args.selector)
            if not matches:
                print(f"Error: selector not found: {args.selector}")
                sys.exit(1)
            print_selected_matches(
                matches,
                args.path,
                line_number,
                use_markdown=use_markdown,
                truncate=truncate,
            )
            return

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    # -------- Non-index: resolve file (directory => pick one) --------
    try:
        filename = pick_file(args.path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    vdump(
        args.verbose,
        resolved_filename=filename,
        picked_from_directory=os.path.isdir(args.path),
    )

    # Read data for sampling
    try:
        data = read_data_for_sampling(filename)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if data is None:
        return
    if not data:
        print("The data is empty.")
        sys.exit(1)

    # Interactive loop
    if args.interactive:
        try:
            while True:
                print("\n" + "=" * 80 + "\n")
                line_number, entry = sample_one(data)

                if not args.selector:
                    print_record(
                        entry,
                        filename,
                        line_number=line_number,
                        use_markdown=use_markdown,
                        truncate=truncate,
                    )
                else:
                    matches = find_matches(entry, args.selector)
                    if not matches:
                        print(f"Selector not found in this sample: {args.selector}")
                    else:
                        print_selected_matches(
                            matches,
                            filename,
                            line_number,
                            use_markdown=use_markdown,
                            truncate=truncate,
                        )

                print("\n" + "=" * 80)
                print("\nPress Enter for next sample, or 'q' + Enter to quit...")
                if input().strip().lower() == "q":
                    print("Goodbye!")
                    break
        except KeyboardInterrupt:
            print("\nGoodbye!")
        return

    # Non-interactive: one sample
    line_number, entry = sample_one(data)

    if not args.selector:
        print_record(
            entry,
            filename,
            line_number=line_number,
            use_markdown=False,
            truncate=truncate,
        )
        return

    matches = find_matches(entry, args.selector)
    if not matches:
        print(f"Error: selector not found in sampled record: {args.selector}")
        sys.exit(1)
    print_selected_matches(
        matches, filename, line_number, use_markdown=False, truncate=truncate
    )


if __name__ == "__main__":
    main()
