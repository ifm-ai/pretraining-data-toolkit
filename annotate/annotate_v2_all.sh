#!/usr/bin/env bash
# Each immediate subdirectory of INPUT_ROOT is one dataset.
set -euo pipefail
if (( $# < 2 )); then
  echo "Usage: $0 INPUT_ROOT OUTPUT_ROOT [annotation options...]" >&2
  exit 2
fi
input_root=$1
output_root=$2
shift 2
[[ -d "$input_root" ]] || { echo "Input root does not exist" >&2; exit 2; }
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
found=0
for source in "$input_root"/*/; do
  [[ -d "$source" ]] || continue
  found=1
  name=$(basename -- "$source")
  bash "$script_dir/annotate_v2.sh" --input-dir "$source" --output-dir "$output_root/$name" "$@"
done
(( found )) || { echo "No dataset directories found" >&2; exit 2; }
