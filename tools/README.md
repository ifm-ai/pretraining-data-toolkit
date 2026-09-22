# Dataset tools

`party.py` is the supplied `party.txt` script renamed to Python and exposed as the
`party` command on installation. It samples JSON, JSONL, gzip JSONL, Parquet,
trajectory JSON, and `.ds` token files. Directory sampling selects a file first;
it is not uniform sampling over all documents in a corpus.

```bash
party examples/input/sample.jsonl
party -i 0 examples/input/sample.jsonl text
party -I examples/input/sample.jsonl
party record.json 'messages[].content'
python -m tools.party --help
```

Quote selectors containing brackets so the shell cannot expand them. `-i` selects
a zero-based record; `-I` samples interactively; `-t` truncates printed content.
Parquet needs PyArrow. Rich/colorist add optional presentation. JSON inspection
works with Python's standard library alone. `.ds` decoding additionally needs
Transformers and access to the tokenizer named in the companion `NAME.ds.metadata`:
first line `tokenizer_repo|2` (or `|4`), second line token count. The binary data
uses little-endian unsigned 16-bit or 32-bit token IDs. These paths can load a
whole file into memory and are intended for inspection.

`download_models.py` fetches all six annotation model/tokenizer sources into the
Hugging Face cache and writes an immutable revision manifest:

```bash
python -m tools.download_models --lock-output models.lock.json
export PRETRAINING_MODEL_MANIFEST="$PWD/models.lock.json"
```

Only the JAIS tokenizer is fetched. Model code is included in classifier snapshots.
Default manifest revisions are `main`; use the generated lock to reproduce a run.
The archive itself does not contain the downloaded weights or third-party wheels.
