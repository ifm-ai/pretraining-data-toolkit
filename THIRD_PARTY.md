# Third-party sources

No third-party model weights or dependency wheels are embedded. The repository
contains the supplied project source and declares installable Python dependencies.
The archive had no license statement for the original source; a release owner must
choose one before publishing. Model licenses and access conditions remain upstream.

| Artifact | Original source | Role |
| --- | --- | --- |
| TopicClassifier | https://huggingface.co/WebOrganizer/TopicClassifier | Topic ID |
| FormatClassifier | https://huggingface.co/WebOrganizer/FormatClassifier | Format ID |
| FineWeb-Edu classifier | https://huggingface.co/HuggingFaceTB/fineweb-edu-classifier | Educational score |
| JAIS tokenizer | https://huggingface.co/core42/jais-13b | Token count only; original URL currently redirects |
| OH/ELI5 fastText | https://huggingface.co/mlfoundations/fasttext-oh-eli5 | `__label__hq` score |
| PreSelect fastText | https://huggingface.co/hkust-nlp/preselect-fasttext-classifier | `__label__1` score |

Public model cards/listings were checked during preparation for WebOrganizer,
OH/ELI5, PreSelect and JAIS. The FineWeb-Edu page could not be retrieved in this
environment; its supplied identifier is retained and remains to be download-tested.
The public listings expose both exact fastText filenames used by this project.

WebOrganizer's model card documents `trust_remote_code=True` and xFormers for its
optional efficient attention. The release uses the documented ordinary attention
path by default. Custom model code is downloaded from the selected Hub revision.
Runtime compatibility with the chosen Transformers/PyTorch versions must be tested
on the deployment machine. See the model cards for citations and license terms.
