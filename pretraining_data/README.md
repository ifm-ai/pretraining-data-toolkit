# Shared support

`runtime.py` provides validated local/SLURM ranks, input/output separation, and the
small timer previously obtained from DataTrove. `models.json` lists every external
annotation model; `models.py` loads that manifest or `PRETRAINING_MODEL_MANIFEST`.
The downloader writes exact Hub commit revisions for future runs. All of this
support code is included in both source and wheel distributions.
