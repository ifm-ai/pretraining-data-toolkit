from typing import List
import fasttext
from huggingface_hub import hf_hub_download
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from dataclasses import dataclass, field
from .resumable_array import ResumableArrayTask
from .jsonl_rw import JsonlReader
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
import signal
import faulthandler
import sys
import logging
import re
import numpy as np


from pretraining_data.runtime import TimingStats, get_rank_world, separate_output

from pretraining_data.models import model_kwargs as source_kwargs

CLASSIFIER_PATTERN = r"/([^/]+)$"

# --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Timers ---
timers = {
    "read": TimingStats(unit="batch"),
    "jais_tokenize": TimingStats(unit="batch"),
    "tokenize": TimingStats(unit="batch"),
    "to_device": TimingStats(unit="batch"),
    "inference": TimingStats(unit="batch"),
    "postprocess": TimingStats(unit="batch"),
    "write": TimingStats(unit="batch"),
}


# --- FastText loading and scoring ---
def load_model(repo_id: str, filename: str):
    model_path = hf_hub_download(repo_id=repo_id, filename=filename, **source_kwargs(repo_id))
    return fasttext.load_model(model_path)


# ---------------- Dataset adapters ----------------


class DatasetAdapter:
    def sanitize_batch(self, df):  # mutate/return df
        return df

    def override_schema(self, schema: pa.Schema) -> pa.Schema:
        return schema


# HPLT: fixes meta.hplt_attr
class HPLTAdapter(DatasetAdapter):
    @staticmethod
    def _is_na(v):
        return v is None or (isinstance(v, float) and np.isnan(v))

    @staticmethod
    def _to_int_or_none(v):
        return int(v) if isinstance(v, (int, np.integer)) else None

    @staticmethod
    def _to_str_or_none(v):
        return v if isinstance(v, str) else None

    @classmethod
    def _sanitize_num_list(cls, x):
        if not isinstance(x, list):
            return None
        out = []
        for v in x:
            if cls._is_na(v):
                out.append(None)
            elif isinstance(v, (int, float, np.integer, np.floating)):
                out.append(float(v))
            else:
                out.append(None)
        return out

    @classmethod
    def _sanitize_seg_langs(cls, x):
        if not isinstance(x, list):
            return None
        out = []
        for v in x:
            if cls._is_na(v):
                out.append(None)
            elif isinstance(v, str):
                out.append(v)
            else:
                out.append(None)
        return out

    @classmethod
    def _sanitize_pii(cls, x):
        # Expect list -> list -> list -> int
        if not isinstance(x, list):
            return None
        L1 = []
        for a in x:
            if cls._is_na(a):
                L1.append(None)
                continue
            if not isinstance(a, list):
                return None
            L2 = []
            for b in a:
                if cls._is_na(b):
                    L2.append(None)
                    continue
                if not isinstance(b, list):
                    return None
                L3 = []
                for c in b:
                    if cls._is_na(c):
                        L3.append(None)
                    elif isinstance(c, (int, np.integer)):
                        L3.append(int(c))
                    else:
                        L3.append(None)
                L2.append(L3)
            L1.append(L2)
        return L1

    @classmethod
    def _sanitize_meta_hplt_attr(cls, val):
        if cls._is_na(val) or not isinstance(val, dict):
            return None
        return {
            "c": cls._to_str_or_none(val.get("c")),
            "collection": cls._to_str_or_none(val.get("collection")),
            "doc_scores": cls._sanitize_num_list(val.get("doc_scores")),
            "f": cls._to_str_or_none(val.get("f")),
            "filter": cls._to_str_or_none(val.get("filter")),
            "o": cls._to_int_or_none(val.get("o")),
            "pii": cls._sanitize_pii(val.get("pii")),
            "prob": cls._sanitize_num_list(val.get("prob")),
            "robotstxt": cls._to_str_or_none(val.get("robotstxt")),
            "rs": cls._to_int_or_none(val.get("rs")),
            "s": cls._to_int_or_none(val.get("s")),
            "seg_langs": cls._sanitize_seg_langs(val.get("seg_langs")),
            "ts": cls._to_str_or_none(val.get("ts")),
        }

    @staticmethod
    def _type_meta_hplt_attr() -> pa.DataType:
        return pa.struct(
            [
                pa.field("c", pa.string(), nullable=True),
                pa.field("collection", pa.string(), nullable=True),
                pa.field(
                    "doc_scores",
                    pa.list_(pa.field("element", pa.float64(), nullable=True)),
                    nullable=True,
                ),
                pa.field("f", pa.string(), nullable=True),
                pa.field("filter", pa.string(), nullable=True),
                pa.field("o", pa.int64(), nullable=True),
                pa.field(
                    "pii",
                    pa.list_(
                        pa.field(
                            "element",
                            pa.list_(
                                pa.field(
                                    "element",
                                    pa.list_(
                                        pa.field("element", pa.int64(), nullable=True)
                                    ),
                                    nullable=True,
                                )
                            ),
                            nullable=True,
                        )
                    ),
                    nullable=True,
                ),
                pa.field(
                    "prob",
                    pa.list_(pa.field("element", pa.float64(), nullable=True)),
                    nullable=True,
                ),
                pa.field("robotstxt", pa.string(), nullable=True),
                pa.field("rs", pa.int64(), nullable=True),
                pa.field("s", pa.int64(), nullable=True),
                pa.field(
                    "seg_langs",
                    pa.list_(pa.field("element", pa.string(), nullable=True)),
                    nullable=True,
                ),
                pa.field("ts", pa.string(), nullable=True),
            ]
        )

    def sanitize_batch(self, df):
        if "meta.hplt_attr" in df.columns:
            df["meta.hplt_attr"] = df["meta.hplt_attr"].apply(
                self._sanitize_meta_hplt_attr
            )
        return df

    def override_schema(self, schema: pa.Schema) -> pa.Schema:
        if "meta.hplt_attr" not in schema.names:
            return schema
        return pa.schema(
            [
                pa.field("meta.hplt_attr", self._type_meta_hplt_attr(), nullable=True)
                if f.name == "meta.hplt_attr"
                else f
                for f in schema
            ]
        )


# S2ORC: fixes meta.openaccessinfo
class S2ORCAdapter(DatasetAdapter):
    @staticmethod
    def _is_na(v):
        return v is None or (isinstance(v, float) and np.isnan(v))

    @classmethod
    def _to_str(cls, v):
        if cls._is_na(v):
            return None
        if isinstance(v, (str, int, float, np.integer, np.floating)):
            return str(v)
        return None

    @classmethod
    def _sanitize_openaccessinfo(cls, val):
        if cls._is_na(val) or not isinstance(val, dict):
            return None
        ext = val.get("externalids")
        ext_out = None
        if isinstance(ext, dict):
            ext_out = {
                "ACL": cls._to_str(ext.get("ACL")),
                "ArXiv": cls._to_str(ext.get("ArXiv")),
                "DOI": cls._to_str(ext.get("DOI")),
                "MAG": cls._to_str(ext.get("MAG")),
                "PubMedCentral": cls._to_str(ext.get("PubMedCentral")),
            }
        return {
            "externalids": ext_out,
            "license": cls._to_str(val.get("license")),
            "status": cls._to_str(val.get("status")),
            "url": cls._to_str(val.get("url")),
        }

    @staticmethod
    def _type_openaccessinfo() -> pa.DataType:
        return pa.struct(
            [
                pa.field(
                    "externalids",
                    pa.struct(
                        [
                            pa.field("ACL", pa.string(), nullable=True),
                            pa.field("ArXiv", pa.string(), nullable=True),
                            pa.field("DOI", pa.string(), nullable=True),
                            pa.field("MAG", pa.string(), nullable=True),
                            pa.field("PubMedCentral", pa.string(), nullable=True),
                        ]
                    ),
                    nullable=True,
                ),
                pa.field("license", pa.string(), nullable=True),
                pa.field("status", pa.string(), nullable=True),
                pa.field("url", pa.string(), nullable=True),
            ]
        )

    def sanitize_batch(self, df):
        if "meta.openaccessinfo" in df.columns:
            df["meta.openaccessinfo"] = df["meta.openaccessinfo"].apply(
                self._sanitize_openaccessinfo
            )
        return df

    def override_schema(self, schema: pa.Schema) -> pa.Schema:
        if "meta.openaccessinfo" not in schema.names:
            return schema
        return pa.schema(
            [
                pa.field(
                    "meta.openaccessinfo", self._type_openaccessinfo(), nullable=True
                )
                if f.name == "meta.openaccessinfo"
                else f
                for f in schema
            ]
        )


# ---------------- MegaMath ----------------
class MegaMathAdapter(DatasetAdapter):
    def __init__(self):
        self.frozen_columns = None  # set from batch 1 for this file

    @staticmethod
    def _is_na(v):
        return v is None or (isinstance(v, float) and np.isnan(v))

    @staticmethod
    def _to_int64(v):
        if isinstance(v, (int, np.integer)):
            return int(v)
        return None

    @staticmethod
    def _to_float64(v):
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        return None

    @staticmethod
    def _to_str(v):
        if v is None:
            return None
        if isinstance(v, (str, int, float, np.integer, np.floating)):
            return str(v)
        return None

    @classmethod
    def _sanitize_media(cls, x):
        if not isinstance(x, list):
            return None
        out = []
        for v in x:
            if cls._is_na(v):
                out.append(None)
            elif isinstance(v, (int, np.integer)):
                out.append(int(v))
            else:
                out.append(None)
        return out

    def sanitize_batch(self, df):
        # Type coercions to keep Arrow happy
        if "meta.media" in df.columns:
            df["meta.media"] = df["meta.media"].apply(self._sanitize_media)
        for c in ("meta.lang_score", "meta.math_score"):
            if c in df.columns:
                df[c] = df[c].apply(self._to_float64)
        for c in ("meta.token_count", "meta.minhash_cluster_id"):
            if c in df.columns:
                df[c] = df[c].apply(self._to_int64)
        if "meta.dup_signals" in df.columns:
            df["meta.dup_signals"] = df["meta.dup_signals"].apply(
                lambda v: None
                if not isinstance(v, dict)
                else {"dup_doc_count": self._to_int64(v.get("dup_doc_count"))}
            )
        for c in (
            "meta.lang",
            "meta.url",
            "meta.timestamp",
            "meta.cc-path",
            "meta.domain",
            "meta.file_path",
            "meta.data_label",
        ):
            if c in df.columns:
                df[c] = df[c].apply(self._to_str)

        # Freeze columns on first batch; add None for any missing frozen columns later
        if self.frozen_columns is None:
            self.frozen_columns = list(df.columns)
        else:
            for c in self.frozen_columns:
                if c not in df.columns:
                    df[c] = None
            # keep any extra columns; the writer’s fixed schema will ignore/drop them

        return df

    def override_schema(self, schema: pa.Schema) -> pa.Schema:
        # Explicit types for stability
        new_fields = []
        for f in schema:
            n = f.name
            if n == "meta.media":
                new_fields.append(
                    pa.field(
                        n,
                        pa.list_(pa.field("element", pa.int32(), nullable=True)),
                        nullable=True,
                    )
                )
            elif n == "meta.dup_signals":
                new_fields.append(
                    pa.field(
                        n,
                        pa.struct(
                            [pa.field("dup_doc_count", pa.int64(), nullable=True)]
                        ),
                        nullable=True,
                    )
                )
            elif n in ("meta.lang_score", "meta.math_score"):
                new_fields.append(pa.field(n, pa.float64(), nullable=True))
            elif n in (
                "meta.lang",
                "meta.url",
                "meta.timestamp",
                "meta.cc-path",
                "meta.domain",
                "meta.file_path",
                "meta.data_label",
            ):
                new_fields.append(pa.field(n, pa.string(), nullable=True))
            else:
                new_fields.append(f)
        return pa.schema(new_fields)


# ---------------- PhilPapers ----------------
class PhilPapersAdapter(DatasetAdapter):
    @staticmethod
    def _is_na(v):
        return v is None or (isinstance(v, float) and np.isnan(v))

    @staticmethod
    def _to_int64(v):
        if isinstance(v, (int, np.integer)):
            return int(v)
        return None

    @staticmethod
    def _to_str(v):
        if v is None:
            return None
        if isinstance(v, (str, int, float, np.integer, np.floating)):
            return str(v)
        return None

    @classmethod
    def _sanitize_media(cls, x):
        if not isinstance(x, list):
            return None
        out = []
        for v in x:
            if cls._is_na(v):
                out.append(None)
            elif isinstance(v, (int, np.integer)):
                out.append(int(v))
            else:
                out.append(None)
        return out

    def sanitize_batch(self, df):
        if "meta.media" in df.columns:
            df["meta.media"] = df["meta.media"].apply(self._sanitize_media)
        if "meta.dup_signals" in df.columns:
            df["meta.dup_signals"] = df["meta.dup_signals"].apply(
                lambda v: None
                if not isinstance(v, dict)
                else {"dup_doc_count": self._to_int64(v.get("dup_doc_count"))}
            )
        for c in (
            "meta.title",
            "meta.type",
            "meta.creator",
            "meta.subject",
            "meta.date",
            "meta.identifier",
            "meta.description",
            "meta.datestamp",
            "meta.file_path",
            "meta.data_label",
        ):
            if c in df.columns:
                df[c] = df[c].apply(self._to_str)
        return df

    def override_schema(self, schema: pa.Schema) -> pa.Schema:
        new_fields = []
        for f in schema:
            n = f.name
            if n == "meta.media":
                new_fields.append(
                    pa.field(
                        n,
                        pa.list_(pa.field("element", pa.int32(), nullable=True)),
                        nullable=True,
                    )
                )
            elif n == "meta.dup_signals":
                new_fields.append(
                    pa.field(
                        n,
                        pa.struct(
                            [pa.field("dup_doc_count", pa.int64(), nullable=True)]
                        ),
                        nullable=True,
                    )
                )
            elif n in (
                "meta.title",
                "meta.type",
                "meta.creator",
                "meta.subject",
                "meta.date",
                "meta.identifier",
                "meta.description",
                "meta.datestamp",
                "meta.file_path",
                "meta.data_label",
            ):
                new_fields.append(pa.field(n, pa.string(), nullable=True))
            else:
                new_fields.append(f)
        return pa.schema(new_fields)


# --- USPTO adapter ----------------
class USPTOAdapter(DatasetAdapter):
    """
    Sanitizes USPTO flattened records so Arrow gets stable scalar types.
    Goal: avoid lists/dicts leaking into string columns (fixes: "Expected bytes, got a 'list' object").
    """

    def __init__(self):
        # freeze columns per file to keep schema stable across batches
        self.frozen_columns = None

    @staticmethod
    def _is_na(v):
        import numpy as _np

        return v is None or (isinstance(v, float) and _np.isnan(v))

    @staticmethod
    def _to_int64(v):
        import numpy as _np

        if isinstance(v, (int, _np.integer)):
            return int(v)
        if isinstance(v, (float, _np.floating)):
            # sometimes you get 11.0 instead of 11
            try:
                iv = int(v)
                if iv == v:
                    return iv
            except Exception:
                return None
        # fall back to parsing strings like "11"
        if isinstance(v, str):
            try:
                return int(v.strip())
            except Exception:
                return None
        return None

    @staticmethod
    def _to_str(v):
        # normalize everything to a Python str (never bytes, never list/dict)
        # so Arrow infers/accepts pa.string()
        if v is None:
            return None
        # accept normal scalars
        if isinstance(v, str):
            return v
        if isinstance(v, (int, float, np.integer, np.floating)):
            return str(v)
        if isinstance(v, bytes):
            try:
                return v.decode("utf-8", errors="replace")
            except Exception:
                return v.decode("latin-1", errors="replace")
        # collapse simple lists/tuples into a readable string
        if isinstance(v, (list, tuple)):
            # join elements as strings; None becomes empty
            return " | ".join("" if e is None else str(e) for e in v)
        # fallback for dicts or anything else
        try:
            import json

            return json.dumps(v, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(v)

    def sanitize_batch(self, df):
        # Known numeric/int fields
        int_fields = {
            "meta.token_count",
            "meta.minhash_cluster_id",
            "meta.data_rank",
        }

        # Known string-ish fields we always want as strings
        str_fields = {
            "text",
            "subset",
            "meta.id",
            "meta.file_path",
            "meta.data_label",
            "meta.source_file",
        }

        # 1) Normalize meta.dup_signals to struct with dup_doc_count:int
        if "meta.dup_signals" in df.columns:
            df["meta.dup_signals"] = df["meta.dup_signals"].apply(
                lambda v: None
                if not isinstance(v, dict)
                else {"dup_doc_count": self._to_int64(v.get("dup_doc_count"))}
            )

        # 2) Int fields
        for c in int_fields:
            if c in df.columns:
                df[c] = df[c].apply(self._to_int64)

        # 3) Force strings for core text/meta identifiers
        for c in str_fields:
            if c in df.columns:
                df[c] = df[c].apply(self._to_str)

        # 4) For all other meta.* columns, force to string unless already handled
        for c in df.columns:
            if c.startswith("meta."):
                if c in int_fields or c == "meta.dup_signals":
                    continue
                df[c] = df[c].apply(self._to_str)

        # 5) Freeze columns based on first batch in the file, fill missing later
        if self.frozen_columns is None:
            self.frozen_columns = list(df.columns)
        else:
            for c in self.frozen_columns:
                if c not in df.columns:
                    df[c] = None
            # keep extra columns; writer schema will drop if not in first batch schema

        return df

    def override_schema(self, schema: pa.Schema) -> pa.Schema:
        # Force stable types for troublesome fields
        int_fields = {
            "meta.token_count",
            "meta.minhash_cluster_id",
            "meta.data_rank",
        }
        str_fields = {
            "text",
            "subset",
            "meta.id",
            "meta.file_path",
            "meta.data_label",
            "meta.source_file",
        }

        new_fields = []
        for f in schema:
            n = f.name
            if n == "meta.dup_signals":
                new_fields.append(
                    pa.field(
                        n,
                        pa.struct(
                            [pa.field("dup_doc_count", pa.int64(), nullable=True)]
                        ),
                        nullable=True,
                    )
                )
            elif n in int_fields:
                new_fields.append(pa.field(n, pa.int64(), nullable=True))
            elif n in str_fields or n.startswith("meta."):
                # All remaining meta.* should be strings (prevents Arrow from inferring binary)
                new_fields.append(pa.field(n, pa.string(), nullable=True))
            else:
                new_fields.append(f)

        return pa.schema(new_fields)


# --- pick_adapter update: add USPTO detection ---
def pick_adapter(path: Path) -> DatasetAdapter | None:
    p = str(path).lower()
    if "hplt" in p:
        return HPLTAdapter()
    if "s2orc" in p:
        return S2ORCAdapter()
    if "megamath" in p:
        return MegaMathAdapter()
    if "phil_papers" in p or "phil-papers" in p or "philpapers" in p:
        return PhilPapersAdapter()
    if "uspto" in p:
        return USPTOAdapter()
    return None


class FastTextScorer:
    def __init__(self, model, positive_label):
        self.model = model
        self.positive_label = positive_label

    def score_texts(self, texts: List[str]) -> List[float]:
        scores = []
        for text in texts:
            clean_text = text.replace("\n", " ").replace("\r", " ")
            labels, label_scores = self.model.predict(clean_text, k=-1)
            label_score_dict = {
                label: float(score) for label, score in zip(labels, label_scores)
            }
            scores.append(label_score_dict.get(self.positive_label, 0.0))
        return scores


# --- Data Classes and Utility Functions (unchanged except for model groups) ---
@dataclass(frozen=True)
class OutputSpec:
    key: str
    extension: str


@dataclass(frozen=True)
class ModelConfig:
    name: str
    outputs: List[OutputSpec]


@dataclass(frozen=True)
class ModelGroup:
    tokenizer_name: str  # For FastText, set to None
    prompt_template: str
    models: List[ModelConfig]
    max_length: int = 8192


@dataclass
class PipelineOptions:
    batch_size: int = 8
    text_field: str = "text"
    compression: str = "gzip"
    memory_efficient_attention: bool = False
    model_groups: List[ModelGroup] = field(
        default_factory=lambda: [
            ModelGroup(
                tokenizer_name="WebOrganizer/TopicClassifier",
                prompt_template="{url}\n\n{text}",
                models=[
                    ModelConfig(
                        name="WebOrganizer/TopicClassifier",
                        outputs=[
                            OutputSpec("domains_topics", "__choice.npy"),
                        ],
                    ),
                    ModelConfig(
                        name="WebOrganizer/FormatClassifier",
                        outputs=[
                            OutputSpec("domains_formats", "__choice.npy"),
                        ],
                    ),
                ],
                max_length=8192,
            ),
            ModelGroup(
                tokenizer_name="HuggingFaceTB/fineweb-edu-classifier",
                prompt_template="{text}",
                models=[
                    ModelConfig(
                        name="HuggingFaceTB/fineweb-edu-classifier",
                        outputs=[
                            OutputSpec("annotations", ".npy"),
                        ],
                    ),
                ],
                max_length=512,
            ),
            ModelGroup(
                tokenizer_name="core42/jais-13b",
                prompt_template="{text}",
                models=[
                    ModelConfig(
                        name="core42/jais-13b",
                        outputs=[
                            OutputSpec("tokens", ".npy"),
                        ],
                    ),
                ],
            ),
            # --- FastText model groups ---
            ModelGroup(
                tokenizer_name=None,
                prompt_template="{text}",
                models=[
                    ModelConfig(
                        name="fasttext_eli5",
                        outputs=[
                            OutputSpec("fasttext_eli5", ".float"),
                        ],
                    ),
                ],
            ),
            ModelGroup(
                tokenizer_name=None,
                prompt_template="{text}",
                models=[
                    ModelConfig(
                        name="fasttext_preselect",
                        outputs=[
                            OutputSpec("fasttext_preselect", ".float"),
                        ],
                    ),
                ],
            ),
        ]
    )


def make_output_path_parquet(
    input_dir: Path, input_path: Path, output_dir: Path
) -> Path:
    rel = input_path.relative_to(input_dir)
    rel_parent = rel.parent
    out_path = rel_parent / f"{rel.stem}.parquet"
    return output_dir / out_path


def text_adapter(data, text_key: str):
    if text_key == "messages":
        return data["messages"][0]["content"]
    else:
        return data[text_key]


# --- ModelWorker and TimingReader (unchanged) ---
class ModelWorker(threading.Thread):
    def __init__(self, name, model, device):
        super().__init__()
        self.name = name
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.model = model
        self.input_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.daemon = True
        self.start()

    def run(self):
        while True:
            job = self.input_queue.get()
            if job is None:
                break
            batch_idx, toks = job
            try:
                with torch.cuda.stream(self.stream), torch.inference_mode():
                    logits = self.model(**toks).logits.float().cpu().numpy()
                self.result_queue.put((batch_idx, logits))
            except Exception as exc:
                self.result_queue.put(exc)

    def submit(self, batch_idx, toks):
        self.input_queue.put((batch_idx, toks))

    def get_result(self):
        result = self.result_queue.get()
        if isinstance(result, Exception):
            raise RuntimeError(f"Model worker {self.name} failed") from result
        return result

    def stop(self):
        self.input_queue.put(None)


class PipelineProcessingTask(ResumableArrayTask):
    """Processes JSONL files through model groups, writing batch-by-batch to parquet."""

    def __init__(
        self,
        files,
        input_dir,
        output_dir,
        options,
        checkpoint_dir=None,
        atomic=False,
        device=None,
    ):
        super().__init__(files, input_dir, output_dir, checkpoint_dir)
        self.options = options
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.atomic = atomic
        if not str(self.device).startswith("cuda") or not torch.cuda.is_available():
            raise RuntimeError("Annotation requires a CUDA GPU; CPU tools are installed separately")
        self.scorer_eli5 = FastTextScorer(load_model(
            "mlfoundations/fasttext-oh-eli5", "openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train.bin"
        ), "__label__hq")
        self.scorer_preselect = FastTextScorer(load_model(
            "hkust-nlp/preselect-fasttext-classifier", "PreSelect-classifier.bin"
        ), "__label__1")

        self.tokenizers = {
            g.tokenizer_name: AutoTokenizer.from_pretrained(
                g.tokenizer_name, use_fast=True, **source_kwargs(g.tokenizer_name)
            )
            for g in options.model_groups
            if g.tokenizer_name
        }

        self.model_workers = {}
        for group in options.model_groups:
            if not group.tokenizer_name or group.tokenizer_name == "core42/jais-13b":
                continue
            for m in group.models:
                key = (group.tokenizer_name, m.name)
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32,
                }
                if m.name in (
                    "WebOrganizer/TopicClassifier",
                    "WebOrganizer/FormatClassifier",
                ):
                    model_kwargs["use_memory_efficient_attention"] = options.memory_efficient_attention
                    model_kwargs["unpad_inputs"] = options.memory_efficient_attention
                model = (
                    AutoModelForSequenceClassification.from_pretrained(
                        m.name, **model_kwargs, **source_kwargs(m.name)
                    )
                    .to(self.device)
                    .eval()
                )
                self.model_workers[key] = ModelWorker(key, model, self.device)

    def output_path(self, input_file: str) -> Path:
        return make_output_path_parquet(
            self.input_dir, self.input_dir / input_file, self.output_dir
        )

    def is_done(self, input_file: str) -> bool:
        return self.output_path(input_file).exists()

    def process_item(self, file_name: str):
        faulthandler.enable(all_threads=True)
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
        signal.signal(signal.SIGINT, signal.default_int_handler)
        input_path = self.input_dir / file_name
        output_path = self.output_path(file_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Pick adapter once per file
        adapter = pick_adapter(input_path)
        if adapter:
            logger.info(f"Using adapter: {adapter.__class__.__name__} for {input_path}")

        # Prepare column names for each model/classifier
        model_columns = []
        fasttext_columns = []
        float_score_columns = []
        for group in self.options.model_groups:
            for m in group.models:
                if m.name == "fasttext_eli5":
                    col_name = "fasttext_eli5_score"
                    fasttext_columns.append(col_name)
                    float_score_columns.append(col_name)
                elif m.name == "fasttext_preselect":
                    col_name = "fasttext_preselect_score"
                    fasttext_columns.append(col_name)
                    float_score_columns.append(col_name)
                elif m.name == "HuggingFaceTB/fineweb-edu-classifier":
                    col_name = "fineweb_edu_classifier_score"
                    float_score_columns.append(col_name)
                elif m.name == "core42/jais-13b":
                    col_name = "jais_token_count"
                else:
                    col_name = re.search(CLASSIFIER_PATTERN, m.name).group(1)
                model_columns.append(col_name)

        input_schema = None
        writer = None

        try:
            with JsonlReader(
                input_path,
                compression=self.options.compression,
                batch_size=self.options.batch_size,
            ) as reader:
                for bi, batch in enumerate(tqdm(reader, desc=f"Processing {file_name}")):
                    with timers["read"]:
                        input_rows = []
                        for data in batch:
                            row = {}
                            for k, v in data.items():
                                if isinstance(v, dict):
                                    for subk, subv in v.items():
                                        row[f"{k}.{subk}"] = subv
                                else:
                                    row[k] = v
                            input_rows.append(row)
                        input_df = pd.DataFrame(input_rows)

                    # --- Model inference as before ---
                    group_inputs = []
                    for group in self.options.model_groups:
                        with timers["tokenize"]:
                            inputs = [
                                group.prompt_template.format(
                                    url=(data.get("meta") or {}).get("url", ""),
                                    text=text_adapter(data, self.options.text_field),
                                )
                                for data in batch
                            ]
                        group_inputs.append(inputs)

                    col_results = {}
                    for gi, group in enumerate(self.options.model_groups):
                        # --- FastText inference ---
                        if group.models[0].name == "fasttext_eli5":
                            col_name = "fasttext_eli5_score"
                            with timers["inference"]:
                                col_results[col_name] = self.scorer_eli5.score_texts(
                                    group_inputs[gi]
                                )
                            continue
                        if group.models[0].name == "fasttext_preselect":
                            col_name = "fasttext_preselect_score"
                            with timers["inference"]:
                                col_results[col_name] = self.scorer_preselect.score_texts(
                                    group_inputs[gi]
                                )
                            continue

                        # --- Tokenizer-based models ---
                        if not group.tokenizer_name:
                            continue
                        tokenizer = self.tokenizers[group.tokenizer_name]
                        inputs = group_inputs[gi]

                        if group.tokenizer_name == "core42/jais-13b":

                            def encode_and_get_length(text):
                                return len(tokenizer.encode(text)) + 1

                            with timers["jais_tokenize"]:
                                with ThreadPoolExecutor(max_workers=8) as executor:
                                    toks_len = list(
                                        executor.map(encode_and_get_length, inputs)
                                    )
                            col_results["jais_token_count"] = toks_len
                            continue

                        with timers["tokenize"]:

                            def encode_one(text):
                                return tokenizer.encode_plus(
                                    text,
                                    padding="max_length",
                                    truncation=True,
                                    max_length=group.max_length,
                                    return_attention_mask=True,
                                    return_tensors=None,
                                )

                            with ThreadPoolExecutor(max_workers=8) as executor:
                                results = list(executor.map(encode_one, inputs))
                            input_ids = [r["input_ids"] for r in results]
                            attention_masks = [r["attention_mask"] for r in results]

                        with timers["to_device"]:
                            toks = {
                                "input_ids": torch.tensor(
                                    input_ids, dtype=torch.int64, pin_memory=True
                                ),
                                "attention_mask": torch.tensor(
                                    attention_masks, dtype=torch.int64, pin_memory=True
                                ),
                            }
                            toks = {
                                k: v.to(self.device, non_blocking=True)
                                for k, v in toks.items()
                            }

                            torch.cuda.synchronize()

                        for m in group.models:
                            key = (group.tokenizer_name, m.name)
                            with timers["inference"]:
                                self.model_workers[key].submit(bi, toks)

                    with timers["postprocess"]:
                        for group in self.options.model_groups:
                            if (
                                not group.tokenizer_name
                                or group.tokenizer_name == "core42/jais-13b"
                            ):
                                continue
                            for m in group.models:
                                key = (group.tokenizer_name, m.name)
                                result_batch_id, logits = self.model_workers[
                                    key
                                ].get_result()
                                assert result_batch_id == bi
                                if m.name == "HuggingFaceTB/fineweb-edu-classifier":
                                    # Save as float score
                                    col_name = "fineweb_edu_classifier_score"
                                    scores = [float(logit.item()) for logit in logits]
                                    col_results[col_name] = scores
                                else:
                                    choices = logits.argmax(axis=-1).tolist()
                                    col_name = re.search(CLASSIFIER_PATTERN, m.name).group(
                                        1
                                    )
                                    col_results[col_name] = choices

                    # --- Add model columns to input DataFrame ---
                    for col in model_columns:
                        input_df[col] = col_results.get(col, [None] * len(input_df))

                    if adapter:
                        input_df = adapter.sanitize_batch(input_df)

                    # --- Write batch to Parquet ---
                    with timers["write"]:
                        # Use a temp file for atomic write
                        tmp_output_path = output_path.with_suffix(
                            output_path.suffix + ".tmp"
                        )
                        if writer is None:
                            # Infer schema from the first batch (no index)
                            input_schema = pa.Schema.from_pandas(
                                input_df, preserve_index=False
                            )

                            # Override fields where needed:
                            fields = []
                            for field in input_schema:
                                if field.name in float_score_columns:
                                    # enforce float32 and nullable for score columns
                                    fields.append(
                                        pa.field(field.name, pa.float32(), nullable=True)
                                    )
                                else:
                                    fields.append(field)

                            input_schema = pa.schema(fields)
                            # Apply adapter overrides
                            if adapter:
                                input_schema = adapter.override_schema(input_schema)

                            writer = pq.ParquetWriter(
                                str(tmp_output_path), input_schema, compression="zstd"
                            )

                        # Build and write record batch using the frozen schema
                        record_batch = pa.RecordBatch.from_pandas(
                            input_df, schema=writer.schema, preserve_index=False
                        )
                        writer.write_batch(record_batch)

                    # Print all intermediate timing stats after every batch
                    print(f"Batch {bi} timing stats:")
                    for stage, timer in timers.items():
                        print(f"  {stage}: {timer}")
                    sys.stdout.flush()

        finally:
            if writer is not None:
                writer.close()
        if writer is not None:
            tmp_output_path.replace(output_path)

        # Log timer stats for this file
        logger.info(f"Timing stats for {file_name}:")
        for stage, timer in timers.items():
            logger.info(f"  {stage}: {timer}")

    def run(self):
        failures = []
        try:
            for fname in tqdm(self.items, desc="Files"):
                if self.is_done(fname):
                    logger.info("Skipping %s", fname)
                    continue
                try:
                    self.process_item(fname)
                except Exception:
                    failures.append(str(fname))
                    logger.exception("Failed to process %s", fname)
        finally:
            for worker in self.model_workers.values():
                worker.stop()
            for worker in self.model_workers.values():
                worker.join()
        if failures:
            raise RuntimeError(f"Failed to annotate {len(failures)} file(s): {failures}")


# --- CLI Interface and get_shard unchanged ---


def get_shard(
    input_dir: Path,
    rank: int,
    world_size: int,
    glob_pattern: str = "**/*",
) -> List[Path]:
    if not input_dir.is_dir():
        raise ValueError(
            f"Input directory {input_dir} does not exist or is not a directory."
        )
    all_files = sorted(p.relative_to(input_dir) for p in input_dir.glob(glob_pattern) if p.is_file())
    shard_files = [f for i, f in enumerate(all_files) if i % world_size == rank]
    return shard_files


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--glob", type=str, default="**/*.jsonl")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--text-field", type=str, default="text")
    parser.add_argument(
        "--compression", type=str, default="none", choices=["gzip", "none"]
    )

    parser.add_argument("--memory-efficient-attention", action="store_true", help="Requires compatible xformers")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size must be positive")
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    separate_output(args.input_dir, args.output_dir)

    rank, world_size = get_rank_world()

    input_files = get_shard(args.input_dir, rank, world_size, args.glob)

    if not input_files:
        logger.info("No input files assigned to this rank")
        return
    outputs = [make_output_path_parquet(args.input_dir, args.input_dir / f, args.output_dir) for f in input_files]
    if len(outputs) != len(set(outputs)):
        raise ValueError("Input filenames map to duplicate output paths")
    pipeline_opts = PipelineOptions(
        batch_size=args.batch_size,
        text_field=args.text_field,
        compression=args.compression,
        memory_efficient_attention=args.memory_efficient_attention,
    )

    task = PipelineProcessingTask(
        files=input_files,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        options=pipeline_opts,
    )
    task.run()


if __name__ == "__main__":
    main()
