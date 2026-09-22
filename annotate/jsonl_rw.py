import os
import gzip
import orjson
from pathlib import Path
import logging
import threading
import queue
try:
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    Dataset = object
    DataLoader = None
import numpy as np


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class JsonlWriterV0:
    """
    Writes JSON lines to a file, optionally compressed with gzip.
    Batches of data can be written at once using write_batch().
    """

    def __init__(self, output_filename: str, compression: str = "gzip"):
        self.output_filename = output_filename
        os.makedirs(
            Path(output_filename).parent, exist_ok=True
        )  # ensures subdirs exist
        self.compression = compression
        self.file = None

    def __enter__(self):
        if self.compression == "gzip":
            self.file = gzip.open(self.output_filename, "wb")
        else:
            self.file = open(self.output_filename, "wb")
        return self

    def write_batch(self, batch):
        """
        Writes out each item in the given batch as a separate JSON line.
        """
        for doc in batch:
            line = orjson.dumps(doc, option=orjson.OPT_APPEND_NEWLINE)
            self.file.write(line)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()


class JsonlWriter:
    """
    Writes JSON lines to a (gzip-compressed) file by first writing to a
    temporary file named "<original>_tmp.jsonl.gz" and then renaming it to
    "<original>.jsonl.gz" upon exit.
    Batches of data can be written at once using write_batch().
    """

    def __init__(self, output_filename: str, compression: str = "gzip"):
        self.output_filename = Path(output_filename)
        # Ensure parent directories exist.
        os.makedirs(self.output_filename.parent, exist_ok=True)

        self.tmp_filename = self.output_filename.with_name(self.output_filename.name + ".tmp")

        self.compression = compression
        self.file = None
        self.data_written = False

    def __enter__(self):
        if self.compression == "gzip":
            self.file = gzip.open(self.tmp_filename, "wb")
        else:
            self.file = open(self.tmp_filename, "wb")
        return self

    def write_batch(self, batch):
        """
        Writes out each item in the given batch as a separate JSON line
        to the temporary file.
        """
        if not isinstance(batch, list):
            raise TypeError("batch must be a list")
        for item in batch:
            if not isinstance(item, dict):
                raise TypeError("Each item in batch must be a dictionary")
        for doc in batch:
            line = orjson.dumps(doc, option=orjson.OPT_APPEND_NEWLINE)
            self.file.write(line)
            self.data_written = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        # Rename the temporary file to the final output name.
        if self.data_written and exc_type is None:
            self.tmp_filename.rename(self.output_filename)
        else:
            # Remove the temporary file if no data was written
            self.tmp_filename.unlink(missing_ok=True)


class JsonlReader:
    """
    Reads JSON lines from a file, optionally compressed with gzip, yielding one
    batch at a time in an iterator. The batch_size is for buffering and can
    be adjusted for performance.
    """

    def __init__(self, file_name: str, compression: str = "gzip", batch_size: int = 64):
        self.file_name = file_name
        self.compression = compression
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.file = None

    def __enter__(self):
        if self.compression == "gzip":
            self.file = gzip.open(self.file_name, "rb")
        else:
            self.file = open(self.file_name, "rb")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def __iter__(self):
        """
        Yields documents in batches
        """
        buffer = []
        try:
            for line in self.file:
                try:
                    document = orjson.loads(line)
                except (EOFError, orjson.JSONDecodeError) as exc:
                    raise ValueError(f"Malformed JSONL in {self.file_name}") from exc
                buffer.append(document)
                if len(buffer) >= self.batch_size:
                    yield buffer
                    buffer = []

            # Yield any leftover documents in the buffer.
            if len(buffer) > 0:
                yield buffer
        except gzip.BadGzipFile:
            raise ValueError(f"File '{self.file_name}' is not a valid gzip stream")


# --- Output Buffer ---


class OutputBuffer:
    """Accumulates outputs for a given (model, output_spec) pair and writes as NumPy."""

    def __init__(self, file_path: Path, atomic: bool = True):
        self.file_path = file_path
        self.buffer = []
        self.atomic = atomic

    def add(self, data):
        if isinstance(data, np.ndarray):
            self.buffer.extend(data.tolist())
        elif isinstance(data, list):
            self.buffer.extend(data)
        else:
            self.buffer.append(data)

    def save(self):
        if not self.buffer:
            return
        array = np.array(self.buffer)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if self.atomic:
            tmp = self.file_path.with_suffix(self.file_path.suffix + ".tmp")
            with tmp.open("wb") as handle:
                np.save(handle, array)
            tmp.rename(self.file_path)
        else:
            np.save(self.file_path, array)
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


class PrefetchingJsonlReader:
    """Strict JSONL reader with bounded prefetch and cancellation on early exit."""
    _SENTINEL = object()

    def __init__(self, file_name, compression="gzip", batch_size=64, prefetch=2):
        self.reader = JsonlReader(file_name, compression, batch_size)
        self._queue = queue.Queue(maxsize=max(1, prefetch))
        self._stop = threading.Event()
        self._worker = None
        self._exception = None

    def __enter__(self):
        self.reader.__enter__()
        self._worker = threading.Thread(target=self._worker_fn, daemon=True)
        self._worker.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._worker:
            self._worker.join()
        self.reader.__exit__(*exc)

    def _put(self, item):
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                pass
        return False

    def _worker_fn(self):
        try:
            for batch in self.reader:
                # JsonlReader reuses its buffer after yielding.
                if not self._put(list(batch)):
                    return
        except Exception as exc:
            self._exception = exc
        finally:
            self._put(self._SENTINEL)

    def __iter__(self):
        while True:
            batch = self._queue.get()
            if batch is self._SENTINEL:
                if self._exception:
                    raise self._exception
                return
            yield batch


class _JsonlFileDataset(Dataset):
    """
    A torch Dataset for jsonl (optionally gzipped) files.
    Indexing is efficient (seek) for uncompressed; for .gz, loads line offsets at startup.
    """

    def __init__(self, filename, compression="gzip", skip_bad_lines=True):
        self.filename = str(filename)
        self.compression = compression
        self.skip_bad_lines = skip_bad_lines
        self.line_offsets = []
        if compression == "gzip":
            # Build offsets on open: yes, this loads the offsets into memory, slow startup, but needed for random access!
            self.line_offsets = []
            with gzip.open(self.filename, "rb") as f:
                offset = 0
                for line in f:
                    self.line_offsets.append(offset)
                    offset += len(line)
            self.n = len(self.line_offsets)
        else:
            # Count lines (and seek offsets for random access)
            self.line_offsets = []
            with open(self.filename, "rb") as f:
                offset = 0
                for line in f:
                    self.line_offsets.append(offset)
                    offset += len(line)
            self.n = len(self.line_offsets)

    def __len__(self):
        return self.n

    def _get_line(self, idx):
        offset = self.line_offsets[idx]
        if self.compression == "gzip":
            with gzip.open(self.filename, "rb") as f:
                f.seek(offset)
                line = f.readline()
                return line
        else:
            with open(self.filename, "rb") as f:
                f.seek(offset)
                line = f.readline()
                return line

    def __getitem__(self, idx):
        line = self._get_line(idx)
        try:
            return orjson.loads(line)
        except (EOFError, orjson.JSONDecodeError):
            if self.skip_bad_lines:
                # Signal to collate_fn to drop this item
                return None
            else:
                raise


def _collate_skip_none(batch):
    """Collate function: skips lines for which decoding failed (None)."""
    return [item for item in batch if item is not None]


class TorchJsonlReader:
    """
    Drop-in replacement for JsonlReader supporting:
    - multiprocessing with PyTorch DataLoader (num_workers)
    - batch iteration
    Usage:
        with TorchJsonlReader(filename, batch_size=32, num_workers=2, pin_memory=True) as reader:
            for batch in reader:
                ...
    """

    def __init__(
        self,
        filename,
        compression="gzip",
        batch_size=64,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    ):
        self.filename = filename
        self.compression = compression
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.dataloader = None
        self._dataset = None

    def __enter__(self):
        if DataLoader is None:
            raise ImportError("TorchJsonlReader requires the annotate dependency group")
        self._dataset = _JsonlFileDataset(self.filename, compression=self.compression)
        self.dataloader = DataLoader(
            self._dataset,
            batch_size=self.batch_size,
            collate_fn=_collate_skip_none,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
            drop_last=False,
        )
        self._iterator = iter(self.dataloader)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._iterator = None
        self.dataloader = None
        self._dataset = None

    def __iter__(self):
        for batch in self._iterator:
            if batch:  # skip empty batches (all decode errors)
                yield batch


# Example Usage:
if __name__ == "__main__":
    import sys

    jsonl_path = sys.argv[1]
    with TorchJsonlReader(
        jsonl_path, batch_size=32, num_workers=4, compression=""
    ) as reader:
        for batch in reader:
            print(f"Batch of {len(batch)} docs")
