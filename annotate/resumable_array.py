import os
import tempfile
import shutil
from typing import List, Optional
from pathlib import Path


def partition_items(items: List[str], n_partitions: int) -> List[List[str]]:
    """Partition a list of items into n roughly equal sublists."""
    if n_partitions <= 0:
        raise ValueError("n_partitions must be positive")
    part_size = (len(items) + n_partitions - 1) // n_partitions
    return [items[i * part_size : (i + 1) * part_size] for i in range(n_partitions)]


class ResumableArrayTask:
    """
    Helper for checkpointed, per-item work (can be file, id, etc).
    User should subclass and override 'process_item(input, output)'.
    """

    def __init__(
        self,
        items: List[str],
        input_dir: Path,
        output_dir: Path,
        checkpoint_dir: Optional[Path] = None,
        output_ext: str = ".done",
        atomic: bool = True,
    ):
        self.items = items
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.checkpoint_dir = checkpoint_dir or output_dir

        if not self.input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {self.input_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.output_ext = output_ext
        self.atomic = atomic

    def get_output_path(self, item: str) -> str:
        """User may override for custom mapping."""
        base = os.path.basename(item)
        return os.path.join(self.output_dir, base + self.output_ext)

    def process_item(self, input_item: str, output_file: str):
        """
        Override this to define what 'processing' means.
        Should write output to output_file.
        """
        raise NotImplementedError

    def is_done(self, item: str, output_file: str) -> bool:
        return os.path.exists(output_file)

    def mark_done(self, item: str, output_file: str):
        # Optional: write a marker, log, etc.
        pass

    def run(self):
        failures = []
        for item in self.items:
            out_file = self.get_output_path(item)
            if self.is_done(item, out_file):
                print(f"Skipping already-done: {item}")
                continue

            print(f"Processing: {item}")
            temp_path = None
            try:
                if self.atomic:
                    # Write to temp file first, then move atomically
                    out_dir = os.path.dirname(out_file)
                    with tempfile.NamedTemporaryFile(dir=out_dir, delete=False) as tf:
                        temp_path = tf.name  # may pass this into process_item
                    self.process_item(item, temp_path)
                    shutil.move(temp_path, out_file)
                else:
                    self.process_item(item, out_file)
                self.mark_done(item, out_file)
            except Exception as e:
                failures.append(item)
                print(f"ERROR on {item}: {e}")
            finally:
                if temp_path:
                    Path(temp_path).unlink(missing_ok=True)
        if failures:
            raise RuntimeError(f"Failed items: {failures}")
