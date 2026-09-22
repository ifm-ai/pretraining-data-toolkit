"""Shared, dependency-free runtime helpers."""
import os
from pathlib import Path
from time import perf_counter


def get_rank_world():
    """Explicit RANK/WORLD_SIZE, or a contiguous SLURM array, or one process."""
    if "RANK" in os.environ or "WORLD_SIZE" in os.environ:
        if not {"RANK", "WORLD_SIZE"} <= os.environ.keys():
            raise ValueError("Set both RANK and WORLD_SIZE")
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    elif "SLURM_ARRAY_TASK_ID" in os.environ:
        world = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        first = int(os.environ.get("SLURM_ARRAY_TASK_MIN", "0"))
        step = int(os.environ.get("SLURM_ARRAY_TASK_STEP", "1"))
        last = int(os.environ.get("SLURM_ARRAY_TASK_MAX", first + step * (world - 1)))
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if step <= 0 or last != first + step * (world - 1) or (task_id-first) % step:
            raise ValueError("Sparse SLURM arrays require explicit RANK/WORLD_SIZE")
        rank = (task_id-first) // step
    else:
        rank, world = 0, 1
    if world <= 0 or not 0 <= rank < world:
        raise ValueError(f"Invalid rank/world: {rank}/{world}")
    return rank, world


def separate_output(input_dir, output_dir):
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    if source == output or source in output.parents:
        raise ValueError("Output must be outside the input tree")


class TimingStats:
    def __init__(self, unit="batch"):
        self.unit, self.total, self.count = unit, 0.0, 0
    def __enter__(self):
        self.start = perf_counter()
        return self
    def __exit__(self, *exc):
        self.total += perf_counter() - self.start
        self.count += 1
    def __str__(self):
        return f"{self.total:.3f}s / {self.count} {self.unit}(s)"
