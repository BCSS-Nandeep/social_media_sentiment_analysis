"""Shared utilities: logging, reproducibility, device and resource helpers."""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Iterator, Sequence, TypeVar

import numpy as np
import psutil
import torch

T = TypeVar("T")

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure root logging once and return the pipeline logger."""
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt="%H:%M:%S")
    for noisy in ("transformers", "urllib3", "filelock", "huggingface_hub", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("benchmark")


def set_seed(seed: int) -> None:
    """Seed every RNG that influences results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve the compute device with a graceful CPU fallback."""
    logger = logging.getLogger("benchmark")
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but not available — falling back to CPU.")
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        logger.info("GPU detected: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")
    logger.info("No GPU detected — running on CPU.")
    return torch.device("cpu")


def get_peak_ram_mb() -> float:
    """Peak resident memory of this process in MB (cross-platform best effort)."""
    mem = psutil.Process().memory_info()
    peak = getattr(mem, "peak_wset", None)  # Windows
    if peak is None:
        try:
            import resource  # Unix only

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        except Exception:  # pragma: no cover - platform dependent
            peak = mem.rss
    return float(peak) / (1024**2)


def get_current_ram_mb() -> float:
    """Current resident memory of this process in MB."""
    return float(psutil.Process().memory_info().rss) / (1024**2)


def reset_gpu_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_gpu_peak_mb(device: torch.device) -> float:
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device)) / (1024**2)
    return 0.0


def get_model_size_mb(model: torch.nn.Module) -> float:
    """In-memory size of all model parameters and buffers, in MB."""
    size = sum(p.numel() * p.element_size() for p in model.parameters())
    size += sum(b.numel() * b.element_size() for b in model.buffers())
    return size / (1024**2)


def resolve_path(path_str: str | Path, base: Path) -> Path:
    """Resolve *path_str* against CWD first, then against the project base dir."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate
    return base / p


def chunked(seq: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    """Yield consecutive slices of *seq* of length *size*."""
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


def format_seconds(seconds: float) -> str:
    """Human-readable duration, e.g. '1m 23.4s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60)
    return f"{int(minutes)}m {rem:.1f}s"
