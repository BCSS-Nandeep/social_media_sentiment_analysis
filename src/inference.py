"""Stage 2 — English sentiment classification with pretrained models.

Both benchmark models ship with real fine-tuned sentiment heads, so no
training is required. Each model's own label space is mapped onto the
canonical ``[Negative, Neutral, Positive]`` space via its ``id2label``.

SieBERT is binary (no Neutral class): a synthetic Neutral probability of
``NEUTRAL_UNCERTAINTY_SCALE * min(p_neg, p_pos)`` is inserted before row-wise
renormalization, so near-ties between the two poles read as Neutral.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import (
    ID2LABEL,
    LABEL2ID,
    LABELS,
    NEUTRAL_UNCERTAINTY_SCALE,
    SentimentModelConfig,
)
from src.utils import chunked, get_gpu_peak_mb, get_model_size_mb, reset_gpu_peak

logger = logging.getLogger("benchmark.inference")


@dataclass
class InferenceResult:
    """Per-post predictions plus aggregate runtime statistics for one model."""

    labels: list[str]        # predicted class names (canonical label space)
    confidences: np.ndarray  # probability of the predicted class, shape (N,)
    probs: np.ndarray        # canonical probability matrix, shape (N, 3)
    times_ms: np.ndarray     # amortized per-post inference time, shape (N,)
    total_time_s: float      # wall-clock inference time for the whole dataset
    gpu_peak_mb: float       # peak GPU memory allocated during inference


def build_label_mapping(id2label: dict[int, str]) -> dict[int, int]:
    """Map a model's class indices onto canonical ``LABELS`` indices.

    Matches by name prefix (``negative``/``neutral``/``positive``, any case).
    Generic ``LABEL_i`` names fall back to the negative/neutral/positive
    convention for 3-class models and negative/positive for binary models.
    """
    mapping: dict[int, int] = {}
    generic = all(str(name).upper().startswith("LABEL_") for name in id2label.values())
    if generic:
        order = [0, 1, 2] if len(id2label) == 3 else [0, 2]
        return {i: order[i] for i in range(len(id2label))}

    for idx, name in id2label.items():
        lowered = str(name).strip().lower()
        for canonical in LABELS:
            if lowered.startswith(canonical.lower()[:3]):
                mapping[int(idx)] = LABEL2ID[canonical]
                break
        else:
            raise ValueError(f"Cannot map model label {name!r} onto {LABELS}")
    return mapping


def to_canonical_probs(raw_probs: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """Project model probabilities into the canonical 3-class space.

    Binary models get a synthetic Neutral column (see module docstring), then
    every row is renormalized to sum to 1.
    """
    canonical = np.zeros((raw_probs.shape[0], len(LABELS)))
    for model_idx, canon_idx in mapping.items():
        canonical[:, canon_idx] += raw_probs[:, model_idx]

    neutral_idx = LABEL2ID["Neutral"]
    if neutral_idx not in mapping.values():  # binary model
        neg = canonical[:, LABEL2ID["Negative"]]
        pos = canonical[:, LABEL2ID["Positive"]]
        canonical[:, neutral_idx] = NEUTRAL_UNCERTAINTY_SCALE * np.minimum(neg, pos)
    return canonical / canonical.sum(axis=1, keepdims=True)


class SentimentClassifier:
    """A tokenizer + pretrained sentiment model wrapped for benchmarking."""

    def __init__(
        self, cfg: SentimentModelConfig, device: torch.device, max_length: int = 128
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.max_length = max_length

        local = cfg.checkpoint_dir
        use_local = (local / "config.json").exists()
        source = str(local) if use_local else cfg.hf_id
        logger.info(
            "Loading %s from %s%s",
            cfg.display_name, source, " (local checkpoint)" if use_local else "",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source)
        self.model.to(device)
        self.model.eval()

        self.label_mapping = build_label_mapping(self.model.config.id2label)
        self.is_binary = LABEL2ID["Neutral"] not in self.label_mapping.values()
        if self.is_binary:
            logger.warning(
                "%s is a binary (Positive/Negative) model — Neutral is "
                "approximated from prediction uncertainty (scale %.1f).",
                cfg.display_name, NEUTRAL_UNCERTAINTY_SCALE,
            )

    @property
    def size_mb(self) -> float:
        return get_model_size_mb(self.model)

    def predict(self, texts: list[str], batch_size: int = 16) -> InferenceResult:
        """Run batched inference and time every batch."""
        all_probs: list[np.ndarray] = []
        times_ms: list[float] = []
        reset_gpu_peak(self.device)
        total_start = time.perf_counter()

        n_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = tqdm(
            chunked(texts, batch_size),
            total=n_batches,
            desc=f"Inference [{self.cfg.display_name}]",
            unit="batch",
        )
        with torch.inference_mode():
            for batch in iterator:
                batch = list(batch)
                encoded = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)

                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                t0 = time.perf_counter()
                logits = self.model(**encoded).logits
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
                times_ms.extend([elapsed_ms / len(batch)] * len(batch))

        total_time = time.perf_counter() - total_start
        probs = to_canonical_probs(np.vstack(all_probs), self.label_mapping)
        pred_ids = probs.argmax(axis=1)

        return InferenceResult(
            labels=[ID2LABEL[int(i)] for i in pred_ids],
            confidences=probs[np.arange(len(pred_ids)), pred_ids],
            probs=probs,
            times_ms=np.asarray(times_ms),
            total_time_s=total_time,
            gpu_peak_mb=get_gpu_peak_mb(self.device),
        )

    def free(self) -> None:
        """Release model memory (important when benchmarking models serially)."""
        del self.model
        del self.tokenizer
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
