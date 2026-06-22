"""
Detecção por modelo ML (CNN + log-mel spectrogram).
"""

from __future__ import annotations

import numpy as np

from client.detectors.base import DetectionResult
from client.ml.classifier import LABEL_TO_EVENT, _get_classifier
from shared.config import settings
from shared.models import EventType


def detect_ml_events(audio: np.ndarray, sample_rate: int) -> DetectionResult:
    clf = _get_classifier()
    if clf is None:
        return DetectionResult(
            detected=False,
            event_type=EventType.SCREAM,
            confidence=0.0,
            message="modelo ML não carregado",
        )

    if audio.size == 0:
        return DetectionResult(
            detected=False,
            event_type=EventType.SCREAM,
            confidence=0.0,
            message="sem áudio",
        )

    label, confidence, scores = clf.predict(audio, sample_rate)
    scores_txt = ", ".join(f"{k}={v:.2f}" for k, v in scores.items())
    event = LABEL_TO_EVENT.get(label)

    if event is None or confidence < settings.ml_confidence_threshold:
        return DetectionResult(
            detected=False,
            event_type=EventType.SCREAM,
            confidence=round(confidence, 3),
            message=f"ML: {label} ({scores_txt})",
        )

    return DetectionResult(
        detected=True,
        event_type=event,
        confidence=round(confidence, 3),
        message=f"ML: {label} ({scores_txt})",
    )
