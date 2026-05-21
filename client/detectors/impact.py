"""
Detecção de impacto / transiente por pico súbito de energia.
"""

import numpy as np

from client.detectors.base import DetectionResult
from shared.config import settings
from shared.models import EventType


def detect_impact(
    audio: np.ndarray,
    sample_rate: int,
) -> DetectionResult:
    if audio.size == 0:
        return DetectionResult(False, EventType.IMPACT, 0.0, "sem áudio")

    x = audio.astype(np.float64)
    frame_len = max(1, int(sample_rate * 0.05))
    n_frames = len(x) // frame_len
    if n_frames < 2:
        return DetectionResult(False, EventType.IMPACT, 0.0, "áudio curto")

    frames = x[: n_frames * frame_len].reshape(n_frames, frame_len)
    energies = np.mean(frames**2, axis=1)
    peak = float(np.max(energies))
    median = float(np.median(energies) + 1e-9)
    ratio = peak / median

    detected = peak >= settings.impact_energy_threshold and ratio > 4.0
    score = min(1.0, ratio / 10.0)

    return DetectionResult(
        detected=detected,
        event_type=EventType.IMPACT,
        confidence=round(score, 3),
        message=f"pico={peak:.4f}, razão={ratio:.2f}",
    )
