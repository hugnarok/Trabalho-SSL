from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from client.capture import StreamCapture
from client.detectors.base import DetectionResult
from client.detectors.help_request import detect_help_request
from client.detectors.impact import detect_impact
from client.detectors.scream import detect_scream
from shared.config import settings


@dataclass
class DetectionState:
    status: str = "iniciando..."
    best: Optional[DetectionResult] = None
    last_audio: Optional[np.ndarray] = None


class DetectionWorker(threading.Thread):
    """Roda FFT/energia em thread separada — não trava o preview."""

    def __init__(self, capture: StreamCapture):
        super().__init__(daemon=True, name="detection-worker")
        self._capture = capture
        self._lock = threading.Lock()
        self._state = DetectionState()
        self._running = False

    def run(self) -> None:
        self._running = True
        while self._running:
            audio = self._capture.read_audio_chunk()
            results = [
                detect_scream(audio, self._capture.sample_rate),
                detect_impact(audio, self._capture.sample_rate),
            ]
            if settings.help_keyword_enabled:
                results.append(
                    detect_help_request(audio, self._capture.sample_rate)
                )

            detected = [r for r in results if r.detected]
            best = max(detected, key=lambda r: r.confidence) if detected else None
            status = best.event_type.value if best else "monitorando"

            with self._lock:
                self._state = DetectionState(
                    status=status,
                    best=best,
                    last_audio=audio.copy(),
                )
            time.sleep(settings.detection_interval_seconds)

    def stop(self) -> None:
        self._running = False

    def snapshot(self) -> DetectionState:
        with self._lock:
            return DetectionState(
                status=self._state.status,
                best=self._state.best,
                last_audio=(
                    self._state.last_audio.copy()
                    if self._state.last_audio is not None
                    else None
                ),
            )
