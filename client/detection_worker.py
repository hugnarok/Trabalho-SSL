from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from client.capture import StreamCapture
from client.detectors.base import DetectionResult
from client.detectors.help_request import detect_help_request
from client.detectors.impact import detect_impact
from client.detectors.ml_events import detect_ml_events
from client.detectors.scream import detect_scream
from client.ml.classifier import is_model_available
from shared.config import settings
from shared.models import EventType


@dataclass
class DetectionState:
    status: str = "iniciando..."
    best: Optional[DetectionResult] = None
    candidate: Optional[DetectionResult] = None
    last_audio: Optional[np.ndarray] = None


class DetectionWorker(threading.Thread):
    """Detectores heurísticos + ML + confirmação em sequência."""

    def __init__(self, capture: StreamCapture):
        super().__init__(daemon=True, name="detection-worker")
        self._capture = capture
        self._lock = threading.Lock()
        self._state = DetectionState()
        self._running = False
        self._streak_type: Optional[EventType] = None
        self._streak_count = 0

    def _collect_results(self, audio: np.ndarray, sample_rate: int) -> List[DetectionResult]:
        results: List[DetectionResult] = []
        ml_ok = settings.ml_enabled and is_model_available()

        if ml_ok:
            ml_result = detect_ml_events(audio, sample_rate)
            if ml_result.detected:
                results.append(ml_result)

        use_heuristic = settings.ml_heuristic_fallback or not ml_ok
        if use_heuristic:
            results.extend(
                [
                    detect_scream(audio, sample_rate),
                    detect_impact(audio, sample_rate),
                ]
            )

        if settings.help_keyword_enabled:
            results.append(detect_help_request(audio, sample_rate))

        return results

    def _min_confidence_for(self, event_type: EventType) -> float:
        if event_type == EventType.HELP_REQUEST:
            return max(settings.min_alert_confidence, settings.help_min_confidence)
        return settings.min_alert_confidence

    def _confirmations_required_for(self, event_type: EventType) -> int:
        if event_type == EventType.HELP_REQUEST:
            return max(1, settings.help_confirmations_required)
        return max(1, settings.detection_confirmations_required)

    def _update_streak(self, candidate: Optional[DetectionResult]) -> Optional[DetectionResult]:
        if candidate is None:
            self._streak_type = None
            self._streak_count = 0
            return None

        required = self._confirmations_required_for(candidate.event_type)
        min_conf = self._min_confidence_for(candidate.event_type)

        if candidate.confidence < min_conf:
            self._streak_type = None
            self._streak_count = 0
            return None

        if self._streak_type == candidate.event_type:
            self._streak_count += 1
        else:
            self._streak_type = candidate.event_type
            self._streak_count = 1

        if self._streak_count >= required:
            return candidate
        return None

    def _status_text(
        self,
        confirmed: Optional[DetectionResult],
        candidate: Optional[DetectionResult],
    ) -> str:
        if confirmed:
            return confirmed.event_type.value
        if candidate:
            req = self._confirmations_required_for(candidate.event_type)
            return f"{candidate.event_type.value}? ({self._streak_count}/{req})"
        return "monitorando"

    def run(self) -> None:
        self._running = True
        while self._running:
            audio = self._capture.read_audio_chunk()
            results = self._collect_results(audio, self._capture.sample_rate)

            detected = [r for r in results if r.detected]
            candidate = (
                max(detected, key=lambda r: r.confidence) if detected else None
            )
            confirmed = self._update_streak(candidate)
            status = self._status_text(confirmed, candidate)

            with self._lock:
                self._state = DetectionState(
                    status=status,
                    best=confirmed,
                    candidate=candidate,
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
                candidate=self._state.candidate,
                last_audio=(
                    self._state.last_audio.copy()
                    if self._state.last_audio is not None
                    else None
                ),
            )
