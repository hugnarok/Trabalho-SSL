"""
Visualização da forma de onda em tempo real (conceitos SSL: amostragem, energia, limiar).

Faixa empilhada abaixo do preview da câmera; vermelho na zona de perigo.
"""

from __future__ import annotations

from typing import Literal, Optional

import cv2
import numpy as np

from shared.config import settings

DangerLevel = Literal["ok", "warning", "danger"]

_WAVE_HEIGHT = 110
_DISPLAY_SECONDS = 2.0

_COLOR_OK = (80, 200, 80)
_COLOR_WARN = (80, 220, 255)
_COLOR_DANGER = (80, 80, 255)
_COLOR_BELOW = (100, 255, 120)
_COLOR_ABOVE = (80, 80, 255)
_BG_OK = (32, 36, 32)
_BG_WARN = (32, 40, 48)
_BG_DANGER = (32, 32, 52)


def danger_level_from_state(
    candidate: Optional[object],
    best: Optional[object],
) -> DangerLevel:
    if best is not None:
        return "danger"
    if candidate is not None:
        return "warning"
    return "ok"


def _background_for(level: DangerLevel) -> tuple[int, int, int]:
    if level == "danger":
        return _BG_DANGER
    if level == "warning":
        return _BG_WARN
    return _BG_OK


def _accent_for(level: DangerLevel) -> tuple[int, int, int]:
    if level == "danger":
        return _COLOR_DANGER
    if level == "warning":
        return _COLOR_WARN
    return _COLOR_OK


def _peak_per_column(segment: np.ndarray, width: int) -> np.ndarray:
    if segment.size == 0:
        return np.zeros(width, dtype=np.float64)
    edges = np.linspace(0, len(segment), width + 1, dtype=int)
    peaks = np.empty(width, dtype=np.float64)
    for i in range(width):
        chunk = segment[edges[i] : edges[i + 1]]
        peaks[i] = float(np.max(np.abs(chunk))) if chunk.size else 0.0
    return peaks


def _rms_envelope(segment: np.ndarray, width: int) -> np.ndarray:
    if segment.size == 0:
        return np.zeros(width, dtype=np.float64)
    edges = np.linspace(0, len(segment), width + 1, dtype=int)
    env = np.empty(width, dtype=np.float64)
    for i in range(width):
        chunk = segment[edges[i] : edges[i + 1]]
        env[i] = float(np.sqrt(np.mean(chunk**2))) if chunk.size else 0.0
    return env


def render_waveform_overlay(
    audio: np.ndarray,
    width: int,
    sample_rate: int,
    danger_level: DangerLevel = "ok",
    display_seconds: float = _DISPLAY_SECONDS,
    height: int = _WAVE_HEIGHT,
) -> np.ndarray:
    """Retorna faixa BGR (height × width) com forma de onda + envelope RMS + limiar."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = _background_for(danger_level)

    n_show = min(len(audio), int(display_seconds * sample_rate))
    segment = audio[-n_show:].astype(np.float64) if n_show > 0 else np.array([], dtype=np.float64)

    mid = height // 2
    plot_h = mid - 14
    threshold = settings.scream_energy_threshold
    scale = max(float(np.max(np.abs(segment))) if segment.size else 0.0, threshold * 2.5, 1e-9)

    cv2.line(canvas, (0, mid), (width, mid), (70, 70, 70), 1)

    thr_y = int(threshold / scale * plot_h)
    thr_y = min(plot_h, max(1, thr_y))
    for y_off in (-thr_y, thr_y):
        y = mid + y_off
        for x in range(0, width, 8):
            cv2.line(canvas, (x, y), (min(x + 4, width - 1), y), (100, 100, 140), 1)

    if segment.size == 0:
        cv2.putText(
            canvas,
            "Aguardando audio...",
            (8, mid + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        return canvas

    peaks = _peak_per_column(segment, width)
    envelope = _rms_envelope(segment, width)

    env_pts = []
    for x, rms_val in enumerate(envelope):
        y = mid - int(rms_val / scale * plot_h)
        env_pts.append((x, y))
    if len(env_pts) >= 2:
        pts = np.array(env_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], False, (60, 140, 200), 1, cv2.LINE_AA)

    for x, amp in enumerate(peaks):
        y = int(amp / scale * plot_h)
        above = amp >= threshold
        color = _COLOR_ABOVE if above else _COLOR_BELOW
        if danger_level == "danger":
            color = _COLOR_ABOVE
        elif danger_level == "warning" and above:
            color = _COLOR_WARN
        cv2.line(canvas, (x, mid - y), (x, mid + y), color, 1)

    rms_global = float(np.sqrt(np.mean(segment**2)))
    accent = _accent_for(danger_level)
    cv2.putText(
        canvas,
        "Forma de onda",
        (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        accent,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"RMS={rms_global:.4f}  limiar={threshold:.3f}",
        (8, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    zone = {"ok": "normal", "warning": "suspeita", "danger": "PERIGO"}[danger_level]
    cv2.putText(
        canvas,
        zone.upper(),
        (width - 90, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        accent,
        1,
        cv2.LINE_AA,
    )
    return canvas
