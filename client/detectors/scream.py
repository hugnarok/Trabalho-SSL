"""
Detecção de grito por energia em banda aguda (esboço — calibrar limiares).

Conceitos SSL: amostragem, janela, FFT, energia espectral.
"""

import numpy as np
from scipy.fft import rfft, rfftfreq

from client.detectors.base import DetectionResult
from shared.config import settings
from shared.models import EventType


def detect_scream(
    audio: np.ndarray,
    sample_rate: int,
) -> DetectionResult:
    if audio.size == 0:
        return DetectionResult(False, EventType.SCREAM, 0.0, "sem áudio")

    # Normalização
    x = audio.astype(np.float64)
    x = x / (np.max(np.abs(x)) + 1e-9)

    # Energia RMS no domínio do tempo
    rms = float(np.sqrt(np.mean(x**2)))

    # Energia na banda aguda (aprox. 2–8 kHz) via FFT
    n = min(len(x), sample_rate * 2)
    segment = x[:n]
    spectrum = np.abs(rfft(segment))
    freqs = rfftfreq(n, 1 / sample_rate)
    high_band = (freqs >= 2000) & (freqs <= 8000)
    high_energy = float(np.mean(spectrum[high_band]) / (np.mean(spectrum) + 1e-9))

    # Score heurístico (substituir por modelo treinado depois)
    score = min(1.0, 0.5 * (rms / settings.scream_energy_threshold) + 0.5 * high_energy)
    detected = rms >= settings.scream_energy_threshold and high_energy > 1.2

    return DetectionResult(
        detected=detected,
        event_type=EventType.SCREAM,
        confidence=round(score, 3),
        message=f"RMS={rms:.4f}, banda_aguda={high_energy:.2f}",
    )
