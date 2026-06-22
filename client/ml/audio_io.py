"""Carrega WAV/FLAC do dataset (PCM, float32, ADPCM, etc.)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Retorna áudio mono float32 em [-1, 1] e taxa de amostragem."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sr)
