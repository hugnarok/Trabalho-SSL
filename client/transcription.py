from __future__ import annotations

import json
import sys
from typing import Optional

import numpy as np

from shared.config import settings

_model = None
_model_error_logged = False


def _log_once(message: str) -> None:
    global _model_error_logged
    if not _model_error_logged:
        print(message, file=sys.stderr)
        _model_error_logged = True


def _load_model():
    global _model
    if _model is not None:
        return _model

    model_path = settings.vosk_model_path
    if not model_path.exists():
        _log_once(
            f"[transcrição] Modelo Vosk não encontrado em: {model_path}\n"
            "  Execute: bash scripts/setup_vosk.sh"
        )
        return None

    try:
        from vosk import Model
    except ImportError:
        _log_once("[transcrição] Pacote 'vosk' não instalado. Rode: pip install vosk")
        return None

    _model = Model(str(model_path))
    return _model


def transcribe_audio(audio: np.ndarray, sample_rate: int) -> str:
    """
    Converte o trecho de áudio (onda amostrada) em texto — ASR offline (Vosk).
    """
    if not settings.transcription_enabled:
        return ""

    if audio.size == 0:
        return ""

    model = _load_model()
    if model is None:
        return ""

    try:
        from vosk import KaldiRecognizer
    except ImportError:
        _log_once("[transcrição] Pacote 'vosk' não instalado.")
        return ""

    recognizer = KaldiRecognizer(model, sample_rate)
    recognizer.SetWords(False)

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    # Processa em blocos (~1 s) para áudios longos (ex.: 5 s)
    block = sample_rate * 2 * 2  # bytes: 1 s de int16 mono
    for i in range(0, len(pcm), block):
        recognizer.AcceptWaveform(pcm[i : i + block])

    result = json.loads(recognizer.FinalResult())
    text = (result.get("text") or "").strip()
    return text


def find_keyword(text: str, keywords: tuple[str, ...]) -> Optional[str]:
    normalized = text.lower()
    for kw in keywords:
        if kw in normalized:
            return kw
    return None
