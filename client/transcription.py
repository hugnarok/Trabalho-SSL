from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import List, Optional

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


@dataclass
class TranscriptResult:
    text: str
    words: List[dict]


def transcribe_audio_detailed(
    audio: np.ndarray, sample_rate: int
) -> TranscriptResult:
    """ASR offline (Vosk) com texto e palavras (confiança por palavra, se disponível)."""
    if not settings.transcription_enabled or audio.size == 0:
        return TranscriptResult("", [])

    model = _load_model()
    if model is None:
        return TranscriptResult("", [])

    try:
        from vosk import KaldiRecognizer
    except ImportError:
        _log_once("[transcrição] Pacote 'vosk' não instalado.")
        return TranscriptResult("", [])

    recognizer = KaldiRecognizer(model, sample_rate)
    recognizer.SetWords(True)

    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
    block = sample_rate * 2 * 2
    for i in range(0, len(pcm), block):
        recognizer.AcceptWaveform(pcm[i : i + block])

    result = json.loads(recognizer.FinalResult())
    text = (result.get("text") or "").strip()
    words = list(result.get("result") or [])
    return TranscriptResult(text=text, words=words)


def transcribe_audio(audio: np.ndarray, sample_rate: int) -> str:
    """Converte o trecho de áudio em texto."""
    return transcribe_audio_detailed(audio, sample_rate).text


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def find_keyword(text: str, keywords: tuple[str, ...]) -> Optional[str]:
    """
    Casa frases inteiras ou palavras isoladas (evita 'ajuda' dentro de 'ajudar').
    Palavras mais longas têm prioridade ('me ajuda' antes de 'ajuda').
    """
    normalized = _normalize_text(text)
    if not normalized:
        return None

    ordered = sorted(
        (k.strip().lower() for k in keywords if k.strip()),
        key=len,
        reverse=True,
    )
    for kw in ordered:
        if " " in kw:
            if kw in normalized:
                return kw
        elif re.search(rf"(?<![\wáàâãéêíóôõúç]){re.escape(kw)}(?![\wáàâãéêíóôõúç])", normalized):
            return kw
    return None


def keyword_word_confidence(words: List[dict], keyword: str) -> Optional[float]:
    """Confiança Vosk da palavra que disparou a chave (frases usam o menor conf entre termos)."""
    if not words:
        return None

    parts = [p for p in keyword.lower().split() if p]
    confs: List[float] = []
    for part in parts:
        for w in words:
            token = (w.get("word") or "").lower()
            if token == part and "conf" in w:
                confs.append(float(w["conf"]))
                break
    if not confs:
        return None
    return min(confs)
