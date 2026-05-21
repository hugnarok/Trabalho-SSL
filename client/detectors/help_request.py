"""
Pedido de socorro — transcrição do áudio (Vosk) + palavras-chave.
"""

import numpy as np

from client.detectors.base import DetectionResult
from client.transcription import find_keyword, transcribe_audio
from shared.config import settings
from shared.models import EventType

DEFAULT_KEYWORDS = (
    "socorro",
    "me ajuda",
    "ajuda",
    "polícia",
    "policia",
)


def _keywords() -> tuple[str, ...]:
    raw = settings.help_keywords.strip()
    if not raw:
        return DEFAULT_KEYWORDS
    return tuple(k.strip().lower() for k in raw.split(",") if k.strip())


def detect_help_request(
    audio: np.ndarray,
    sample_rate: int,
) -> DetectionResult:
    if not settings.help_keyword_enabled:
        return DetectionResult(
            detected=False,
            event_type=EventType.HELP_REQUEST,
            confidence=0.0,
            message="detecção de socorro desativada",
        )

    text = transcribe_audio(audio, sample_rate)
    keywords = _keywords()

    if not text:
        return DetectionResult(
            detected=False,
            event_type=EventType.HELP_REQUEST,
            confidence=0.0,
            message="sem fala detectada (instale modelo Vosk ou fale mais alto)",
        )

    matched = find_keyword(text, keywords)
    if matched:
        return DetectionResult(
            detected=True,
            event_type=EventType.HELP_REQUEST,
            confidence=0.92,
            message=f'transcrição: "{text}" | palavra-chave: {matched}',
        )

    return DetectionResult(
        detected=False,
        event_type=EventType.HELP_REQUEST,
        confidence=0.15,
        message=f'transcrição: "{text}"',
    )
