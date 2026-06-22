"""
Pedido de socorro — transcrição do áudio (Vosk) + palavras-chave.
"""

import numpy as np

from client.detectors.base import DetectionResult
from client.transcription import (
    find_keyword,
    keyword_word_confidence,
    transcribe_audio_detailed,
)
from shared.config import settings
from shared.models import EventType

# Frases de socorro — evite palavras soltas genéricas ("ajuda" sozinha gera falso positivo).
DEFAULT_KEYWORDS = (
    "socorro",
    "me ajuda",
    "me ajudem",
    "preciso de ajuda",
    "alguém me ajuda",
    "chama a polícia",
    "chama a policia",
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

    transcript = transcribe_audio_detailed(audio, sample_rate)
    text = transcript.text
    keywords = _keywords()

    if not text:
        return DetectionResult(
            detected=False,
            event_type=EventType.HELP_REQUEST,
            confidence=0.0,
            message="sem fala detectada (instale modelo Vosk ou fale mais alto)",
        )

    matched = find_keyword(text, keywords)
    if not matched:
        return DetectionResult(
            detected=False,
            event_type=EventType.HELP_REQUEST,
            confidence=0.0,
            message=f'fala sem pedido de socorro: "{text}"',
        )

    word_conf = keyword_word_confidence(transcript.words, matched)
    min_word = settings.help_min_word_confidence
    if word_conf is not None and word_conf < min_word:
        return DetectionResult(
            detected=False,
            event_type=EventType.HELP_REQUEST,
            confidence=round(word_conf, 3),
            message=(
                f'transcrição: "{text}" | chave "{matched}" com confiança baixa '
                f"({word_conf:.2f} < {min_word})"
            ),
        )

    confidence = 0.92
    if word_conf is not None:
        confidence = max(settings.help_min_confidence, min(0.98, word_conf))

    return DetectionResult(
        detected=True,
        event_type=EventType.HELP_REQUEST,
        confidence=confidence,
        message=f'transcrição: "{text}" | pedido de socorro: {matched}',
    )
