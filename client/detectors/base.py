from dataclasses import dataclass

from shared.models import EventType


@dataclass
class DetectionResult:
    detected: bool
    event_type: EventType
    confidence: float
    message: str = ""
