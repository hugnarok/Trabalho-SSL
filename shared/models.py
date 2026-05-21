from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    SCREAM = "grito"
    IMPACT = "impacto"
    HELP_REQUEST = "socorro"
    AGGRESSION = "agressao"  # vídeo / multimodal (futuro)


class AlertPayload(BaseModel):
    camera_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: EventType
    confidence: float = Field(ge=0.0, le=1.0)
    message: str = ""


class AlertResponse(BaseModel):
    alert_id: str
    received_at: datetime
    snapshot_path: Optional[str] = None
    audio_path: Optional[str] = None
