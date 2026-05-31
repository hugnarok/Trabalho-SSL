from __future__ import annotations

from typing import Optional

import requests

from client.detectors.base import DetectionResult
from shared.config import settings
from shared.models import AlertResponse


def send_alert(
    result: DetectionResult,
    snapshot_jpeg: Optional[bytes],
    audio_wav: Optional[bytes],
    transcricao: str = "",
    video_mp4: Optional[bytes] = None,
) -> AlertResponse:
    url = f"{settings.central_url.rstrip('/')}/api/alerts"
    data = {
        "camera_id": settings.camera_label,
        "event_type": result.event_type.value,
        "confidence": str(result.confidence),
        "message": result.message,
        "transcricao": transcricao,
    }
    files = {}
    if snapshot_jpeg:
        files["snapshot"] = ("snapshot.jpg", snapshot_jpeg, "image/jpeg")
    if video_mp4:
        files["video"] = ("clip.mp4", video_mp4, "video/mp4")
    if audio_wav:
        files["audio"] = ("clip.wav", audio_wav, "audio/wav")

    headers = {}
    if settings.central_api_key:
        headers["X-API-Key"] = settings.central_api_key

    response = requests.post(
        url, data=data, files=files or None, headers=headers, timeout=60
    )
    response.raise_for_status()
    return AlertResponse(**response.json())
