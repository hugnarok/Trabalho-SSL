from __future__ import annotations

import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from shared.config import settings
from shared.models import EventType


def ensure_alerts_dir() -> Path:
    path = settings.alerts_dir
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_alert(
    camera_id: str,
    event_type: EventType,
    confidence: float,
    message: str,
    snapshot_bytes: Optional[bytes],
    audio_bytes: Optional[bytes],
    transcricao: str = "",
) -> dict:
    alert_id = str(uuid.uuid4())
    base = ensure_alerts_dir() / alert_id
    base.mkdir(parents=True, exist_ok=True)

    snapshot_path: Optional[str] = None
    audio_path: Optional[str] = None

    if snapshot_bytes:
        snap_file = base / "snapshot.jpg"
        snap_file.write_bytes(snapshot_bytes)
        snapshot_path = str(snap_file.relative_to(settings.alerts_dir.parent.parent))

    if audio_bytes:
        audio_file = base / "clip.wav"
        audio_file.write_bytes(audio_bytes)
        audio_path = str(audio_file.relative_to(settings.alerts_dir.parent.parent))

    meta = {
        "alert_id": alert_id,
        "camera_id": camera_id,
        "event_type": event_type.value,
        "confidence": confidence,
        "message": message,
        "transcricao": transcricao.strip(),
        "received_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_path": snapshot_path,
        "audio_path": audio_path,
    }
    (base / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def list_alerts(limit: int = 50) -> List[dict]:
    root = ensure_alerts_dir()
    metas: List[dict] = []
    for folder in sorted(root.iterdir(), reverse=True):
        if not folder.is_dir():
            continue
        meta_file = folder / "meta.json"
        if meta_file.exists():
            metas.append(json.loads(meta_file.read_text(encoding="utf-8")))
        if len(metas) >= limit:
            break
    return metas


def clear_all_alerts() -> int:
    """Remove todas as pastas de alerta. Retorna quantidade removida."""
    root = ensure_alerts_dir()
    removed = 0
    for folder in root.iterdir():
        if folder.is_dir():
            shutil.rmtree(folder)
            removed += 1
    return removed
