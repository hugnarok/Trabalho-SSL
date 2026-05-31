from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

from fastapi import HTTPException
from fastapi.responses import FileResponse

from shared.config import settings

MEDIA_TYPES = {
    ".mp4": "video/mp4",
    ".wav": "audio/wav",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}


def _load_meta(alert_id: str) -> dict:
    meta_file = settings.alerts_dir / alert_id / "meta.json"
    if not meta_file.is_file():
        raise HTTPException(status_code=404, detail="Alerta não encontrado")
    return json.loads(meta_file.read_text(encoding="utf-8"))


def _resolve_media_file(alert_id: str, kind: str) -> Tuple[Path, str]:
    """
    kind: video | audio | snapshot
    """
    meta = _load_meta(alert_id)
    key = {
        "video": "video_path",
        "audio": "audio_path",
        "snapshot": "snapshot_path",
    }.get(kind)
    if not key:
        raise HTTPException(status_code=400, detail="Tipo de mídia inválido")

    rel = meta.get(key)
    if not rel:
        raise HTTPException(status_code=404, detail="Arquivo não disponível neste alerta")

    root = settings.alerts_dir.parent.parent.resolve()
    full = (root / rel).resolve()
    if not str(full).startswith(str(root)):
        raise HTTPException(status_code=403, detail="Caminho inválido")
    if not full.is_file():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado no disco")

    media_type = MEDIA_TYPES.get(full.suffix.lower(), "application/octet-stream")
    return full, media_type


def media_file_response(alert_id: str, kind: str) -> FileResponse:
    path, media_type = _resolve_media_file(alert_id, kind)
    return FileResponse(
        path,
        media_type=media_type,
        filename=path.name,
    )
