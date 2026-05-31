"""
Servidor da central — recebe alertas com snapshot e áudio.

Executar:
    uvicorn central.app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from central.media import media_file_response
from central.storage import clear_all_alerts, list_alerts, save_alert
from shared.config import settings
from shared.models import AlertResponse, EventType

app = FastAPI(
    title="Central de Alertas — SSL Trabalho Final",
    description="Recebe alertas de violência (grito, impacto, pedido de socorro).",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.now(timezone.utc).isoformat()}


@app.get("/api/alerts")
def get_alerts(limit: int = 50):
    return {"alerts": list_alerts(limit=limit)}


@app.delete("/api/alerts")
def delete_alerts():
    removed = clear_all_alerts()
    return {"ok": True, "removed": removed}


@app.get("/central", response_class=HTMLResponse)
def central_dashboard():
    """Interface mínima da central (atende requisito de UI do trabalho)."""
    return """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <title>Central de Alertas</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 2rem; background: #1a1a2e; color: #eee; }
    h1 { color: #e94560; }
    .toolbar { display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
    .btn-clear {
      background: #e94560; color: #fff; border: none; padding: 0.5rem 1rem;
      border-radius: 6px; cursor: pointer; font-size: 0.95rem;
    }
    .btn-clear:hover { background: #c73e54; }
    .btn-clear:disabled { opacity: 0.5; cursor: not-allowed; }
    .alert { background: #16213e; padding: 1rem; margin: 0.5rem 0; border-radius: 8px; }
    img, video { max-width: 480px; border-radius: 4px; display: block; margin-top: 0.5rem; }
    a { color: #7eb8da; }
  </style>
</head>
<body>
  <h1>Central de Alertas — Monitoramento</h1>
  <div class="toolbar">
    <p style="margin:0">Atualiza a cada 5s. <a href="/docs">API Swagger</a></p>
    <button type="button" class="btn-clear" id="btnClear">Limpar alertas</button>
  </div>
  <div id="list">Carregando...</div>
  <script>
    async function clearAlerts() {
      if (!confirm('Remover todos os alertas salvos? Esta ação não pode ser desfeita.')) return;
      const btn = document.getElementById('btnClear');
      btn.disabled = true;
      try {
        const r = await fetch('/api/alerts', { method: 'DELETE' });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || 'Falha ao limpar');
        await load();
        alert(data.removed > 0 ? `${data.removed} alerta(s) removido(s).` : 'Nenhum alerta para remover.');
      } catch (e) {
        alert('Erro ao limpar: ' + e.message);
      } finally {
        btn.disabled = false;
      }
    }
    document.getElementById('btnClear').addEventListener('click', clearAlerts);

    let listSignature = '';

    async function load() {
      const r = await fetch('/api/alerts');
      const data = await r.json();
      const el = document.getElementById('list');
      if (!data.alerts.length) {
        listSignature = '';
        el.innerHTML = '<p>Nenhum alerta ainda.</p>';
        return;
      }
      const signature = data.alerts.map(a => a.alert_id + (a.received_at || '')).join('|');
      if (signature === listSignature) return;
      listSignature = signature;

      el.innerHTML = data.alerts.map(a => `
        <div class="alert">
          <strong>${a.event_type}</strong> — câmera ${a.camera_id} — confiança ${a.confidence}
          <br><small>${a.received_at}</small>
          <p>${a.message || ''}</p>
          ${a.transcricao ? `<p><strong>Fala no momento:</strong> ${a.transcricao}</p>` : ''}
          ${a.video_path ? `<video controls playsinline preload="auto" src="/api/alerts/${a.alert_id}/video#t=0.1"></video><a href="/api/alerts/${a.alert_id}/video" target="_blank">Abrir vídeo</a>` : ''}
          ${a.snapshot_path ? `<img src="/api/alerts/${a.alert_id}/snapshot" alt="snapshot" />` : ''}
          ${a.audio_path ? `<audio controls preload="metadata" src="/api/alerts/${a.alert_id}/audio"></audio>` : ''}
        </div>
      `).join('');
    }
    load(); setInterval(load, 5000);
  </script>
</body>
</html>
"""


@app.api_route(
    "/api/alerts/{alert_id}/video",
    methods=["GET", "HEAD"],
)
def alert_video(alert_id: str):
    return media_file_response(alert_id, "video")


@app.get("/api/alerts/{alert_id}/audio")
def alert_audio(alert_id: str):
    return media_file_response(alert_id, "audio")


@app.get("/api/alerts/{alert_id}/snapshot")
def alert_snapshot(alert_id: str):
    return media_file_response(alert_id, "snapshot")


@app.get("/files/{file_path:path}")
def serve_file(file_path: str):
    """Compatibilidade com meta.json antigo."""
    from central.media import MEDIA_TYPES

    root = settings.alerts_dir.parent.parent.resolve()
    full = (root / file_path).resolve()
    if not str(full).startswith(str(root)) or not full.is_file():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    media_type = MEDIA_TYPES.get(full.suffix.lower(), "application/octet-stream")
    return FileResponse(full, media_type=media_type)


@app.post("/api/alerts", response_model=AlertResponse)
async def receive_alert(
    camera_id: str = Form(...),
    event_type: EventType = Form(...),
    confidence: float = Form(...),
    message: str = Form(""),
    transcricao: str = Form(""),
    snapshot: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
):
    snapshot_bytes = await snapshot.read() if snapshot else None
    video_bytes = await video.read() if video else None
    audio_bytes = await audio.read() if audio else None

    meta = save_alert(
        camera_id=camera_id,
        event_type=event_type,
        confidence=confidence,
        message=message,
        snapshot_bytes=snapshot_bytes,
        audio_bytes=audio_bytes,
        transcricao=transcricao,
        video_bytes=video_bytes,
    )
    return AlertResponse(
        alert_id=meta["alert_id"],
        received_at=datetime.fromisoformat(meta["received_at"]),
        snapshot_path=meta.get("snapshot_path"),
        audio_path=meta.get("audio_path"),
    )
