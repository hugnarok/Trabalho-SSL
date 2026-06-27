"""
Cliente de monitoramento — câmera + microfone + detecção + envio à central.

Executar:
    python -m client.main
"""

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from client.alert_sender import send_alert
from client.capture import StreamCapture
from client.detection_worker import DetectionWorker
from client.transcription import transcribe_audio
from client.waveform_overlay import danger_level_from_state, render_waveform_overlay
from shared.config import settings


def _send_alert_async(
    capture: StreamCapture,
    worker: DetectionWorker,
    last_alert_at: float,
) -> float:
    state = worker.snapshot()
    if not state.best or state.last_audio is None:
        return last_alert_at
    now = time.time()
    if (now - last_alert_at) < settings.alert_cooldown_seconds:
        return last_alert_at

    snapshot = None
    video_mp4 = None
    if settings.alert_save_video:
        video_mp4 = capture.get_alert_video_bytes()
        if not video_mp4:
            print("  [aviso] Clip de vídeo vazio; verifique buffer da câmera.")
    if settings.alert_save_snapshot:
        frame = capture.get_snapshot_frame()
        if frame is not None:
            snapshot = capture.frame_to_jpeg_bytes(frame)

    wav = capture.audio_to_wav_bytes(state.last_audio, capture.sample_rate)

    transcricao = ""
    if settings.transcription_enabled:
        transcricao = transcribe_audio(state.last_audio, capture.sample_rate)

    print(
        f"[ALERTA] {state.best.event_type.value} conf={state.best.confidence} — {state.best.message}"
    )
    if transcricao:
        print(f'  Fala transcrita: "{transcricao}"')
    try:
        resp = send_alert(
            state.best, snapshot, wav, transcricao=transcricao, video_mp4=video_mp4
        )
        print(f"  -> Central OK: alert_id={resp.alert_id}")
        return now
    except Exception as exc:
        print(f"  -> Falha ao enviar à central: {exc}")
        return last_alert_at


def run():
    capture = StreamCapture()
    capture.open()
    worker = DetectionWorker(capture)
    worker.start()

    last_alert_at = 0.0
    alert_lock = threading.Lock()
    had_active_alert = False
    frame_interval = 1.0 / max(1, settings.display_fps)
    window = "Monitoramento de Ondas Sonoras"

    print("Cliente iniciado. Central:", settings.central_url)
    print(
        f"Preview ~{settings.display_fps} FPS | "
        f"câmera {settings.camera_width}x{settings.camera_height} | "
        f"detecção a cada {settings.detection_interval_seconds}s"
    )
    print("Pressione 'q' na janela de vídeo para encerrar.\n")

    try:
        while True:
            loop_start = time.perf_counter()
            frame = capture.get_latest_frame()
            state = worker.snapshot()
            active = state.best is not None
            fire_alert = active and not had_active_alert
            had_active_alert = active

            if frame is not None:
                preview = capture.make_preview(
                    frame,
                    state.status,
                    alert_active=state.candidate is not None or state.best is not None,
                )
                audio_live = capture.read_audio_chunk()
                level = danger_level_from_state(state.candidate, state.best)
                wave = render_waveform_overlay(
                    audio_live,
                    preview.shape[1],
                    capture.sample_rate,
                    danger_level=level,
                )
                combined = np.vstack([preview, wave])
                cv2.imshow(window, combined)

            delay_ms = max(1, int(frame_interval * 1000))
            key = cv2.waitKey(delay_ms) & 0xFF
            if key == ord("q"):
                break

            if fire_alert:

                def _dispatch():
                    nonlocal last_alert_at
                    with alert_lock:
                        last_alert_at = _send_alert_async(
                            capture, worker, last_alert_at
                        )

                threading.Thread(target=_dispatch, daemon=True).start()

            elapsed = time.perf_counter() - loop_start
            slack = frame_interval - elapsed
            if slack > 0.001:
                time.sleep(slack)

    finally:
        worker.stop()
        worker.join(timeout=2.0)
        capture.release()


if __name__ == "__main__":
    run()
