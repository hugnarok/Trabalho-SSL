from __future__ import annotations

import io
import os
import platform
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np
import sounddevice as sd

from shared.config import settings


class AudioRingBuffer:
    """Buffer circular — evita concatenação a cada callback (mais leve)."""

    def __init__(
        self,
        sample_rate: int,
        chunk_seconds: float,
        max_buffer_seconds: float = 12.0,
    ):
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_seconds)
        self._max_samples = int(sample_rate * max_buffer_seconds)
        self._buf = np.zeros(self._max_samples, dtype=np.float32)
        self._write = 0
        self._filled = 0
        self._lock = threading.Lock()
        self._stream: Optional[sd.InputStream] = None

    def _callback(self, indata, _frames, _time, status):
        if status:
            print(f"[áudio] {status}", file=sys.stderr)
        chunk = indata[:, 0]
        n = len(chunk)
        with self._lock:
            if n >= self._max_samples:
                self._buf[:] = chunk[-self._max_samples :]
                self._write = 0
                self._filled = self._max_samples
                return
            space_end = self._max_samples - self._write
            if n <= space_end:
                self._buf[self._write : self._write + n] = chunk
                self._write = (self._write + n) % self._max_samples
            else:
                self._buf[self._write :] = chunk[:space_end]
                rest = n - space_end
                self._buf[:rest] = chunk[space_end:]
                self._write = rest
            self._filled = min(self._filled + n, self._max_samples)

    def start(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=int(self.sample_rate * 0.05),
            callback=self._callback,
        )
        self._stream.start()

    def get_chunk(self) -> np.ndarray:
        with self._lock:
            if self._filled == 0:
                return np.zeros(self.chunk_samples, dtype=np.float32)
            if self._filled < self.chunk_samples:
                out = np.zeros(self.chunk_samples, dtype=np.float32)
                out[-self._filled :] = self._read_last(self._filled)
                return out
            return self._read_last(self.chunk_samples)

    def _read_last(self, count: int) -> np.ndarray:
        start = (self._write - count) % self._max_samples
        if start + count <= self._max_samples:
            return self._buf[start : start + count].copy()
        part1 = self._max_samples - start
        return np.concatenate([self._buf[start:], self._buf[: count - part1]])

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


def _frames_to_mp4_bytes_opencv(frames: List[np.ndarray], fps: int) -> Optional[bytes]:
    """Fallback se imageio não estiver disponível."""
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    fps = max(5, min(fps, 60))
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    path = tmp.name
    tmp.close()
    try:
        codec = "avc1" if platform.system() == "Darwin" else "mp4v"
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*codec), float(fps), (w, h)
        )
        if not writer.isOpened():
            return None
        for frame in frames:
            writer.write(frame)
        writer.release()
        data = open(path, "rb").read()
        return data if len(data) > 100 else None
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _frames_to_mp4_bytes(frames: List[np.ndarray], fps: int) -> Optional[bytes]:
    """H.264 (yuv420p) — compatível com Chrome/Safari no painel da central."""
    if not frames:
        return None
    fps = max(5, min(fps, 60))
    try:
        import imageio.v3 as iio
    except ImportError:
        return _frames_to_mp4_bytes_opencv(frames, fps)

    rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    path = tempfile.mktemp(suffix=".mp4")
    try:
        iio.imwrite(
            path,
            rgb,
            fps=fps,
            codec="libx264",
            extension=".mp4",
            ffmpeg_params=["-movflags", "+faststart", "-pix_fmt", "yuv420p"],
        )
        data = open(path, "rb").read()
        return data if len(data) > 100 else None
    except Exception as exc:
        print(f"[vídeo] imageio falhou ({exc}); usando fallback OpenCV.", file=sys.stderr)
        return _frames_to_mp4_bytes_opencv(frames, fps)
    finally:
        if os.path.exists(path):
            os.unlink(path)


class _VideoReaderThread(threading.Thread):
    """Lê a câmera e mantém buffer circular para clip de vídeo no alerta."""

    def __init__(self, cap: cv2.VideoCapture, buffer_seconds: float, fps: int):
        super().__init__(daemon=True, name="video-capture")
        self._cap = cap
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._frames: deque = deque(maxlen=max(30, int(buffer_seconds * fps) + 10))
        self._running = False

    def run(self) -> None:
        self._running = True
        while self._running:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._latest = frame
                    self._frames.append(frame.copy())
            else:
                time.sleep(0.01)

    def stop(self) -> None:
        self._running = False

    def get_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest

    def get_latest_copy(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def get_clip_mp4_bytes(self, seconds: float, fps: int) -> Optional[bytes]:
        with self._lock:
            if not self._frames:
                return None
            n = min(len(self._frames), max(1, int(seconds * fps)))
            clip = list(self._frames)[-n:]
        return _frames_to_mp4_bytes(clip, fps)


class StreamCapture:
    """Webcam (thread) + microfone (callback) desacoplados do preview."""

    def __init__(
        self,
        camera_index: Optional[int] = None,
        sample_rate: Optional[int] = None,
        chunk_seconds: Optional[float] = None,
    ):
        self.camera_index = camera_index if camera_index is not None else settings.camera_id
        self.sample_rate = sample_rate or settings.sample_rate
        self.chunk_seconds = chunk_seconds or settings.audio_chunk_seconds
        self.chunk_samples = int(self.sample_rate * self.chunk_seconds)
        self._cap: Optional[cv2.VideoCapture] = None
        self._video_thread: Optional[_VideoReaderThread] = None
        self._audio: Optional[AudioRingBuffer] = None
        self._preview_size: Optional[Tuple[int, int]] = None

    @staticmethod
    def _open_camera(index: int) -> cv2.VideoCapture:
        if platform.system() == "Darwin":
            cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(index)

    def open(self) -> None:
        self._cap = self._open_camera(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a câmera índice {self.camera_index}. "
                "Verifique permissões e se outro app está usando a câmera."
            )

        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if settings.camera_width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.camera_width)
        if settings.camera_height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.camera_height)
        if settings.camera_fps > 0:
            self._cap.set(cv2.CAP_PROP_FPS, settings.camera_fps)

        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        pw = settings.preview_width
        if pw > 0 and w > 0:
            ph = max(1, int(h * (pw / w)))
            self._preview_size = (pw, ph)
        else:
            self._preview_size = (w, h) if w > 0 and h > 0 else None

        fps = max(5, settings.camera_fps)
        self._video_thread = _VideoReaderThread(
            self._cap,
            buffer_seconds=settings.video_buffer_seconds,
            fps=fps,
        )
        self._video_thread.start()

        self._audio = AudioRingBuffer(self.sample_rate, self.chunk_seconds)
        self._audio.start()

    def get_latest_frame(self) -> Optional[np.ndarray]:
        if self._video_thread is None:
            return None
        return self._video_thread.get_latest()

    def get_snapshot_frame(self) -> Optional[np.ndarray]:
        if self._video_thread is None:
            return None
        return self._video_thread.get_latest_copy()

    def get_alert_video_bytes(self) -> Optional[bytes]:
        if self._video_thread is None:
            return None
        fps = max(5, settings.camera_fps)
        return self._video_thread.get_clip_mp4_bytes(
            settings.alert_video_seconds,
            fps,
        )

    def make_preview(self, frame: np.ndarray, status_text: str, alert_active: bool) -> np.ndarray:
        out = frame.copy()
        if self._preview_size and (
            out.shape[1] != self._preview_size[0]
            or out.shape[0] != self._preview_size[1]
        ):
            out = cv2.resize(
                out,
                self._preview_size,
                interpolation=cv2.INTER_AREA,
            )
        color = (0, 0, 255) if alert_active else (0, 255, 0)
        cv2.putText(
            out,
            f"Status: {status_text}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )
        return out

    def read_audio_chunk(self) -> np.ndarray:
        if self._audio is None:
            return np.zeros(self.chunk_samples, dtype=np.float32)
        return self._audio.get_chunk()

    def frame_to_jpeg_bytes(self, frame: np.ndarray) -> bytes:
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise RuntimeError("Falha ao codificar frame JPEG")
        return buf.tobytes()

    @staticmethod
    def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
        audio = np.clip(audio, -1.0, 1.0)
        pcm = (audio * 32767).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return buffer.getvalue()

    def release(self) -> None:
        if self._video_thread is not None:
            self._video_thread.stop()
            self._video_thread.join(timeout=2.0)
            self._video_thread = None
        if self._audio is not None:
            self._audio.stop()
            self._audio = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        cv2.destroyAllWindows()
