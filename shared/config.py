from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Central
    central_url: str = "http://127.0.0.1:8000"
    central_api_key: str = ""

    # Cliente
    camera_id: int = 0
    camera_label: str = "cam-01"
    sample_rate: int = 16_000
    audio_chunk_seconds: float = 5.0
    detection_interval_seconds: float = 0.5
    display_fps: int = 30
    preview_width: int = 640
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    alert_cooldown_seconds: float = 10.0

    # Anti falso positivo
    detection_confirmations_required: int = 2
    min_alert_confidence: float = 0.55
    scream_energy_threshold: float = 0.025
    scream_high_band_ratio: float = 1.35
    scream_band_low_hz: float = 2000.0
    scream_band_high_hz: float = 8000.0
    impact_energy_threshold: float = 0.018
    impact_peak_ratio: float = 5.0
    impact_window_seconds: float = 0.05

    help_keyword_enabled: bool = True
    transcription_enabled: bool = True
    vosk_model_path: Path = ROOT_DIR / "data" / "models" / "vosk-model-small-pt-0.3"
    help_keywords: str = "socorro,me ajuda,ajuda,polícia,policia"

    # Evidência no alerta (vídeo em vez de foto por padrão)
    alert_save_video: bool = True
    alert_video_seconds: float = 5.0
    video_buffer_seconds: float = 6.0
    alert_save_snapshot: bool = False

    # Armazenamento
    alerts_dir: Path = ROOT_DIR / "data" / "alerts"


settings = Settings()
