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

    # Detecção (limiares iniciais — calibrar com testes)
    scream_energy_threshold: float = 0.02
    impact_energy_threshold: float = 0.015
    help_keyword_enabled: bool = True
    transcription_enabled: bool = True
    vosk_model_path: Path = ROOT_DIR / "data" / "models" / "vosk-model-small-pt-0.3"
    help_keywords: str = "socorro,me ajuda,ajuda,polícia,policia"

    # Armazenamento
    alerts_dir: Path = ROOT_DIR / "data" / "alerts"


settings = Settings()
