from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from client.ml.features import log_mel_spectrogram
from shared.config import settings
from shared.models import EventType

_CLASSIFIER: Optional["SoundClassifier"] = None
_LOAD_ERROR_LOGGED = False

# Mapeamento rótulos do treino -> EventType do sistema
LABEL_TO_EVENT = {
    "grito": EventType.SCREAM,
    "impacto": EventType.IMPACT,
    "normal": None,
}


class SoundClassifier:
    def __init__(self, model_path: Path, meta_path: Path):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Instale PyTorch: pip install torch") from exc

        from client.ml.model_arch import build_cnn

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.labels: List[str] = meta["labels"]
        self.sample_rate = int(meta.get("sample_rate", 16000))
        self.n_mels = int(meta.get("n_mels", 64))
        self.n_frames = int(meta.get("n_frames", 128))

        self._torch = torch
        self._device = torch.device("cpu")
        self._model = build_cnn(len(self.labels), self.n_mels, self.n_frames)
        state = torch.load(model_path, map_location=self._device)
        self._model.load_state_dict(state)
        self._model.eval()

    def predict(self, audio: np.ndarray, sample_rate: int) -> Tuple[str, float, Dict[str, float]]:
        spec = log_mel_spectrogram(audio, sample_rate, self.n_frames)
        x = self._torch.from_numpy(spec).unsqueeze(0).to(self._device)
        with self._torch.no_grad():
            logits = self._model(x)
            probs = self._torch.softmax(logits, dim=1).cpu().numpy()[0]

        idx = int(np.argmax(probs))
        label = self.labels[idx]
        scores = {lbl: float(probs[i]) for i, lbl in enumerate(self.labels)}
        return label, float(probs[idx]), scores


def is_model_available() -> bool:
    return _get_classifier() is not None


def _get_classifier() -> Optional[SoundClassifier]:
    global _CLASSIFIER, _LOAD_ERROR_LOGGED
    if _CLASSIFIER is not None:
        return _CLASSIFIER

    model_path = settings.ml_model_path
    meta_path = model_path.with_suffix(".json")
    if not model_path.is_file() or not meta_path.is_file():
        if not _LOAD_ERROR_LOGGED:
            print(
                f"[ML] Modelo não encontrado em {model_path}\n"
                "  Treine com: python -m scripts.train_sound_model",
                file=sys.stderr,
            )
            _LOAD_ERROR_LOGGED = True
        return None

    try:
        _CLASSIFIER = SoundClassifier(model_path, meta_path)
        return _CLASSIFIER
    except Exception as exc:
        if not _LOAD_ERROR_LOGGED:
            print(f"[ML] Falha ao carregar modelo: {exc}", file=sys.stderr)
            _LOAD_ERROR_LOGGED = True
        return None


def reset_classifier_cache() -> None:
    global _CLASSIFIER, _LOAD_ERROR_LOGGED
    _CLASSIFIER = None
    _LOAD_ERROR_LOGGED = False
