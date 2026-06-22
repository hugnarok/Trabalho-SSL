#!/usr/bin/env python3
"""
Treina CNN leve (log-mel) para grito / impacto / normal.

Dados:
  - data/datasets/prepared/{grito,impacto,normal}/  (após prepare_nigens)
  - ou data/samples/ml/{grito,impacto,normal}/       (gravações próprias)

Uso:
  python -m scripts.train_sound_model
  python -m scripts.train_sound_model --epochs 25 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from client.ml.audio_io import load_wav  # noqa: E402
from client.ml.features import log_mel_spectrogram, resample_audio  # noqa: E402
from client.ml.model_arch import build_cnn  # noqa: E402
from shared.config import settings  # noqa: E402

LABELS = ["normal", "grito", "impacto"]
DATA_DIRS = [
    ROOT / "data" / "datasets" / "prepared",
    ROOT / "data" / "samples" / "ml",
]


def collect_dataset() -> list[tuple[Path, int]]:
    items: list[tuple[Path, int]] = []
    for base in DATA_DIRS:
        if not base.is_dir():
            continue
        for label_idx, label in enumerate(LABELS):
            folder = base / label
            if not folder.is_dir():
                continue
            for wav in folder.glob("*.wav"):
                items.append((wav, label_idx))
    return items


def split_data(
    items: list[tuple[Path, int]],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    random.shuffle(items)
    n = len(items)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test = items[:n_test]
    val = items[n_test : n_test + n_val]
    train = items[n_test + n_val :]
    return train, val, test


def build_tensor(path: Path, n_frames: int = 128) -> np.ndarray:
    audio, sr = load_wav(path)
    audio = resample_audio(audio, sr, settings.sample_rate)
    max_len = int(settings.sample_rate * settings.audio_chunk_seconds)
    if len(audio) > max_len:
        audio = audio[-max_len:]
    return log_mel_spectrogram(audio, settings.sample_rate, n_frames)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("Instale PyTorch: pip install torch")
        sys.exit(1)

    items = collect_dataset()
    if len(items) < 30:
        print("Poucos arquivos para treino (< 30).")
        print("Rode: python -m scripts.prepare_nigens")
        print("Ou coloque WAVs em data/samples/ml/{grito,impacto,normal}/")
        sys.exit(1)

    train_items, val_items, test_items = split_data(items)
    print(f"Amostras: train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    def to_tensors(subset):
        xs, ys = [], []
        for path, label in subset:
            xs.append(build_tensor(path))
            ys.append(label)
        return (
            torch.from_numpy(np.stack(xs)),
            torch.tensor(ys, dtype=torch.long),
        )

    print("Extraindo espectrogramas (pode demorar)...")
    x_train, y_train = to_tensors(train_items)
    x_val, y_val = to_tensors(val_items)
    x_test, y_test = to_tensors(test_items)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=args.batch_size)

    device = torch.device("cpu")
    model = build_cnn(len(LABELS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    def run_epoch(loader, train_mode: bool) -> tuple[float, float]:
        model.train(train_mode)
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.set_grad_enabled(train_mode):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                if train_mode:
                    optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                if train_mode:
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * len(yb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        return total_loss / max(total, 1), correct / max(total, 1)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(train_loader, True)
        va_loss, va_acc = run_epoch(val_loader, False)
        print(
            f"Época {epoch:02d} | loss train={tr_loss:.4f} acc={tr_acc:.3f} "
            f"| val loss={va_loss:.4f} acc={va_acc:.3f}"
        )

    te_loss, te_acc = run_epoch(test_loader, False)
    print(f"\nTeste: loss={te_loss:.4f} accuracy={te_acc:.3f}")

    out_dir = settings.ml_model_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), settings.ml_model_path)
    meta = {
        "labels": LABELS,
        "sample_rate": settings.sample_rate,
        "n_mels": 64,
        "n_frames": 128,
        "test_accuracy": te_acc,
    }
    meta_path = settings.ml_model_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nModelo salvo: {settings.ml_model_path}")
    print(f"Meta: {meta_path}")
    print("Reinicie o cliente para carregar o ML.")


if __name__ == "__main__":
    random.seed(42)
    main()
