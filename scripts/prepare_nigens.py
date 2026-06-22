#!/usr/bin/env python3
"""
Organiza o NIGENS em pastas para treino: grito | impacto | normal.

Normalmente chamado por: python -m scripts.download_nigens --prepare
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NIGENS_DIR = ROOT / "data" / "datasets" / "nigens"
RAW = NIGENS_DIR / "raw"
OUT = ROOT / "data" / "datasets" / "prepared"

GRITO_KEYS = ("female_scream", "male_scream", "femalescream", "malescream", "scream")
IMPACTO_KEYS = ("crash", "knocking", "knock")
NORMAL_KEYS = (
    "footsteps",
    "female_speech",
    "male_speech",
    "femalespeech",
    "malespeech",
    "piano",
    "general",
    "engine",
    "running_engine",
    "rain",
    "wind",
    "alarm",
    "phone",
    "fire",
    "dog",
    "baby",
)


def _norm(name: str) -> str:
    """Nome de pasta/chave sem separadores (femaleSpeech == female_speech)."""
    return "".join(c for c in name.lower() if c.isalnum())


def _match_folder(name: str, keys: tuple[str, ...]) -> bool:
    folder = _norm(name)
    return any(_norm(k) in folder or folder in _norm(k) for k in keys)


def _discover_audio_roots() -> Path:
    """Encontra a raiz onde estão as pastas de classes (ex.: female_scream)."""
    if not RAW.exists():
        return RAW

    # Pastas de classe diretamente em raw/
    for child in RAW.iterdir():
        if child.is_dir() and _match_folder(child.name, GRITO_KEYS + IMPACTO_KEYS + NORMAL_KEYS):
            return RAW

    # Um nível abaixo (ex.: raw/NIGENS/female_scream)
    for child in RAW.iterdir():
        if not child.is_dir():
            continue
        for sub in child.iterdir():
            if sub.is_dir() and _match_folder(
                sub.name, GRITO_KEYS + IMPACTO_KEYS + NORMAL_KEYS
            ):
                return child

    return RAW


def _class_folders(base: Path) -> list[Path]:
    folders = []
    if not base.is_dir():
        return folders
    for path in sorted(base.rglob("*")):
        if not path.is_dir():
            continue
        if list(path.glob("*.wav")):
            folders.append(path)
    # Preferir pastas folha (com wavs, sem subpastas com wavs)
    leaf = []
    for f in folders:
        if not any(sub.is_dir() and list(sub.glob("*.wav")) for sub in f.iterdir()):
            leaf.append(f)
    return leaf if leaf else folders


def main():
    base = _discover_audio_roots()
    if not base.exists():
        print(f"Pasta não encontrada: {RAW}")
        print("Rode primeiro: python -m scripts.download_nigens")
        sys.exit(1)

    for label in ("grito", "impacto", "normal"):
        dest = OUT / label
        dest.mkdir(parents=True, exist_ok=True)
        for old in dest.glob("*.wav"):
            old.unlink()

    counts = {"grito": 0, "impacto": 0, "normal": 0}
    skipped: list[tuple[str, int]] = []
    seen_names = set()

    for folder in _class_folders(base):
        name = folder.name
        if name in seen_names:
            continue
        wavs = sorted(folder.glob("*.wav"))
        if not wavs:
            continue

        target = None
        if _match_folder(name, GRITO_KEYS):
            target = "grito"
        elif _match_folder(name, IMPACTO_KEYS):
            target = "impacto"
        elif _match_folder(name, NORMAL_KEYS):
            target = "normal"

        if not target:
            skipped.append((name, len(wavs)))
            continue

        seen_names.add(name)
        dest = OUT / target
        for i, wav in enumerate(wavs):
            out_name = f"{name}_{i:04d}{wav.suffix}"
            shutil.copy2(wav, dest / out_name)
            counts[target] += 1

    print(f"Base escaneada: {base}")
    print(f"Saída: {OUT}")
    for k, v in counts.items():
        print(f"  {k}: {v} arquivos")

    if skipped:
        print("\nPastas ignoradas (não mapeadas para grito/impacto/normal):")
        for name, n in skipped:
            print(f"  {name}: {n} arquivos")

    total = sum(counts.values())
    if total < 30:
        print(
            "\nPoucos arquivos encontrados. Estrutura esperada após extração:\n"
            "  raw/female_scream/*.wav\n"
            "  raw/crash/*.wav\n"
            "Rode: python -m scripts.download_nigens"
        )
        sys.exit(1)

    print(f"\nTotal: {total} clipes prontos para treino.")


if __name__ == "__main__":
    main()
