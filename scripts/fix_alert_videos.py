#!/usr/bin/env python3
"""Corrige MP4s antigos (moov no final) para tocar no navegador."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from central.video_utils import ensure_faststart_mp4
from shared.config import settings


def main():
    root = settings.alerts_dir
    fixed = 0
    for folder in root.iterdir():
        if not folder.is_dir():
            continue
        clip = folder / "clip.mp4"
        if clip.is_file():
            data = clip.read_bytes()
            moov_at = data.find(b"moov")
            if moov_at > 1000 and moov_at > len(data) // 2:
                if ensure_faststart_mp4(clip):
                    print(f"OK: {folder.name}")
                    fixed += 1
            else:
                print(f"já OK: {folder.name}")
    print(f"\n{fixed} vídeo(s) corrigido(s).")


if __name__ == "__main__":
    main()
