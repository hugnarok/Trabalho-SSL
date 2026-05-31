from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def ensure_faststart_mp4(path: Path) -> bool:
    """
    Move o atom 'moov' para o início do MP4 (faststart).
    Necessário para <video> no Chrome/Safari reproduzir sem baixar o arquivo inteiro antes.
    """
    try:
        import imageio_ffmpeg
    except ImportError:
        print(
            "[central] imageio-ffmpeg não instalado; vídeo pode não tocar no navegador.",
            file=sys.stderr,
        )
        return False

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = path.with_suffix(".faststart.mp4")
    try:
        result = subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(path),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(tmp),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"[central] ffmpeg faststart falhou: {result.stderr[:300]}", file=sys.stderr)
            return False
        tmp.replace(path)
        return True
    finally:
        if tmp.exists():
            tmp.unlink()
