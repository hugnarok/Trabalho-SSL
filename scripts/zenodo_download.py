"""
Download de arquivos do Zenodo via API REST (Python 3.9+).
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional


def fetch_record_files(record_id: str) -> List[Dict[str, Any]]:
    url = f"https://zenodo.org/api/records/{record_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "TrabalhoFinal-SSL/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    files = data.get("files", [])
    if not files:
        raise RuntimeError(f"Nenhum arquivo no registro Zenodo {record_id}")
    return files


def _format_size(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} TB"


def download_file(
    url: str,
    dest: Path,
    expected_size: Optional[int] = None,
    chunk_size: int = 1024 * 1024,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    resume_from = 0
    if part.exists():
        resume_from = part.stat().st_size
    if dest.exists() and expected_size and dest.stat().st_size >= expected_size:
        print(f"Já baixado: {dest} ({_format_size(dest.stat().st_size)})")
        return dest

    if dest.exists() and not expected_size:
        print(f"Já existe: {dest}")
        return dest

    headers = {"User-Agent": "TrabalhoFinal-SSL/1.0"}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"
        print(f"Retomando download em {_format_size(resume_from)}...")

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = expected_size
            if total is None and "Content-Length" in resp.headers:
                total = int(resp.headers["Content-Length"]) + resume_from

            mode = "ab" if resume_from else "wb"
            downloaded = resume_from
            last_print = time.time()

            with open(part, mode) as out:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    now = time.time()
                    if now - last_print >= 2.0:
                        if total:
                            pct = 100 * downloaded / total
                            print(
                                f"\r  {_format_size(downloaded)} / {_format_size(total)} "
                                f"({pct:.1f}%)",
                                end="",
                                flush=True,
                            )
                        else:
                            print(f"\r  {_format_size(downloaded)}", end="", flush=True)
                        last_print = now

            print()

    except urllib.error.HTTPError as exc:
        if exc.code == 416 and part.exists():
            print("Download já completo (resume).")
        else:
            raise

    part.replace(dest)
    print(f"Salvo: {dest} ({_format_size(dest.stat().st_size)})")
    return dest


def download_zenodo_record(
    record_id: str,
    output_dir: Path,
    filename: Optional[str] = None,
) -> Path:
    """
    Baixa um arquivo do registro Zenodo (usa o primeiro ou o que bater filename).
    """
    files = fetch_record_files(record_id)
    chosen = None
    for f in files:
        key = f.get("key", "")
        if filename is None or key == filename:
            chosen = f
            break
    if chosen is None:
        names = [f.get("key") for f in files]
        raise RuntimeError(f"Arquivo '{filename}' não encontrado. Disponíveis: {names}")

    key = chosen["key"]
    size = int(chosen.get("size", 0))
    url = chosen["links"]["self"]
    dest = output_dir / key
    print(f"Zenodo {record_id}: {key} ({_format_size(size)})")
    return download_file(url, dest, expected_size=size or None)
