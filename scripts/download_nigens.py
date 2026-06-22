#!/usr/bin/env python3
"""
Baixa o NIGENS do Zenodo, extrai e (opcional) prepara + treina o modelo ML.

Uso:
  python -m scripts.download_nigens
  python -m scripts.download_nigens --prepare --train
  python -m scripts.download_nigens --skip-download   # só extrair/preparar
"""
from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.zenodo_download import download_zenodo_record  # noqa: E402

ZENODO_RECORD = "2535878"
ZIP_NAME = "NIGENS.zip"
NIGENS_DIR = ROOT / "data" / "datasets" / "nigens"
DOWNLOADS_DIR = NIGENS_DIR / "downloads"
RAW_DIR = NIGENS_DIR / "raw"
PREPARED_DIR = ROOT / "data" / "datasets" / "prepared"
# NIGENS tem 1017 WAV + anotações; exige ~800+ WAV para considerar extração OK
MIN_WAVS_FOR_COMPLETE = 800


def _count_wavs() -> int:
    if not RAW_DIR.exists():
        return 0
    return sum(1 for _ in RAW_DIR.rglob("*.wav"))


def _extraction_complete() -> bool:
    return _count_wavs() >= MIN_WAVS_FOR_COMPLETE


def extract_zip(zip_path: Path, dest: Path, clean: bool = False) -> None:
    """
    Extrai com `unzip` do sistema (suporta Deflate64 etc.).
    O zipfile do Python 3.9 falha em parte do NIGENS.zip.
    """
    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP não encontrado: {zip_path}")

    if clean and dest.exists():
        print(f"Removendo extração anterior em {dest}...")
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Extraindo {zip_path.name} → {dest} (pode demorar alguns minutos)...")

    unzip_bin = shutil.which("unzip")
    if unzip_bin:
        print("Usando unzip do sistema (compatível com este ZIP)...")
        result = subprocess.run(
            [unzip_bin, "-o", str(zip_path), "-d", str(dest)],
            check=False,
        )
        if result.returncode in (0, 1):
            # unzip retorna 1 se alguns arquivos falharam com aviso — ainda assim pode estar OK
            n = _count_wavs()
            print(f"\nExtração concluída. Arquivos .wav encontrados: {n}")
            if n >= MIN_WAVS_FOR_COMPLETE:
                return
            if result.returncode == 0:
                return
        print(f"unzip terminou com código {result.returncode}")

    if platform.system() == "Darwin":
        ditto = shutil.which("ditto")
        if ditto:
            print("Tentando ditto (macOS)...")
            if clean and dest.exists():
                shutil.rmtree(dest)
                dest.mkdir(parents=True)
            subprocess.run(
                ["ditto", "-x", "-k", str(zip_path), str(dest)],
                check=True,
            )
            print(f"Extração concluída. WAVs: {_count_wavs()}")
            return

    raise RuntimeError(
        "Não foi possível extrair o ZIP.\n"
        "No macOS, instale unzip (já vem no sistema) ou extraia manualmente:\n"
        f"  unzip -o '{zip_path}' -d '{dest}'"
    )


def run_prepare() -> None:
    from scripts.prepare_nigens import main as prepare_main

    prepare_main()


def run_train(epochs: int) -> None:
    cmd = [sys.executable, "-m", "scripts.train_sound_model", "--epochs", str(epochs)]
    print("Executando:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))


def main():
    parser = argparse.ArgumentParser(description="Download NIGENS (Zenodo) para treino ML")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Pula download (usa ZIP já em downloads/)",
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Pula extração se já houver WAV suficientes em raw/",
    )
    parser.add_argument(
        "--clean-extract",
        action="store_true",
        help="Apaga raw/ antes de extrair de novo",
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Organiza pastas grito/impacto/normal após extrair",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Treina o modelo após preparar",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--all",
        action="store_true",
        help="Equivalente a --prepare --train",
    )
    args = parser.parse_args()

    if args.all:
        args.prepare = True
        args.train = True

    zip_path = DOWNLOADS_DIR / ZIP_NAME

    print("=" * 60)
    print("NIGENS — base online para ML (grito, impacto, normal)")
    print("Fonte: https://zenodo.org/records/2535878")
    print("Licença: CC BY-NC-ND 4.0 (uso acadêmico não comercial)")
    print("=" * 60)

    if not args.skip_download:
        try:
            download_zenodo_record(ZENODO_RECORD, DOWNLOADS_DIR, filename=ZIP_NAME)
        except Exception as exc:
            print(f"\nErro no download: {exc}", file=sys.stderr)
            print(
                "\nAlternativa manual:\n"
                f"  1) Baixe {ZIP_NAME} em https://zenodo.org/records/2535878\n"
                f"  2) Coloque em {DOWNLOADS_DIR}/\n"
                "  3) Rode: python -m scripts.download_nigens --skip-download --prepare"
            )
            sys.exit(1)
    elif not zip_path.is_file():
        print(f"ZIP não encontrado: {zip_path}")
        print("Remova --skip-download ou baixe o arquivo manualmente.")
        sys.exit(1)

    if not args.skip_extract:
        n_wavs = _count_wavs()
        if not _extraction_complete() or args.clean_extract:
            if n_wavs > 0 and not args.clean_extract:
                print(
                    f"Extração incompleta detectada ({n_wavs} WAV). "
                    "Completando com unzip (-o)..."
                )
            extract_zip(zip_path, RAW_DIR, clean=args.clean_extract)
        else:
            print(f"Extração OK — {n_wavs} arquivos .wav em {RAW_DIR}")
    else:
        print("Extração ignorada (--skip-extract).")

    if args.prepare:
        print("\n--- Preparando classes ---")
        run_prepare()

    if args.train:
        if not PREPARED_DIR.exists() or not any(PREPARED_DIR.glob("*/")):
            print("Dados preparados não encontrados. Rode com --prepare primeiro.")
            sys.exit(1)
        print("\n--- Treinando modelo ---")
        run_train(args.epochs)

    print("\n" + "=" * 60)
    if args.train:
        print("Pronto! Reinicie o cliente: python -m client.main")
    elif args.prepare:
        print("Próximo passo: python -m scripts.train_sound_model")
    else:
        print("Próximo passo: python -m scripts.download_nigens --prepare --train")
    print("=" * 60)


if __name__ == "__main__":
    main()
