#!/usr/bin/env bash
# Baixa o modelo Vosk em português (pequeno) para transcrição offline.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_DIR="$ROOT/data/models"
MODEL_NAME="vosk-model-small-pt-0.3"
ZIP_URL="https://alphacephei.com/vosk/models/${MODEL_NAME}.zip"
TARGET="$MODEL_DIR/$MODEL_NAME"

mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

if [ -d "$TARGET" ]; then
  echo "Modelo já existe em: $TARGET"
  exit 0
fi

echo "Baixando $MODEL_NAME (~50 MB)..."
curl -L -o "${MODEL_NAME}.zip" "$ZIP_URL"
unzip -q "${MODEL_NAME}.zip"
rm "${MODEL_NAME}.zip"
echo "Pronto: $TARGET"
