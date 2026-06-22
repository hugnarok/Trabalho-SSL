"""
Analisa um arquivo WAV e imprime métricas dos detectores (sem câmera).

Uso:
    python -m client.calibrate caminho/para/audio.wav
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client.ml.audio_io import load_wav
from client.detectors.help_request import detect_help_request
from client.detectors.impact import detect_impact
from client.detectors.scream import detect_scream
from shared.config import settings


def main():
    if len(sys.argv) < 2:
        print("Uso: python -m client.calibrate <arquivo.wav>")
        sys.exit(1)

    path = Path(sys.argv[1]).expanduser()
    if not path.is_file():
        print(f"Arquivo não encontrado: {path}")
        print("\nUse um caminho real para um .wav, por exemplo:")
        print("  python -m client.calibrate data/datasets/prepared/grito/femaleScream_0000.wav")
        print("  python -m client.calibrate data/samples/ml/grito/meu_teste.wav")
        sys.exit(1)

    audio, rate = load_wav(path)
    print(f"Arquivo: {path}")
    print(f"Amostras: {len(audio)} | Taxa: {rate} Hz | Duração: {len(audio)/rate:.2f} s\n")
    print("--- Limiares atuais (.env) ---")
    print(f"  SCREAM_ENERGY_THRESHOLD = {settings.scream_energy_threshold}")
    print(f"  SCREAM_HIGH_BAND_RATIO  = {settings.scream_high_band_ratio}")
    print(f"  IMPACT_ENERGY_THRESHOLD = {settings.impact_energy_threshold}")
    print(f"  IMPACT_PEAK_RATIO       = {settings.impact_peak_ratio}")
    print(f"  MIN_ALERT_CONFIDENCE    = {settings.min_alert_confidence}")
    print(f"  CONFIRMAÇÕES NECESSÁRIAS = {settings.detection_confirmations_required}\n")

    for name, fn in (
        ("GRITO", detect_scream),
        ("IMPACTO", detect_impact),
        ("SOCORRO", lambda a, r: detect_help_request(a, r)),
    ):
        r = fn(audio, rate)
        flag = "SIM" if r.detected else "não"
        print(f"[{name}] dispararia? {flag} | conf={r.confidence} | {r.message}")


if __name__ == "__main__":
    main()
