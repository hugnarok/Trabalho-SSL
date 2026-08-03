# Calibration — Reducing False Positives

## How the system decides to send an alert

1. Every `DETECTION_INTERVAL_SECONDS` (e.g., 0.5 s), the system analyzes the last `AUDIO_CHUNK_SECONDS` (5 s) of audio.
2. Detectors (`scream`, `impact`, `help_request`) return a candidate.
3. An alert fires only if the **same type** appears **`DETECTION_CONFIRMATIONS_REQUIRED` times in a row** (e.g., 2 → ~1 s confirmation).
4. Confidence must be ≥ `MIN_ALERT_CONFIDENCE`.

On screen you will see `grito? (1/2)` before confirmation; after confirmation it becomes `grito` and sends to the central server.

## Adjusting `.env`

| Variable | Too many false positives | Not detecting when it should |
|----------|--------------------------|------------------------------|
| `SCREAM_ENERGY_THRESHOLD` | **Increase** (e.g., 0.03) | **Decrease** (e.g., 0.02) |
| `SCREAM_HIGH_BAND_RATIO` | **Increase** (e.g., 1.5) | **Decrease** (e.g., 1.2) |
| `IMPACT_ENERGY_THRESHOLD` | **Increase** | **Decrease** |
| `IMPACT_PEAK_RATIO` | **Increase** (e.g., 6) | **Decrease** (e.g., 4) |
| `MIN_ALERT_CONFIDENCE` | **Increase** (e.g., 0.65) | **Decrease** (e.g., 0.45) |
| `DETECTION_CONFIRMATIONS_REQUIRED` | **3** | **1** |
| `ALERT_COOLDOWN_SECONDS` | **Increase** (e.g., 20) | — |

## Help request (not the CNN model)

The classifier trained on NIGENS recognizes **scream**, **impact**, and **normal** — not help requests.

Help detection uses **Vosk (transcription)** + **keywords** in `HELP_KEYWORDS`. An alert fires only if transcribed text contains a configured phrase (e.g., `socorro`, `me ajuda`, `preciso de ajuda`).

| Variable | Effect |
|----------|--------|
| `HELP_KEYWORDS` | Comma-separated list; avoid `ajuda` alone (triggers on "ajudar") |
| `HELP_MIN_WORD_CONFIDENCE` | Minimum Vosk word confidence (e.g., 0.45) |
| `HELP_MIN_CONFIDENCE` | Minimum help alert confidence (e.g., 0.85) |
| `HELP_CONFIRMATIONS_REQUIRED` | Consecutive readings with same phrase (e.g., 2) |

Test without camera: speak clearly *"socorro"* or *"me ajuda"* into the microphone, or use `python -m client.calibrate file.wav`.

If **any voice** triggers **scream** (not help), increase `ML_CONFIDENCE_THRESHOLD` (e.g., 0.75) or set `ML_HEURISTIC_FALLBACK=false`.

## Testing with an audio file

```bash
source .venv/bin/activate
python -m client.calibrate data/samples/your_audio.wav
```

The script prints RMS, high band, peak, and impact ratio **without** opening the camera.

## Video in alerts

| Variable | Description |
|----------|-------------|
| `ALERT_SAVE_VIDEO` | `true` = sends MP4 clip (default) |
| `ALERT_VIDEO_SECONDS` | Clip duration (3 or 5) |
| `VIDEO_BUFFER_SECONDS` | Must be ≥ clip duration (e.g., 6) |
| `ALERT_SAVE_SNAPSHOT` | `true` for JPEG **in addition** to video |

## Practical tips

- Calibrate in **silence** and **ambient noise** first; note values in the terminal (alert `message`).
- Keyboard/door impacts often trigger `impact` — increase `IMPACT_PEAK_RATIO`.
- Music/TV may trigger `scream` — increase `SCREAM_HIGH_BAND_RATIO`.
- For stable demos: `DETECTION_CONFIRMATIONS_REQUIRED=3` and thresholds slightly above real environment levels.
