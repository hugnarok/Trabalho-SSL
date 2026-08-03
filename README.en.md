# SafeAlert — Acoustic Monitoring System

Distributed system for detecting risk situations in academic environments. It operates via **camera and microphone** and identifies three event types: **screams**, **impact sounds**, and **verbal help requests**. When an anomaly is confirmed, it automatically sends a retroactive video clip, audio, and incident metadata to the central server.

> **Ethical scope:** triage and evidence logging tool for operational support. It does not perform person identification and does not replace forensic analysis. Use only with participant consent and a legal basis (LGPD in Brazil).

> **Language:** [Português (BR)](../README.md) · **English**

---

## Table of Contents

- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the system](#running-the-system)
- [Threshold calibration](#threshold-calibration)
- [Machine Learning](#machine-learning)
- [Full configuration (.env)](#full-configuration-env)
- [Repository structure](#repository-structure)
- [Utility scripts](#utility-scripts)
- [Technical documentation](#technical-documentation)
- [Symposium paper](#symposium-paper)
- [License and ethical use](#license-and-ethical-use)

---

## How it works

The system consists of two independent processes communicating over HTTP:

```
┌─────────────────────────────────┐        HTTP multipart        ┌──────────────────────────┐
│  Edge client                    │ ────────────────────────────►│  Central server          │
│  Camera + Microphone            │  meta.json + clip.mp4 + wav  │  FastAPI + Dashboard     │
│  CNN · RMS heuristic · Vosk     │                              │  data/alerts/<uuid>/     │
└─────────────────────────────────┘                              └──────────────────────────┘
```

**Client:** captures audio in a circular buffer and continuous video frames. Every 0.5 s, three detectors run in parallel on the same audio window:

| Detector | Method | Role |
|---|---|---|
| CNN (log-mel spectrogram) | Machine learning (PyTorch) | Semantic classification: scream / impact / normal |
| Energy heuristic | RMS threshold + high-band ratio | Fallback and redundancy — model-independent |
| Offline ASR (Vosk) | Speech recognition (PT) | Keyword detection: "socorro", "me ajuda", etc. |

Two consecutive cycles with the same event type confirm an alert. The client then extracts the **last 5 seconds of video** from the buffer (retroactive capture — the event has already occurred) and sends the package to the central server.

**Central server:** receives alerts, persists files to disk, and displays them on the operator panel with inline video and audio players.

---

## Requirements

| Item | Detail |
|---|---|
| Python | 3.9 or higher (tested: 3.11 on Linux/macOS, 3.9 on macOS) |
| Webcam | Any USB or built-in camera |
| Microphone | Any system audio input |
| RAM | Minimum 2 GB free (Vosk model + PyTorch) |
| Disk | ~3 GB for NIGENS dataset + models (training only) |
| Tested OS | macOS · Linux (Ubuntu 22+) · Windows (adjust `CAMERA_ID`) |

---

## Installation

### 1. Clone and create virtual environment

```bash
git clone <repository-url>
cd Trabalho-SSL
python3 -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **macOS note:** Vosk 0.3.45 has no macOS wheel; `requirements.txt` uses `>=0.3.44`.

### 2. Copy configuration

```bash
cp .env.example .env
```

Edit `.env` if needed (camera, central URL, thresholds). Defaults work for local use.

### 3. Download speech transcription model (Vosk, ~50 MB)

```bash
bash scripts/setup_vosk.sh
```

The script downloads `vosk-model-small-pt-0.3` to `data/models/` automatically. If it already exists, it does nothing.

### 4. (Recommended) Train the CNN acoustic classifier

Without the CNN model, the system runs on energy heuristics and ASR only — still functional, but with less classification richness.

```bash
# Download NIGENS (~2 GB), prepare clips, and train the CNN in one command:
python -m scripts.download_nigens --all --epochs 25
```

See the [Machine Learning](#machine-learning) section for details.

---

## Running the system

Open **two terminals** with the virtual environment activated:

**Terminal 1 — Alert central:**

```bash
uvicorn central.app:app --reload --host 127.0.0.1 --port 8000
```

| Address | Description |
|---|---|
| http://127.0.0.1:8000/central | Operator panel (alerts + videos) |
| http://127.0.0.1:8000/docs | Interactive REST API documentation |

**Terminal 2 — Client (camera + microphone):**

```bash
python -m client.main
```

A preview window opens with the camera image and real-time audio waveform. Wave color changes green → yellow → red according to detected risk level. Press **`q`** to exit.

> **macOS:** on first run, the system will request camera and microphone permission for Terminal (or Python). Grant access and restart the client if needed.

> **Windows:** if the camera does not open, set `CAMERA_ID=1` (or 2) in `.env`.

---

## Threshold calibration

Energy heuristic thresholds (`SCREAM_ENERGY_THRESHOLD`, `IMPACT_PEAK_RATIO`, etc.) should be tuned for the specific acoustic environment where the system will be deployed.

Use the calibration script with a reference WAV file:

```bash
# Test with a scream file:
python -m client.calibrate data/datasets/prepared/grito/femaleScream_0000.wav

# Test with a normal file (should not trigger):
python -m client.calibrate data/datasets/prepared/normal/footsteps_0000.wav

# Test with your own audio:
python -m client.calibrate my_audio.wav
```

The script prints current threshold values and indicates whether each detector would fire for that audio, without needing a camera or central connection. Adjust values in `.env` and re-run until thresholds are adequate.

See also: [docs/calibration.en.md](docs/calibration.en.md).

---

## Machine Learning

The CNN classifier converts audio windows into log-mel spectrograms and classifies them into three categories: **normal**, **scream**, and **impact**.

### Training with NIGENS (recommended)

```bash
# All at once: download + preparation + training
python -m scripts.download_nigens --all --epochs 25

# Or separate steps:
python -m scripts.download_nigens           # download and extract only (~2 GB)
python -m scripts.download_nigens --prepare # organize into scream/impact/normal
python -m scripts.train_sound_model --epochs 25  # train the CNN
```

The NIGENS dataset (CC BY-NC-ND 4.0 license) is downloaded directly from Zenodo. **Do not commit dataset or model files** — they are in `.gitignore`. Each team member should download and train locally.

### NIGENS → project class mapping

| NIGENS folders | Class |
|---|---|
| `femaleScream`, `maleScream` | `grito` (scream) |
| `crash`, `knock` | `impacto` (impact) |
| `femaleSpeech`, `maleSpeech`, `general`, `piano`, `footsteps`, `phone`, `engine`, `alarm`, `fire`, `dog`, `baby` | `normal` |

### Training with custom audio

Place `.wav` files in:

```
data/samples/ml/grito/
data/samples/ml/impacto/
data/samples/ml/normal/
```

Then run `python -m scripts.train_sound_model --epochs 25` normally. Both directories (`datasets/prepared/` and `samples/ml/`) are combined automatically.

### Model evaluation

```bash
python -m scripts.evaluate_model
```

Prints accuracy, F1-score per class, and confusion matrix. Useful to verify generalization before production use.

### Generated artifacts

| File | Description |
|---|---|
| `data/models/sound_classifier.pt` | CNN weights (PyTorch) |
| `data/models/sound_classifier.json` | Metadata: classes, sample rate, metrics |

### ML configuration variables

| Variable | Default | Description |
|---|---|---|
| `ML_ENABLED` | `true` | Enables CNN classifier if model exists |
| `ML_CONFIDENCE_THRESHOLD` | `0.65` | Minimum probability to trigger CNN alert |
| `ML_HEURISTIC_FALLBACK` | `true` | Keeps RMS heuristic running in parallel with CNN |

Set `ML_HEURISTIC_FALLBACK=false` to use CNN only (not recommended without a well-calibrated model).

Full ML guide: [docs/ml.en.md](docs/ml.en.md).

---

## Full configuration (.env)

| Variable | Default | Description |
|---|---|---|
| `CENTRAL_URL` | `http://127.0.0.1:8000` | Central API URL |
| `CAMERA_ID` | `0` | Webcam index (0 = system default) |
| `CAMERA_LABEL` | `cam-01` | Camera identifier in alerts |
| `SAMPLE_RATE` | `16000` | Audio sample rate in Hz |
| `AUDIO_CHUNK_SECONDS` | `5.0` | Audio window duration per cycle |
| `DETECTION_INTERVAL_SECONDS` | `0.5` | Interval between detection cycles |
| `DISPLAY_FPS` | `30` | Local preview window FPS |
| `CAMERA_WIDTH` / `CAMERA_HEIGHT` | `640` / `480` | Video capture resolution |
| `ALERT_COOLDOWN_SECONDS` | `15.0` | Minimum interval between consecutive alerts |
| `DETECTION_CONFIRMATIONS_REQUIRED` | `2` | Consecutive cycles required to confirm alert |
| `MIN_ALERT_CONFIDENCE` | `0.55` | Global minimum confidence for any alert |
| `SCREAM_ENERGY_THRESHOLD` | `0.025` | RMS energy threshold for scream detector |
| `SCREAM_HIGH_BAND_RATIO` | `1.35` | Min. 2–8 kHz band energy ratio for scream |
| `IMPACT_ENERGY_THRESHOLD` | `0.018` | RMS threshold for impact detector |
| `IMPACT_PEAK_RATIO` | `5.0` | Min. peak relative to median for impact |
| `ALERT_SAVE_VIDEO` | `true` | Saves 5 s retroactive MP4 clip with alert |
| `ALERT_VIDEO_SECONDS` | `5.0` | Evidence clip duration |
| `VIDEO_BUFFER_SECONDS` | `6.0` | Circular video buffer size |
| `ALERT_SAVE_SNAPSHOT` | `false` | Saves JPEG frame (alternative to video) |
| `TRANSCRIPTION_ENABLED` | `true` | Enables Vosk keyword detector |
| `VOSK_MODEL_PATH` | `data/models/vosk-model-small-pt-0.3` | Vosk model path |
| `HELP_KEYWORDS` | `socorro,me ajuda,...` | Keywords/phrases triggering help alert |
| `HELP_MIN_CONFIDENCE` | `0.85` | Min. keyword match confidence |
| `HELP_MIN_WORD_CONFIDENCE` | `0.45` | Min. per-word Vosk confidence |
| `HELP_CONFIRMATIONS_REQUIRED` | `2` | Confirmations for help alert |
| `ML_ENABLED` | `true` | Enables CNN classifier |
| `ML_CONFIDENCE_THRESHOLD` | `0.65` | CNN probability threshold |
| `ML_HEURISTIC_FALLBACK` | `true` | Keeps heuristic in parallel with CNN |

---

## Repository structure

```
Trabalho-SSL/
│
├── client/                     # Edge client module
│   ├── main.py                 # Entry point: main loop + preview
│   ├── capture.py              # Circular audio/video buffer (StreamCapture)
│   ├── detection_worker.py     # Parallel detection thread + streak counter
│   ├── alert_sender.py         # Multipart packaging and send to central
│   ├── transcription.py        # Vosk interface (offline ASR)
│   ├── calibrate.py            # Threshold calibration tool via WAV
│   ├── waveform_overlay.py     # Waveform visualization in preview
│   ├── detectors/
│   │   ├── base.py             # DetectionResult dataclass
│   │   ├── scream.py           # Heuristic: RMS + high band (2–8 kHz)
│   │   ├── impact.py           # Heuristic: peak vs. median energy
│   │   ├── help_request.py     # Vosk keywords
│   │   └── ml_events.py        # CNN classifier interface
│   └── ml/
│       ├── model_arch.py       # CNN architecture (PyTorch)
│       ├── features.py         # Log-mel spectrogram extraction
│       ├── classifier.py       # Singleton: model load and inference
│       └── audio_io.py         # WAV read and resampling
│
├── central/                    # Server module
│   ├── app.py                  # FastAPI REST API + operator HTML panel
│   ├── storage.py              # Alert persistence in data/alerts/<uuid>/
│   ├── media.py                # Media read utilities
│   └── video_utils.py          # MP4 encoding with faststart
│
├── shared/                     # Shared code
│   ├── config.py               # Pydantic Settings — reads .env variables
│   └── models.py               # Types: EventType, DetectionResult, AlertResponse
│
├── scripts/                    # Training and maintenance tools
│   ├── download_nigens.py      # NIGENS download + extract + preparation
│   ├── prepare_nigens.py       # Organizes NIGENS clips by class
│   ├── train_sound_model.py    # Trains CNN and saves weights
│   ├── evaluate_model.py       # Evaluation: F1, accuracy, confusion matrix
│   ├── fix_alert_videos.py     # Fixes old MP4s without faststart flag
│   ├── setup_vosk.sh           # Downloads Vosk PT model (~50 MB)
│   └── zenodo_download.py      # Zenodo download helper
│
├── docs/
│   ├── architecture.en.md      # Detailed architecture (English)
│   ├── ml.en.md                  # Full ML pipeline (English)
│   ├── calibration.en.md         # Calibration guide (English)
│   ├── references.en.md          # Bibliography (English)
│   ├── symposium-paper.md        # Symposium submission draft
│   ├── arquitetura.md            # Architecture (Portuguese)
│   └── ml.md                     # ML guide (Portuguese)
│
├── data/                       # Generated at runtime — not versioned
│   ├── alerts/                 # Alerts received by central
│   ├── datasets/               # NIGENS + prepared clips (download locally)
│   ├── models/                 # CNN + Vosk models (generate locally)
│   └── samples/                # Optional test audio/video
│
├── requirements.txt
├── .env.example
├── README.md                   # Portuguese
└── README.en.md                # English (this file)
```

---

## Utility scripts

| Script | Usage | Description |
|---|---|---|
| `scripts/setup_vosk.sh` | `bash scripts/setup_vosk.sh` | Downloads Vosk PT model to `data/models/` |
| `scripts/download_nigens.py` | `python -m scripts.download_nigens --all` | NIGENS download + prep + training |
| `scripts/train_sound_model.py` | `python -m scripts.train_sound_model --epochs 25` | Trains CNN with data in `data/datasets/prepared/` and `data/samples/ml/` |
| `scripts/evaluate_model.py` | `python -m scripts.evaluate_model` | Evaluates model: F1 per class, confusion matrix |
| `client/calibrate.py` | `python -m client.calibrate audio.wav` | Tests detectors on a WAV file without camera |
| `scripts/fix_alert_videos.py` | `python -m scripts.fix_alert_videos` | Fixes old MP4s that won't play in browser |

---

## Technical documentation

- [System architecture](docs/architecture.en.md) — data flow, components, design decisions
- [Machine Learning pipeline](docs/ml.en.md) — log-mel spectrogram, training, evaluation
- [Threshold calibration](docs/calibration.en.md) — reducing false positives
- [Bibliography](docs/references.en.md)

---

## Symposium paper

A draft paper for robotics/intelligent systems symposium submission is available at:

**[docs/symposium-paper.md](docs/symposium-paper.md)**

It includes abstract, architecture summary, experimental results, and ethical considerations in academic English.

---

## License and ethical use

The source code of this project is for academic use. The NIGENS dataset is distributed under **CC BY-NC-ND 4.0** (non-commercial, no derivatives, with attribution). Trained models derived from NIGENS are subject to the same license.

**Usage guidelines:**

- Run the system only in test environments with explicit participant consent.
- Do not store third-party data without a legal basis under **Law No. 13.709/2018 (LGPD)** in Brazil.
- The system does not perform person identification and must not be used as unrestricted surveillance.
- For real deployment on a university campus, consult the institution's legal office.

---

## Authors and Contact

<div align="center">
  <br><br>
     <i>Deivy Rossi Teixeira de Melo — Undergraduate — 5th Semester, Computer Engineering @ CEFET-MG</i>
  <br><br>
  
  [![Gmail][gmail-badge]][gmail-autor1]
  [![Linkedin][linkedin-badge]][linkedin-autor1]
  [![GitHub][github-badge]][github-autor1]
  [![Instagram][instagram-badge]][instagram-autor1]
  
  <br><br>
     <i>Fernando Horita Siratuti — Undergraduate — 5th Semester, Computer Engineering @ CEFET-MG</i>
  <br><br>
  
  [![Gmail][gmail-badge]][gmail-autor2]
  [![Linkedin][linkedin-badge]][linkedin-autor2]
  [![GitHub][github-badge]][github-autor2]
  [![Instagram][instagram-badge]][instagram-autor2]
  
  <br><br>
     <i>Hugo Henrique Marques — Undergraduate — 5th Semester, Computer Engineering @ CEFET-MG</i>
  <br><br>
  
  [![Gmail][gmail-badge]][gmail-autor3]
  [![Linkedin][linkedin-badge]][linkedin-autor3]
  [![GitHub][github-badge]][github-autor3]
  [![Instagram][instagram-badge]][instagram-autor3]
  
  <br><br>
     <i>Vinicius Ramalho de Oliveira — Undergraduate — 5th Semester, Computer Engineering @ CEFET-MG</i>
  <br><br>
  
  [![Gmail][gmail-badge]][gmail-autor4]
  [![Linkedin][linkedin-badge]][linkedin-autor4]
  [![GitHub][github-badge]][github-autor4]
  [![Instagram][instagram-badge]][instagram-autor4]

</div>

[gmail-badge]: https://img.shields.io/badge/-Gmail-c14438?style=flat-square&logo=Gmail&logoColor=white
[linkedin-badge]: https://img.shields.io/badge/-LinkedIn-blue?style=flat-square&logo=Linkedin&logoColor=white
[github-badge]: https://img.shields.io/badge/-GitHub-181717?style=flat-square&logo=github&logoColor=white
[instagram-badge]: https://img.shields.io/badge/-Instagram-E4405F?style=flat-square&logo=instagram&logoColor=white

[gmail-autor1]: mailto:deivyrossi@gmail.com
[linkedin-autor1]: https://www.linkedin.com/in/deivy-rossi-380263279/
[github-autor1]: https://github.com/deivyrossi
[instagram-autor1]: https://www.instagram.com/deivyrossi/

[gmail-autor2]: mailto:siratutifernando@gmail.com
[linkedin-autor2]: https://www.linkedin.com/in/fernando-siratuti-503ba8301/
[github-autor2]: https://github.com/fernando-horita-siratuti
[instagram-autor2]: https://www.instagram.com/siratuti_/

[gmail-autor3]: mailto:hugohmarques4@gmail.com
[linkedin-autor3]: https://www.linkedin.com/in/hugo-h-marques-980629216/
[github-autor3]: https://github.com/hugnarok
[instagram-autor3]: https://www.instagram.com/hugomarques_02/

[gmail-autor4]: mailto:ramalhooliveiravini@gmail.com
[linkedin-autor4]: https://www.linkedin.com/in/vin%C3%ADcius-ramalho-de-oliveira-4464b8327/
[github-autor4]: https://github.com/ViniciusRO22
[instagram-autor4]: https://www.instagram.com/_vinicius.ro_/
