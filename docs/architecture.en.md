# System Architecture — SafeAlert

> **Academic scope:** triage and evidence logging tool for operational support in controlled environments. Does not replace forensic analysis or perform judicial identification of individuals.

---

## Overview

SafeAlert adopts an **asynchronous client–server architecture** with two independent processes:

```
┌────────────────────────────────────────────┐
│  CLIENT (edge)                             │
│                                            │
│  Camera → Circular frame buffer            │
│  Microphone → AudioRingBuffer (12 s)       │
│                                            │
│  DetectionWorker (dedicated thread)        │
│  ├── CNNClassifier   ─┐                    │
│  ├── RMS heuristic   ─┼─ fusion → streak   │
│  └── Vosk ASR        ─┘  counter (k=2)    │
│                  │                         │
│            confirmed alert                 │
│                  │                         │
│  AlertSender: WAV + MP4 (retroactive) + JSON│
└────────────────────┬───────────────────────┘
                     │ HTTP POST multipart/form-data
                     ▼
┌────────────────────────────────────────────┐
│  CENTRAL (server)                          │
│                                            │
│  FastAPI (Uvicorn ASGI)                    │
│  POST /api/alerts → storage.py             │
│  GET  /central    → HTML Dashboard         │
│                                            │
│  data/alerts/<uuid>/                       │
│  ├── meta.json                             │
│  ├── clip.mp4  (5 s retroactive video)     │
│  └── clip.wav  (event audio)               │
└────────────────────────────────────────────┘
```

The two modules communicate exclusively via REST API. The client has no direct access to the central database or filesystem, and the central server does not control client capture — coupling is intentionally minimal.

---

## Client Module

### Multimodal Capture (`capture.py` — `StreamCapture`)

Audio and video capture run in **independent threads**, never blocking the main loop.

**Audio — `AudioRingBuffer`:**

- Fixed-size circular buffer (default: 12 s × 16,000 Hz = 192,000 samples).
- Fed by the asynchronous `sounddevice` callback on the audio I/O thread.
- Writes protected by `threading.Lock` for safe access from the detection thread.
- Reads return a copy of the most recent segment without pausing capture.
- No dynamic allocations at runtime: the write pointer advances modulo the fixed size, overwriting old samples.

**Video — `_VideoReaderThread`:**

- Continuous frame reading loop via OpenCV (`cv2.VideoCapture`).
- Frames stored in `collections.deque(maxlen=N)` — when full, the oldest frame is discarded automatically.
- `maxlen` is computed as `VIDEO_BUFFER_SECONDS × CAMERA_FPS` (default: 6 s × 30 FPS = 180 frames).
- **Retroactive capture:** when an alert is confirmed, the last N frames are extracted from the deque. The evidence clip contains the seconds _before_ the trigger, not after — i.e., it captures the event that caused the alert.
- H.264/MP4 encoding via `imageio-ffmpeg` with `faststart` flag (`moov` atom at file start), enabling immediate browser playback without full download. Fallback to OpenCV `mp4v` codec when FFmpeg is unavailable.

### Parallel Detection Pipeline (`detection_worker.py` — `DetectionWorker`)

Runs as a dedicated thread, executing the detection cycle every `DETECTION_INTERVAL_SECONDS` (default: 0.5 s).

```
Audio window (5 s)
        │
        ├──► CNNClassifier.predict()     → DetectionResult(scream | impact | normal, conf)
        ├──► detect_scream()             → DetectionResult(scream, conf)
        ├──► detect_impact()             → DetectionResult(impact, conf)
        └──► detect_help_request()       → DetectionResult(help, conf)
                    │
              fusion: candidate = max(detected, key=confidence)
                    │
              _update_streak(candidate)
                    │
              if streak >= k:
                    └──► state.best = candidate  →  alert trigger
```

**Streak mechanism (anti-noise):**

- Maintains a counter of consecutive cycles with the same event type.
- Alert is promoted to `state.best` only after `DETECTION_CONFIRMATIONS_REQUIRED` consecutive confirmations (default: k=2).
- If the event type changes between cycles, the counter resets.
- Cost: adds `k × DETECTION_INTERVAL_SECONDS` latency to triggering (default: +1 s).
- Benefit: eliminates triggers from isolated transient acoustic events (cough, single door knock).

**Detector fusion:**

When multiple detectors fire in the same cycle, the candidate with highest `confidence` is selected. Heuristic and CNN compete — if CNN identifies scream at 0.87 and heuristic identifies impact at 0.91, the cycle candidate will be impact/heuristic.

### Detectors

#### Scream Heuristic (`detectors/scream.py`)

Combines two criteria computed over the full audio window:

1. **RMS energy:** `E_RMS ≥ SCREAM_ENERGY_THRESHOLD` — indicates significant volume.
2. **High-band ratio:** energy in 2–8 kHz band (via STFT) divided by total energy. Screams concentrate energy in high frequencies. If ratio exceeds `SCREAM_HIGH_BAND_RATIO`, criterion is satisfied.

Trigger: **both** criteria must be satisfied (logical AND). Final `score` is a 50/50 weighted average of the two ratios normalized by thresholds.

#### Impact Heuristic (`detectors/impact.py`)

Analyzes energy distribution in 50 ms short windows over the audio segment:

- Computes energy **peak** and **median** across windows.
- A peak above `IMPACT_PEAK_RATIO × median` (default: 5×) is classified as impact.
- Logic captures abrupt-onset events: knocks, falls, hits — characterized by energy peaks far above ambient level.

#### Help Request Detector (`detectors/help_request.py`)

Runs in background via Vosk (offline ASR):

- Transcribes audio continuously without sending data to external servers.
- Compares transcribed text with `HELP_KEYWORDS` using word-boundary regex (`\b`), preventing "ajuda" from matching inside "ajudar".
- Filters words below `HELP_MIN_WORD_CONFIDENCE` (default: 0.45) before comparison.
- Compound phrases (e.g., "me ajuda") take priority over isolated words.

#### CNN Classifier (`detectors/ml_events.py` + `ml/`)

- Loaded as singleton with lazy loading on first use.
- Extracts log-mel spectrogram from audio window: Hann window STFT (`N_FFT=512`, `Hop=256`), 64 mel filter bank, log compression, z-score normalization.
- Output: tensor `1 × 64 × 128` passed to CNN.
- Prediction returns softmax probabilities for three classes. If max probability exceeds `ML_CONFIDENCE_THRESHOLD` and class is not `normal`, returns `DetectionResult`.
- On load error (model not found, incompatible version), fails silently — energy heuristic assumes fallback role.

### Alert Sending (`alert_sender.py`)

When an alert is confirmed (`state.best` non-null and cooldown expired), `AlertSender`:

1. Captures retroactive MP4 clip from video buffer.
2. Converts audio segment to in-memory WAV.
3. Builds `multipart/form-data` payload with: `meta` (JSON with type, confidence, UTC timestamp, camera label, transcription), `video` (MP4 bytes), `audio` (WAV bytes).
4. Executes POST in separate thread (`daemon=True`) to avoid blocking capture loop.
5. After successful send, records returned `alert_id` and updates cooldown timestamp.

---

## Central Module

### REST API (`central/app.py`)

Built on FastAPI with Uvicorn (ASGI), supports multiple simultaneous requests without I/O blocking.

| Endpoint | Method | Description |
|---|---|---|
| `/api/alerts` | POST | Receives multipart alert; saves to disk; returns `alert_id` |
| `/api/alerts` | GET | Lists alerts chronologically (most recent first) |
| `/api/alerts/{id}` | GET | Returns metadata for a specific alert |
| `/api/alerts/{id}/{media}` | GET | Returns media file (`clip.mp4`, `clip.wav`, `snapshot.jpg`) |
| `/api/alerts` | DELETE | Removes all alerts (development/testing) |
| `/central` | GET | Operator HTML panel |
| `/docs` | GET | Interactive OpenAPI documentation |

Media route validates path against _path traversal_: resolved path must remain inside `data/alerts/`.

### Persistence (`central/storage.py`)

Each alert is saved in folder `data/alerts/<uuid>/` containing:

```
data/alerts/
└── 3f7a2c1e-…/
    ├── meta.json      # type, confidence, timestamp, camera, transcription
    ├── clip.mp4       # retroactive video clip (H.264, faststart)
    └── clip.wav       # event audio (PCM 16 kHz)
```

UUID is generated by `uuid.uuid4()` on receipt, ensuring uniqueness without client coordination. Alert listing reads `meta.json` from all folders and sorts by `received_at`.

### Operator Dashboard

Static HTML panel served at `/central` with automatic refresh every 5 s via async _polling_. Displays per alert: event type, confidence level, timestamp, source camera, transcription (if available), inline video player, and inline audio player.

---

## Configuration and Modularity

All configuration is read from environment variables (`.env` file) via `pydantic-settings` (`shared/config.py`). This ensures:

- No values are hardcoded in production code.
- Different environments (lab A, lab B, production) use distinct `.env` files without code changes.
- Parameter documentation is centralized in `.env.example`.

Shared types and models between client and central (`EventType`, `DetectionResult`, `AlertResponse`) live in `shared/models.py`, avoiding duplication and ensuring contract consistency.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Retroactive circular video buffer | Captures the event that caused the alert, not the period after triggering |
| Three parallel detectors | No single detector is sufficient; each covers gaps in the others |
| Streak counter (k=2) | Reduces false positives from transient events without excessive latency |
| Offline ASR (Vosk) | Eliminates voice transmission to external servers — LGPD requirement |
| Heuristic as CNN fallback | CNN has low F1 for impact (0.13); heuristic guarantees abrupt peak detection |
| 5 s max clip | Minimizes storage and frames system as triage, not continuous surveillance |
| FastAPI + Uvicorn | Native async I/O supports multiple simultaneous clients without blocking |

---

## Complete Alert Flow

```
t=0.0 s  Acoustic event occurs (e.g., scream)
t=0.5 s  DetectionWorker processes window → candidate: scream, conf=0.88 (streak=1)
t=1.0 s  DetectionWorker confirms → scream, conf=0.91 (streak=2 ≥ k=2)
          → state.best = DetectionResult(SCREAM, 0.91)
t=1.0 s  Main loop detects state.best → triggers _send_alert_async in thread
t=1.0 s  AlertSender extracts last 5 s from video buffer (frames from t=-5 s to t=0 s)
t=1.0 s  AlertSender converts 5 s audio → WAV; builds multipart
t=1.1 s  POST /api/alerts → Central receives, saves to data/alerts/<uuid>/
t=1.1 s  Operator sees alert on dashboard (next refresh within 5 s)
```

Typical total latency from acoustic event to dashboard availability: **6–7 seconds** (dominated by 5 s audio buffer + 1 s confirmation).

---

## Future Work

- [ ] Add `class_weight` to training `CrossEntropyLoss` to improve F1 on minority classes
- [ ] Implement API key authentication on admin routes (`DELETE`, `POST`)
- [ ] Replace UUID-based alert ordering with `received_at` ordering
- [ ] Add exponential backoff retry in `AlertSender`
- [ ] Integrate human pose estimation (OpenPose / MoveNet) for true multimodal fusion
- [ ] Replace filesystem persistence with database (SQLite or PostgreSQL) for scalability
