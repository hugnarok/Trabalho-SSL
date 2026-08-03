# SafeAlert: A Distributed Edge–Server System for Acoustic Risk Detection in Academic Environments

**Authors:** Deivy Rossi Teixeira de Melo, Fernando Horita Siratuti, Hugo Henrique Marques, Vinicius Ramalho de Oliveira  
**Affiliation:** Computer Engineering Undergraduate Program, CEFET-MG, Brazil

---

## Abstract

We present **SafeAlert**, a distributed monitoring system designed for academic campuses that combines camera and microphone sensing to detect three acoustic risk events: **screams**, **impact sounds**, and **verbal help requests**. The edge client runs three parallel detectors—a lightweight CNN on log-mel spectrograms, RMS-based heuristics, and offline speech recognition (Vosk)—and confirms alerts only after consecutive detections of the same event type. Upon confirmation, the system sends a retroactive 5-second video clip, audio, and metadata to a central FastAPI server with an operator dashboard. The architecture prioritizes privacy (offline ASR, no person identification), low coupling between edge and server, and operational triage rather than continuous surveillance. Experiments on the NIGENS dataset show strong performance on normal/scream classes (F1 = 0.91 / 0.72) but limited impact detection (F1 = 0.13), motivating a hybrid fusion strategy with energy heuristics. SafeAlert is relevant to robotics and intelligent systems symposiums as an edge AI application combining multimodal sensing, real-time inference, and human-centered safety monitoring.

**Keywords:** edge computing, acoustic event detection, multimodal sensing, campus safety, speech recognition, convolutional neural networks

---

## 1. Introduction

Violence and distress situations in educational environments require fast operational response, yet continuous video surveillance raises ethical and legal concerns. SafeAlert addresses this gap with a **triage-oriented** system: it does not identify individuals or replace forensic analysis, but detects salient acoustic events and packages short evidence clips for human operators.

The system targets three event classes aligned with operational needs:

| Event | Detection approach |
|-------|-------------------|
| Scream | CNN + RMS/high-band energy heuristic |
| Impact | CNN + peak-to-median energy heuristic |
| Help request | Offline ASR (Vosk) + keyword matching |

Contributions:

1. A **client–server edge architecture** with retroactive video buffering and HTTP multipart alert delivery.
2. A **parallel multi-detector fusion** pipeline with streak-based confirmation to reduce false positives.
3. An **offline ASR module** for Portuguese help-request keywords, avoiding cloud speech APIs.
4. Empirical evaluation of a **lightweight CNN** trained on NIGENS, with documented limitations and fallback design.

---

## 2. Related Work

Environmental sound classification has been studied on datasets such as ESC-50 [5] and AudioSet [6]. Mel-frequency representations [3] remain standard for speech and environmental sounds. Our CNN follows this tradition with a compact architecture suitable for CPU-only edge devices.

Campus safety systems often rely on CCTV; SafeAlert complements video with **acoustic semantics** (scream vs. impact vs. spoken distress) while limiting stored footage to short retroactive clips—a design choice aligned with privacy-by-design principles and data minimization (cf. GDPR/LGPD frameworks [9]).

---

## 3. System Architecture

SafeAlert comprises two independent processes communicating via REST:

```
┌─────────────────────────────┐     HTTP multipart      ┌──────────────────────────┐
│  Edge client                │ ───────────────────────►│  Central server          │
│  Camera + Microphone        │  meta.json + MP4 + WAV  │  FastAPI + Dashboard     │
│  CNN · RMS heuristic · Vosk │                         │  data/alerts/<uuid>/     │
└─────────────────────────────┘                         └──────────────────────────┘
```

### 3.1 Edge Client

- **StreamCapture:** circular audio buffer (12 s) and deque-based video buffer (6 s at 30 FPS).
- **DetectionWorker:** runs every 0.5 s on a 5 s audio window; fuses detector outputs by maximum confidence; requires k=2 consecutive confirmations.
- **AlertSender:** extracts the last 5 s of video retroactively, encodes H.264/MP4 with faststart, and POSTs asynchronously.

### 3.2 Central Server

- FastAPI endpoints for alert ingestion, listing, and media retrieval.
- Operator HTML dashboard with inline video/audio players (5 s polling).
- File-based persistence under `data/alerts/<uuid>/`.

Detailed architecture: [architecture.en.md](architecture.en.md).

---

## 4. Detection Methods

### 4.1 CNN Classifier

Input: 5 s mono PCM at 16 kHz → log-mel spectrogram (64 mel bins, 128 frames) → 3-block CNN → softmax over {normal, scream, impact}. Model size ≈ 500 KB; inference < 50 ms on CPU.

Training data: NIGENS [CC BY-NC-ND 4.0], mapped to project classes (see [ml.en.md](ml.en.md)).

### 4.2 Energy Heuristics

- **Scream:** RMS energy AND high-frequency band ratio (2–8 kHz).
- **Impact:** short-window energy peak vs. median (captures abrupt transients).

Heuristics serve as **mandatory fallback** when the CNN underperforms on impact sounds.

### 4.3 Help Request (Vosk ASR)

Offline transcription with keyword matching (`socorro`, `me ajuda`, etc.) using word-boundary regex to avoid spurious matches. No audio leaves the edge device for ASR.

---

## 5. Experimental Results

Validation on NIGENS-derived test split (134 samples):

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|-----|---------|
| Normal | 0.88 | 0.95 | 0.91 | 110 |
| Scream | 0.82 | 0.64 | 0.72 | 14 |
| Impact | 0.20 | 0.10 | 0.13 | 10 |
| **Weighted avg.** | **0.82** | **0.85** | **0.83** | **134** |

**Interpretation:** high overall accuracy reflects class imbalance; 36% of screams are missed by CNN alone. Impact F1 is near failure due to only 10 test samples. Production deployment therefore **requires** heuristic fusion and streak confirmation.

Typical end-to-end latency from acoustic event to dashboard visibility: **6–7 seconds** (5 s analysis window + 1 s confirmation + network).

---

## 6. Ethical Considerations

- **Scope:** evidence triage for operators; not person identification or judicial attribution.
- **Consent & law:** deploy only with participant consent and institutional legal review (Brazil: LGPD, Law 13.709/2018).
- **Data minimization:** 5 s retroactive clips; no continuous archive by default.
- **Offline ASR:** voice data not sent to third-party cloud services.

---

## 7. Conclusion and Future Work

SafeAlert demonstrates a practical edge–server pipeline for acoustic risk triage in academic settings, combining ML, heuristics, and offline ASR. Key future directions:

- Class-weighted loss and data augmentation for minority classes.
- Domain-specific recordings from target campus environments.
- Human pose estimation (MoveNet/OpenPose) for true audio–video fusion.
- API authentication and database-backed alert storage for multi-site deployment.

---

## References

1. Davis, S.; Mermelstein, E. Comparison of parametric representations for monosyllabic word recognition. *IEEE TASSP*, 1980.
2. Piczak, K. J. ESC: Dataset for Environmental Sound Classification. *ACM Multimedia*, 2015.
3. Gemmeke, J. F. et al. Audio Set. *IEEE ICASSP*, 2017.
4. NIGENS General Sound Events Database. Zenodo, 2019. https://zenodo.org/records/2535878
5. Brazil. Lei nº 13.709/2018 (LGPD).
6. Vosk Offline Speech Recognition. https://alphacephei.com/vosk/

*Full bibliography: [references.en.md](references.en.md)*

---

## Repository & Demo

- Source code: `Trabalho-SSL/`
- English documentation: [README.en.md](../README.en.md)
- Quick start: `uvicorn central.app:app` + `python -m client.main`
