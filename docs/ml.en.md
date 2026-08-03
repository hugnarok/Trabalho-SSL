# Machine Learning Pipeline — SafeAlert

This document details SafeAlert's machine-learning-based acoustic classification pipeline: from feature extraction to real-time edge inference.

---

## Pipeline Overview

```
Raw audio (PCM, 16 kHz, mono)
        │
        ▼
  Hann windowing (N_FFT=512, Hop=256)
        │
        ▼
  STFT → power spectrum
        │
        ▼
  64 mel filter bank (20 Hz – 8,000 Hz)
        │
        ▼
  Log compression + z-score normalization
        │
        ▼
  Tensor 1 × 64 × 128  (channels × filters × frames)
        │
        ▼
  Lightweight CNN (3 Conv blocks → Dense 128 → Softmax 3)
        │
        ▼
  [normal | scream | impact]  +  probability
```

---

## Feature Extraction (`client/ml/features.py`)

### Why log-mel spectrogram?

The human ear does not perceive frequencies linearly: differences at low frequencies are much more perceptible than the same differences at high frequencies. The mel scale approximates this perception curve. Projecting the power spectrum onto this scale gives the CNN a representation that emphasizes spectrally relevant regions for sound event perception.

### Parameters used

| Parameter | Value | Rationale |
|---|---|---|
| Sample rate | 16,000 Hz | Covers frequencies up to 8 kHz — voice and human scream region |
| Window size (`N_FFT`) | 512 samples (32 ms) | Adequate spectral resolution for environmental sounds |
| Hop | 256 samples (16 ms) | 50% overlap — balance between temporal resolution and cost |
| Window | Hann | Reduces spectral leakage at frame boundaries |
| Mel filters | 64 | Sufficient mel resolution to distinguish scream/impact/normal |
| Segment duration | 5 s | Produces fixed 64 × 128 frame tensor (hardware-independent) |

### Normalization

The resulting log-mel spectrogram is z-score normalized (mean 0, std 1) computed over the segment itself. This makes the model robust to volume variations across environments with very different noise levels.

### Implementation

The mel filter bank is computed analytically by `_get_mel_fb()` and cached as a global variable after the first call. This eliminates recalculation per audio window and removes the `librosa` dependency at inference time, reducing edge device installation size.

---

## CNN Architecture (`client/ml/model_arch.py`)

The network was designed for edge inference: small enough to run on CPU without dedicated accelerator, with latency below 50 ms.

```
Input: 1 × 64 × 128

Block 1: Conv2d(1→16, 3×3) → BatchNorm → ReLU → MaxPool2d(2×2)
         Output: 16 × 32 × 64

Block 2: Conv2d(16→32, 3×3) → BatchNorm → ReLU → MaxPool2d(2×2)
         Output: 32 × 16 × 32

Block 3: Conv2d(32→64, 3×3) → BatchNorm → ReLU → AdaptiveAvgPool2d(4×4)
         Output: 64 × 4 × 4 = 1,024 values

Flatten → Linear(1024→128) → Dropout(0.3) → ReLU

Output: Linear(128→3) → Softmax
       [P(normal), P(scream), P(impact)]
```

**Architecture decisions:**

- **BatchNorm** after each convolution: stabilizes training with limited data and reduces learning rate tuning needs.
- **AdaptiveAvgPool2d(4,4)** in third block (instead of fixed MaxPool): ensures fixed output tensor size regardless of input length — important for segments with slightly different durations.
- **Dropout(0.3)** in dense layer: regularization for small datasets.
- Total size: ~500 KB on disk (float32). Fits in memory on any device with more than 50 MB RAM available.

---

## NIGENS Dataset

**NIGENS** (NIGENS General Sound Events Database) is a public corpus of general sound events, distributed under **CC BY-NC-ND 4.0** (non-commercial, no modification, with attribution). Available at: https://zenodo.org/records/2535878

### Class mapping

| NIGENS folders | SafeAlert class | Rationale |
|---|---|---|
| `femaleScream`, `maleScream` | `grito` (scream) | Female and male distress screams |
| `crash`, `knock` | `impacto` (impact) | Impact, fall, knock sounds |
| `femaleSpeech`, `maleSpeech` | `normal` | Everyday speech — prevents voice from triggering alerts |
| `general`, `piano`, `footsteps` | `normal` | Common campus ambient sounds |
| `phone`, `engine`, `alarm` | `normal` | Reduces false positives from mechanical noise |
| `fire`, `dog`, `baby` | `normal` | Biologically salient but benign sounds |

### Class imbalance

NIGENS has many more normal sound samples than screams and impacts. In the validation set used to evaluate the model, there were 110 normal samples, 14 scream, and 10 impact. This imbalance inflates global accuracy and hurts recall on risk classes — see Results section.

**Implemented mitigation:** the `normal` class is built from multiple subcategories to increase acoustic diversity and reduce overfitting to NIGENS-specific noise style.

**Pending mitigation:** adding `weight` to `CrossEntropyLoss` inversely proportional to each class frequency would be the most impactful fix for scream and impact F1.

---

## Step-by-Step: Training

### Option 1 — Automatic download and training (recommended)

```bash
# Downloads NIGENS (~2 GB), organizes clips, and trains CNN in one command:
python -m scripts.download_nigens --all --epochs 25
```

Internally runs three steps in sequence:
1. `download_nigens.py` — downloads and extracts Zenodo ZIP
2. `prepare_nigens.py` — copies and renames clips to `data/datasets/prepared/grito/`, `impacto/`, `normal/`
3. `train_sound_model.py` — trains CNN and saves weights

### Option 2 — Separate steps

```bash
# 1. Download and extraction only:
python -m scripts.download_nigens

# 2. Clip organization (required after download):
python -m scripts.download_nigens --prepare

# 3. Training:
python -m scripts.train_sound_model --epochs 25 --batch-size 16 --lr 0.001
```

### Option 3 — Manual download (if automatic fails)

1. Visit https://zenodo.org/records/2535878 and download `NIGENS.zip`
2. Place in `data/datasets/nigens/downloads/NIGENS.zip`
3. Run:
   ```bash
   python -m scripts.download_nigens --skip-download --all --epochs 25
   ```

> **macOS:** ZIP extraction uses system `unzip` (required for Deflate64 support). Verify with `which unzip`.

### Option 4 — Custom audio

Place `.wav` files (mono, any sample rate — resampled to 16 kHz) in:

```
data/samples/ml/grito/
data/samples/ml/impacto/
data/samples/ml/normal/
```

Both directories (`datasets/prepared/` and `samples/ml/`) are combined automatically by the training script. This allows augmenting rare classes with recordings from the target environment.

### Training script parameters

```bash
python -m scripts.train_sound_model \
  --epochs 25 \        # number of epochs
  --batch-size 16 \    # batch size
  --lr 0.001 \         # learning rate (Adam)
  --val-split 0.15 \   # validation fraction
  --test-split 0.15    # test fraction
```

The script prints per-epoch metrics (train and validation loss) and, at the end, full classification report (precision, recall, F1 per class) on the test set.

### Generated artifacts

| File | Description |
|---|---|
| `data/models/sound_classifier.pt` | CNN weights (PyTorch state dict) |
| `data/models/sound_classifier.json` | Metadata: classes, sample rate, final accuracy |

---

## Model Evaluation

```bash
python -m scripts.evaluate_model
```

Displays:
- Global accuracy on test set
- Per-class report (precision, recall, F1, support)
- Text confusion matrix

Use to verify generalization before production deployment.

---

## Real-Time Inference (`client/ml/classifier.py`)

The classifier is loaded as a **singleton** with _lazy loading_: weight file is read from disk only on first call, not at client startup. This allows the client to start even if the model has not been trained yet — energy heuristic assumes fallback role.

Inference flow per cycle:

1. `DetectionWorker` calls `detect_ml_events(audio, sample_rate)`
2. `ml_events.py` calls `SoundClassifier.predict(audio, sr)`
3. `classifier.py` extracts log-mel spectrogram via `features.log_mel_spectrogram()`
4. Passes normalized tensor through CNN
5. Obtains softmax probabilities
6. If `max(probs) >= ML_CONFIDENCE_THRESHOLD` and winning class ≠ `normal`, returns `DetectionResult` with type and confidence
7. Otherwise, returns `DetectionResult(detected=False)`

Typical latency (CPU, no GPU): < 50 ms on x86-64 (Intel Core i7) and ARM M1.

---

## Results Obtained

| Class | Precision | Recall | F1-Score | N samples |
|---|---|---|---|---|
| Normal | 0.88 | 0.95 | 0.91 | 110 |
| Scream | 0.82 | 0.64 | 0.72 | 14 |
| Impact | 0.20 | 0.10 | 0.13 | 10 |
| **Weighted avg.** | **0.82** | **0.85** | **0.83** | **134** |

**Interpretation:**

- Global accuracy of 85.07% is misleading — it mainly reflects performance on the majority class (normal).
- Scream recall of 0.64 means 36% of real screams are missed by CNN alone. For a safety system, this is unacceptable as the sole detector.
- Impact F1 of 0.13 indicates near-complete failure to generalize for that class, given training sample scarcity.
- These results justify keeping energy heuristics as mandatory fallback.

**Variance note:** with only 10 impact samples in the test set, one additional or missing sample changes F1 by ~0.10. Values should be interpreted as trend indicators.

---

## Future Improvements

### High priority (direct impact on results)

1. **Class weights in loss:** `CrossEntropyLoss(weight=torch.tensor([1/110, 1/14, 1/10]))` — penalizes errors on rare classes without needing more data.

2. **Early stopping:** save weights when `val_loss` is minimum, not from last epoch. Prevents overfitting on small datasets.

3. **Data augmentation:** apply to minority classes:
   - _Time stretching_ (±10%): changes speed without changing pitch
   - _Pitch shifting_ (±2 semitones): varies fundamental frequency
   - Gaussian noise addition (SNR 10–20 dB): simulates noisy environments
   - _SpecAugment_: masks time and frequency bands in spectrogram

4. **Real data collection:** record screams and impacts in the target environment (campus corridor, parking lot) to reduce domain gap between NIGENS and deployment context.

### Medium priority

5. **Separate stride vs. counter:** maintain independent streak counters per event type instead of one global counter — prevents rapid alternation between detectors from blocking any confirmation.

6. **Semantic priority fusion:** `HELP_REQUEST > SCREAM > IMPACT` as tie-breaker beyond confidence, reflecting semantic specificity of each event.

### Low priority

7. **Computer vision integration:** human pose estimation (MoveNet, OpenPose) to detect aggressive postures — true audio–video multimodal fusion.
