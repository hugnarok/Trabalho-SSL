"""
Extração de features para ML — log-mel spectrogram (conceitos SSL: STFT, escala mel, energia).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.fft import rfft
from scipy.signal import resample_poly

N_MELS = 64
N_FFT = 512
HOP = 256
TARGET_SR = 16_000


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    fmax = sr / 2.0
    mel_max = _hz_to_mel(np.array([fmax]))[0]
    mel_points = np.linspace(0, mel_max, n_mels + 2)
    hz_points = 700.0 * (10 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        if center <= left:
            center = left + 1
        if right <= center:
            right = center + 1
        for k in range(left, center):
            if center > left:
                fb[i, k] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                fb[i, k] = (right - k) / (right - center)
    return fb


_MEL_FB: Optional[np.ndarray] = None


def _get_mel_fb(sr: int) -> np.ndarray:
    global _MEL_FB
    if _MEL_FB is None:
        _MEL_FB = _mel_filterbank(sr, N_FFT, N_MELS)
    return _MEL_FB


def resample_audio(audio: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    if sr == target_sr:
        return audio.astype(np.float32)
    if len(audio) < 2:
        return audio.astype(np.float32)
    g = np.gcd(sr, target_sr)
    return resample_poly(audio, target_sr // g, sr // g).astype(np.float32)


def log_mel_spectrogram(
    audio: np.ndarray,
    sample_rate: int,
    target_frames: int = 128,
) -> np.ndarray:
    """
    Retorna tensor (1, n_mels, target_frames) normalizado em log.
    """
    x = resample_audio(audio, sample_rate)
    x = x.astype(np.float64)
    peak = np.max(np.abs(x)) + 1e-9
    x = x / peak

    n_frames = 1 + max(0, (len(x) - N_FFT) // HOP)
    if n_frames < 1:
        return np.zeros((1, N_MELS, target_frames), dtype=np.float32)

    mel_fb = _get_mel_fb(TARGET_SR)
    specs = []
    for i in range(n_frames):
        start = i * HOP
        frame = x[start : start + N_FFT]
        if len(frame) < N_FFT:
            frame = np.pad(frame, (0, N_FFT - len(frame)))
        windowed = frame * np.hanning(N_FFT)
        power = np.abs(rfft(windowed)) ** 2
        mel = mel_fb @ power[: mel_fb.shape[1]]
        specs.append(np.log(mel + 1e-6))

    spec = np.stack(specs, axis=1)
    if spec.shape[1] < target_frames:
        pad = target_frames - spec.shape[1]
        spec = np.pad(spec, ((0, 0), (0, pad)), mode="constant")
    else:
        spec = spec[:, -target_frames:]

    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec[np.newaxis, :, :].astype(np.float32)
