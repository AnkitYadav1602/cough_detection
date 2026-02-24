"""
feature_extraction.py
---------------------
Per-cough acoustic feature extraction: MFCC (13), RMS, ZCR, spectral centroid,
and optional spectral entropy. Aggregation across all coughs for global stats.
"""

import numpy as np
import librosa

from audio_processing import DEFAULT_SR

# Feature extraction defaults
N_MFCC = 13
N_FFT = 2048
HOP_LENGTH = 512


def extract_mfcc(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    n_mfcc: int = N_MFCC,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """Compute MFCCs and return mean across time (per coefficient)."""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfcc, axis=1)


def extract_rms(
    y: np.ndarray,
    frame_length: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> float:
    """Compute RMS energy and return mean."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    return float(np.mean(rms)) if len(rms) > 0 else 0.0


def extract_zcr(y: np.ndarray, frame_length: int = N_FFT, hop_length: int = HOP_LENGTH) -> float:
    """Compute zero-crossing rate and return mean."""
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    return float(np.mean(zcr)) if len(zcr) > 0 else 0.0


def extract_spectral_centroid(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> float:
    """Compute spectral centroid (Hz) and return mean."""
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    return float(np.mean(cent)) if len(cent) > 0 else 0.0


def extract_spectral_entropy(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> float:
    """Compute spectral entropy (normalized) per frame and return mean."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    # Normalize to get distribution per frame
    p = S / (np.sum(S, axis=0, keepdims=True) + 1e-10)
    p = np.clip(p, 1e-10, 1)
    entropy = -np.sum(p * np.log2(p), axis=0)
    # Normalize by max possible entropy (log2(n_bins))
    n_bins = S.shape[0]
    max_ent = np.log2(n_bins)
    norm_entropy = entropy / max_ent if max_ent > 0 else entropy
    return float(np.mean(norm_entropy)) if len(norm_entropy) > 0 else 0.0


def features_per_cough(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    include_entropy: bool = True,
) -> dict:
    """
    Extract all features for a single cough segment.
    Returns dict with mfcc_mean (length 13), rms, zcr, centroid, entropy (optional).
    """
    if len(y) < 256:
        # Too short: return zeros/neutral
        return {
            "mfcc_mean": np.zeros(N_MFCC),
            "rms": 0.0,
            "zcr": 0.0,
            "centroid": 0.0,
            "entropy": 0.0,
        }
    mfcc_mean = extract_mfcc(y, sr=sr)
    rms = extract_rms(y)
    zcr = extract_zcr(y)
    centroid = extract_spectral_centroid(y, sr=sr)
    entropy = extract_spectral_entropy(y, sr=sr) if include_entropy else 0.0
    return {
        "mfcc_mean": mfcc_mean,
        "rms": rms,
        "zcr": zcr,
        "centroid": centroid,
        "entropy": entropy,
    }


def aggregate_cough_features(
    cough_arrays: list[np.ndarray],
    wet_dry_labels: list[int],
    sr: int = DEFAULT_SR,
) -> dict:
    """
    Given list of cough waveforms and per-cough wet (1) / dry (0) labels,
    compute aggregate statistics:
    - mean of each MFCC (1..13)
    - mean_energy (RMS), mean_zcr, mean_centroid, mean_entropy
    - cough_count, wet_count, dry_count, wet_percentage
    """
    n = len(cough_arrays)
    if n == 0:
        return {
            "cough_count": 0,
            "wet_count": 0,
            "dry_count": 0,
            "wet_percentage": 0.0,
            "mean_energy": 0.0,
            "mean_zcr": 0.0,
            "mean_centroid": 0.0,
            "mean_entropy": 0.0,
            "mfcc_1_mean": 0.0,
            "mfcc_2_mean": 0.0,
            "mfcc_3_mean": 0.0,
            "mfcc_4_mean": 0.0,
            "mfcc_5_mean": 0.0,
            "mfcc_6_mean": 0.0,
            "mfcc_7_mean": 0.0,
            "mfcc_8_mean": 0.0,
            "mfcc_9_mean": 0.0,
            "mfcc_10_mean": 0.0,
            "mfcc_11_mean": 0.0,
            "mfcc_12_mean": 0.0,
            "mfcc_13_mean": 0.0,
        }
    wet_dry_labels = list(wet_dry_labels) if wet_dry_labels else [0] * n
    assert len(wet_dry_labels) == n

    all_feats = [features_per_cough(c, sr=sr) for c in cough_arrays]
    wet_count = sum(1 for l in wet_dry_labels if l == 1)
    dry_count = n - wet_count
    wet_pct = (wet_count / n * 100.0) if n > 0 else 0.0

    mean_energy = np.mean([f["rms"] for f in all_feats])
    mean_zcr = np.mean([f["zcr"] for f in all_feats])
    mean_centroid = np.mean([f["centroid"] for f in all_feats])
    mean_entropy = np.mean([f["entropy"] for f in all_feats])
    mfcc_means = np.mean([f["mfcc_mean"] for f in all_feats], axis=0)

    out = {
        "cough_count": n,
        "wet_count": wet_count,
        "dry_count": dry_count,
        "wet_percentage": wet_pct,
        "mean_energy": float(mean_energy),
        "mean_zcr": float(mean_zcr),
        "mean_centroid": float(mean_centroid),
        "mean_entropy": float(mean_entropy),
    }
    for i in range(N_MFCC):
        out[f"mfcc_{i+1}_mean"] = float(mfcc_means[i])
    return out
