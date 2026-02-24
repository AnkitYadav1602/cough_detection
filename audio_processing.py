"""
audio_processing.py
-------------------
Energy-based cough detection and segmentation from WAV files.
Frames audio, computes short-time energy, applies threshold, and groups
consecutive high-energy frames into cough events. Filters out segments < 200 ms.
"""

import numpy as np
import librosa


# Default audio processing parameters
DEFAULT_SR = 22050
FRAME_LENGTH_MS = 25
HOP_LENGTH_MS = 10
MIN_COUGH_DURATION_MS = 200
MIN_SILENCE_MS = 100
ENERGY_MULTIPLIER = 2.0  # threshold = mean + multiplier * std


def load_audio(file_path: str, sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    """
    Load WAV file and return (samples, sample_rate).
    Resamples to target sr if needed.
    """
    y, orig_sr = librosa.load(file_path, sr=sr, mono=True)
    return y, orig_sr if orig_sr != sr else sr


def frame_audio(
    y: np.ndarray,
    sr: int,
    frame_length_ms: float = FRAME_LENGTH_MS,
    hop_length_ms: float = HOP_LENGTH_MS,
) -> tuple[np.ndarray, int, int]:
    """
    Convert signal to overlapping frames.
    Returns (frames 2D array, frame_length_samples, hop_length_samples).
    """
    frame_length = int(sr * frame_length_ms / 1000)
    hop_length = int(sr * hop_length_ms / 1000)
    frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    return frames, frame_length, hop_length


def short_time_energy(frames: np.ndarray) -> np.ndarray:
    """Compute short-time energy per frame: sum of squared samples."""
    return np.sum(frames ** 2, axis=0)


def detect_cough_segments(
    y: np.ndarray,
    sr: int = DEFAULT_SR,
    frame_length_ms: float = FRAME_LENGTH_MS,
    hop_length_ms: float = HOP_LENGTH_MS,
    min_cough_duration_ms: float = MIN_COUGH_DURATION_MS,
    min_silence_ms: float = MIN_SILENCE_MS,
    energy_multiplier: float = ENERGY_MULTIPLIER,
) -> list[tuple[int, int]]:
    """
    Energy-based cough segmentation.
    - Frame audio and compute short-time energy.
    - Threshold = mean(energy) + energy_multiplier * std(energy).
    - Group consecutive high-energy frames into events.
    - Merge events separated by less than min_silence_ms.
    - Drop segments shorter than min_cough_duration_ms.
    Returns list of (start_sample, end_sample) per cough.
    """
    frames, frame_length, hop_length = frame_audio(
        y, sr, frame_length_ms, hop_length_ms
    )
    ste = short_time_energy(frames)

    # Adaptive threshold
    thresh = np.mean(ste) + energy_multiplier * np.std(ste)
    if np.isnan(thresh) or thresh <= 0:
        thresh = np.max(ste) * 0.1

    # Binary mask: 1 where energy above threshold
    above = ste >= thresh

    # Convert frame indices to sample indices
    min_cough_frames = max(1, int(min_cough_duration_ms / hop_length_ms))
    min_silence_frames = max(0, int(min_silence_ms / hop_length_ms))

    segments = []
    in_segment = False
    seg_start_frame = 0

    for i in range(len(above)):
        if above[i] and not in_segment:
            in_segment = True
            seg_start_frame = i
        elif not above[i] and in_segment:
            in_segment = False
            seg_end_frame = i
            if seg_end_frame - seg_start_frame >= min_cough_frames:
                start_sample = seg_start_frame * hop_length
                end_sample = min(
                    seg_end_frame * hop_length + frame_length,
                    len(y),
                )
                segments.append((start_sample, end_sample))

    if in_segment:
        seg_end_frame = len(above)
        if seg_end_frame - seg_start_frame >= min_cough_frames:
            start_sample = seg_start_frame * hop_length
            end_sample = min(seg_end_frame * hop_length + frame_length, len(y))
            segments.append((start_sample, end_sample))

    # Merge segments that are very close (within min_silence)
    merged = []
    for start, end in segments:
        if merged and (start - merged[-1][1]) / sr * 1000 < min_silence_ms:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    return merged


def get_cough_audio_segments(
    file_path: str,
    sr: int = DEFAULT_SR,
    **kwargs,
) -> tuple[list[np.ndarray], int, list[tuple[int, int]]]:
    """
    Load WAV, detect coughs, return list of cough waveforms and metadata.
    Returns (list of cough arrays, sample_rate, list of (start, end) in samples).
    """
    y, sr = load_audio(file_path, sr=sr)
    segments = detect_cough_segments(y, sr=sr, **kwargs)
    coughs = [y[s:e] for s, e in segments]
    return coughs, sr, segments
