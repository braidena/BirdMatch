"""
audio_features.py
==================
PROVIDED MODULE - you do not need to read or modify this file to complete
the assignment. It is documented here so you understand what the functions
give you, since everything you build depends on their output.

This module hides the "signal processing" layer of the pipeline: turning a
raw audio recording into a compact numeric fingerprint that can be compared
across recordings of different lengths. Internally it does:

    waveform  --STFT-->  spectrogram  --Mel filterbank-->  mel spectrogram
              --log + DCT (cepstral step)-->  MFCC feature matrix

You are NOT responsible for implementing any of that. Your job (in
dtw.py, search.py, and benchmark.py) starts from the feature matrices this
module produces.

Public API
----------
load_audio(path, sr=22050) -> (waveform: np.ndarray, sr: int)
extract_features(waveform, sr, n_mfcc=20) -> np.ndarray of shape (n_frames, n_mfcc)
features_from_file(path, sr=22050, n_mfcc=20) -> np.ndarray  (load + extract in one call)
plot_spectrogram(waveform, sr, title=None, ax=None)           (for debugging/plots)
save_features(path, feature_matrix) / load_features(path)      (npy cache helpers)
"""

from __future__ import annotations

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


DEFAULT_SR = 22050
DEFAULT_N_MFCC = 20
DEFAULT_N_FFT = 1024
DEFAULT_HOP_LENGTH = 256


def load_audio(path: str, sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    """
    Load an audio file from disk as a mono waveform.

    Parameters
    ----------
    path : str
        Path to a .wav/.mp3/.flac/... audio file.
    sr : int
        Target sample rate. The file is resampled to this rate if needed,
        so every clip you load ends up on the same time grid.

    Returns
    -------
    waveform : np.ndarray, shape (n_samples,)
        The audio signal as floating point samples in [-1, 1].
    sr : int
        The sample rate of `waveform` (equal to the `sr` argument).
    """
    waveform, sr = librosa.load(path, sr=sr, mono=True)
    return waveform, sr


def extract_features(
    waveform: np.ndarray,
    sr: int,
    n_mfcc: int = DEFAULT_N_MFCC,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    """
    Convert a raw waveform into an MFCC feature matrix.

    This is the black box: internally it computes the Short-Time Fourier
    Transform (STFT) of the waveform, maps the resulting spectrogram onto
    the Mel frequency scale (which approximates how pitch is perceived),
    takes the log magnitude, and applies a discrete cosine transform (DCT)
    to decorrelate the Mel bands into "cepstral" coefficients. The result
    is a sequence of feature vectors, one per short time window (frame),
    each summarizing the spectral shape of the audio at that instant.

    Parameters
    ----------
    waveform : np.ndarray, shape (n_samples,)
        Audio signal, e.g. from `load_audio`.
    sr : int
        Sample rate of `waveform`.
    n_mfcc : int
        Number of MFCC coefficients to keep per frame (feature dimension).
    n_fft : int
        FFT window size used internally for the STFT.
    hop_length : int
        Number of samples between successive frames (controls time
        resolution of the output).

    Returns
    -------
    features : np.ndarray, shape (n_frames, n_mfcc)
        One row per time frame, one column per MFCC coefficient.
        `n_frames` depends on the length of the input clip, which is why
        two recordings of the same bird will generally NOT produce
        feature matrices of the same shape -- this is exactly the
        problem DTW is designed to handle.
    """
    if waveform.size == 0:
        raise ValueError("extract_features: waveform is empty")

    mfcc = librosa.feature.mfcc(
        y=waveform, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    # librosa returns shape (n_mfcc, n_frames); we transpose so that each
    # row is one time frame, which is the convention used throughout the
    # rest of this assignment (a feature matrix is a *sequence* of frames).
    features = mfcc.T.astype(np.float64)

    # Per-coefficient z-score normalization across time. This keeps the
    # scale of each MFCC coefficient comparable, which matters a lot for
    # any distance-based comparison (Euclidean, DTW, ...).
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True) + 1e-8
    features = (features - mean) / std

    return features


def features_from_file(
    path: str,
    sr: int = DEFAULT_SR,
    n_mfcc: int = DEFAULT_N_MFCC,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
) -> np.ndarray:
    """Convenience wrapper: load_audio + extract_features in one call."""
    waveform, sr = load_audio(path, sr=sr)
    return extract_features(
        waveform, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )


def plot_spectrogram(waveform: np.ndarray, sr: int, title: str | None = None, ax=None):
    """
    Plot a log-frequency spectrogram of `waveform`. Useful for sanity
    checking a clip and for figures in your write-up. Not required by
    the autograder.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(waveform, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP_LENGTH)),
        ref=np.max,
    )
    img = librosa.display.specshow(
        D, sr=sr, hop_length=DEFAULT_HOP_LENGTH, x_axis="time", y_axis="log", ax=ax
    )
    if title:
        ax.set_title(title)
    return ax


def save_features(path: str, feature_matrix: np.ndarray) -> None:
    """Save a feature matrix to disk as a .npy file (creates parent dirs)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.save(path, feature_matrix)


def load_features(path: str) -> np.ndarray:
    """Load a feature matrix previously saved with `save_features`."""
    return np.load(path)


def summarize_features(feature_matrix: np.ndarray) -> np.ndarray:
    """
    Collapse a variable-length feature matrix (n_frames, n_mfcc) into a
    single FIXED-LENGTH vector of shape (2 * n_mfcc,), by concatenating
    the per-coefficient mean and standard deviation across all frames.

    Why you need this: DTW compares two variable-length sequences frame by
    frame, which is accurate but expensive (see dtw.py). Indexing
    structures like LSH, k-d trees, or FAISS all expect every item to be a
    point in the SAME fixed-dimensional space -- they have no notion of
    "align two sequences of different lengths" built in. This function
    gives you that fixed-dimensional point: a rough summary of a clip's
    overall spectral content, thrown away all timing/order information.

    This is the standard "cheap filter, expensive rerank" trick used by
    real large-scale audio search systems (and by search engines in
    general): use a fast approximate representation to narrow millions of
    candidates down to a shortlist, then run the accurate-but-slow
    comparison (DTW) only on that shortlist. You'll build exactly this
    two-stage pipeline in search.py.

    Parameters
    ----------
    feature_matrix : np.ndarray, shape (n_frames, n_mfcc)

    Returns
    -------
    summary : np.ndarray, shape (2 * n_mfcc,)
        [mean of each coefficient across frames] concatenated with
        [std of each coefficient across frames].
    """
    if feature_matrix.ndim != 2 or feature_matrix.shape[0] == 0:
        raise ValueError("summarize_features: expected a non-empty (n_frames, n_mfcc) array")
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    return np.concatenate([mean, std]).astype(np.float64)


# ---------------------------------------------------------------------------
# Dataset preparation helpers. The dataset was built with these exact
# settings, so any NEW clip you want to use as a query must go through
# `clip_features` with the same arguments to be comparable.
# ---------------------------------------------------------------------------

DEFAULT_MAX_SECONDS = 10.0
DEFAULT_POOL = 4


def loudest_window(waveform: np.ndarray, sr: int, seconds: float) -> np.ndarray:
    """
    Return the `seconds`-long stretch of `waveform` with the most energy
    (clips shorter than `seconds` are returned unchanged). Recordings often
    start or end with silence or handling noise; the loudest stretch is
    usually the bird.
    """
    win = int(seconds * sr)
    if waveform.size <= win:
        return waveform
    hop = sr // 2
    energy = waveform.astype(np.float64) ** 2
    csum = np.concatenate([[0.0], np.cumsum(energy)])
    starts = np.arange(0, waveform.size - win + 1, hop)
    totals = csum[starts + win] - csum[starts]
    best = int(starts[np.argmax(totals)])
    return waveform[best:best + win]


def pool_frames(features: np.ndarray, pool: int) -> np.ndarray:
    """
    Average every `pool` consecutive frames of a (n_frames, n_mfcc) matrix.
    DTW costs O(n*m), so pooling by 4 makes it about 16x cheaper while
    keeping the overall shape of the call.
    """
    if pool <= 1:
        return features
    n = (features.shape[0] // pool) * pool
    if n == 0:
        return features
    return features[:n].reshape(n // pool, pool, features.shape[1]).mean(axis=1)


def clip_features(
    path: str,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    pool: int = DEFAULT_POOL,
) -> np.ndarray:
    """
    LEGACY pipeline (per-clip z-scored MFCC). Kept for reference only: it
    erased each clip's average spectrum and gave poor species matching.
    Use `variant_clip_features` below.
    """
    waveform, sr = load_audio(path)
    waveform = loudest_window(waveform, sr, max_seconds)
    feats = extract_features(waveform, sr)
    return pool_frames(feats, pool).astype(np.float32)


# ---------------------------------------------------------------------------
# Feature variants (no per-clip normalization)
#
# Why: z-scoring every clip across time removes the clip's average spectrum,
# which is the most species-specific thing in it. These variants keep the
# absolute spectral shape, restrict to the 1-10 kHz band where most songbird
# energy lives (this also drops low-frequency wind/traffic rumble that would
# otherwise make recordings from the same noisy place look alike), and are
# standardized ONCE across the whole reference set (`fit_standardizer`)
# instead of once per clip.
# ---------------------------------------------------------------------------

FMIN, FMAX = 1000.0, 10000.0
VARIANTS = ("mfcc", "mfcc_cmn", "pcen", "logmel_floor")


def variant_features(waveform: np.ndarray, sr: int, variant: str) -> np.ndarray:
    """
    Unpooled, un-standardized features, shape (n_frames, d).

    mfcc         MFCC coefficients 1..20 (c0, the overall loudness, is dropped)
    mfcc_cmn     same, minus each clip's own mean per coefficient
    pcen         per-channel energy normalization of a 40-band mel spectrogram
                 (designed for bioacoustics; suppresses steady background noise)
    logmel_floor 40-band log-mel spectrogram minus its per-band median over
                 time (removes the stationary noise floor)
    """
    kw = dict(y=waveform, sr=sr, n_fft=DEFAULT_N_FFT, hop_length=DEFAULT_HOP_LENGTH,
              fmin=FMIN, fmax=min(FMAX, sr / 2 - 1))
    if variant in ("mfcc", "mfcc_cmn"):
        m = librosa.feature.mfcc(n_mfcc=21, n_mels=64, **kw)[1:].T
        if variant == "mfcc_cmn":
            m = m - m.mean(axis=0, keepdims=True)
        return m.astype(np.float64)
    if variant == "pcen":
        S = librosa.feature.melspectrogram(n_mels=40, power=1.0, **kw)
        return librosa.pcen(S * (2 ** 31), sr=sr, hop_length=DEFAULT_HOP_LENGTH).T.astype(np.float64)
    if variant == "logmel_floor":
        S = librosa.feature.melspectrogram(n_mels=40, power=2.0, **kw)
        L = np.log10(S + 1e-10)
        return (L - np.median(L, axis=1, keepdims=True)).T.astype(np.float64)
    raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")


def raw_variant_features(path, variants=VARIANTS, max_seconds=DEFAULT_MAX_SECONDS, pool=DEFAULT_POOL):
    """Decode `path` once and return {variant: pooled un-standardized features}."""
    waveform, sr = load_audio(path)
    waveform = loudest_window(waveform, sr, max_seconds)
    return {v: pool_frames(variant_features(waveform, sr, v), pool) for v in variants}


def fit_standardizer(feature_matrices) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean and std over ALL frames of the given matrices
    (pass the reference set only, so queries never influence it)."""
    allf = np.vstack(list(feature_matrices))
    return allf.mean(axis=0), allf.std(axis=0) + 1e-8


def apply_standardizer(features: np.ndarray, stats: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    mean, std = stats
    return ((features - mean) / std).astype(np.float32)


def variant_clip_features(path, variant, stats, max_seconds=DEFAULT_MAX_SECONDS, pool=DEFAULT_POOL):
    """Features for ONE new clip (e.g. an unknown query song), processed
    exactly like the dataset. `stats` comes from feature_stats.npz."""
    raw = raw_variant_features(path, (variant,), max_seconds, pool)[variant]
    return apply_standardizer(raw, stats)
