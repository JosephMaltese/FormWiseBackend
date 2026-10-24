"""Conditioning for noisy per-frame angle measurements.

Pose landmarks jitter by a few degrees frame to frame, which is enough to fool
a naive "did the angle change by 5 degrees?" rep counter. Every angle series is
gap-filled and smoothed before any rep logic looks at it.
"""
from __future__ import annotations

import numpy as np


def fill_gaps(values: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN holes; edges are held at the nearest value."""
    series = np.asarray(values, dtype=float).copy()
    missing = np.isnan(series)
    if missing.all() or not missing.any():
        return series
    index = np.arange(series.size)
    series[missing] = np.interp(index[missing], index[~missing], series[~missing])
    return series


def moving_median(values: np.ndarray, window: int) -> np.ndarray:
    """Odd-length centred median filter; kills single-frame landmark spikes."""
    series = np.asarray(values, dtype=float)
    if window <= 1 or series.size == 0:
        return series.copy()
    width = window if window % 2 else window + 1
    width = min(width, series.size if series.size % 2 else series.size - 1) or 1
    if width <= 1:
        return series.copy()
    padded = np.pad(series, width // 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, width)
    return np.median(windows, axis=1)


def ema(values: np.ndarray, alpha: float) -> np.ndarray:
    series = np.asarray(values, dtype=float)
    if series.size == 0:
        return series.copy()
    out = np.empty_like(series)
    out[0] = series[0]
    for i in range(1, series.size):
        out[i] = alpha * series[i] + (1.0 - alpha) * out[i - 1]
    return out


def smooth(values: np.ndarray, median_window: int = 5, alpha: float = 0.45) -> np.ndarray:
    """Gap-fill, de-spike, then low-pass with a zero-phase EMA.

    The EMA is run forwards and backwards and averaged so that smoothing does
    not shift rep timings -- phase lag would quietly inflate every tempo
    measurement.
    """
    series = fill_gaps(values)
    if series.size == 0 or np.isnan(series).all():
        return series
    series = moving_median(series, median_window)
    forward = ema(series, alpha)
    backward = ema(series[::-1], alpha)[::-1]
    return (forward + backward) / 2.0


def velocity(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Degrees per second, via central differences."""
    series = np.asarray(values, dtype=float)
    stamps = np.asarray(times, dtype=float)
    if series.size < 2:
        return np.zeros_like(series)
    return np.gradient(series, stamps)
