"""Per-frame metric storage and per-rep summary statistics.

The pipeline makes a single pass over the video recording whatever angles an
exercise declares, then every downstream question -- rep boundaries, faults,
scoring -- is answered from this table. Keeping measurement and interpretation
apart is what stops each new exercise from becoming another 250-line state
machine.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from app.core.signals import smooth

STAT_KEYS = ("min", "max", "mean", "range", "start", "end", "at_bottom")


@dataclass(frozen=True)
class MetricStats:
    """Summary of one metric over one rep window."""

    minimum: float
    maximum: float
    mean: float
    start: float
    end: float
    at_bottom: float

    @property
    def span(self) -> float:
        if np.isnan(self.minimum) or np.isnan(self.maximum):
            return float("nan")
        return self.maximum - self.minimum

    def get(self, key: str) -> float:
        match key:
            case "min":
                return self.minimum
            case "max":
                return self.maximum
            case "mean":
                return self.mean
            case "range":
                return self.span
            case "start":
                return self.start
            case "end":
                return self.end
            case "at_bottom":
                return self.at_bottom
        raise KeyError(f"unknown stat {key!r}; expected one of {STAT_KEYS}")


_EMPTY_STATS = MetricStats(*([float("nan")] * 6))


class MetricTable:
    """Time-aligned, smoothed metric series for every analysed frame."""

    def __init__(
        self,
        times: Iterable[float],
        frame_indices: Iterable[int],
        raw: dict[str, Iterable[float]],
        detected: Iterable[bool],
        *,
        median_window: int = 5,
        alpha: float = 0.45,
    ) -> None:
        self.times = np.asarray(list(times), dtype=float)
        self.frame_indices = np.asarray(list(frame_indices), dtype=int)
        self.detected = np.asarray(list(detected), dtype=bool)
        self.raw = {name: np.asarray(list(values), dtype=float) for name, values in raw.items()}
        self.smoothed = {
            name: smooth(values, median_window=median_window, alpha=alpha)
            for name, values in self.raw.items()
        }

    def __len__(self) -> int:
        return int(self.times.size)

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(self.smoothed)

    @property
    def detection_rate(self) -> float:
        if self.detected.size == 0:
            return 0.0
        return float(self.detected.mean())

    def series(self, name: str) -> np.ndarray:
        return self.smoothed[name]

    def value_at(self, name: str, index: int) -> float:
        series = self.smoothed.get(name)
        if series is None or not 0 <= index < series.size:
            return float("nan")
        return float(series[index])

    def stats(self, name: str, start: int, end: int, bottom: int) -> MetricStats:
        """Summarise ``name`` over the inclusive window ``[start, end]``."""
        series = self.smoothed.get(name)
        if series is None or series.size == 0:
            return _EMPTY_STATS
        lo = max(0, min(start, series.size - 1))
        hi = max(lo, min(end, series.size - 1))
        window = series[lo : hi + 1]
        if window.size == 0 or np.isnan(window).all():
            return _EMPTY_STATS
        return MetricStats(
            minimum=float(np.nanmin(window)),
            maximum=float(np.nanmax(window)),
            mean=float(np.nanmean(window)),
            start=float(series[lo]),
            end=float(series[hi]),
            at_bottom=self.value_at(name, bottom),
        )

    def stats_bundle(self, start: int, end: int, bottom: int) -> dict[str, MetricStats]:
        return {name: self.stats(name, start, end, bottom) for name in self.smoothed}

    def whole_set_stats(self) -> dict[str, MetricStats]:
        last = max(0, len(self) - 1)
        return self.stats_bundle(0, last, last)
