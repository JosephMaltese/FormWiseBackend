"""Collects per-frame measurements into a :class:`MetricTable`.

Pulled out of the pipeline so the measurement -> rep -> score path can be
exercised against synthetic landmark sequences, with no video and no pose model
in the loop.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from app.core.landmarks import Side
from app.core.metrics import MetricTable
from app.exercises.base import ExerciseAnalyzer


class MetricRecorder:
    def __init__(self, analyzer: ExerciseAnalyzer, aspect: float = 1.0) -> None:
        self._analyzer = analyzer
        self._aspect = aspect
        self._times: list[float] = []
        self._frame_indices: list[int] = []
        self._detected: list[bool] = []
        self._series: dict[str, list[float]] = {name: [] for name in analyzer.metric_names}
        self.points: list[np.ndarray | None] = []
        self.sides: dict[int, Side] = {}

    def add(self, frame_index: int, timestamp: float, points: np.ndarray | None) -> Side | None:
        if points is None:
            values = {name: math.nan for name in self._series}
            side = None
        else:
            values, side = self._analyzer.measure_frame(points, self._aspect)

        self._frame_indices.append(frame_index)
        self._times.append(timestamp)
        self._detected.append(side is not None)
        self.points.append(points)
        if side is not None:
            self.sides[frame_index] = side
        for name, bucket in self._series.items():
            bucket.append(values[name])
        return side

    def __len__(self) -> int:
        return len(self._times)

    @property
    def frame_indices(self) -> list[int]:
        return self._frame_indices

    @property
    def dominant_side(self) -> Side | None:
        if not self.sides:
            return None
        return Counter(self.sides.values()).most_common(1)[0][0]

    def table(self, median_window: int = 5, alpha: float = 0.45) -> MetricTable:
        return MetricTable(
            times=self._times,
            frame_indices=self._frame_indices,
            raw=self._series,
            detected=self._detected,
            median_window=median_window,
            alpha=alpha,
        )
