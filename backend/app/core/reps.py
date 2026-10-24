"""Repetition segmentation from a single primary joint-angle signal.

A rep is a round trip of the primary angle: ``peak -> trough -> peak``. Extrema
are confirmed with a prominence (hysteresis) band rather than by comparing
consecutive frames, so a lifter pausing at the bottom or a landmark wobbling by
a couple of degrees cannot manufacture a rep.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

ExtremumKind = Literal["min", "max"]


@dataclass(frozen=True)
class Extremum:
    kind: ExtremumKind
    index: int
    value: float
    time: float


@dataclass(frozen=True)
class Rep:
    """One detected repetition, indexed into the sampled frame timeline."""

    number: int
    start_index: int
    bottom_index: int
    end_index: int
    start_time: float
    bottom_time: float
    end_time: float
    start_angle: float
    bottom_angle: float
    end_angle: float

    @property
    def top_angle(self) -> float:
        """The conservative lockout: the weaker of the two end positions."""
        return min(self.start_angle, self.end_angle)

    @property
    def range_of_motion(self) -> float:
        return self.top_angle - self.bottom_angle

    @property
    def descent_seconds(self) -> float:
        return self.bottom_time - self.start_time

    @property
    def ascent_seconds(self) -> float:
        return self.end_time - self.bottom_time

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def find_extrema(values: Sequence[float], prominence: float) -> list[Extremum]:
    """Alternating minima/maxima confirmed by a ``prominence`` degree reversal."""
    series = np.asarray(values, dtype=float)
    if series.size == 0 or prominence <= 0:
        return []

    extrema: list[Extremum] = []
    high = low = float(series[0])
    high_index = low_index = 0
    direction = 0  # -1 falling, +1 rising, 0 unknown

    for index in range(series.size):
        value = float(series[index])
        if np.isnan(value):
            continue
        if value > high:
            high, high_index = value, index
        if value < low:
            low, low_index = value, index

        if direction >= 0 and value <= high - prominence:
            extrema.append(Extremum("max", high_index, high, 0.0))
            direction = -1
            low, low_index = value, index
        elif direction <= 0 and value >= low + prominence:
            extrema.append(Extremum("min", low_index, low, 0.0))
            direction = 1
            high, high_index = value, index

    return extrema


def segment_reps(
    values: Sequence[float],
    times: Sequence[float],
    min_range_of_motion: float,
    prominence: float,
) -> list[Rep]:
    """Group confirmed extrema into ``max -> min -> max`` repetitions.

    A trailing half-rep (the video stops before the lifter returns to the top)
    is intentionally dropped -- it has no measurable concentric phase.
    """
    stamps = np.asarray(times, dtype=float)
    extrema = [
        Extremum(e.kind, e.index, e.value, float(stamps[e.index]))
        for e in find_extrema(values, prominence)
        if e.index < stamps.size
    ]

    reps: list[Rep] = []
    for top, bottom, next_top in zip(extrema, extrema[1:], extrema[2:]):
        if (top.kind, bottom.kind, next_top.kind) != ("max", "min", "max"):
            continue
        if min(top.value, next_top.value) - bottom.value < min_range_of_motion:
            continue
        reps.append(
            Rep(
                number=len(reps) + 1,
                start_index=top.index,
                bottom_index=bottom.index,
                end_index=next_top.index,
                start_time=top.time,
                bottom_time=bottom.time,
                end_time=next_top.time,
                start_angle=top.value,
                bottom_angle=bottom.value,
                end_angle=next_top.value,
            )
        )
    return reps
