import numpy as np
import pytest

from app.core.reps import find_extrema, segment_reps
from app.core.signals import smooth


def synthetic_set(reps: int, top: float, bottom: float, period: float, fps: float = 15.0,
                  noise: float = 0.0, seed: int = 0):
    """A clean ``top -> bottom -> top`` angle signal, optionally with jitter."""
    samples = int(reps * period * fps) + 1
    times = np.arange(samples) / fps
    midpoint = (top + bottom) / 2.0
    amplitude = (top - bottom) / 2.0
    values = midpoint + amplitude * np.cos(2 * np.pi * times / period)
    if noise:
        values = values + np.random.default_rng(seed).normal(0, noise, values.size)
    return values, times


def test_extrema_alternate():
    values, _ = synthetic_set(4, 170.0, 60.0, period=2.0)
    kinds = [extremum.kind for extremum in find_extrema(values, prominence=15.0)]
    assert kinds == ["max", "min"] * (len(kinds) // 2) or kinds[:2] == ["max", "min"]
    assert all(a != b for a, b in zip(kinds, kinds[1:]))


def test_counts_full_reps_and_drops_the_trailing_half():
    values, times = synthetic_set(4, 170.0, 60.0, period=2.0)
    reps = segment_reps(values, times, min_range_of_motion=40.0, prominence=15.0)
    # Four cycles starting at the top yield three completed round trips plus a
    # final descent the clip never finishes.
    assert len(reps) == 3
    assert [rep.number for rep in reps] == [1, 2, 3]


def test_rep_timings_and_range_are_recovered():
    values, times = synthetic_set(3, 180.0, 80.0, period=3.0)
    rep = segment_reps(values, times, min_range_of_motion=40.0, prominence=15.0)[0]
    assert rep.range_of_motion == pytest.approx(100.0, abs=2.0)
    assert rep.bottom_angle == pytest.approx(80.0, abs=2.0)
    assert rep.descent_seconds == pytest.approx(1.5, abs=0.15)
    assert rep.ascent_seconds == pytest.approx(1.5, abs=0.15)
    assert rep.duration == pytest.approx(3.0, abs=0.2)


def test_landmark_jitter_does_not_invent_reps():
    """The old frame-to-frame delta counter fired on noise this size."""
    values = 150.0 + np.random.default_rng(3).normal(0, 4.0, 300)
    times = np.arange(values.size) / 15.0
    assert segment_reps(values, times, min_range_of_motion=40.0, prominence=15.0) == []


def test_partial_movement_below_min_rom_is_not_a_rep():
    values, times = synthetic_set(4, 150.0, 135.0, period=2.0)
    assert segment_reps(values, times, min_range_of_motion=40.0, prominence=5.0) == []


def test_pause_at_the_bottom_still_counts_once():
    descend = np.linspace(170.0, 70.0, 20)
    hold = np.full(25, 70.0) + np.random.default_rng(1).normal(0, 1.5, 25)
    ascend = np.linspace(70.0, 170.0, 20)
    values = smooth(np.concatenate([descend, hold, ascend, descend, hold, ascend]))
    times = np.arange(values.size) / 15.0
    reps = segment_reps(values, times, min_range_of_motion=40.0, prominence=15.0)
    assert len(reps) == 1  # the second round trip never returns to a confirmed top
    assert reps[0].bottom_angle == pytest.approx(70.0, abs=3.0)


def test_empty_input_is_safe():
    assert segment_reps([], [], min_range_of_motion=40.0, prominence=15.0) == []
