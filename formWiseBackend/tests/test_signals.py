import numpy as np
import pytest

from app.core.signals import ema, fill_gaps, moving_median, smooth, velocity


def test_fill_gaps_interpolates_interior_and_holds_edges():
    filled = fill_gaps(np.array([np.nan, 1.0, np.nan, 3.0, np.nan]))
    assert filled == pytest.approx([1.0, 1.0, 2.0, 3.0, 3.0])


def test_fill_gaps_passes_through_all_nan():
    assert np.isnan(fill_gaps(np.array([np.nan, np.nan]))).all()


def test_moving_median_removes_a_single_frame_spike():
    noisy = np.array([10.0, 10.0, 95.0, 10.0, 10.0])
    assert moving_median(noisy, 3) == pytest.approx([10.0] * 5)


def test_ema_tracks_a_step_without_overshooting():
    stepped = ema(np.array([0.0, 0.0, 10.0, 10.0, 10.0]), alpha=0.5)
    assert stepped[0] == 0.0
    assert 0.0 < stepped[2] < 10.0
    assert stepped[-1] == pytest.approx(10.0, abs=2.0)


def test_smooth_does_not_shift_the_peak():
    """Zero-phase filtering matters: a lagged peak skews every rep timing."""
    times = np.linspace(0, 4, 200)
    clean = np.sin(2 * np.pi * times / 2.0)
    rng = np.random.default_rng(7)
    smoothed = smooth(clean + rng.normal(0, 0.05, clean.size))
    assert abs(int(np.argmax(smoothed)) - int(np.argmax(clean))) <= 2


def test_velocity_of_a_linear_ramp_is_constant():
    times = np.linspace(0.0, 1.0, 11)
    assert velocity(times * 30.0, times) == pytest.approx(np.full(11, 30.0))
