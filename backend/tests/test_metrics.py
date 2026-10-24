import math

import numpy as np
import pytest

from app.core.metrics import STAT_KEYS, MetricTable


def table(values, detected=None) -> MetricTable:
    times = [index / 10.0 for index in range(len(values))]
    return MetricTable(
        times=times,
        frame_indices=list(range(len(values))),
        raw={"angle": values},
        detected=detected if detected is not None else [True] * len(values),
        median_window=1,
        alpha=1.0,  # no smoothing, so the assertions are about the table itself
    )


def test_stats_summarise_a_rep_window():
    stats = table([100.0, 80.0, 60.0, 80.0, 100.0]).stats("angle", 0, 4, 2)
    assert stats.minimum == 60.0
    assert stats.maximum == 100.0
    assert stats.start == 100.0
    assert stats.end == 100.0
    assert stats.at_bottom == 60.0
    assert stats.span == 40.0


def test_every_documented_stat_key_resolves():
    stats = table([10.0, 20.0, 30.0]).stats("angle", 0, 2, 1)
    assert all(not math.isnan(stats.get(key)) for key in STAT_KEYS)


def test_unknown_stat_key_is_rejected():
    with pytest.raises(KeyError, match="unknown stat"):
        table([1.0, 2.0]).stats("angle", 0, 1, 0).get("median")


def test_window_bounds_are_clamped():
    stats = table([10.0, 20.0, 30.0]).stats("angle", -5, 99, 1)
    assert (stats.start, stats.end) == (10.0, 30.0)


def test_missing_metric_returns_nan_stats_instead_of_raising():
    stats = table([10.0, 20.0]).stats("elbow_angle", 0, 1, 0)
    assert math.isnan(stats.mean) and math.isnan(stats.span)


def test_nan_holes_are_interpolated_before_statistics():
    stats = table([10.0, float("nan"), 30.0]).stats("angle", 0, 2, 1)
    assert stats.at_bottom == pytest.approx(20.0)


def test_an_all_nan_metric_stays_nan():
    nan = float("nan")
    stats = table([nan, nan, nan]).stats("angle", 0, 2, 1)
    assert math.isnan(stats.minimum)


def test_detection_rate_reports_usable_frames():
    assert table([1.0] * 4, detected=[True, True, False, False]).detection_rate == 0.5
    assert table([], detected=[]).detection_rate == 0.0


def test_whole_set_stats_cover_the_entire_clip():
    stats = table([10.0, 50.0, 30.0]).whole_set_stats()["angle"]
    assert (stats.minimum, stats.maximum) == (10.0, 50.0)


def test_value_at_is_bounds_safe():
    subject = table([10.0, 20.0])
    assert subject.value_at("angle", 1) == 20.0
    assert math.isnan(subject.value_at("angle", 99))
    assert math.isnan(subject.value_at("angle", -1))


def test_smoothing_is_applied_to_the_series_the_reps_are_read_from():
    spiky = MetricTable(
        times=[index / 10.0 for index in range(5)],
        frame_indices=list(range(5)),
        raw={"angle": [100.0, 100.0, 180.0, 100.0, 100.0]},
        detected=[True] * 5,
        median_window=3,
        alpha=0.5,
    )
    assert np.max(spiky.series("angle")) < 120.0
    assert np.max(spiky.raw["angle"]) == 180.0
