"""End-to-end checks per exercise, driven by synthetic landmark sequences."""
from __future__ import annotations

import pytest

from app.core.landmarks import Landmark, Side
from app.exercises import available_specs, get_analyzer, resolve_key
from tests.synthetic import (
    BENCH_CHEST,
    BENCH_CHEST_HIPS_UP,
    BENCH_LOCKOUT,
    CURL_EXTENDED,
    CURL_FLEXED,
    CURL_FLEXED_SWINGING,
    SQUAT_BOTTOM,
    SQUAT_BOTTOM_FOLDED,
    SQUAT_BOTTOM_HEEL_UP,
    SQUAT_SHALLOW,
    SQUAT_STAND,
    analyse,
    cycle,
    to_points,
)


def codes(reps) -> set[str]:
    return {fault.code for rep in reps for fault in rep.faults}


# -- registry --------------------------------------------------------------
def test_every_registered_rubric_sums_to_100():
    for spec in available_specs():
        assert get_analyzer(spec.key).build_rubric()  # raises if the weights are wrong


def test_legacy_exercise_labels_still_resolve():
    """The original API took 'BENCH PRESS'-style strings; keep accepting them."""
    assert resolve_key("BENCH PRESS") == "bench_press"
    assert resolve_key("BICEP CURL") == "bicep_curl"
    assert resolve_key("SQUAT") == "squat"
    assert resolve_key("deadlift") is None


def test_overlay_metrics_are_actually_measured():
    for spec in available_specs():
        analyzer = get_analyzer(spec.key)
        for overlay in spec.overlays:
            assert overlay.metric in analyzer.metric_names
            assert 2 <= len(overlay.joints) <= 3


def test_fault_rules_reference_declared_metrics():
    for spec in available_specs():
        analyzer = get_analyzer(spec.key)
        for rule in analyzer.fault_rules:
            assert rule.metric in analyzer.metric_names


# -- squat -----------------------------------------------------------------
def test_squat_counts_reps_and_scores_a_clean_set_highly():
    reps, breakdown, _ = analyse(get_analyzer("squat"), cycle(SQUAT_STAND, SQUAT_BOTTOM, reps=4))
    assert len(reps) == 3
    assert breakdown.total >= 90
    assert codes(reps) == set()
    assert all(rep.quality.value == "full" for rep in reps)


def test_squat_flags_a_shallow_set_and_scores_it_lower():
    deep, deep_score, _ = analyse(get_analyzer("squat"), cycle(SQUAT_STAND, SQUAT_BOTTOM, reps=4))
    shallow, shallow_score, _ = analyse(
        get_analyzer("squat"), cycle(SQUAT_STAND, SQUAT_SHALLOW, reps=4)
    )
    assert "shallow_depth" in codes(shallow)
    assert shallow_score.total < deep_score.total - 10


def test_squat_flags_heel_lift():
    reps, _, _ = analyse(get_analyzer("squat"), cycle(SQUAT_STAND, SQUAT_BOTTOM_HEEL_UP, reps=4))
    assert "heel_lift" in codes(reps)


def test_squat_flags_a_folded_torso():
    reps, _, _ = analyse(get_analyzer("squat"), cycle(SQUAT_STAND, SQUAT_BOTTOM_FOLDED, reps=4))
    assert "excessive_lean" in codes(reps)


def test_squat_standing_still_is_not_a_rep():
    frames = [to_points(SQUAT_STAND)] * 120
    reps, breakdown, _ = analyse(get_analyzer("squat"), frames)
    assert reps == []
    assert breakdown.total == 0.0


# -- bicep curl ------------------------------------------------------------
def test_curl_counts_reps_and_measures_full_range():
    reps, breakdown, _ = analyse(
        get_analyzer("bicep_curl"), cycle(CURL_EXTENDED, CURL_FLEXED, reps=4)
    )
    assert len(reps) == 3
    assert reps[0].rep.bottom_angle < 45
    assert reps[0].rep.top_angle > 155
    assert breakdown.total >= 90


def test_curl_flags_torso_swing():
    clean, clean_score, _ = analyse(
        get_analyzer("bicep_curl"), cycle(CURL_EXTENDED, CURL_FLEXED, reps=4)
    )
    swung, swung_score, _ = analyse(
        get_analyzer("bicep_curl"), cycle(CURL_EXTENDED, CURL_FLEXED_SWINGING, reps=4)
    )
    assert "torso_swing" in codes(swung)
    assert codes(clean) == set()
    assert swung_score.total < clean_score.total - 20


def test_curl_phases_are_the_reverse_of_a_squat():
    """Flexion is the concentric here, so the lowering is the *ascent*."""
    reps, _, _ = analyse(
        get_analyzer("bicep_curl"), cycle(CURL_EXTENDED, CURL_FLEXED, reps=3, seconds_per_rep=4.0)
    )
    rep = reps[0]
    assert rep.concentric_seconds == pytest.approx(rep.rep.descent_seconds)
    assert rep.eccentric_seconds == pytest.approx(rep.rep.ascent_seconds)


# -- bench press -----------------------------------------------------------
def test_bench_counts_reps_on_a_clean_set():
    reps, breakdown, _ = analyse(
        get_analyzer("bench_press"), cycle(BENCH_LOCKOUT, BENCH_CHEST, reps=4)
    )
    assert len(reps) == 3
    assert codes(reps) == set()
    assert breakdown.total >= 90


def test_bench_flags_hips_leaving_the_bench():
    reps, breakdown, _ = analyse(
        get_analyzer("bench_press"), cycle(BENCH_LOCKOUT, BENCH_CHEST_HIPS_UP, reps=4)
    )
    assert "hip_lift" in codes(reps)
    assert breakdown.total < 80


def test_bench_legs_are_scored_from_the_whole_set():
    _, breakdown, _ = analyse(
        get_analyzer("bench_press"), cycle(BENCH_LOCKOUT, BENCH_CHEST, reps=4)
    )
    leg_drive = next(result for result in breakdown.criteria if result.key == "leg_drive")
    assert leg_drive.ratio == 1.0


# -- side selection --------------------------------------------------------
def test_the_better_lit_side_is_tracked():
    """Filming from the right is as valid as filming from the left."""
    analyzer = get_analyzer("squat")
    mirrored = {
        Landmark(int(landmark) + 1): position for landmark, position in SQUAT_STAND.items()
    }
    points = to_points(mirrored)
    points[int(Landmark.LEFT_HIP)][3] = 0.05
    assert analyzer.select_side(points) is Side.RIGHT


def test_an_invisible_subject_yields_no_measurements():
    analyzer = get_analyzer("squat")
    points = to_points(SQUAT_STAND, visibility=0.05)
    metrics, side = analyzer.measure_frame(points)
    assert side is None
    assert all(value != value for value in metrics.values())  # all NaN


def test_measurements_are_mirror_invariant():
    """A left-side and right-side recording of the same squat must agree."""
    analyzer = get_analyzer("squat")
    mirrored = {
        Landmark(int(landmark) + 1): (1.0 - x, y) for landmark, (x, y) in SQUAT_BOTTOM.items()
    }
    left, left_side = analyzer.measure_frame(to_points(SQUAT_BOTTOM))
    right, right_side = analyzer.measure_frame(to_points(mirrored))
    assert (left_side, right_side) == (Side.LEFT, Side.RIGHT)
    for metric, value in left.items():
        assert right[metric] == pytest.approx(value, abs=1e-4), metric


def test_aspect_ratio_correction_changes_measured_angles():
    """Raw normalised coordinates squash angles on non-square video."""
    analyzer = get_analyzer("squat")
    points = to_points(SQUAT_BOTTOM)
    square, _ = analyzer.measure_frame(points, aspect=1.0)
    wide, _ = analyzer.measure_frame(points, aspect=16 / 9)
    assert square["knee_angle"] != pytest.approx(wide["knee_angle"], abs=1.0)
