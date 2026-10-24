"""Pipeline wiring: landmark interpolation and per-frame overlay state."""
from __future__ import annotations

import numpy as np
import pytest

from app.core.evaluation import ConcentricDirection
from app.core.landmarks import LANDMARK_COUNT
from app.video.pipeline import LandmarkTrack, _frame_state, _timeline
from tests.test_scoring import major, make_analysis, minor


def points(value: float) -> np.ndarray:
    return np.full((LANDMARK_COUNT, 4), value, dtype=np.float32)


def test_sampled_frames_are_returned_exactly():
    track = LandmarkTrack([0, 2, 4], [points(0.0), points(1.0), points(2.0)], max_gap=4)
    assert track.at(2)[0][0] == pytest.approx(1.0)


def test_intermediate_frames_are_interpolated():
    """Holding the previous sample instead would make the skeleton stutter."""
    track = LandmarkTrack([0, 4], [points(0.0), points(4.0)], max_gap=8)
    assert track.at(1)[0][0] == pytest.approx(1.0)
    assert track.at(3)[0][0] == pytest.approx(3.0)


def test_a_gap_wider_than_the_limit_is_not_interpolated_across():
    track = LandmarkTrack([0, 100], [points(0.0), points(100.0)], max_gap=4)
    assert track.at(50)[0][0] == pytest.approx(0.0)


def test_frames_without_a_detection_stay_empty():
    track = LandmarkTrack([0, 2], [None, None], max_gap=4)
    assert track.at(0) is None
    assert track.at(1) is None


def test_frames_beyond_the_last_sample_fall_back_within_the_gap():
    track = LandmarkTrack([0, 2], [points(0.0), points(2.0)], max_gap=4)
    assert track.at(3)[0][0] == pytest.approx(2.0)
    assert track.at(99) is None


def test_an_empty_track_is_safe():
    assert LandmarkTrack([], [], max_gap=4).at(0) is None


# -- overlay state ---------------------------------------------------------
def test_phase_labels_follow_the_concentric_direction():
    reps = [make_analysis(1)]  # descends 0.0 -> 2.0s, ascends 2.0 -> 3.5s
    squat = _frame_state(1.0, reps, ConcentricDirection.ASCENDING, {})
    curl = _frame_state(1.0, reps, ConcentricDirection.DESCENDING, {})
    assert (squat.phase, curl.phase) == ("Eccentric", "Concentric")
    assert _frame_state(3.0, reps, ConcentricDirection.ASCENDING, {}).phase == "Concentric"


def test_frames_between_reps_are_labelled_setup():
    state = _frame_state(99.0, [make_analysis(1)], ConcentricDirection.ASCENDING, {})
    assert state.phase == "Setup"
    assert state.rep_number is None


def test_frame_state_carries_the_rep_faults_for_the_chips():
    reps = [make_analysis(1, faults=[major()])]
    assert _frame_state(1.0, reps, ConcentricDirection.ASCENDING, {}).faults == ("Heels lifting",)


def test_timeline_colours_reps_by_worst_fault():
    reps = [
        make_analysis(1),
        make_analysis(2, faults=[minor()]),
        make_analysis(3, faults=[major(), minor()]),
    ]
    assert [entry.color_key for entry in _timeline(reps)] == ["good", "warn", "bad"]
