"""Synthetic landmark sequences: a lifter rendered as 33 points, no video.

Interpolating between a start pose and a bottom pose gives a smooth, fully
controllable angle trajectory, which lets the measurement -> rep -> fault ->
score path be tested without a pose model or a fixture video.
"""
from __future__ import annotations

import numpy as np

from app.core.landmarks import LANDMARK_COUNT, Landmark
from app.core.scoring import ScoringContext
from app.exercises.base import ExerciseAnalyzer
from app.exercises.recorder import MetricRecorder
from app.video.pipeline import evaluate_reps

SAMPLE_FPS = 15.0

Pose = dict[Landmark, tuple[float, float]]


def to_points(pose: Pose, visibility: float = 0.95) -> np.ndarray:
    points = np.zeros((LANDMARK_COUNT, 4), dtype=np.float32)
    for landmark, (x, y) in pose.items():
        points[int(landmark)] = (x, y, 0.0, visibility)
    return points


def blend(start: Pose, end: Pose, weight: float) -> Pose:
    return {
        landmark: (
            start[landmark][0] + (end[landmark][0] - start[landmark][0]) * weight,
            start[landmark][1] + (end[landmark][1] - start[landmark][1]) * weight,
        )
        for landmark in start
    }


def cycle(start: Pose, bottom: Pose, reps: int, seconds_per_rep: float = 3.0,
          fps: float = SAMPLE_FPS) -> list[np.ndarray]:
    """``start -> bottom -> start`` repeated, ending back at the start pose."""
    half = max(2, int(seconds_per_rep * fps / 2))
    frames: list[np.ndarray] = [to_points(start)]
    for _ in range(reps):
        for step in range(1, half + 1):
            frames.append(to_points(blend(start, bottom, step / half)))
        for step in range(1, half + 1):
            frames.append(to_points(blend(bottom, start, step / half)))
    # A couple of settled frames so the final top is confirmed as an extremum.
    frames.extend([to_points(start)] * 3)
    return frames


def analyse(analyzer: ExerciseAnalyzer, frames: list[np.ndarray], fps: float = SAMPLE_FPS):
    """Run the real measurement, rep-detection and scoring path over ``frames``."""
    recorder = MetricRecorder(analyzer)
    for index, points in enumerate(frames):
        recorder.add(index, index / fps, points)
    table = recorder.table()
    reps = evaluate_reps(analyzer, table)
    breakdown = analyzer.build_rubric().score(
        ScoringContext(reps=reps, table=table, direction=analyzer.spec.concentric)
    )
    return reps, breakdown, table


# -- squat -----------------------------------------------------------------
SQUAT_STAND: Pose = {
    Landmark.LEFT_SHOULDER: (0.50, 0.30),
    Landmark.LEFT_HIP: (0.50, 0.52),
    Landmark.LEFT_KNEE: (0.50, 0.72),
    Landmark.LEFT_ANKLE: (0.50, 0.92),
    Landmark.LEFT_HEEL: (0.48, 0.93),
    Landmark.LEFT_FOOT_INDEX: (0.58, 0.93),
}

SQUAT_BOTTOM: Pose = {
    Landmark.LEFT_SHOULDER: (0.38, 0.52),
    Landmark.LEFT_HIP: (0.44, 0.73),
    Landmark.LEFT_KNEE: (0.56, 0.74),
    Landmark.LEFT_ANKLE: (0.50, 0.92),
    Landmark.LEFT_HEEL: (0.48, 0.93),
    Landmark.LEFT_FOOT_INDEX: (0.58, 0.93),
}

SQUAT_SHALLOW: Pose = {
    Landmark.LEFT_SHOULDER: (0.42, 0.42),
    Landmark.LEFT_HIP: (0.47, 0.64),
    Landmark.LEFT_KNEE: (0.55, 0.74),
    Landmark.LEFT_ANKLE: (0.50, 0.92),
    Landmark.LEFT_HEEL: (0.48, 0.93),
    Landmark.LEFT_FOOT_INDEX: (0.58, 0.93),
}

#: Bottom position with the heel raised off the floor.
SQUAT_BOTTOM_HEEL_UP: Pose = {**SQUAT_BOTTOM, Landmark.LEFT_HEEL: (0.48, 0.885)}

#: Bottom position with the torso folded forward over the knees.
SQUAT_BOTTOM_FOLDED: Pose = {**SQUAT_BOTTOM, Landmark.LEFT_SHOULDER: (0.20, 0.66)}


# -- bicep curl ------------------------------------------------------------
CURL_EXTENDED: Pose = {
    Landmark.LEFT_SHOULDER: (0.50, 0.30),
    Landmark.LEFT_ELBOW: (0.50, 0.46),
    Landmark.LEFT_WRIST: (0.50, 0.62),
    Landmark.LEFT_HIP: (0.50, 0.60),
}

CURL_FLEXED: Pose = {
    Landmark.LEFT_SHOULDER: (0.50, 0.30),
    Landmark.LEFT_ELBOW: (0.50, 0.46),
    Landmark.LEFT_WRIST: (0.56, 0.33),
    Landmark.LEFT_HIP: (0.50, 0.60),
}

#: Flexed, but with the torso rocked back to launch the weight.
CURL_FLEXED_SWINGING: Pose = {**CURL_FLEXED, Landmark.LEFT_SHOULDER: (0.58, 0.31)}


# -- bench press -----------------------------------------------------------
BENCH_LOCKOUT: Pose = {
    Landmark.LEFT_SHOULDER: (0.50, 0.50),
    Landmark.LEFT_ELBOW: (0.50, 0.33),
    Landmark.LEFT_WRIST: (0.50, 0.16),
    Landmark.LEFT_HIP: (0.68, 0.52),
    Landmark.LEFT_KNEE: (0.84, 0.62),
    Landmark.LEFT_ANKLE: (0.80, 0.80),
}

BENCH_CHEST: Pose = {
    Landmark.LEFT_SHOULDER: (0.50, 0.50),
    Landmark.LEFT_ELBOW: (0.42, 0.515),
    Landmark.LEFT_WRIST: (0.47, 0.34),
    Landmark.LEFT_HIP: (0.68, 0.52),
    Landmark.LEFT_KNEE: (0.84, 0.62),
    Landmark.LEFT_ANKLE: (0.80, 0.80),
}

#: Bottom position reached by arching the hips up off the bench.
BENCH_CHEST_HIPS_UP: Pose = {**BENCH_CHEST, Landmark.LEFT_HIP: (0.68, 0.43)}
