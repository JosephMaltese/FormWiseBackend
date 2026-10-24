"""Thin wrapper around MediaPipe Pose.

Isolating the dependency here keeps the measurement, rep-detection and scoring
layers importable (and unit-testable) without loading a TensorFlow graph.
"""
from __future__ import annotations

from types import TracebackType

import cv2 as cv
import mediapipe as mp
import numpy as np

from app.config import Settings
from app.core.landmarks import LANDMARK_COUNT

_mp_pose = mp.solutions.pose


class PoseEstimator:
    """Detects 33 body landmarks per frame, as ``(33, 4)`` x/y/z/visibility."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pose: object | None = None

    def __enter__(self) -> PoseEstimator:
        self._pose = _mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self._settings.pose_model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=self._settings.min_detection_confidence,
            min_tracking_confidence=self._settings.min_tracking_confidence,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None

    def estimate(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        if self._pose is None:
            raise RuntimeError("PoseEstimator must be used as a context manager")
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        if results.pose_landmarks is None:
            return None
        points = np.empty((LANDMARK_COUNT, 4), dtype=np.float32)
        for index, landmark in enumerate(results.pose_landmarks.landmark):
            points[index] = (landmark.x, landmark.y, landmark.z, landmark.visibility)
        return points
