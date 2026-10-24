from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.landmarks import LANDMARK_COUNT, Landmark  # noqa: E402


@pytest.fixture
def blank_points() -> np.ndarray:
    """A fully visible, physically meaningless landmark array to mutate."""
    points = np.zeros((LANDMARK_COUNT, 4), dtype=np.float32)
    points[:, 3] = 1.0
    return points


@pytest.fixture
def place():
    def _place(points: np.ndarray, landmark: Landmark, x: float, y: float,
               visibility: float = 1.0) -> None:
        points[int(landmark)] = (x, y, 0.0, visibility)

    return _place
