"""Side-aware access to BlazePose's 33 landmarks.

Indices mirror MediaPipe Pose. They are duplicated here on purpose so that the
analysis layer stays importable (and unit-testable) without MediaPipe.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np

from app.core.geometry import Point


class Landmark(IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


LANDMARK_COUNT = 33


class Side(str, Enum):
    LEFT = "left"
    RIGHT = "right"

    @property
    def other(self) -> Side:
        return Side.RIGHT if self is Side.LEFT else Side.LEFT


#: Joint names an exercise can ask for, resolved against the active side.
JOINTS: dict[str, tuple[Landmark, Landmark]] = {
    "shoulder": (Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER),
    "elbow": (Landmark.LEFT_ELBOW, Landmark.RIGHT_ELBOW),
    "wrist": (Landmark.LEFT_WRIST, Landmark.RIGHT_WRIST),
    "index": (Landmark.LEFT_INDEX, Landmark.RIGHT_INDEX),
    "hip": (Landmark.LEFT_HIP, Landmark.RIGHT_HIP),
    "knee": (Landmark.LEFT_KNEE, Landmark.RIGHT_KNEE),
    "ankle": (Landmark.LEFT_ANKLE, Landmark.RIGHT_ANKLE),
    "heel": (Landmark.LEFT_HEEL, Landmark.RIGHT_HEEL),
    "toe": (Landmark.LEFT_FOOT_INDEX, Landmark.RIGHT_FOOT_INDEX),
    "ear": (Landmark.LEFT_EAR, Landmark.RIGHT_EAR),
}

#: Torso/limb skeleton used for rendering. The face mesh is deliberately
#: omitted -- it adds clutter without telling a lifter anything.
SKELETON_EDGES: tuple[tuple[Landmark, Landmark], ...] = (
    (Landmark.LEFT_SHOULDER, Landmark.RIGHT_SHOULDER),
    (Landmark.LEFT_SHOULDER, Landmark.LEFT_ELBOW),
    (Landmark.LEFT_ELBOW, Landmark.LEFT_WRIST),
    (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_ELBOW),
    (Landmark.RIGHT_ELBOW, Landmark.RIGHT_WRIST),
    (Landmark.LEFT_SHOULDER, Landmark.LEFT_HIP),
    (Landmark.RIGHT_SHOULDER, Landmark.RIGHT_HIP),
    (Landmark.LEFT_HIP, Landmark.RIGHT_HIP),
    (Landmark.LEFT_HIP, Landmark.LEFT_KNEE),
    (Landmark.LEFT_KNEE, Landmark.LEFT_ANKLE),
    (Landmark.LEFT_ANKLE, Landmark.LEFT_HEEL),
    (Landmark.LEFT_HEEL, Landmark.LEFT_FOOT_INDEX),
    (Landmark.LEFT_ANKLE, Landmark.LEFT_FOOT_INDEX),
    (Landmark.RIGHT_HIP, Landmark.RIGHT_KNEE),
    (Landmark.RIGHT_KNEE, Landmark.RIGHT_ANKLE),
    (Landmark.RIGHT_ANKLE, Landmark.RIGHT_HEEL),
    (Landmark.RIGHT_HEEL, Landmark.RIGHT_FOOT_INDEX),
    (Landmark.RIGHT_ANKLE, Landmark.RIGHT_FOOT_INDEX),
)

_LEFT_LANDMARKS = frozenset(left for left, _ in JOINTS.values())


def resolve(joint: str, side: Side) -> Landmark:
    left, right = JOINTS[joint]
    return left if side is Side.LEFT else right


@dataclass(frozen=True)
class PoseSample:
    """One frame of landmarks, read through the lens of a chosen body side.

    Coordinates are returned in units of frame *height*: MediaPipe normalises x
    and y independently, so on anything other than a square video the raw values
    squash every angle toward the long axis. ``aspect`` (width / height) undoes
    that before any trigonometry happens.
    """

    points: np.ndarray  # shape (33, 4): x, y, z, visibility
    side: Side
    aspect: float = 1.0

    def xy(self, joint: str) -> Point:
        return self.landmark_xy(resolve(joint, self.side))

    def mirrored_xy(self, joint: str) -> Point:
        return self.landmark_xy(resolve(joint, self.side.other))

    def landmark_xy(self, landmark: Landmark) -> Point:
        row = self.points[int(landmark)]
        return (float(row[0]) * self.aspect, float(row[1]))

    def visibility(self, joint: str) -> float:
        return float(self.points[int(resolve(joint, self.side))][3])

    def has(self, *joints: str, threshold: float) -> bool:
        return all(self.visibility(joint) >= threshold for joint in joints)


def side_visibility(points: np.ndarray, joints: tuple[str, ...], side: Side) -> float:
    """Mean visibility of ``joints`` on ``side`` for a landmark array."""
    if not joints:
        return 0.0
    indices = [int(resolve(joint, side)) for joint in joints]
    return float(np.mean(points[indices, 3]))


def choose_side(
    points: np.ndarray, joints: tuple[str, ...], threshold: float
) -> Side | None:
    """Pick the body side the camera can actually see.

    Lifters film themselves from whichever side is convenient; locking the
    analysis to the left limbs (as the original implementation did) throws away
    half of all uploads.
    """
    scores = {side: side_visibility(points, joints, side) for side in Side}
    best = max(scores, key=lambda side: scores[side])
    return best if scores[best] >= threshold else None
