"""Planar geometry helpers for pose landmarks.

All coordinates are MediaPipe-normalised image coordinates: ``x`` grows to the
right, ``y`` grows *downward*. Angles are returned in degrees.
"""
from __future__ import annotations

import math

Point = tuple[float, float]


def joint_angle(a: Point, b: Point, c: Point) -> float:
    """Interior angle at ``b`` formed by ``a-b-c``, in ``[0, 180]``."""
    bax, bay = a[0] - b[0], a[1] - b[1]
    bcx, bcy = c[0] - b[0], c[1] - b[1]
    mag = math.hypot(bax, bay) * math.hypot(bcx, bcy)
    if mag == 0.0:
        return float("nan")
    cosine = (bax * bcx + bay * bcy) / mag
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def segment_angle_to_vertical(start: Point, end: Point) -> float:
    """Angle of the segment ``start -> end`` away from the image vertical.

    ``0`` means perfectly upright (either direction), ``90`` means horizontal.
    """
    dx, dy = end[0] - start[0], end[1] - start[1]
    if dx == 0.0 and dy == 0.0:
        return float("nan")
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def segment_angle_to_horizontal(start: Point, end: Point) -> float:
    """Angle of the segment ``start -> end`` away from the image horizontal."""
    vertical = segment_angle_to_vertical(start, end)
    return float("nan") if math.isnan(vertical) else 90.0 - vertical


def distance(a: Point, b: Point) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def midpoint(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def facing_sign(back: Point, front: Point) -> float:
    """``+1`` when the subject faces screen-right, ``-1`` when screen-left.

    Derived from two landmarks that are front/back of each other in the
    sagittal plane (heel -> toe, for example).
    """
    delta = front[0] - back[0]
    if delta == 0.0:
        return 1.0
    return 1.0 if delta > 0 else -1.0


def signed_forward_offset(reference: Point, target: Point, facing: float) -> float:
    """Horizontal offset of ``target`` ahead of ``reference`` in the direction
    the subject faces. Positive means the target is further forward."""
    return (target[0] - reference[0]) * facing
