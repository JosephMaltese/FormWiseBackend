import math

import pytest

from app.core.geometry import (
    facing_sign,
    joint_angle,
    segment_angle_to_horizontal,
    segment_angle_to_vertical,
    signed_forward_offset,
)


def test_joint_angle_right_angle():
    assert joint_angle((0.0, 1.0), (0.0, 0.0), (1.0, 0.0)) == pytest.approx(90.0)


def test_joint_angle_straight_limb():
    assert joint_angle((0.0, 0.0), (0.0, 1.0), (0.0, 2.0)) == pytest.approx(180.0)


def test_joint_angle_is_undefined_for_coincident_points():
    assert math.isnan(joint_angle((0.0, 0.0), (0.0, 0.0), (1.0, 0.0)))


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ((0.0, 0.0), (0.0, -1.0), 0.0),    # straight up
        ((0.0, 0.0), (0.0, 1.0), 0.0),     # straight down, same deviation
        ((0.0, 0.0), (1.0, 0.0), 90.0),    # horizontal
        ((0.0, 0.0), (1.0, -1.0), 45.0),
    ],
)
def test_segment_angle_to_vertical(start, end, expected):
    assert segment_angle_to_vertical(start, end) == pytest.approx(expected)


def test_vertical_and_horizontal_are_complementary():
    start, end = (0.2, 0.4), (0.5, 0.1)
    total = segment_angle_to_vertical(start, end) + segment_angle_to_horizontal(start, end)
    assert total == pytest.approx(90.0)


def test_facing_and_forward_offset_are_mirror_symmetric():
    # Facing right: knee ahead of the toe is a positive offset...
    assert facing_sign((0.3, 1.0), (0.5, 1.0)) == 1.0
    assert signed_forward_offset((0.5, 1.0), (0.6, 0.6), facing=1.0) == pytest.approx(0.1)
    # ...and the mirrored setup must give the same answer.
    assert facing_sign((0.5, 1.0), (0.3, 1.0)) == -1.0
    assert signed_forward_offset((0.3, 1.0), (0.2, 0.6), facing=-1.0) == pytest.approx(0.1)
