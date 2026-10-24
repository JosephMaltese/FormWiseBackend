"""Dumbbell / barbell bicep curl."""
from __future__ import annotations

from app.core.evaluation import ConcentricDirection
from app.core.faults import FaultRule, Severity
from app.core.geometry import joint_angle, segment_angle_to_vertical
from app.core.landmarks import PoseSample
from app.core.scoring import (
    ConsistencyCriterion,
    FaultFreeCriterion,
    RangeOfMotionCriterion,
    Rubric,
    TempoCriterion,
)
from app.exercises.base import AngleOverlay, ExerciseAnalyzer, ExerciseSpec, register


@register
class BicepCurlAnalyzer(ExerciseAnalyzer):
    spec = ExerciseSpec(
        key="bicep_curl",
        name="Bicep Curl",
        summary="Elbow flexion under load, judged on range, torso stability and tempo.",
        camera_guidance=(
            "Film from the side, 2-3 m away, with your shoulder, elbow, wrist and "
            "hip all in frame."
        ),
        primary_metric="elbow_angle",
        concentric=ConcentricDirection.DESCENDING,
        full_rom_bottom=45.0,
        full_rom_top=155.0,
        min_range_of_motion=45.0,
        extremum_prominence=18.0,
        tracking_joints=("shoulder", "elbow", "wrist", "hip"),
        overlays=(
            AngleOverlay("elbow_angle", ("shoulder", "elbow", "wrist"), "ELB"),
            AngleOverlay("torso_lean", ("hip", "shoulder"), "TORSO"),
        ),
        metric_labels={
            "elbow_angle": "Elbow angle",
            "torso_lean": "Torso lean from vertical",
            "upper_arm_angle": "Upper-arm angle from vertical",
        },
        aliases=("bicep curl", "curl", "biceps curl"),
    )

    fault_rules = (
        FaultRule(
            code="torso_swing",
            label="Torso swing",
            metric="torso_lean",
            stat="range",
            comparator="gt",
            threshold=9.0,
            severity=Severity.MAJOR,
            cue="you swung your torso to start the curl. Brace your core and keep your "
            "chest still so the biceps move the weight, not momentum.",
        ),
        FaultRule(
            code="elbow_drift",
            label="Elbow drift",
            metric="upper_arm_angle",
            stat="range",
            comparator="gt",
            threshold=16.0,
            severity=Severity.MINOR,
            cue="your elbow travelled forward as you curled, handing the work to your "
            "front delt. Pin your upper arm against your ribs.",
        ),
        FaultRule(
            code="short_curl",
            label="Incomplete contraction",
            metric="elbow_angle",
            stat="at_bottom",
            comparator="gt",
            threshold=60.0,
            severity=Severity.MINOR,
            cue="you stopped short of a full squeeze. Curl until your forearm almost "
            "touches your biceps.",
        ),
        FaultRule(
            code="short_extension",
            label="Incomplete extension",
            metric="elbow_angle",
            stat="end",
            comparator="lt",
            threshold=150.0,
            severity=Severity.MINOR,
            cue="you cut the lowering short. Straighten the arm fully between reps -- "
            "the stretched position is where most of the growth happens.",
        ),
    )

    def measure(self, sample: PoseSample) -> dict[str, float]:
        shoulder = sample.xy("shoulder")
        elbow = sample.xy("elbow")
        wrist = sample.xy("wrist")
        hip = sample.xy("hip")
        return {
            "elbow_angle": joint_angle(shoulder, elbow, wrist),
            "torso_lean": segment_angle_to_vertical(hip, shoulder),
            "upper_arm_angle": segment_angle_to_vertical(shoulder, elbow),
        }

    def build_rubric(self) -> Rubric:
        return Rubric(
            RangeOfMotionCriterion(
                "range_of_motion",
                "Range of motion",
                30,
                full_bottom=self.spec.full_rom_bottom,
                full_top=self.spec.full_rom_top,
            ),
            FaultFreeCriterion("torso_stability", "Torso stability", 25, codes=("torso_swing",)),
            FaultFreeCriterion("elbow_position", "Elbow position", 15, codes=("elbow_drift",)),
            TempoCriterion("tempo", "Lowering tempo", 20, phase="eccentric", ideal=(1.5, 3.5)),
            ConsistencyCriterion("consistency", "Rep consistency", 10, tolerance=12.0),
        )
