"""Back / front squat, filmed from the side."""
from __future__ import annotations

import math

from app.core.evaluation import ConcentricDirection
from app.core.faults import FaultRule, Severity
from app.core.geometry import (
    distance,
    facing_sign,
    joint_angle,
    segment_angle_to_horizontal,
    segment_angle_to_vertical,
    signed_forward_offset,
)
from app.core.landmarks import PoseSample
from app.core.scoring import (
    ConsistencyCriterion,
    FaultFreeCriterion,
    MetricTargetCriterion,
    Rubric,
    TempoCriterion,
)
from app.exercises.base import AngleOverlay, ExerciseAnalyzer, ExerciseSpec, register


@register
class SquatAnalyzer(ExerciseAnalyzer):
    spec = ExerciseSpec(
        key="squat",
        name="Squat",
        summary="Knee and hip flexion, judged on depth, heel contact, torso angle and knee travel.",
        camera_guidance=(
            "Film from the side at hip height, far enough back that your feet and "
            "head stay in frame for the whole set."
        ),
        primary_metric="knee_angle",
        concentric=ConcentricDirection.ASCENDING,
        full_rom_bottom=90.0,
        full_rom_top=165.0,
        min_range_of_motion=35.0,
        extremum_prominence=15.0,
        tracking_joints=("hip", "knee", "ankle", "shoulder"),
        overlays=(
            AngleOverlay("knee_angle", ("hip", "knee", "ankle"), "KNEE"),
            AngleOverlay("hip_angle", ("shoulder", "hip", "knee"), "HIP"),
            AngleOverlay("torso_lean", ("hip", "shoulder"), "TORSO"),
        ),
        metric_labels={
            "knee_angle": "Knee angle",
            "hip_angle": "Hip angle",
            "thigh_angle": "Thigh angle above horizontal",
            "torso_lean": "Torso lean from vertical",
            "foot_angle": "Foot angle from the floor",
            "knee_travel": "Knee travel past the toes (shin lengths)",
        },
        aliases=("squat", "back squat", "barbell squat", "air squat"),
    )

    fault_rules = (
        FaultRule(
            code="heel_lift",
            label="Heels lifting",
            metric="foot_angle",
            stat="range",
            comparator="gt",
            threshold=12.0,
            severity=Severity.MAJOR,
            cue="your heels came off the floor, which dumps the load onto your toes. "
            "Sit back into your midfoot, widen your stance slightly, or work on "
            "ankle mobility.",
        ),
        FaultRule(
            code="shallow_depth",
            label="Above parallel",
            metric="thigh_angle",
            stat="at_bottom",
            comparator="gt",
            threshold=20.0,
            severity=Severity.MAJOR,
            cue="you stopped above parallel. Aim to bring your hip crease level with "
            "the top of your knee.",
        ),
        FaultRule(
            code="excessive_lean",
            label="Excessive forward lean",
            metric="torso_lean",
            stat="max",
            comparator="gt",
            threshold=55.0,
            severity=Severity.MINOR,
            cue="your torso folded forward, turning the squat into a good morning. "
            "Keep your chest up and your weight over the middle of your foot.",
        ),
        FaultRule(
            code="knee_travel",
            label="Knees past the toes",
            metric="knee_travel",
            stat="at_bottom",
            comparator="gt",
            threshold=0.5,
            severity=Severity.MINOR,
            cue="your knees shot well past your toes. Start the descent by sitting "
            "your hips back, not by bending the knees first.",
            unit=" shin lengths",
        ),
        FaultRule(
            code="no_lockout",
            label="Missed lockout",
            metric="knee_angle",
            stat="end",
            comparator="lt",
            threshold=160.0,
            severity=Severity.MINOR,
            cue="you started the next rep before standing all the way up. Finish each "
            "rep with your hips and knees locked.",
        ),
    )

    def measure(self, sample: PoseSample) -> dict[str, float]:
        hip = sample.xy("hip")
        knee = sample.xy("knee")
        ankle = sample.xy("ankle")
        shoulder = sample.xy("shoulder")
        metrics = {
            "knee_angle": joint_angle(hip, knee, ankle),
            "hip_angle": joint_angle(shoulder, hip, knee),
            "thigh_angle": abs(segment_angle_to_horizontal(knee, hip)),
            "torso_lean": segment_angle_to_vertical(hip, shoulder),
            "foot_angle": math.nan,
            "knee_travel": math.nan,
        }

        if sample.has("heel", "toe", threshold=self.visibility_threshold):
            heel = sample.xy("heel")
            toe = sample.xy("toe")
            metrics["foot_angle"] = abs(segment_angle_to_horizontal(heel, toe))
            shin = distance(knee, ankle)
            if shin > 1e-6:
                facing = facing_sign(heel, toe)
                metrics["knee_travel"] = signed_forward_offset(toe, knee, facing) / shin
        return metrics

    def build_rubric(self) -> Rubric:
        return Rubric(
            MetricTargetCriterion(
                "depth",
                "Squat depth",
                25,
                metric="thigh_angle",
                stat="at_bottom",
                full_credit=5.0,
                no_credit=45.0,
            ),
            FaultFreeCriterion("heel_contact", "Heel contact", 20, codes=("heel_lift",)),
            FaultFreeCriterion("torso_position", "Torso position", 15, codes=("excessive_lean",)),
            FaultFreeCriterion("knee_tracking", "Knee tracking", 10, codes=("knee_travel",)),
            FaultFreeCriterion("lockout", "Lockout", 5, codes=("no_lockout",)),
            TempoCriterion("tempo", "Lowering tempo", 15, phase="eccentric", ideal=(1.5, 3.5)),
            ConsistencyCriterion("consistency", "Rep consistency", 10, tolerance=12.0),
        )
