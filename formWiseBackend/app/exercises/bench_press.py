"""Barbell / dumbbell bench press."""
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
    SetupCriterion,
    TempoCriterion,
)
from app.exercises.base import AngleOverlay, ExerciseAnalyzer, ExerciseSpec, register


@register
class BenchPressAnalyzer(ExerciseAnalyzer):
    spec = ExerciseSpec(
        key="bench_press",
        name="Bench Press",
        summary="Horizontal press, judged on depth, lockout, bar path and bench contact.",
        camera_guidance=(
            "Film square from the side of the bench so your shoulder, elbow, wrist, "
            "hip, knee and ankle are all visible."
        ),
        primary_metric="elbow_angle",
        concentric=ConcentricDirection.ASCENDING,
        full_rom_bottom=80.0,
        full_rom_top=160.0,
        min_range_of_motion=40.0,
        extremum_prominence=15.0,
        tracking_joints=("shoulder", "elbow", "wrist", "hip"),
        overlays=(
            AngleOverlay("elbow_angle", ("shoulder", "elbow", "wrist"), "ELB"),
            AngleOverlay("torso_angle", ("shoulder", "hip", "knee"), "HIP"),
            AngleOverlay("knee_angle", ("hip", "knee", "ankle"), "KNEE"),
        ),
        metric_labels={
            "elbow_angle": "Elbow angle",
            "torso_angle": "Shoulder-hip-knee angle",
            "knee_angle": "Knee angle",
            "forearm_lean": "Forearm lean from vertical",
        },
        aliases=("bench press", "bench", "barbell bench press"),
    )

    fault_rules = (
        FaultRule(
            code="hip_lift",
            label="Hips leaving the bench",
            metric="torso_angle",
            stat="range",
            comparator="gt",
            threshold=12.0,
            severity=Severity.MAJOR,
            cue="your hips rose off the bench to help the bar up. Keep your glutes "
            "planted and drive through your feet instead.",
        ),
        FaultRule(
            code="shallow_press",
            label="Shallow depth",
            metric="elbow_angle",
            stat="at_bottom",
            comparator="gt",
            threshold=95.0,
            severity=Severity.MINOR,
            cue="the bar stopped well above your chest. Lower until it touches, or "
            "drop the weight until you can.",
        ),
        FaultRule(
            code="no_lockout",
            label="Missed lockout",
            metric="elbow_angle",
            stat="end",
            comparator="lt",
            threshold=155.0,
            severity=Severity.MINOR,
            cue="you started the next rep before locking out. Finish with straight "
            "elbows and a stable bar.",
        ),
        FaultRule(
            code="forearm_flare",
            label="Forearms off vertical",
            metric="forearm_lean",
            stat="at_bottom",
            comparator="gt",
            threshold=22.0,
            severity=Severity.MINOR,
            cue="at the bottom your forearms were angled rather than stacked under "
            "the bar. Adjust your grip width or tuck your elbows to about 45°.",
        ),
    )

    def measure(self, sample: PoseSample) -> dict[str, float]:
        shoulder = sample.xy("shoulder")
        elbow = sample.xy("elbow")
        wrist = sample.xy("wrist")
        hip = sample.xy("hip")
        knee = sample.xy("knee")
        ankle = sample.xy("ankle")
        leg_visible = sample.has("knee", "ankle", threshold=self.visibility_threshold)
        return {
            "elbow_angle": joint_angle(shoulder, elbow, wrist),
            "torso_angle": joint_angle(shoulder, hip, knee) if leg_visible else float("nan"),
            "knee_angle": joint_angle(hip, knee, ankle) if leg_visible else float("nan"),
            "forearm_lean": segment_angle_to_vertical(elbow, wrist),
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
            FaultFreeCriterion("bench_contact", "Bench contact", 25, codes=("hip_lift",)),
            FaultFreeCriterion("bar_path", "Bar path", 10, codes=("forearm_flare",)),
            SetupCriterion("leg_drive", "Leg position", 5, metric="knee_angle", low=45.0, high=135.0),
            TempoCriterion("tempo", "Lowering tempo", 20, phase="eccentric", ideal=(1.5, 3.5)),
            ConsistencyCriterion("consistency", "Rep consistency", 10, tolerance=12.0),
        )
