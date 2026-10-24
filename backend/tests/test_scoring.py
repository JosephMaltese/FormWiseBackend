import pytest

from app.core.evaluation import (
    ConcentricDirection,
    RepAnalysis,
    RepQuality,
    classify_quality,
    phase_durations,
)
from app.core.faults import Fault, FaultRule, Severity
from app.core.metrics import MetricStats, MetricTable
from app.core.reps import Rep
from app.core.scoring import (
    ConsistencyCriterion,
    FaultFreeCriterion,
    MetricTargetCriterion,
    RangeOfMotionCriterion,
    Rubric,
    ScoringContext,
    SetupCriterion,
    TempoCriterion,
    grade_for,
    ramp,
)


def make_rep(number: int = 1, *, bottom: float = 80.0, top: float = 170.0,
             descent: float = 2.0, ascent: float = 1.5) -> Rep:
    return Rep(
        number=number,
        start_index=0, bottom_index=10, end_index=20,
        start_time=0.0, bottom_time=descent, end_time=descent + ascent,
        start_angle=top, bottom_angle=bottom, end_angle=top,
    )


def make_analysis(number: int = 1, *, faults: list[Fault] | None = None, bottom: float = 80.0,
                  top: float = 170.0, stats: dict[str, MetricStats] | None = None,
                  eccentric: float = 2.0, concentric: float = 1.5) -> RepAnalysis:
    return RepAnalysis(
        rep=make_rep(number, bottom=bottom, top=top),
        stats=stats or {},
        faults=faults or [],
        quality=RepQuality.FULL if bottom <= 90 and top >= 165 else RepQuality.PARTIAL,
        eccentric_seconds=eccentric,
        concentric_seconds=concentric,
    )


def context(reps) -> ScoringContext:
    table = MetricTable([0.0, 1.0], [0, 1], {"knee_angle": [170.0, 80.0]}, [True, True])
    return ScoringContext(reps=reps, table=table, direction=ConcentricDirection.ASCENDING)


def major(code: str = "heel_lift") -> Fault:
    return Fault(code, "Heels lifting", Severity.MAJOR, "plant your feet", 20.0, 12.0)


def minor(code: str = "excessive_lean") -> Fault:
    return Fault(code, "Forward lean", Severity.MINOR, "chest up", 60.0, 55.0)


# -- primitives ------------------------------------------------------------
def test_ramp_gives_partial_credit_between_thresholds():
    assert ramp(5.0, full_credit=5.0, no_credit=45.0) == 1.0
    assert ramp(25.0, full_credit=5.0, no_credit=45.0) == pytest.approx(0.5)
    assert ramp(60.0, full_credit=5.0, no_credit=45.0) == 0.0
    assert ramp(float("nan"), 5.0, 45.0) == 0.0


def test_grades_track_the_bands():
    assert grade_for(95.0) == "Excellent"
    assert grade_for(80.0) == "Solid"
    assert grade_for(0.0) == "Poor"


# -- rubric wiring ---------------------------------------------------------
def test_rubric_rejects_weights_that_do_not_sum_to_100():
    with pytest.raises(ValueError, match="sum to 100"):
        Rubric(TempoCriterion("tempo", "Tempo", 60))


def test_zero_reps_scores_zero_without_dividing_by_zero():
    """The original scorers raised ZeroDivisionError on a set with no reps."""
    rubric = Rubric(
        RangeOfMotionCriterion("rom", "ROM", 50, full_bottom=90.0, full_top=165.0),
        TempoCriterion("tempo", "Tempo", 50),
    )
    breakdown = rubric.score(context([]))
    assert breakdown.total == 0.0
    assert all(result.detail == "No reps detected." for result in breakdown.criteria)


def test_perfect_set_cannot_exceed_100():
    rubric = Rubric(
        RangeOfMotionCriterion("rom", "ROM", 40, full_bottom=90.0, full_top=165.0),
        FaultFreeCriterion("clean", "Clean", 30, codes=("heel_lift",)),
        TempoCriterion("tempo", "Tempo", 30, ideal=(1.5, 3.5)),
    )
    reps = [make_analysis(n, bottom=70.0, top=175.0) for n in range(1, 6)]
    assert rubric.score(context(reps)).total == 100.0


# -- individual criteria ---------------------------------------------------
def test_fault_free_criterion_weights_major_above_minor():
    criterion = FaultFreeCriterion("clean", "Clean", 100, codes=("heel_lift", "excessive_lean"))
    all_major = criterion.evaluate(context([make_analysis(1, faults=[major()])]))
    all_minor = criterion.evaluate(context([make_analysis(1, faults=[minor()])]))
    assert all_major.ratio == 0.0
    assert all_minor.ratio == pytest.approx(0.5)


def test_fault_free_criterion_ignores_unrelated_faults():
    criterion = FaultFreeCriterion("clean", "Clean", 100, codes=("heel_lift",))
    assert criterion.evaluate(context([make_analysis(1, faults=[minor()])])).ratio == 1.0


def test_fault_free_criterion_exposes_the_codes_it_covers():
    criterion = FaultFreeCriterion("clean", "Clean", 100, codes=("heel_lift",))
    assert criterion.evaluate(context([make_analysis()])).covers == ("heel_lift",)


def test_tempo_rewards_control_and_penalises_a_dropped_eccentric():
    criterion = TempoCriterion("tempo", "Tempo", 100, ideal=(1.5, 3.5), bounds=(0.4, 7.0))
    assert criterion.evaluate(context([make_analysis(eccentric=2.5)])).ratio == 1.0
    assert criterion.evaluate(context([make_analysis(eccentric=0.5)])).ratio < 0.2
    assert criterion.evaluate(context([make_analysis(eccentric=6.5)])).ratio < 0.2


def test_consistency_penalises_depth_drifting_across_a_set():
    criterion = ConsistencyCriterion("c", "Consistency", 100, tolerance=10.0, no_credit=40.0)
    steady = [make_analysis(n, bottom=80.0) for n in range(1, 5)]
    fading = [make_analysis(n, bottom=80.0 + 25.0 * n) for n in range(1, 5)]
    collapsing = [make_analysis(n, bottom=80.0 + 40.0 * n) for n in range(1, 5)]
    assert criterion.evaluate(context(steady)).ratio == 1.0
    assert 0.0 < criterion.evaluate(context(fading)).ratio < 0.5
    assert criterion.evaluate(context(collapsing)).ratio == 0.0


def test_single_rep_is_treated_as_consistent():
    criterion = ConsistencyCriterion("c", "Consistency", 100)
    assert criterion.evaluate(context([make_analysis()])).ratio == 1.0


def test_metric_target_criterion_averages_partial_credit():
    stats = {"thigh_angle": MetricStats(5.0, 40.0, 20.0, 40.0, 40.0, 25.0)}
    criterion = MetricTargetCriterion(
        "depth", "Depth", 100, metric="thigh_angle", stat="at_bottom",
        full_credit=5.0, no_credit=45.0,
    )
    assert criterion.evaluate(context([make_analysis(stats=stats)])).ratio == pytest.approx(0.5)


def test_criteria_report_unmeasurable_metrics_rather_than_crashing():
    nan = float("nan")
    stats = {"thigh_angle": MetricStats(nan, nan, nan, nan, nan, nan)}
    criterion = MetricTargetCriterion(
        "depth", "Depth", 100, metric="thigh_angle", stat="at_bottom",
        full_credit=5.0, no_credit=45.0,
    )
    result = criterion.evaluate(context([make_analysis(stats=stats)]))
    assert result.ratio == 0.0
    assert "camera angle" in result.detail


def test_setup_criterion_checks_a_whole_set_average():
    table = MetricTable([0.0, 1.0], [0, 1], {"knee_angle": [90.0, 90.0]}, [True, True])
    criterion = SetupCriterion("legs", "Legs", 100, metric="knee_angle", low=45.0, high=135.0)
    ctx = ScoringContext([make_analysis()], table, ConcentricDirection.ASCENDING)
    assert criterion.evaluate(ctx).ratio == 1.0


def test_range_of_motion_rewards_depth_and_lockout_separately():
    criterion = RangeOfMotionCriterion("rom", "ROM", 100, full_bottom=90.0, full_top=165.0)
    deep_no_lockout = criterion.evaluate(context([make_analysis(bottom=70.0, top=140.0)]))
    full = criterion.evaluate(context([make_analysis(bottom=70.0, top=175.0)]))
    assert 0.4 < deep_no_lockout.ratio < 0.7
    assert full.ratio == 1.0


# -- fault rules -----------------------------------------------------------
def test_fault_rule_fires_only_past_its_threshold():
    rule = FaultRule("heel_lift", "Heels lifting", "foot_angle", "range", "gt", 12.0,
                     Severity.MAJOR, "plant your feet")
    tripped = rule.evaluate({"foot_angle": MetricStats(0.0, 20.0, 8.0, 0.0, 0.0, 18.0)})
    assert tripped is not None and tripped.measured == 20.0
    assert rule.evaluate({"foot_angle": MetricStats(0.0, 6.0, 3.0, 0.0, 0.0, 5.0)}) is None


def test_fault_rule_stays_silent_when_the_metric_was_never_visible():
    nan = float("nan")
    rule = FaultRule("heel_lift", "Heels lifting", "foot_angle", "range", "gt", 12.0,
                     Severity.MAJOR, "plant your feet")
    assert rule.evaluate({"foot_angle": MetricStats(nan, nan, nan, nan, nan, nan)}) is None
    assert rule.evaluate({}) is None


def test_fault_load_saturates_at_one():
    analysis = make_analysis(faults=[major(), minor()])
    assert analysis.fault_load == 1.0
    assert not analysis.is_clean


# -- phase mapping ---------------------------------------------------------
def test_phase_mapping_follows_the_declared_concentric_direction():
    rep = make_rep(descent=2.0, ascent=1.0)
    assert phase_durations(rep, ConcentricDirection.ASCENDING) == (2.0, 1.0)
    assert phase_durations(rep, ConcentricDirection.DESCENDING) == (1.0, 2.0)


def test_quality_requires_both_depth_and_lockout():
    assert classify_quality(make_rep(bottom=80.0, top=170.0), 90.0, 165.0) is RepQuality.FULL
    assert classify_quality(make_rep(bottom=120.0, top=170.0), 90.0, 165.0) is RepQuality.PARTIAL
    assert classify_quality(make_rep(bottom=80.0, top=140.0), 90.0, 165.0) is RepQuality.PARTIAL
