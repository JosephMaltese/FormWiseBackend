from app.core.cues import build_cues
from app.core.faults import Fault, Severity
from app.core.scoring import CriterionResult, ScoreBreakdown
from tests.test_scoring import make_analysis

GUIDANCE = "Film from the side."


def breakdown(*criteria: CriterionResult) -> ScoreBreakdown:
    total = sum(result.earned for result in criteria)
    return ScoreBreakdown(total, "Solid", list(criteria))


def heel_fault() -> Fault:
    return Fault("heel_lift", "Heels lifting", Severity.MAJOR, "plant your feet", 20.0, 12.0)


def test_no_reps_produces_a_single_actionable_cue():
    cues, strengths = build_cues([], breakdown(), 0.95, GUIDANCE)
    assert [cue.code for cue in cues] == ["no_reps"]
    assert GUIDANCE in cues[0].message
    assert strengths == []


def test_poor_detection_is_surfaced_before_anything_else():
    cues, _ = build_cues([make_analysis()], breakdown(), 0.2, GUIDANCE)
    assert cues[0].code == "low_visibility"
    assert cues[0].severity == "major"


def test_faults_are_grouped_with_the_reps_they_affected():
    reps = [make_analysis(1, faults=[heel_fault()]), make_analysis(2),
            make_analysis(3, faults=[heel_fault()])]
    cues, _ = build_cues(reps, breakdown(), 1.0, GUIDANCE)
    heel = next(cue for cue in cues if cue.code == "heel_lift")
    assert heel.affected_reps == (1, 3)
    assert heel.message.startswith("Reps 1 and 3:")


def test_a_fault_on_every_rep_says_so():
    reps = [make_analysis(n, faults=[heel_fault()]) for n in (1, 2)]
    cues, _ = build_cues(reps, breakdown(), 1.0, GUIDANCE)
    assert cues[0].message.startswith("Every rep:")


def test_weak_criteria_are_explained_when_no_fault_covers_them():
    weak = CriterionResult("depth", "Squat depth", 25, 0.2, "Averaged 40 degrees.")
    cues, _ = build_cues([make_analysis()], breakdown(weak), 1.0, GUIDANCE)
    assert any(cue.code == "depth" for cue in cues)


def test_a_criterion_is_not_repeated_after_its_fault_cue():
    """'Your heels lifted' should not be followed by 'heel contact cost you 10'."""
    weak = CriterionResult("heel_contact", "Heel contact", 20, 0.5, "Flagged on 1 of 2 reps.",
                           covers=("heel_lift",))
    reps = [make_analysis(1, faults=[heel_fault()]), make_analysis(2)]
    cues, _ = build_cues(reps, breakdown(weak), 1.0, GUIDANCE)
    assert [cue.code for cue in cues] == ["heel_lift"]


def test_strong_criteria_are_reported_as_strengths():
    strong = CriterionResult("tempo", "Lowering tempo", 20, 1.0, "Averaged 2.3s.")
    _, strengths = build_cues([make_analysis()], breakdown(strong), 1.0, GUIDANCE)
    assert strengths == ["Lowering tempo: Averaged 2.3s."]
