# FormWiseBackend

A FastAPI service that scores weightlifting form from a phone video. It runs
MediaPipe BlazePose over the clip, measures the joint angles that matter for the
chosen lift, segments the set into reps, flags specific form faults, and returns
a 0–100 score with a per-rep breakdown plus an annotated copy of the video.

Supported lifts: **squat**, **bench press**, **bicep curl**.

## Quick start

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Interactive API docs: <http://127.0.0.1:8000/docs>

```bash
pytest          # 112 tests, ~1s (no video or pose model needed)
ruff check .
```

## Using the API

Analysis takes seconds to minutes. Uploads return a job ID, and clients poll for
the result while processing runs in the background.

```bash
# 1. submit a set
curl -X POST http://127.0.0.1:8000/api/v1/analyses \
     -F "file=@set.mp4" -F "exercise=squat"
# -> 202 {"id": "...", "status": "queued", "status_url": "...", "annotated_video_url": "..."}

# 2. poll until status is "completed" (progress is 0..1 while it runs)
curl http://127.0.0.1:8000/api/v1/analyses/$ID

# 3. fetch the annotated clip
curl http://127.0.0.1:8000/api/v1/analyses/$ID/video -o annotated.mp4
```

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/analyses` | Upload a set, start the job (`202`) |
| `GET` | `/api/v1/analyses/{id}` | Progress while running, full report when done |
| `GET` | `/api/v1/analyses/{id}/video` | Annotated MP4 |
| `DELETE` | `/api/v1/analyses/{id}` | Drop the job and its video |
| `GET` | `/api/v1/exercises` | Supported lifts, camera guidance, scoring weights |
| `GET` | `/api/v1/health` | Liveness + active job count |

`exercise` accepts either a key (`squat`) or a label (`SQUAT`, `Bench Press`).

### What the report contains

```jsonc
{
  "score": { "total": 85.5, "grade": "Solid", "criteria": [
      { "label": "Squat depth",  "earned": 22.4, "weight": 25,
        "detail": "Averaged 9° across 4 rep(s); full credit at 5°." },
      { "label": "Heel contact", "earned": 10.0, "weight": 20,
        "detail": "Flagged on 2 of 4 rep(s)." }
  ]},
  "reps": { "attempted": 4, "full_range": 4, "partial_range": 0, "clean": 1, "with_faults": 3 },
  "rep_details": [
    { "number": 1, "bottom_angle": 62.7, "eccentric_seconds": 2.6, "concentric_seconds": 1.46,
      "quality": "full", "faults": [{ "code": "heel_lift", "severity": "major", "cue": "..." }] }
  ],
  "cues": [{ "severity": "major", "message": "Reps 1 and 3: your heels came off the floor, ..." }],
  "strengths": ["Rep consistency: Bottom position varied by ±4° across the set."],
  "tempo": { "average_eccentric_seconds": 2.35, "average_concentric_seconds": 1.63 },
  "video": { "detection_rate": 1.0, "tracked_side": "left", "frames_analyzed": 304 }
}
```

Every point is attributable: each criterion reports what it measured and why it
scored what it did, and `detection_rate` tells the client when a bad recording
makes the numbers unreliable.

## How it works

```
app/
  api/routes/       HTTP layer — upload validation, polling, file serving
  services/         job store (Protocol + in-memory impl), uploads, orchestration
  video/            pose estimation, video I/O + codec negotiation, overlay, pipeline
  core/             pure analysis: geometry, smoothing, rep segmentation,
                    faults, scoring rubric, coaching cues  (no OpenCV, no MediaPipe)
  exercises/        one declarative module per lift
```

The pipeline makes **two passes** over the clip:

1. **Measure** — sample frames down to ~15 fps, run pose estimation once, and
   record every angle the exercise declared into a `MetricTable`. Landmarks are
   cached.
2. **Annotate** — re-read at full frame rate and draw the overlay using the
   final score, rep boundaries, and detected faults.

Measurement and interpretation are shared across exercises. Each lift declares
its metrics, fault rules, and scoring rubric. A full exercise is ~130 lines:

```python
spec = ExerciseSpec(primary_metric="knee_angle", concentric=ASCENDING,
                    full_rom_bottom=90.0, full_rom_top=165.0, ...)

fault_rules = (
    FaultRule(code="heel_lift", metric="foot_angle", stat="range", comparator="gt",
              threshold=12.0, severity=Severity.MAJOR, cue="your heels came off the floor..."),
)

def build_rubric(self):          # weights are validated to sum to 100
    return Rubric(
        MetricTargetCriterion("depth", "Squat depth", 25, metric="thigh_angle",
                              stat="at_bottom", full_credit=5.0, no_credit=45.0),
        FaultFreeCriterion("heel_contact", "Heel contact", 20, codes=("heel_lift",)),
        ...
    )
```

Adding a lift means writing that file and importing it in
`app/exercises/__init__.py` — no changes to the pipeline, scoring, rendering or
API.

### Measurement details

- **Rep detection** uses prominence-based extrema (a confirmed reversal of
  ≥15°) on a gap-filled, median-filtered, zero-phase-smoothed signal. Landmark
  jitter and pauses at the bottom cannot manufacture reps, and smoothing does
  not shift rep timings (which would skew every tempo measurement).
- **Side selection** picks the side of the body the camera can see most clearly
  in each clip.
- **Aspect correction** rescales normalised coordinates into units of frame
  height before any trigonometry — otherwise every angle on a non-square video
  is squashed toward the long axis.
- **Partial credit**: criteria ramp between thresholds, so small differences in
  range of motion are reflected in the score.

### The annotated video

Body-only skeleton (the face mesh adds clutter), live arcs for the tracked joint
angles, a HUD with score/grade/rep counter/phase, fault chips for the rep on
screen, and a timeline strip colouring each rep green/amber/red so the clip can
be scrubbed straight to a mistake. Codecs are negotiated (`avc1` → `H264` →
`mp4v`); if `ffmpeg` is on `PATH` the result is re-encoded to H.264 + faststart
for in-browser streaming.

## Configuration

Environment variables, prefixed `FORMCOACH_` (or a `.env` file). See
`app/config.py`.

| Variable | Default | Notes |
| --- | --- | --- |
| `FORMCOACH_CORS_ORIGINS` | `http://localhost:3000` | Comma-separated |
| `FORMCOACH_MEDIA_ROOT` | `media` | Uploads and annotated output |
| `FORMCOACH_MAX_UPLOAD_BYTES` | 200 MB | Enforced while streaming to disk |
| `FORMCOACH_MAX_VIDEO_SECONDS` | 300 | Checked before the job is queued |
| `FORMCOACH_TARGET_SAMPLE_FPS` | 15 | Pose sampling rate |
| `FORMCOACH_POSE_MODEL_COMPLEXITY` | 1 | 0 = fastest, 2 = most accurate |
| `FORMCOACH_MAX_CONCURRENT_ANALYSES` | 2 | Worker-thread cap |
| `FORMCOACH_ANALYSIS_TTL_SECONDS` | 3600 | Jobs and videos are swept after this |

