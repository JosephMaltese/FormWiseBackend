"""HTTP contract tests.

The pose pipeline is stubbed out: these cover routing, validation, job state
and cleanup, not the computer vision (which ``test_exercises.py`` covers).
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.schemas.analysis import (
    AnalysisReport,
    RepSummary,
    ScoreOut,
    TempoSummary,
    VideoSummary,
)

PIXELS = 64


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    """A tiny real mp4 so the route's decodability probe has something to read."""
    path = tmp_path / "set.mp4"
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"mp4v"), 10.0, (PIXELS, PIXELS))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.zeros((PIXELS, PIXELS, 3), dtype=np.uint8))
    writer.release()
    return path


def stub_report(exercise: str = "squat") -> AnalysisReport:
    return AnalysisReport(
        exercise=exercise,
        exercise_name="Squat",
        primary_metric="knee_angle",
        primary_metric_label="Knee angle",
        score=ScoreOut(total=88.0, grade="Solid", criteria=[]),
        reps=RepSummary(attempted=3, full_range=3, partial_range=0, clean=2, with_faults=1),
        rep_details=[],
        tempo=TempoSummary(),
        metrics=[],
        cues=[],
        strengths=[],
        video=VideoSummary(
            duration_seconds=2.0, source_fps=10.0, width=PIXELS, height=PIXELS,
            frames_total=20, frames_analyzed=20, sample_fps=10.0, detection_rate=1.0,
            tracked_side="left",
        ),
        camera_guidance="Film from the side.",
    )


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    def fake_analyze(source, destination, exercise, settings, progress=None):
        if progress:
            progress("measuring", 0.5)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"annotated-video-bytes")
        return stub_report(exercise)

    monkeypatch.setattr("app.services.analysis_service.analyze_video", fake_analyze)
    settings = Settings(media_root=tmp_path / "media", max_upload_bytes=4096,
                        max_video_seconds=60.0)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def submit(client: TestClient, video: Path, exercise: str = "squat"):
    with video.open("rb") as handle:
        return client.post(
            "/api/v1/analyses",
            files={"file": (video.name, handle, "video/mp4")},
            data={"exercise": exercise},
        )


def wait_for(client: TestClient, analysis_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/v1/analyses/{analysis_id}").json()
        if body["status"] in {"completed", "failed"}:
            return body
        time.sleep(0.05)
    raise AssertionError(f"analysis {analysis_id} did not finish")


# -- discovery -------------------------------------------------------------
def test_health_lists_registered_exercises(client: TestClient):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert set(body["exercises"]) == {"squat", "bench_press", "bicep_curl"}


def test_exercises_expose_their_scoring_rubric(client: TestClient):
    squat = client.get("/api/v1/exercises/squat").json()
    assert squat["key"] == "squat"
    assert sum(criterion["weight"] for criterion in squat["scoring"]) == 100
    assert client.get("/api/v1/exercises/deadlift").status_code == 404


def test_legacy_exercise_label_is_accepted(client: TestClient, sample_video: Path):
    response = submit(client, sample_video, exercise="SQUAT")
    assert response.status_code == 202
    assert response.json()["exercise"] == "squat"


# -- validation -------------------------------------------------------------
def test_unknown_exercise_is_rejected(client: TestClient, sample_video: Path):
    assert submit(client, sample_video, exercise="deadlift").status_code == 422


def test_undecodable_upload_is_rejected(client: TestClient, tmp_path: Path):
    junk = tmp_path / "not-a-video.mp4"
    junk.write_bytes(b"definitely not an mp4")
    assert submit(client, junk).status_code == 400


def test_oversized_upload_is_rejected(client: TestClient, tmp_path: Path):
    big = tmp_path / "big.mp4"
    big.write_bytes(b"0" * 5000)  # the fixture caps uploads at 4096 bytes
    response = submit(client, big)
    assert response.status_code == 400
    assert "upload limit" in response.json()["detail"]


def test_unsupported_file_type_is_rejected(client: TestClient, tmp_path: Path):
    document = tmp_path / "notes.txt"
    document.write_bytes(b"hello")
    with document.open("rb") as handle:
        response = client.post(
            "/api/v1/analyses",
            files={"file": ("notes.txt", handle, "text/plain")},
            data={"exercise": "squat"},
        )
    assert response.status_code == 400


def test_rejected_uploads_do_not_linger_on_disk(client: TestClient, tmp_path: Path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"nope")
    submit(client, junk)
    assert list((tmp_path / "media" / "uploads").iterdir()) == []


# -- job lifecycle -----------------------------------------------------------
def test_analysis_runs_and_returns_a_report(client: TestClient, sample_video: Path):
    created = submit(client, sample_video).json()
    assert created["status"] == "queued"

    body = wait_for(client, created["id"])
    assert body["status"] == "completed"
    assert body["progress"] == 1.0
    assert body["report"]["score"]["total"] == 88.0
    assert body["annotated_video_url"].endswith(f"/analyses/{created['id']}/video")


def test_annotated_video_is_served_once_the_job_finishes(client: TestClient, sample_video: Path):
    created = submit(client, sample_video).json()
    wait_for(client, created["id"])
    response = client.get(f"/api/v1/analyses/{created['id']}/video")
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert response.content == b"annotated-video-bytes"


def test_the_raw_upload_is_deleted_after_analysis(client: TestClient, sample_video: Path,
                                                  tmp_path: Path):
    created = submit(client, sample_video).json()
    wait_for(client, created["id"])
    assert list((tmp_path / "media" / "uploads").iterdir()) == []


def test_unknown_ids_are_404(client: TestClient):
    assert client.get("/api/v1/analyses/nope").status_code == 404
    assert client.get("/api/v1/analyses/nope/video").status_code == 404
    assert client.delete("/api/v1/analyses/nope").status_code == 404


def test_deleting_an_analysis_removes_its_video(client: TestClient, sample_video: Path,
                                                tmp_path: Path):
    created = submit(client, sample_video).json()
    wait_for(client, created["id"])
    assert client.delete(f"/api/v1/analyses/{created['id']}").status_code == 204
    assert client.get(f"/api/v1/analyses/{created['id']}").status_code == 404
    assert list((tmp_path / "media" / "annotated").iterdir()) == []


def test_a_failed_analysis_reports_the_reason(client: TestClient, sample_video: Path,
                                              monkeypatch: pytest.MonkeyPatch):
    def explode(*args, **kwargs):
        from app.video.pipeline import AnalysisError

        raise AnalysisError("Video is 900s long; the limit is 60s.")

    monkeypatch.setattr("app.services.analysis_service.analyze_video", explode)
    created = submit(client, sample_video).json()
    body = wait_for(client, created["id"])
    assert body["status"] == "failed"
    assert "900s" in body["error"]
    assert body["annotated_video_url"] is None
    assert client.get(f"/api/v1/analyses/{created['id']}/video").status_code == 409


def test_unexpected_crashes_do_not_leak_internals(client: TestClient, sample_video: Path,
                                                  monkeypatch: pytest.MonkeyPatch):
    def explode(*args, **kwargs):
        raise RuntimeError("cuda: out of memory at 0x7fff")

    monkeypatch.setattr("app.services.analysis_service.analyze_video", explode)
    created = submit(client, sample_video).json()
    body = wait_for(client, created["id"])
    assert body["status"] == "failed"
    assert "cuda" not in body["error"]


def test_cors_headers_are_exposed_to_the_configured_origin(client: TestClient):
    response = client.get("/api/v1/health", headers={"Origin": "http://localhost:3000"})
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"
