from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from app.schemas.analysis import AnalysisStatus
from app.services.store import InMemoryAnalysisStore, new_record


def record(tmp_path: Path, ttl: int = 3600):
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "in.mp4"
    output = tmp_path / "out.mp4"
    source.write_bytes(b"source")
    output.write_bytes(b"annotated")
    return new_record("squat", source, output, ttl_seconds=ttl, source_filename="in.mp4")


def test_records_round_trip(tmp_path: Path):
    store = InMemoryAnalysisStore()
    saved = store.save(record(tmp_path))
    assert store.get(saved.id) == saved
    assert store.get("missing") is None


def test_updates_are_applied_and_timestamped(tmp_path: Path):
    store = InMemoryAnalysisStore()
    saved = store.save(record(tmp_path))
    updated = store.update(saved.id, progress=0.5, stage="annotating")
    assert (updated.progress, updated.stage) == (0.5, "annotating")
    assert updated.updated_at >= saved.updated_at
    assert store.update("missing", progress=1.0) is None


def test_active_count_ignores_finished_jobs(tmp_path: Path):
    store = InMemoryAnalysisStore()
    saved = store.save(record(tmp_path))
    assert store.count_active() == 1
    store.update(saved.id, status=AnalysisStatus.COMPLETED)
    assert store.count_active() == 0


def test_delete_removes_the_record_and_its_files(tmp_path: Path):
    store = InMemoryAnalysisStore()
    saved = store.save(record(tmp_path))
    assert store.delete(saved.id) is True
    assert not saved.output_path.exists()
    assert store.delete(saved.id) is False


def test_purge_only_collects_expired_terminal_jobs(tmp_path: Path):
    store = InMemoryAnalysisStore()
    stale = store.save(record(tmp_path / "stale", ttl=-1))
    running = store.save(record(tmp_path / "running", ttl=-1))
    fresh = store.save(record(tmp_path / "fresh"))
    store.update(stale.id, status=AnalysisStatus.COMPLETED)
    store.update(fresh.id, status=AnalysisStatus.COMPLETED)

    assert store.purge_expired() == 1
    assert store.get(stale.id) is None
    assert store.get(running.id) is not None  # still queued, so it is kept
    assert store.get(fresh.id) is not None    # not expired yet


def test_expiry_is_derived_from_the_ttl(tmp_path: Path):
    subject = record(tmp_path, ttl=120)
    assert subject.expires_at - subject.created_at == timedelta(seconds=120)
    assert subject.created_at <= datetime.now(UTC)


def test_has_video_requires_a_completed_job_and_a_file(tmp_path: Path):
    store = InMemoryAnalysisStore()
    saved = store.save(record(tmp_path))
    assert saved.has_video is False
    completed = store.update(saved.id, status=AnalysisStatus.COMPLETED)
    assert completed.has_video is True
    completed.output_path.unlink()
    assert completed.has_video is False
