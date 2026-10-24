"""Application settings, loaded from the environment (prefix ``FORMCOACH_``)."""
from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FORMCOACH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- HTTP -------------------------------------------------------------
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    api_prefix: str = "/api/v1"

    # --- Uploads ----------------------------------------------------------
    media_root: Path = Path("media")
    max_upload_bytes: int = 200 * 1024 * 1024
    max_video_seconds: float = 300.0
    allowed_content_types: set[str] = Field(
        default_factory=lambda: {
            "video/mp4",
            "video/quicktime",
            "video/x-msvideo",
            "video/webm",
            "application/octet-stream",
        }
    )

    # --- Pose estimation --------------------------------------------------
    # Frames are sampled down to roughly this rate before being sent through
    # the pose model; the annotated video is still written at full frame rate.
    target_sample_fps: float = 15.0
    pose_model_complexity: int = Field(default=1, ge=0, le=2)
    min_detection_confidence: float = Field(default=0.5, gt=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, gt=0.0, le=1.0)
    landmark_visibility_threshold: float = Field(default=0.45, ge=0.0, le=1.0)

    # --- Signal conditioning ----------------------------------------------
    smoothing_median_window: int = Field(default=5, ge=1)
    smoothing_alpha: float = Field(default=0.45, gt=0.0, le=1.0)

    # --- Job handling -----------------------------------------------------
    # Analyses run in a bounded thread pool today. Swapping this layer for
    # Celery means replacing ``AnalysisRunner.submit`` with ``task.delay(...)``
    # and ``InMemoryAnalysisStore`` with a Redis-backed store; nothing else in
    # the codebase knows how jobs are executed or where they live.
    max_concurrent_analyses: int = Field(default=2, ge=1)
    analysis_ttl_seconds: int = Field(default=3600, ge=60)
    cleanup_interval_seconds: int = Field(default=300, ge=30)

    # --- Rendering --------------------------------------------------------
    draw_overlay: bool = True
    # Optional re-encode to browser-friendly H.264. Ignored when ffmpeg is absent.
    ffmpeg_path: str | None = None

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @property
    def upload_dir(self) -> Path:
        return self.media_root / "uploads"

    @property
    def output_dir(self) -> Path:
        return self.media_root / "annotated"

    def ensure_directories(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def resolve_ffmpeg(self) -> str | None:
        return self.ffmpeg_path or shutil.which("ffmpeg")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
