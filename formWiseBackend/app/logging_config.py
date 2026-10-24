"""Logging setup shared by the app and any future worker process."""
from __future__ import annotations

import logging
import os


def configure_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=(level or os.getenv("FORMCOACH_LOG_LEVEL", "INFO")).upper(),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # MediaPipe and absl are extremely chatty on first inference.
    logging.getLogger("absl").setLevel(logging.ERROR)
