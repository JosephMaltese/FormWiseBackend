"""Compatibility entry point for ``uvicorn main:app``.

The application itself lives in :mod:`app.main`; prefer ``uvicorn app.main:app``.
"""
from app.main import app

__all__ = ["app"]
