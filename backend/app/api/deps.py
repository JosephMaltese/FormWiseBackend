"""Request-scoped dependencies."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings
from app.services.analysis_service import AnalysisService
from app.services.store import AnalysisStore


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_store(request: Request) -> AnalysisStore:
    return request.app.state.store


def get_service(request: Request) -> AnalysisService:
    return request.app.state.service


SettingsDep = Annotated[Settings, Depends(get_settings)]
StoreDep = Annotated[AnalysisStore, Depends(get_store)]
ServiceDep = Annotated[AnalysisService, Depends(get_service)]
