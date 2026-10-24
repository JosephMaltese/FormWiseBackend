"""Exercise registry.

Importing this package registers every supported exercise.
"""
from app.exercises.base import (
    AngleOverlay,
    ExerciseAnalyzer,
    ExerciseSpec,
    available_specs,
    get_analyzer,
    register,
    resolve_key,
)
from app.exercises.bench_press import BenchPressAnalyzer
from app.exercises.bicep_curl import BicepCurlAnalyzer
from app.exercises.squat import SquatAnalyzer

__all__ = [
    "AngleOverlay",
    "ExerciseAnalyzer",
    "ExerciseSpec",
    "BenchPressAnalyzer",
    "BicepCurlAnalyzer",
    "SquatAnalyzer",
    "available_specs",
    "get_analyzer",
    "register",
    "resolve_key",
]
