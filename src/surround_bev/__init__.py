"""SurroundBEV: camera-only bird's-eye-view perception on nuScenes.

Milestone 0 exposes configuration loading, nuScenes sample access, and a
6-camera mosaic. Milestone 1 adds ground-truth 3D box projection into those
cameras.
"""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
"""Absolute path to the repository root (parent of ``src/``)."""

__all__ = ["REPO_ROOT"]
