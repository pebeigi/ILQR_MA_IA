"""Filesystem-safe names for calibration artifacts."""

from __future__ import annotations

_WINDOWS_INVALID = '<>:"|?*\\'


def safe_case_filename(case_id: str) -> str:
    """Return a slug safe on Windows, macOS, and Linux."""
    name = case_id.replace("/", "_").replace("->", "_to_")
    for ch in _WINDOWS_INVALID:
        name = name.replace(ch, "_")
    return name
