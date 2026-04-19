"""
In-memory session store for the DICOM pipeline.

Holds per-session state (loaded volume, metadata, and later the full
vessel mesh) keyed by a short UUID. Single-process only — swap for
Redis if we ever need multi-worker deployment.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional, Tuple

import numpy as np
import SimpleITK as sitk

_sessions: Dict[str, Dict[str, Any]] = {}


def create_session(data: Dict[str, Any]) -> str:
    """Register a new session and return its id."""
    session_id = uuid.uuid4().hex[:12]
    _sessions[session_id] = dict(data)
    return session_id


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Fetch session state, or None if unknown."""
    return _sessions.get(session_id)


def update_session(session_id: str, **fields: Any) -> None:
    """Merge fields into an existing session."""
    if session_id in _sessions:
        _sessions[session_id].update(fields)


def drop_session(session_id: str) -> None:
    """Remove a session."""
    _sessions.pop(session_id, None)


def _all_sessions() -> Dict[str, Dict[str, Any]]:
    """Test-only accessor — returns the internal dict by reference."""
    return _sessions


def get_or_compute_window(session_id: str) -> Optional[Tuple[float, float]]:
    """
    Return cached (p1, p99) intensity window for MRA display, computing
    once from non-zero voxels across the whole volume. Returns None if
    the session is unknown or has no volume.
    """
    session = _sessions.get(session_id)
    if session is None or "volume" not in session:
        return None

    cached = session.get("window")
    if cached is not None:
        return cached

    volume: sitk.Image = session["volume"]
    arr = sitk.GetArrayViewFromImage(volume)
    nonzero = arr[arr > 0]
    if nonzero.size == 0:
        # Fall back to full range so constant/zero volumes still render.
        lo, hi = float(arr.min()), float(arr.max())
    else:
        lo = float(np.percentile(nonzero, 1))
        hi = float(np.percentile(nonzero, 99))
    if hi <= lo:
        hi = lo + 1.0

    window = (lo, hi)
    session["window"] = window
    return window
