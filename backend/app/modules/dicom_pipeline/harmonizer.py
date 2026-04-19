"""
Harmonize a cropped mesh to match the IntrA training distribution so the
existing RiskPredictorV2 sees in-distribution input. Also computes z-score
diagnostics against reference IntrA stats.

IntrA reference stats are computed once from the on-disk IntrA .obj files
and cached as JSON next to this module.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import trimesh


logger = logging.getLogger("aneuxplain.harmonizer")

INTRA_OBJ_DIR_DEFAULT = (
    "C:/Users/buzok/OneDrive - Al Akhawayn University in Ifrane/"
    "CapstoneProject/IntrA/annotated/obj"
)

_CACHE_PATH = Path(__file__).parent / "intra_reference_stats.json"


def _mesh_stats(mesh: trimesh.Trimesh) -> Dict[str, float]:
    verts = np.asarray(mesh.vertices)
    bbox_diag = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
    edges = mesh.edges_unique
    if len(edges) == 0:
        mean_edge = 0.0
    else:
        e_vecs = verts[edges[:, 0]] - verts[edges[:, 1]]
        mean_edge = float(np.linalg.norm(e_vecs, axis=1).mean())
    return {
        "vertex_count": float(len(verts)),
        "face_count": float(len(mesh.faces)),
        "bbox_diagonal": bbox_diag,
        "edge_length": mean_edge,
    }


def _summarize(values: np.ndarray) -> Dict[str, float]:
    return {
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
    }


def compute_intra_reference_stats(
    intra_obj_dir: str = INTRA_OBJ_DIR_DEFAULT,
    sample_size: int = 30,
) -> Dict[str, Any]:
    """
    Compute (or load from cache) median/IQR stats across IntrA .obj samples.
    """
    if _CACHE_PATH.exists():
        try:
            with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:  # pragma: no cover
            logger.warning("Failed to read intra stats cache (%s); recomputing", e)

    obj_dir = Path(intra_obj_dir)
    if not obj_dir.exists():
        raise FileNotFoundError(f"IntrA reference dir not found: {intra_obj_dir}")

    obj_files = sorted(obj_dir.rglob("*.obj"))[:sample_size]
    if not obj_files:
        raise FileNotFoundError(f"No .obj files under {intra_obj_dir}")

    records: list[Dict[str, float]] = []
    for p in obj_files:
        try:
            m = trimesh.load(str(p), process=False, force="mesh")
            if not isinstance(m, trimesh.Trimesh) or len(m.vertices) == 0:
                continue
            records.append(_mesh_stats(m))
        except Exception as e:
            logger.warning("Skipping %s: %s", p.name, e)

    if not records:
        raise RuntimeError("No IntrA meshes could be loaded for reference stats")

    keys = ["vertex_count", "face_count", "bbox_diagonal", "edge_length"]
    stats: Dict[str, Any] = {"sample_size": len(records)}
    for k in keys:
        arr = np.array([r[k] for r in records], dtype=np.float64)
        stats[k] = _summarize(arr)

    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
    except Exception as e:  # pragma: no cover
        logger.warning("Failed to write intra stats cache: %s", e)

    return stats


def harmonize_to_intra(
    mesh: trimesh.Trimesh,
    target_vertex_count: Optional[int] = None,
) -> trimesh.Trimesh:
    """
    Center, smooth, and resample a cropped mesh toward the IntrA distribution.
    """
    stats = compute_intra_reference_stats()
    if target_vertex_count is None:
        target_vertex_count = int(round(stats["vertex_count"]["median"]))

    verts = np.asarray(mesh.vertices, dtype=np.float64).copy()
    faces = np.asarray(mesh.faces, dtype=np.int64).copy()
    centroid = verts.mean(axis=0)
    verts = verts - centroid
    work = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

    try:
        trimesh.smoothing.filter_laplacian(work, iterations=15)
    except Exception as e:  # pragma: no cover — smoothing shouldn't fail
        logger.warning("Laplacian smoothing failed: %s", e)

    current_vc = len(work.vertices)
    if current_vc > target_vertex_count:
        try:
            decimated = _decimate(work, target_vertex_count)
            if isinstance(decimated, trimesh.Trimesh) and len(decimated.vertices) > 0:
                work = decimated
        except Exception as e:
            logger.warning("Decimation failed: %s", e)
    elif current_vc < target_vertex_count:
        try:
            while len(work.vertices) < target_vertex_count:
                new_v, new_f = trimesh.remesh.subdivide(
                    np.asarray(work.vertices), np.asarray(work.faces)
                )
                work = trimesh.Trimesh(vertices=new_v, faces=new_f, process=False)
                if len(work.vertices) > 4 * target_vertex_count:
                    break
            decimated = _decimate(work, target_vertex_count)
            if isinstance(decimated, trimesh.Trimesh) and len(decimated.vertices) > 0:
                work = decimated
        except Exception as e:
            logger.warning("Subdivide/decimate failed: %s", e)

    return work


def _decimate(mesh: trimesh.Trimesh, target_vertex_count: int) -> trimesh.Trimesh:
    """
    Best-effort quadric decimation toward a target vertex count. trimesh's
    simplify_quadric_decimation takes target `face_count` or `percent`; we
    estimate faces from the current vertex→face ratio so the output lands
    near the requested vertex budget.
    """
    cur_v = len(mesh.vertices)
    cur_f = len(mesh.faces)
    if cur_v <= target_vertex_count or cur_f == 0:
        return mesh
    # Closed triangle meshes have ~2× as many faces as vertices; use the
    # current ratio to stay accurate for open pieces too.
    ratio = cur_f / max(cur_v, 1)
    target_faces = max(4, int(round(target_vertex_count * ratio)))
    try:
        out = mesh.simplify_quadric_decimation(face_count=target_faces)
        if isinstance(out, trimesh.Trimesh) and len(out.vertices) > 0:
            return out
    except TypeError:
        # Older trimesh where face_count kwarg isn't supported
        percent = max(1e-3, min(1.0, target_faces / cur_f))
        out = mesh.simplify_quadric_decimation(percent=percent)
        if isinstance(out, trimesh.Trimesh) and len(out.vertices) > 0:
            return out
    return mesh


def _robust_std(q25: float, q75: float) -> float:
    iqr = max(q75 - q25, 1e-9)
    return iqr / 1.349


def compare_to_intra_distribution(mesh: trimesh.Trimesh) -> Dict[str, Any]:
    """
    Z-score the input mesh's key stats against the IntrA reference IQR.
    """
    stats = compute_intra_reference_stats()
    ms = _mesh_stats(mesh)

    out: Dict[str, Any] = {}
    for key in ("vertex_count", "bbox_diagonal", "edge_length"):
        ref = stats[key]
        sigma = _robust_std(ref["q25"], ref["q75"])
        z = (ms[key] - ref["median"]) / sigma
        out[key] = {
            "value": float(ms[key]),
            "z_score": float(z),
            "in_distribution": bool(abs(z) < 2.0),
        }
    return out
