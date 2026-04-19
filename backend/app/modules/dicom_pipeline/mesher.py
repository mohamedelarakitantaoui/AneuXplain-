"""
Binary vessel mask → surface mesh via marching cubes.

The mask produced by :mod:`segmenter` lives on a voxel grid. This module
converts it to a watertight-ish triangle surface in physical (mm) space
so downstream modules (cropping, harmonization, the existing PointNet
pipeline) can operate on it.
"""

from __future__ import annotations

import logging

import numpy as np
import SimpleITK as sitk
import trimesh
from skimage import measure

logger = logging.getLogger(__name__)


def mask_to_mesh(
    mask: sitk.Image,
    smoothing_iterations: int = 10,
) -> trimesh.Trimesh:
    """
    Convert a binary vessel mask to a smoothed triangle mesh in mm.

    Steps:
        1. Marching cubes at level 0.5 with voxel spacing from the mask.
        2. Laplacian smoothing (``smoothing_iterations`` passes).
        3. Remove duplicate faces and unreferenced vertices.

    Args:
        mask: Binary ``sitk.Image`` produced by :func:`segment_vessels`.
        smoothing_iterations: Laplacian smoothing pass count. Set to 0
            to skip smoothing entirely.

    Returns:
        A ``trimesh.Trimesh`` with vertices in millimetres.
    """
    arr = sitk.GetArrayFromImage(mask).astype(np.float32)  # (z, y, x)
    if arr.max() <= 0:
        raise ValueError("Mask is empty; nothing to mesh.")

    # SimpleITK spacing is (x, y, z); skimage expects (z, y, x) to match arr axes.
    sx, sy, sz = mask.GetSpacing()
    spacing_zyx = (float(sz), float(sy), float(sx))

    verts_zyx, faces, _normals, _values = measure.marching_cubes(
        arr, level=0.5, spacing=spacing_zyx
    )
    # skimage returns vertices in (z, y, x) order; swap to (x, y, z).
    vertices = verts_zyx[:, [2, 1, 0]]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if smoothing_iterations > 0:
        _laplacian_smooth(mesh, smoothing_iterations)

    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    logger.info(
        "Meshed vessel tree: %d vertices, %d faces (smoothing=%d)",
        len(mesh.vertices), len(mesh.faces), smoothing_iterations,
    )
    return mesh


def _laplacian_smooth(mesh: trimesh.Trimesh, iterations: int) -> None:
    """Laplacian smoothing, in-place. Falls back to manual averaging."""
    try:
        trimesh.smoothing.filter_laplacian(mesh, iterations=iterations)
        return
    except (AttributeError, Exception) as e:  # pragma: no cover
        logger.warning("trimesh.smoothing.filter_laplacian unavailable (%s); manual fallback", e)

    verts = np.asarray(mesh.vertices, dtype=np.float64)
    neighbours = mesh.vertex_neighbors
    for _ in range(iterations):
        new_verts = verts.copy()
        for i, nbrs in enumerate(neighbours):
            if nbrs:
                new_verts[i] = verts[nbrs].mean(axis=0)
        verts = new_verts
    mesh.vertices = verts


def export_mesh_to_obj(mesh: trimesh.Trimesh, out_path: str) -> None:
    """Save a mesh as a standard Wavefront .obj."""
    mesh.export(out_path, file_type="obj")
