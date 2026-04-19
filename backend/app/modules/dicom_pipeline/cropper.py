"""Crop a full vessel-tree mesh to a local region around a clicked point."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import trimesh


def crop_around_point(
    mesh: trimesh.Trimesh,
    click_xyz: Tuple[float, float, float],
    radius_mm: float,
) -> trimesh.Trimesh:
    """
    Extract the connected sub-mesh within a spherical radius of a clicked point.

    Keeps vertices within `radius_mm` of `click_xyz`, then keeps faces whose
    three vertices all survived, then restricts the result to the single
    connected component containing the vertex closest to the click.

    Raises:
        ValueError: if the resulting component has fewer than 200 vertices.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    click = np.asarray(click_xyz, dtype=np.float64).reshape(3)

    dists = np.linalg.norm(vertices - click, axis=1)
    vert_mask = dists <= float(radius_mm)
    if not vert_mask.any():
        raise ValueError("Crop too small (0 vertices). Try a larger radius.")

    face_mask = vert_mask[faces].all(axis=1)
    kept_faces = faces[face_mask]
    if kept_faces.size == 0:
        raise ValueError("Crop too small (0 vertices). Try a larger radius.")

    used_vert_ids = np.unique(kept_faces)
    remap = -np.ones(len(vertices), dtype=np.int64)
    remap[used_vert_ids] = np.arange(len(used_vert_ids))
    new_vertices = vertices[used_vert_ids]
    new_faces = remap[kept_faces]

    sub = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

    # Find the vertex (in sub) closest to the click — anchor for component pick
    sub_dists = np.linalg.norm(np.asarray(sub.vertices) - click, axis=1)
    anchor_vid = int(np.argmin(sub_dists))

    components = trimesh.graph.connected_components(
        edges=sub.edges, nodes=np.arange(len(sub.vertices))
    )
    chosen = None
    for comp in components:
        if anchor_vid in comp:
            chosen = comp
            break
    if chosen is None or len(chosen) == 0:
        raise ValueError("Crop too small (0 vertices). Try a larger radius.")

    comp_mask = np.zeros(len(sub.vertices), dtype=bool)
    comp_mask[chosen] = True
    comp_face_mask = comp_mask[sub.faces].all(axis=1)
    comp_faces = np.asarray(sub.faces)[comp_face_mask]

    used2 = np.unique(comp_faces)
    remap2 = -np.ones(len(sub.vertices), dtype=np.int64)
    remap2[used2] = np.arange(len(used2))
    final_vertices = np.asarray(sub.vertices)[used2]
    final_faces = remap2[comp_faces]

    if len(final_vertices) < 200:
        raise ValueError(
            f"Crop too small ({len(final_vertices)} vertices). Try a larger radius."
        )

    return trimesh.Trimesh(vertices=final_vertices, faces=final_faces, process=False)
