"""Tests for mask_to_mesh and the full-mesh endpoint."""

from __future__ import annotations

import io

import numpy as np
import pytest
import SimpleITK as sitk
import trimesh
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.modules.dicom_pipeline import session_store
from app.modules.dicom_pipeline.mesher import mask_to_mesh
from app.routes.dicom import router as dicom_router


def _cube_mask() -> sitk.Image:
    """Binary mask: 1 inside a [8,24)^3 cube within a (32,32,32) volume."""
    arr = np.zeros((32, 32, 32), dtype=np.uint8)
    arr[8:24, 8:24, 8:24] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.5, 0.5))
    return img


def _sphere_volume() -> sitk.Image:
    """Bright sphere in a noisy volume — suitable for full segment+mesh path."""
    rng = np.random.default_rng(1)
    arr = rng.uniform(0, 20, size=(40, 40, 40)).astype(np.float32)
    zz, yy, xx = np.mgrid[0:40, 0:40, 0:40]
    dist = np.sqrt((zz - 20) ** 2 + (yy - 20) ** 2 + (xx - 20) ** 2)
    arr[dist <= 8] = 900.0
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.5, 0.5))
    return img


def test_mask_to_mesh_produces_valid_mesh():
    mask = _cube_mask()
    mesh = mask_to_mesh(mask, smoothing_iterations=3)

    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.faces) > 0

    # Mesh should roughly span the cube in mm (16 voxels * 0.5 mm = 8 mm per side).
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    for e in extents:
        assert 6.0 < e < 10.0

    # Round-trip via OBJ
    buf = io.StringIO(mesh.export(file_type="obj"))
    roundtrip = trimesh.load(buf, file_type="obj")
    assert len(roundtrip.vertices) > 0


def _make_client() -> TestClient:
    app = FastAPI()
    app.include_router(dicom_router)
    return TestClient(app)


def test_segment_endpoint_includes_mesh_stats():
    session_id = session_store.create_session({
        "volume": _sphere_volume(),
        "metadata": {"source_format": "nifti", "modality": "unknown"},
    })
    client = _make_client()
    try:
        r = client.post(f"/dicom/segment/{session_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["vertex_count"] > 0
        assert body["face_count"] > 0
        assert len(body["mesh_bounds_mm"]) == 2
        assert len(body["mesh_bounds_mm"][0]) == 3
        # Sphere bounds should be roughly [0, 20] mm cubed
        lo, hi = body["mesh_bounds_mm"]
        for a, b in zip(lo, hi):
            assert b - a > 5.0
    finally:
        session_store.drop_session(session_id)


def test_full_mesh_endpoint_returns_obj():
    session_id = session_store.create_session({
        "volume": _sphere_volume(),
        "metadata": {"source_format": "nifti", "modality": "unknown"},
    })
    client = _make_client()
    try:
        client.post(f"/dicom/segment/{session_id}")
        r = client.get(f"/dicom/full-mesh/{session_id}.obj")
        assert r.status_code == 200
        assert "attachment" in r.headers["content-disposition"]
        assert f"vessel_tree_{session_id}.obj" in r.headers["content-disposition"]

        # Parse it back
        mesh = trimesh.load(io.StringIO(r.text), file_type="obj")
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0
    finally:
        session_store.drop_session(session_id)


def test_full_mesh_endpoint_before_segmentation_returns_404():
    session_id = session_store.create_session({
        "volume": _sphere_volume(),
        "metadata": {"source_format": "nifti", "modality": "unknown"},
    })
    client = _make_client()
    try:
        r = client.get(f"/dicom/full-mesh/{session_id}.obj")
        assert r.status_code == 404
        assert "Segmentation must be run" in r.json()["detail"]
    finally:
        session_store.drop_session(session_id)


def test_full_mesh_endpoint_unknown_session_404():
    client = _make_client()
    r = client.get("/dicom/full-mesh/nope.obj")
    assert r.status_code == 404
