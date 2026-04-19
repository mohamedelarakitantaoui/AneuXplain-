"""Tests for cropper, harmonizer, and the /dicom/crop-and-analyze endpoint."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import trimesh
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.modules.dicom_pipeline import session_store
from app.modules.dicom_pipeline.cropper import crop_around_point
from app.modules.dicom_pipeline.harmonizer import (
    _CACHE_PATH,
    compare_to_intra_distribution,
    compute_intra_reference_stats,
    harmonize_to_intra,
)
from app.routes.dicom import router as dicom_router


def _two_cluster_mesh() -> trimesh.Trimesh:
    """Two ico-spheres — one at origin, one far away."""
    s1 = trimesh.creation.icosphere(subdivisions=4, radius=5.0)
    s1.apply_translation([0.0, 0.0, 0.0])
    s2 = trimesh.creation.icosphere(subdivisions=4, radius=5.0)
    s2.apply_translation([100.0, 0.0, 0.0])
    return trimesh.util.concatenate([s1, s2])


def _dense_sphere(radius: float = 10.0, subdivisions: int = 5) -> trimesh.Trimesh:
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)


# ---------------------------------------------------------------------------
# cropper
# ---------------------------------------------------------------------------


def test_crop_around_point_basic():
    mesh = _two_cluster_mesh()
    total = len(mesh.vertices)
    cropped = crop_around_point(mesh, (0.0, 0.0, 0.0), radius_mm=8.0)
    assert len(cropped.vertices) < total
    # All cropped vertices must be near the clicked cluster (origin), not the
    # far cluster at x=100.
    assert np.asarray(cropped.vertices)[:, 0].max() < 50.0
    assert len(cropped.vertices) >= 200


def test_crop_too_small_raises():
    mesh = _dense_sphere()
    with pytest.raises(ValueError, match="Crop too small"):
        # Tiny radius picks up fewer than 200 verts
        crop_around_point(mesh, (0.0, 0.0, 0.0), radius_mm=0.001)


# ---------------------------------------------------------------------------
# harmonizer
# ---------------------------------------------------------------------------


def test_compute_intra_reference_stats_creates_cache():
    # Force recomputation for this test
    if _CACHE_PATH.exists():
        _CACHE_PATH.unlink()
    stats = compute_intra_reference_stats(sample_size=8)
    assert _CACHE_PATH.exists()
    for key in ("vertex_count", "bbox_diagonal", "edge_length"):
        assert key in stats
        assert "median" in stats[key]
    med_vc = stats["vertex_count"]["median"]
    assert 500 < med_vc < 10000, f"Unexpected IntrA median vertex count: {med_vc}"
    # Second call should hit cache and return identical dict
    stats2 = compute_intra_reference_stats(sample_size=8)
    assert stats2 == stats


def test_harmonize_matches_target_vertex_count():
    # Start with a ~12k-vertex mesh
    mesh = trimesh.creation.icosphere(subdivisions=6, radius=10.0)
    assert len(mesh.vertices) > 5000
    target = 2000
    out = harmonize_to_intra(mesh, target_vertex_count=target)
    assert abs(len(out.vertices) - target) / target < 0.05


def test_compare_distribution_shape():
    mesh = trimesh.creation.icosphere(subdivisions=4, radius=8.0)
    res = compare_to_intra_distribution(mesh)
    for key in ("vertex_count", "bbox_diagonal", "edge_length"):
        assert key in res
        entry = res[key]
        assert "value" in entry and "z_score" in entry and "in_distribution" in entry


# ---------------------------------------------------------------------------
# endpoint
# ---------------------------------------------------------------------------


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(dicom_router)
    return TestClient(app)


def test_crop_and_analyze_endpoint_404_unknown_session():
    client = _client()
    r = client.post(
        "/dicom/crop-and-analyze",
        json={"session_id": "nope", "click_point": [0.0, 0.0, 0.0]},
    )
    assert r.status_code == 404


def test_crop_and_analyze_endpoint_400_before_segmentation():
    session_id = session_store.create_session({"volume": "fake-volume-marker"})
    client = _client()
    r = client.post(
        "/dicom/crop-and-analyze",
        json={"session_id": session_id, "click_point": [0.0, 0.0, 0.0]},
    )
    assert r.status_code == 400
    session_store.drop_session(session_id)


def test_crop_and_analyze_endpoint_reuses_analyze_logic():
    mesh = _dense_sphere(radius=10.0, subdivisions=5)
    session_id = session_store.create_session(
        {"volume": "fake", "full_mesh": mesh}
    )

    fake_engine = mock.MagicMock()
    fake_engine._models_loaded = True
    fake_morph = mock.MagicMock()
    fake_clin = mock.MagicMock()

    fake_result = {
        "risk_score": 0.42,
        "risk_level": "MODERATE",
        "interpretation": "test",
        "morphology": None,
        "clinical_report": None,
    }

    with mock.patch(
        "app.routes.dicom.analyze_mesh_file", return_value=fake_result
    ) as mock_analyze, mock.patch(
        "app.main.engine", fake_engine
    ), mock.patch(
        "app.main.morphology_analyzer", fake_morph
    ), mock.patch(
        "app.main.clinical_explainer", fake_clin
    ):
        client = _client()
        r = client.post(
            "/dicom/crop-and-analyze",
            json={
                "session_id": session_id,
                "click_point": [0.0, 0.0, 0.0],
                "crop_radius_mm": 15.0,
            },
        )

    session_store.drop_session(session_id)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["risk_score"] == 0.42
    assert "harmonization" in body
    assert "crop_info" in body
    assert body["crop_info"]["cropped_vertex_count"] > 0

    # analyze_mesh_file was called once with a .obj path
    assert mock_analyze.call_count == 1
    called_path = mock_analyze.call_args[0][0]
    assert str(called_path).endswith(".obj")


def test_cropped_mesh_endpoint_404_unknown_session():
    client = _client()
    r = client.get("/dicom/cropped-mesh/nope.obj")
    assert r.status_code == 404


def test_cropped_mesh_endpoint_404_before_crop():
    session_id = session_store.create_session({"volume": "fake"})
    client = _client()
    r = client.get(f"/dicom/cropped-mesh/{session_id}.obj")
    assert r.status_code == 404
    session_store.drop_session(session_id)


def test_cropped_mesh_endpoint_happy_path(tmp_path):
    obj_path = tmp_path / "cropped.obj"
    trimesh.creation.icosphere(subdivisions=3, radius=5.0).export(str(obj_path))
    session_id = session_store.create_session(
        {"volume": "fake", "cropped_obj_path": str(obj_path)}
    )
    client = _client()
    r = client.get(f"/dicom/cropped-mesh/{session_id}.obj")
    session_store.drop_session(session_id)
    assert r.status_code == 200
    assert r.headers.get("content-disposition", "").endswith(f"cropped_{session_id}.obj")
    assert len(r.content) > 0


def test_cropped_heatmap_endpoint_404_unknown_session():
    client = _client()
    r = client.get("/dicom/cropped-heatmap/nope")
    assert r.status_code == 404


def test_cropped_heatmap_endpoint_404_before_crop():
    session_id = session_store.create_session({"volume": "fake"})
    client = _client()
    r = client.get(f"/dicom/cropped-heatmap/{session_id}")
    assert r.status_code == 404
    session_store.drop_session(session_id)


def test_cropped_heatmap_endpoint_happy_path(tmp_path):
    obj_path = tmp_path / "cropped.obj"
    trimesh.creation.icosphere(subdivisions=3, radius=5.0).export(str(obj_path))
    session_id = session_store.create_session(
        {"volume": "fake", "cropped_obj_path": str(obj_path)}
    )

    fake_engine = mock.MagicMock()
    fake_engine._models_loaded = True
    fake_payload = {"heatmap": [0.1, 0.2, 0.3], "risk_score": 0.42}

    with mock.patch(
        "app.routes.dicom.compute_heatmap_for_mesh", return_value=fake_payload
    ) as mock_heatmap, mock.patch("app.main.engine", fake_engine):
        client = _client()
        r = client.get(f"/dicom/cropped-heatmap/{session_id}")

    session_store.drop_session(session_id)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["heatmap"] == [0.1, 0.2, 0.3]
    assert body["risk_score"] == 0.42
    assert mock_heatmap.call_count == 1
    called_path = mock_heatmap.call_args[0][0]
    assert str(called_path) == str(obj_path)


def test_intra_reference_stats_json_exists_and_sensible():
    """After the compute test above runs, the JSON cache should exist."""
    if not _CACHE_PATH.exists():
        compute_intra_reference_stats(sample_size=8)
    with open(_CACHE_PATH, "r", encoding="utf-8") as f:
        stats = json.load(f)
    assert stats["sample_size"] >= 1
    assert stats["vertex_count"]["median"] > 0
    assert stats["bbox_diagonal"]["median"] > 0
