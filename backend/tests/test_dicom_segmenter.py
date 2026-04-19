"""Tests for segment_vessels and POST /dicom/segment/{session_id}."""

from __future__ import annotations

import numpy as np
import pytest
import SimpleITK as sitk
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.modules.dicom_pipeline import session_store
from app.modules.dicom_pipeline.segmenter import segment_vessels
from app.routes.dicom import router as dicom_router


def _make_tube_volume() -> sitk.Image:
    """
    Build a (40, 40, 40) volume with:
      - Dim background noise everywhere (low intensity)
      - One bright horizontal tube along x at (y=20, z=20), radius 2
      - A few bright speckles (should be removed by opening + largest-CC)
    """
    rng = np.random.default_rng(0)
    arr = rng.uniform(0, 20, size=(40, 40, 40)).astype(np.float32)

    # Main tube along x, centered at y=20,z=20, radius 2
    for y in range(40):
        for z in range(40):
            if (y - 20) ** 2 + (z - 20) ** 2 <= 4:
                arr[z, y, 5:35] = 900.0

    # Isolated speckles (single voxels, far from the tube)
    for coord in [(2, 2, 2), (38, 38, 38), (2, 38, 2), (5, 35, 35)]:
        arr[coord] = 950.0

    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.5, 0.5))
    return img


def test_segment_extracts_tube():
    volume = _make_tube_volume()
    mask = segment_vessels(volume, threshold_percentile=95.0)

    assert isinstance(mask, sitk.Image)
    assert mask.GetSize() == volume.GetSize()
    assert mask.GetSpacing() == volume.GetSpacing()

    mask_arr = sitk.GetArrayFromImage(mask)
    assert mask_arr.dtype == np.uint8
    assert set(np.unique(mask_arr).tolist()).issubset({0, 1})

    voxel_count = int(mask_arr.sum())
    # Tube: ~13 voxels/slice × 30 slices = ~390. Speckles should be gone.
    assert voxel_count > 200
    assert voxel_count < 800

    # Single connected component
    n_components = sitk.GetArrayFromImage(
        sitk.ConnectedComponent(mask)
    ).max()
    assert n_components == 1

    # The tube lives at y=20,z=20 — mask should span that row along x
    assert mask_arr[20, 20, 15] == 1


def test_segment_endpoint_returns_bbox():
    volume = _make_tube_volume()
    session_id = session_store.create_session({
        "volume": volume,
        "metadata": {"source_format": "nifti", "modality": "unknown"},
    })

    app = FastAPI()
    app.include_router(dicom_router)
    client = TestClient(app)

    try:
        r = client.post(f"/dicom/segment/{session_id}")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "segmented"
        assert body["mask_voxel_count"] > 200
        bbox = body["bounding_box"]
        assert len(bbox) == 6
        # Tube spans x in [5, 35), y≈[18,23), z≈[18,23)
        x_min, y_min, z_min, sx, sy, sz = bbox
        assert 4 <= x_min <= 6
        assert 28 <= sx <= 32
        assert 17 <= y_min <= 19
        assert 17 <= z_min <= 19

        # Mask persisted in session store
        session = session_store.get_session(session_id)
        assert "mask" in session
    finally:
        session_store.drop_session(session_id)


def test_segment_endpoint_unknown_session():
    app = FastAPI()
    app.include_router(dicom_router)
    client = TestClient(app)
    r = client.post("/dicom/segment/nope")
    assert r.status_code == 404
