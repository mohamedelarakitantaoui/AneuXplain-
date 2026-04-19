"""Tests for /dicom/slice and /dicom/slice-info."""

from __future__ import annotations

import io

import numpy as np
import pytest
import SimpleITK as sitk
from fastapi.testclient import TestClient
from PIL import Image

from app.modules.dicom_pipeline import session_store
from app.routes.dicom import router as dicom_router

from fastapi import FastAPI


@pytest.fixture
def client() -> TestClient:
    app = FastAPI()
    app.include_router(dicom_router)
    return TestClient(app)


@pytest.fixture
def loaded_session() -> str:
    # Deterministic ramp volume so PNG output is non-trivial.
    arr = np.zeros((8, 12, 16), dtype=np.float32)  # (z, y, x)
    for z in range(arr.shape[0]):
        for y in range(arr.shape[1]):
            for x in range(arr.shape[2]):
                arr[z, y, x] = (x + y * 2 + z * 5) * 10.0
    volume = sitk.GetImageFromArray(arr)
    volume.SetSpacing((0.5, 0.6, 0.7))

    session_id = session_store.create_session({
        "volume": volume,
        "metadata": {
            "source_format": "nifti",
            "modality": "unknown",
            "dimensions": [16, 12, 8],
            "pixel_spacing": [0.5, 0.6],
            "slice_thickness": 0.7,
            "patient_orientation": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        },
    })
    yield session_id
    session_store.drop_session(session_id)


def test_slice_info(client: TestClient, loaded_session: str):
    r = client.get(f"/dicom/slice-info/{loaded_session}")
    assert r.status_code == 200
    body = r.json()
    assert body["axial_count"] == 8
    assert body["sagittal_count"] == 16
    assert body["coronal_count"] == 12
    assert body["dimensions"] == [16, 12, 8]
    assert body["spacing"] == pytest.approx([0.5, 0.6, 0.7])


def test_axial_slice_png(client: TestClient, loaded_session: str):
    r = client.get(f"/dicom/slice/{loaded_session}/axial/3")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(r.content))
    assert img.mode == "L"
    assert img.size == (16, 12)  # (width=x, height=y)


def test_sagittal_slice_png(client: TestClient, loaded_session: str):
    r = client.get(f"/dicom/slice/{loaded_session}/sagittal/5")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(r.content))
    assert img.size == (12, 8)  # (y, z)


def test_coronal_slice_png(client: TestClient, loaded_session: str):
    r = client.get(f"/dicom/slice/{loaded_session}/coronal/4")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(r.content))
    assert img.size == (16, 8)  # (x, z)


def test_window_is_cached(client: TestClient, loaded_session: str):
    client.get(f"/dicom/slice/{loaded_session}/axial/0")
    session = session_store.get_session(loaded_session)
    assert "window" in session
    lo, hi = session["window"]
    assert lo < hi


def test_unknown_session_404(client: TestClient):
    r = client.get("/dicom/slice/nope/axial/0")
    assert r.status_code == 404
    r2 = client.get("/dicom/slice-info/nope")
    assert r2.status_code == 404


def test_invalid_axis_400(client: TestClient, loaded_session: str):
    r = client.get(f"/dicom/slice/{loaded_session}/diagonal/0")
    assert r.status_code == 400
    assert "Invalid axis" in r.json()["detail"]


def test_out_of_range_index_400(client: TestClient, loaded_session: str):
    r = client.get(f"/dicom/slice/{loaded_session}/axial/99")
    assert r.status_code == 400
    assert "out of range" in r.json()["detail"]
    assert "[0, 7]" in r.json()["detail"]  # axial_count=8 → valid [0,7]
