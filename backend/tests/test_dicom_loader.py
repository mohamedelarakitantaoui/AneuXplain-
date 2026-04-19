"""Tests for app.modules.dicom_pipeline.loader.load_medical_volume."""

from __future__ import annotations

import time
import zipfile
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from app.modules.dicom_pipeline import loader
from app.modules.dicom_pipeline.loader import load_medical_volume


_PHI_TAG_NAMES = {
    "patient_name", "patient_id", "patient_birth_date",
    "patient_sex", "patient_age", "name", "dob",
}


# ---------- fixtures ----------

@pytest.fixture
def synthetic_dicom_zip(tmp_path: Path) -> Path:
    """Build a tiny 10x10x10 MR DICOM series and zip it."""
    dcm_dir = tmp_path / "dcm"
    dcm_dir.mkdir()

    arr = np.random.randint(0, 1000, size=(10, 10, 10), dtype=np.int16)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.5, 0.5, 1.0))

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    mod_time = time.strftime("%H%M%S")
    mod_date = time.strftime("%Y%m%d")
    series_uid = f"1.2.826.0.1.3680043.2.1125.{mod_date}.1{mod_time}"

    direction = img.GetDirection()
    orient = "\\".join(
        str(direction[i]) for i in (0, 3, 6, 1, 4, 7)
    )

    series_tags = [
        ("0008|0031", mod_time),
        ("0008|0021", mod_date),
        ("0008|0008", "DERIVED\\SECONDARY"),
        ("0020|000e", series_uid),
        ("0020|0037", orient),
        ("0008|103e", "Synthetic MR"),
        ("0008|0060", "MR"),
        # Deliberately DO NOT set patient tags — fixture must not leak PHI.
    ]

    for i in range(img.GetDepth()):
        slice_img = img[:, :, i]
        for tag, value in series_tags:
            slice_img.SetMetaData(tag, value)
        pos = img.TransformIndexToPhysicalPoint((0, 0, i))
        slice_img.SetMetaData("0020|0032", "\\".join(str(x) for x in pos))
        slice_img.SetMetaData("0020|0013", str(i))
        writer.SetFileName(str(dcm_dir / f"slice_{i:03d}.dcm"))
        writer.Execute(slice_img)

    zip_path = tmp_path / "series.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in sorted(dcm_dir.iterdir()):
            zf.write(f, arcname=f.name)
    return zip_path


@pytest.fixture
def synthetic_nifti(tmp_path: Path) -> Path:
    arr = np.random.rand(12, 11, 10).astype(np.float32)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((0.6, 0.6, 1.2))
    p = tmp_path / "volume.nii.gz"
    sitk.WriteImage(img, str(p))
    return p


# ---------- happy paths ----------

def test_loads_dicom_zip(synthetic_dicom_zip: Path):
    volume, metadata = load_medical_volume(str(synthetic_dicom_zip))

    assert isinstance(volume, sitk.Image)
    assert volume.GetSize() == (10, 10, 10)
    assert metadata["source_format"] == "dicom"
    assert metadata["modality"] == "MR"
    assert metadata["dimensions"] == [10, 10, 10]
    assert len(metadata["pixel_spacing"]) == 2
    assert metadata["slice_thickness"] is not None
    assert len(metadata["patient_orientation"]) == 9


def test_loads_nifti(synthetic_nifti: Path):
    volume, metadata = load_medical_volume(str(synthetic_nifti))

    assert isinstance(volume, sitk.Image)
    # SimpleITK uses (x, y, z); our array was shape (12, 11, 10) → size (10, 11, 12)
    assert volume.GetSize() == (10, 11, 12)
    assert metadata["source_format"] == "nifti"
    assert metadata["modality"] == "unknown"
    assert metadata["dimensions"] == [10, 11, 12]
    assert metadata["pixel_spacing"] == pytest.approx([0.6, 0.6])
    assert metadata["slice_thickness"] == pytest.approx(1.2)


def test_metadata_contains_no_phi(synthetic_dicom_zip: Path):
    _, metadata = load_medical_volume(str(synthetic_dicom_zip))
    keys_lower = {str(k).lower() for k in metadata.keys()}
    for phi in _PHI_TAG_NAMES:
        assert phi not in keys_lower, f"PHI-ish key leaked: {phi}"
    # And no stringified values that look like patient tags
    for v in metadata.values():
        assert "patient" not in str(v).lower()


# ---------- error cases ----------

def test_unsupported_extension(tmp_path: Path):
    p = tmp_path / "thing.obj"
    p.write_bytes(b"xxx")
    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_medical_volume(str(p))


def test_corrupt_zip(tmp_path: Path):
    p = tmp_path / "broken.zip"
    p.write_bytes(b"this is not a zip file")
    with pytest.raises(ValueError, match="valid zip"):
        load_medical_volume(str(p))


def test_zip_without_dicom(tmp_path: Path):
    p = tmp_path / "empty.zip"
    with zipfile.ZipFile(p, "w") as zf:
        zf.writestr("readme.txt", "no DICOMs here")
    with pytest.raises(ValueError, match="No DICOM series"):
        load_medical_volume(str(p))


def test_empty_volume_raises(tmp_path: Path, monkeypatch):
    """If the decoded volume has a zero dimension, loader must reject it."""
    empty = sitk.Image([1, 0, 1], sitk.sitkFloat32)
    monkeypatch.setattr(loader, "_load_nifti", lambda _p: empty)

    p = tmp_path / "fake.nii"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="empty"):
        load_medical_volume(str(p))


def test_unreadable_nifti(tmp_path: Path):
    p = tmp_path / "broken.nii.gz"
    p.write_bytes(b"not a real nifti stream")
    with pytest.raises(ValueError, match="Failed to read NIfTI"):
        load_medical_volume(str(p))
