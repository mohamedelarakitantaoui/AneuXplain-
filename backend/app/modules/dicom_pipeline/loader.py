"""
Medical volume loader.

Accepts a zipped DICOM series (.zip) or a NIfTI file (.nii/.nii.gz) and
returns a SimpleITK image plus an anonymized metadata dict. We deliberately
never read PHI tags (patient name/ID/DOB/etc.) from DICOM headers, even
if present — only imaging parameters are extracted.
"""

from __future__ import annotations

import logging
import os
import tempfile
import zipfile
from typing import Any, Dict, List, Tuple

import SimpleITK as sitk

logger = logging.getLogger(__name__)


def _detect_format(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".zip"):
        return "dicom"
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        return "nifti"
    raise ValueError(
        f"Unsupported file extension for {path!r}. "
        "Expected .zip (DICOM series), .nii, or .nii.gz."
    )


def _find_best_series(extract_dir: str) -> Tuple[List[str], int]:
    """
    Walk extract_dir and return (files_of_largest_series, total_series_count).
    """
    reader = sitk.ImageSeriesReader()
    best_files: List[str] = []
    total_series = 0
    for root, _dirs, _files in os.walk(extract_dir):
        try:
            series_ids = reader.GetGDCMSeriesIDs(root)
        except RuntimeError:
            continue
        for sid in series_ids:
            total_series += 1
            files = reader.GetGDCMSeriesFileNames(root, sid)
            if len(files) > len(best_files):
                best_files = list(files)
    return best_files, total_series


def _load_dicom_from_zip(zip_path: str) -> Tuple[sitk.Image, str]:
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"Not a valid zip archive: {zip_path}")

    extract_dir = tempfile.mkdtemp(prefix="dicom_")
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile as e:
        raise ValueError(f"Corrupted zip archive: {e}") from e

    best_files, total_series = _find_best_series(extract_dir)
    if not best_files:
        raise ValueError("No DICOM series found in archive")
    if total_series > 1:
        logger.warning(
            "Archive contains %d DICOM series; using the largest (%d files)",
            total_series, len(best_files),
        )

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(best_files)
    try:
        volume = reader.Execute()
    except RuntimeError as e:
        raise ValueError(f"Failed to decode DICOM series: {e}") from e

    modality = "unknown"
    try:
        slice_reader = sitk.ImageFileReader()
        slice_reader.SetFileName(best_files[0])
        slice_reader.LoadPrivateTagsOff()
        slice_reader.ReadImageInformation()
        if slice_reader.HasMetaDataKey("0008|0060"):
            modality = slice_reader.GetMetaData("0008|0060").strip() or "unknown"
    except RuntimeError:
        pass

    return volume, modality


def _load_nifti(path: str) -> sitk.Image:
    try:
        return sitk.ReadImage(path)
    except RuntimeError as e:
        raise ValueError(f"Failed to read NIfTI file: {e}") from e


def _build_metadata(
    volume: sitk.Image,
    source_format: str,
    modality: str,
) -> Dict[str, Any]:
    spacing = volume.GetSpacing()
    size = volume.GetSize()
    direction = volume.GetDirection()
    return {
        "source_format": source_format,
        "modality": modality,
        "slice_thickness": float(spacing[2]) if len(spacing) >= 3 else None,
        "pixel_spacing": [float(spacing[0]), float(spacing[1])],
        "dimensions": [int(s) for s in size],
        "patient_orientation": [float(d) for d in direction],
    }


def load_medical_volume(path: str) -> Tuple[sitk.Image, Dict[str, Any]]:
    """
    Load a medical imaging volume from a DICOM zip or a NIfTI file.

    Supported formats:
        - ``.zip``          — a zipped DICOM series (picks the largest
                              series if multiple are present).
        - ``.nii`` / ``.nii.gz`` — a NIfTI volume.

    Returns:
        (volume, metadata). ``metadata`` is anonymized: PatientName,
        PatientID, PatientBirthDate, and other identifying DICOM tags
        are never extracted, even when present.

    Raises:
        ValueError: unsupported extension, corrupt archive, no DICOM
            series found, unreadable NIfTI, or empty volume.
    """
    source_format = _detect_format(path)

    if source_format == "dicom":
        volume, modality = _load_dicom_from_zip(path)
    else:
        volume = _load_nifti(path)
        modality = "unknown"
        logger.warning(
            "NIfTI files carry no modality tag; modality set to 'unknown'"
        )

    size = volume.GetSize()
    if any(d == 0 for d in size):
        raise ValueError(f"Loaded volume is empty (dimensions={list(size)})")

    if modality != "unknown" and modality.upper() != "MR":
        logger.warning(
            "Modality %r is not MR; risk predictor was trained on TOF-MRA — "
            "results may be unreliable",
            modality,
        )

    metadata = _build_metadata(volume, source_format, modality)
    return volume, metadata


def load_dicom_series(zip_path: str) -> sitk.Image:
    """Legacy thin wrapper — prefer :func:`load_medical_volume`."""
    volume, _ = load_medical_volume(zip_path)
    return volume
