"""
DICOM / NIfTI preprocessing endpoints.

Flow:
    POST /dicom/upload             — accept a zipped DICOM series OR a
                                     NIfTI file, decode the volume, stash
                                     it under a session_id.
    POST /dicom/crop-and-analyze   — (stub) given a session_id and a
                                     click point, crop the local region
                                     and hand it to the existing analyze
                                     pipeline.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import SimpleITK as sitk
import trimesh
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from ..analyze_core import analyze_mesh_file
from ..heatmap_core import compute_heatmap_for_mesh
from ..modules.dicom_pipeline import session_store
from ..modules.dicom_pipeline.cropper import crop_around_point
from ..modules.dicom_pipeline.harmonizer import (
    compare_to_intra_distribution,
    harmonize_to_intra,
)
from ..modules.dicom_pipeline.loader import load_medical_volume
from ..modules.dicom_pipeline.mesher import mask_to_mesh
from ..modules.dicom_pipeline.segmenter import segment_vessels


logger = logging.getLogger("aneuxplain.dicom")
router = APIRouter(prefix="/dicom", tags=["dicom"])


class UploadResponse(BaseModel):
    session_id: str
    metadata: Dict[str, Any]
    status: str


class CropAndAnalyzeRequest(BaseModel):
    session_id: str
    click_point: List[float]
    crop_radius_mm: float = 15.0


def _suffix_for(filename: str) -> str:
    """Preserve `.nii.gz` as a compound suffix; otherwise take the last one."""
    lower = filename.lower()
    if lower.endswith(".nii.gz"):
        return ".nii.gz"
    return Path(filename).suffix or ".bin"


@router.post(
    "/upload",
    response_model=UploadResponse,
    summary="Upload a DICOM series (.zip) or NIfTI volume (.nii / .nii.gz)",
)
async def upload_volume(file: UploadFile = File(...)) -> UploadResponse:
    """
    Accept a medical imaging volume and register it in the session store.

    Supported inputs:
        - ``.zip`` containing a DICOM MRA series (the largest series in
          the archive is selected if more than one is present).
        - ``.nii`` / ``.nii.gz`` — a NIfTI volume.

    Returns the session id, anonymized metadata (no patient-identifying
    tags are read from DICOM headers), and ``status: "loaded"``.
    """
    filename = file.filename or "upload.bin"
    suffix = _suffix_for(filename)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = tmp.name
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()

        try:
            volume, metadata = load_medical_volume(tmp_path)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:  # pragma: no cover — safety net
            logger.exception("Unexpected loader failure")
            raise HTTPException(status_code=500, detail=f"Volume load failed: {e}")

        session_id = session_store.create_session({
            "volume": volume,
            "metadata": metadata,
        })
        return UploadResponse(
            session_id=session_id,
            metadata=metadata,
            status="loaded",
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


class SliceInfoResponse(BaseModel):
    axial_count: int
    sagittal_count: int
    coronal_count: int
    dimensions: List[int]
    spacing: List[float]


_VALID_AXES = ("axial", "sagittal", "coronal")


def _extract_slice(volume: sitk.Image, axis: str, index: int) -> np.ndarray:
    """Return a 2D numpy array for the requested orthogonal slice."""
    size = volume.GetSize()  # (x, y, z)
    if axis == "axial":
        max_idx = size[2]
    elif axis == "coronal":
        max_idx = size[1]
    else:  # sagittal
        max_idx = size[0]

    if index < 0 or index >= max_idx:
        raise HTTPException(
            status_code=400,
            detail=f"Index {index} out of range for {axis}: valid [0, {max_idx - 1}]",
        )

    if axis == "axial":
        slice_img = volume[:, :, index]
    elif axis == "coronal":
        slice_img = volume[:, index, :]
    else:
        slice_img = volume[index, :, :]

    return sitk.GetArrayFromImage(slice_img)


def _slice_to_png(slice_arr: np.ndarray, lo: float, hi: float) -> bytes:
    clipped = np.clip(slice_arr.astype(np.float32), lo, hi)
    norm = (clipped - lo) / (hi - lo)
    u8 = (norm * 255.0).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8, mode="L").save(buf, format="PNG")
    return buf.getvalue()


@router.get(
    "/slice-info/{session_id}",
    response_model=SliceInfoResponse,
    summary="Slice counts and spacing for a loaded volume",
)
async def slice_info(session_id: str) -> SliceInfoResponse:
    session = session_store.get_session(session_id)
    if session is None or "volume" not in session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    volume: sitk.Image = session["volume"]
    size = volume.GetSize()
    spacing = volume.GetSpacing()
    return SliceInfoResponse(
        axial_count=int(size[2]),
        sagittal_count=int(size[0]),
        coronal_count=int(size[1]),
        dimensions=[int(s) for s in size],
        spacing=[float(s) for s in spacing],
    )


@router.get(
    "/slice/{session_id}/{axis}/{index}",
    responses={200: {"content": {"image/png": {}}}},
    summary="Windowed 2D slice preview (PNG)",
)
async def get_slice(session_id: str, axis: str, index: int) -> Response:
    if axis not in _VALID_AXES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid axis {axis!r}; expected one of {_VALID_AXES}",
        )

    session = session_store.get_session(session_id)
    if session is None or "volume" not in session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    volume: sitk.Image = session["volume"]
    slice_arr = _extract_slice(volume, axis, index)

    window = session_store.get_or_compute_window(session_id)
    assert window is not None  # session presence already checked
    lo, hi = window

    png_bytes = _slice_to_png(slice_arr, lo, hi)
    return Response(content=png_bytes, media_type="image/png")


class SegmentResponse(BaseModel):
    status: str
    mask_voxel_count: int
    bounding_box: List[int]  # [x_min, y_min, z_min, x_size, y_size, z_size]
    vertex_count: int
    face_count: int
    mesh_bounds_mm: List[List[float]]  # [[xmin,ymin,zmin], [xmax,ymax,zmax]]


@router.post(
    "/segment/{session_id}",
    response_model=SegmentResponse,
    summary="Extract the vascular tree from a loaded volume",
)
async def segment_session(
    session_id: str,
    threshold_percentile: float = 99.0,
    use_vesselness: bool = False,
) -> SegmentResponse:
    session = session_store.get_session(session_id)
    if session is None or "volume" not in session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    volume: sitk.Image = session["volume"]
    try:
        mask = segment_vessels(
            volume,
            threshold_percentile=threshold_percentile,
            use_vesselness=use_vesselness,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    mask_arr = sitk.GetArrayViewFromImage(mask)
    voxel_count = int(mask_arr.sum())
    if voxel_count == 0:
        raise HTTPException(
            status_code=400,
            detail="Segmentation produced an empty mask; try lowering threshold_percentile.",
        )

    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(mask)
    bbox = list(stats.GetBoundingBox(1))  # (x, y, z, sx, sy, sz)

    try:
        mesh = mask_to_mesh(mask)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    session_store.update_session(session_id, mask=mask, full_mesh=mesh)

    bounds = mesh.bounds  # (2, 3) array
    return SegmentResponse(
        status="segmented",
        mask_voxel_count=voxel_count,
        bounding_box=[int(v) for v in bbox],
        vertex_count=int(len(mesh.vertices)),
        face_count=int(len(mesh.faces)),
        mesh_bounds_mm=[
            [float(x) for x in bounds[0]],
            [float(x) for x in bounds[1]],
        ],
    )


@router.get(
    "/full-mesh/{session_id}.obj",
    summary="Download the full vessel-tree mesh as a Wavefront .obj",
)
async def download_full_mesh(session_id: str) -> Response:
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    if "full_mesh" not in session:
        raise HTTPException(
            status_code=404,
            detail="Segmentation must be run before downloading mesh",
        )

    mesh: trimesh.Trimesh = session["full_mesh"]
    obj_bytes = mesh.export(file_type="obj")
    if isinstance(obj_bytes, str):
        obj_bytes = obj_bytes.encode("utf-8")

    return Response(
        content=obj_bytes,
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename=vessel_tree_{session_id}.obj',
        },
    )


@router.post("/crop-and-analyze")
async def crop_and_analyze(payload: CropAndAnalyzeRequest) -> Dict[str, Any]:
    """
    Crop the stored vessel mesh around a clicked point, harmonize to the
    IntrA distribution, and run the shared analyze pipeline on the result.
    """
    session = session_store.get_session(payload.session_id)
    if session is None:
        raise HTTPException(
            status_code=404, detail=f"Session not found: {payload.session_id}"
        )
    if "full_mesh" not in session:
        raise HTTPException(
            status_code=400,
            detail="Segmentation must be run before crop-and-analyze",
        )

    if len(payload.click_point) != 3:
        raise HTTPException(status_code=400, detail="click_point must have 3 floats")
    click = (
        float(payload.click_point[0]),
        float(payload.click_point[1]),
        float(payload.click_point[2]),
    )

    full_mesh: trimesh.Trimesh = session["full_mesh"]
    try:
        cropped = crop_around_point(full_mesh, click, payload.crop_radius_mm)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cropped_vc = int(len(cropped.vertices))
    harmonized = harmonize_to_intra(cropped)
    harmonized_vc = int(len(harmonized.vertices))
    dist = compare_to_intra_distribution(harmonized)

    # Lazy import to avoid a circular import at module load (main imports this
    # router; this function reaches back into main for the globals).
    from ..main import engine, morphology_analyzer, clinical_explainer

    if engine is None or not getattr(engine, "_models_loaded", False):
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Replace any prior cropped mesh for this session before writing a
    # fresh one — we intentionally keep the file around after the request
    # so GET /dicom/cropped-mesh/{session_id}.obj can serve it.
    prev_path = session.get("cropped_obj_path")
    if prev_path:
        try:
            os.unlink(prev_path)
        except OSError:
            pass

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".obj")
    tmp_path = tmp.name
    tmp.close()
    harmonized.export(tmp_path)
    try:
        analysis = analyze_mesh_file(
            tmp_path, engine, morphology_analyzer, clinical_explainer
        )
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    session_store.update_session(payload.session_id, cropped_obj_path=tmp_path)

    response: Dict[str, Any] = dict(analysis)
    response["harmonization"] = {
        "vertex_count": dist["vertex_count"],
        "bbox_diagonal": dist["bbox_diagonal"],
        "edge_length": dist["edge_length"],
        "all_in_distribution": bool(
            dist["vertex_count"]["in_distribution"]
            and dist["bbox_diagonal"]["in_distribution"]
            and dist["edge_length"]["in_distribution"]
        ),
    }
    response["crop_info"] = {
        "click_point": list(click),
        "radius_mm": float(payload.crop_radius_mm),
        "cropped_vertex_count": cropped_vc,
        "harmonized_vertex_count": harmonized_vc,
    }
    return response


@router.get(
    "/cropped-mesh/{session_id}.obj",
    summary="Download the cropped + harmonized mesh as a Wavefront .obj",
)
async def download_cropped_mesh(session_id: str) -> Response:
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    path = session.get("cropped_obj_path")
    if not path or not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="No cropped mesh for this session — run crop-and-analyze first.",
        )
    with open(path, "rb") as f:
        obj_bytes = f.read()
    return Response(
        content=obj_bytes,
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename=cropped_{session_id}.obj',
        },
    )


@router.get(
    "/cropped-heatmap/{session_id}",
    summary="Gradient-based spatial heatmap for the cropped mesh in a session",
)
async def cropped_heatmap(session_id: str) -> Dict[str, Any]:
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    path = session.get("cropped_obj_path")
    if not path or not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail="No cropped mesh for this session — run crop-and-analyze first.",
        )

    from ..main import engine

    if engine is None or not getattr(engine, "_models_loaded", False):
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        return compute_heatmap_for_mesh(path, engine)
    except Exception as e:
        logger.warning("Cropped heatmap computation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Heatmap computation failed: {e}")
