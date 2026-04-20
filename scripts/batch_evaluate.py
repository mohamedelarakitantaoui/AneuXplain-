"""
Batch-evaluate the AneuXplain prototype across the 50-subject Lausanne cohort
(25 patients + 25 controls).

For each subject:
    1. POST /dicom/upload                        (NIfTI volume from angio/)
    2. POST /dicom/segment/{session_id}          (default vessel segmentation)
    3. POST /dicom/crop-and-analyze              (click point from click_points.json)

Per-subject failures are logged and the batch continues. The CSV is written
incrementally so a Ctrl-C / crash won't lose completed subjects, and re-running
the script skips any subject already marked "success".

Inputs:
    C:\\Users\\buzok\\Desktop\\Test-demo\\click_points.json
    C:\\Users\\buzok\\Desktop\\Test-demo\\angio\\*.nii

Outputs:
    C:\\Users\\buzok\\Desktop\\Test-demo\\results\\batch_results.csv
    C:\\Users\\buzok\\Desktop\\Test-demo\\results\\raw_responses\\{subject_id}.json
    C:\\Users\\buzok\\Desktop\\Test-demo\\results\\batch.log
"""

from __future__ import annotations

import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError:
    sys.stderr.write(
        "requests not installed. Run: pip install requests --break-system-packages\n"
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "http://localhost:8000"
TEST_DEMO_DIR = Path(r"C:\Users\buzok\Desktop\Test-demo")
ANGIO_DIR = TEST_DEMO_DIR / "angio"
CLICK_POINTS_PATH = TEST_DEMO_DIR / "click_points.json"
RESULTS_DIR = TEST_DEMO_DIR / "results"
RAW_RESPONSES_DIR = RESULTS_DIR / "raw_responses"
CSV_PATH = RESULTS_DIR / "batch_results.csv"
LOG_PATH = RESULTS_DIR / "batch.log"

CROP_RADIUS_MM = 15.0
HTTP_TIMEOUT_SECONDS = 300

# Maps the morphology measurement "name" field (as returned by the backend)
# to the flat CSV column name.
MORPHOLOGY_NAME_TO_COLUMN = {
    "Neck Width": "neck_width",
    "Dome Height": "dome_height",
    "Max Dome Diameter": "max_dome_diameter",
    "Aspect Ratio": "aspect_ratio",
    "Dome-to-Neck Ratio": "dome_to_neck_ratio",
    "Irregularity Index": "irregularity_index",
    "Volume": "volume",
    "Surface Area": "surface_area",
}
MORPHOLOGY_COLUMNS = list(MORPHOLOGY_NAME_TO_COLUMN.values())

CSV_COLUMNS = [
    "subject_id",
    "group",
    "click_point_x", "click_point_y", "click_point_z",
    "lesion_voxel_count",
    "lesion_extent_mm_max",
    "pipeline_status",
    "error_message",
    "upload_ms", "segment_ms", "analyze_ms", "total_ms",
    "session_id",
    "risk_score",
    "predicted_class",
    *MORPHOLOGY_COLUMNS,
    "click_point_source",
    "mesh_vertex_count",
    "mesh_face_count",
    "mesh_bounds_x_min", "mesh_bounds_x_max",
    "mesh_bounds_y_min", "mesh_bounds_y_max",
    "mesh_bounds_z_min", "mesh_bounds_z_max",
    "cropped_vertex_count",
    "harmonized_vertex_count",
    "harmonization_all_in_distribution",
    "vertex_count_zscore",
    "bbox_diagonal_zscore",
    "edge_length_zscore",
    "high_risk_count",
    "moderate_risk_count",
    "low_risk_count",
    "raw_response_file",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging() -> logging.Logger:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("batch_evaluate")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_angio_file(subject_id: str) -> Path | None:
    """Find the .nii file for a subject, by prefix match."""
    matches = sorted(ANGIO_DIR.glob(f"{subject_id}_*_angio.nii"))
    if not matches:
        return None
    return matches[0]


def load_click_points() -> dict[str, Any]:
    with CLICK_POINTS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_results() -> dict[str, dict[str, str]]:
    """Read CSV if it exists, return {subject_id: row_dict}."""
    if not CSV_PATH.exists():
        return {}
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return {row["subject_id"]: row for row in reader}


def write_results_csv(rows: list[dict[str, Any]]) -> None:
    """Atomic-ish rewrite of the entire CSV."""
    tmp = CSV_PATH.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
    tmp.replace(CSV_PATH)


def extract_value(d: Any, *keys: str) -> Any:
    """Return the first present, non-None top-level key from a dict."""
    if not isinstance(d, dict):
        return None
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def find_first_key_match(d: Any, substrings: list[str]) -> Any:
    """Search top-level keys (case-insensitive) for one containing any substring."""
    if not isinstance(d, dict):
        return None
    for k, v in d.items():
        kl = k.lower()
        if any(sub in kl for sub in substrings) and v is not None:
            return v
    return None


def extract_morphology(analyze_response: dict[str, Any]) -> dict[str, str]:
    """Pull the 8 morphology floats out of the response into flat CSV columns."""
    out: dict[str, str] = {col: "" for col in MORPHOLOGY_COLUMNS}
    morph = analyze_response.get("morphology")
    if not isinstance(morph, dict):
        return out
    for measurement in morph.get("measurements", []):
        name = measurement.get("name")
        col = MORPHOLOGY_NAME_TO_COLUMN.get(name)
        if not col:
            continue
        value = measurement.get("value")
        status = measurement.get("status")
        if value is None or status == "failed":
            continue
        try:
            out[col] = f"{float(value):.6f}"
        except (TypeError, ValueError):
            pass
    return out


def resolve_click_point(
    group: str,
    stored_click: list[float] | None,
    mesh_bounds: list[list[float]] | None,
) -> tuple[list[float], str]:
    """
    Decide which click point to send to /dicom/crop-and-analyze.

    - controls: always mesh center ("mesh_center_control")
    - patients: stored lesion centroid if strictly inside mesh_bounds
                ("lesion_centroid"), else mesh center
                ("mesh_center_fallback_patient")

    Note: process_subject() may additionally set click_point_source to
    "mesh_center_fallback_empty_crop" at runtime if the first
    crop-and-analyze attempt (with click_source == "lesion_centroid")
    returns the backend's "Crop too small" error and is retried with
    mesh_center.
    """
    center: list[float] | None = None
    if isinstance(mesh_bounds, list) and len(mesh_bounds) == 2:
        try:
            (xmin, ymin, zmin) = (float(mesh_bounds[0][0]),
                                  float(mesh_bounds[0][1]),
                                  float(mesh_bounds[0][2]))
            (xmax, ymax, zmax) = (float(mesh_bounds[1][0]),
                                  float(mesh_bounds[1][1]),
                                  float(mesh_bounds[1][2]))
            center = [(xmin + xmax) / 2.0,
                      (ymin + ymax) / 2.0,
                      (zmin + zmax) / 2.0]
        except (TypeError, ValueError, IndexError):
            center = None

    if group == "control":
        if center is not None:
            return center, "mesh_center_control"
        # No usable bounds; fall back to stored click if any, otherwise origin.
        if stored_click is not None and len(stored_click) == 3:
            return [float(c) for c in stored_click], "mesh_center_control"
        return [0.0, 0.0, 0.0], "mesh_center_control"

    # patient
    if (stored_click is not None and len(stored_click) == 3 and center is not None):
        x, y, z = (float(stored_click[0]), float(stored_click[1]), float(stored_click[2]))
        if (xmin < x < xmax) and (ymin < y < ymax) and (zmin < z < zmax):
            return [x, y, z], "lesion_centroid"
        return center, "mesh_center_fallback_patient"
    if center is not None:
        return center, "mesh_center_fallback_patient"
    if stored_click is not None and len(stored_click) == 3:
        return [float(c) for c in stored_click], "lesion_centroid"
    return [0.0, 0.0, 0.0], "mesh_center_fallback_patient"


# ---------------------------------------------------------------------------
# HTTP pipeline
# ---------------------------------------------------------------------------

def upload_volume(nii_path: Path) -> tuple[str, float]:
    """POST /dicom/upload. Returns (session_id, elapsed_ms)."""
    t0 = time.perf_counter()
    with nii_path.open("rb") as f:
        files = {"file": (nii_path.name, f, "application/octet-stream")}
        r = requests.post(
            f"{BASE_URL}/dicom/upload",
            files=files,
            timeout=HTTP_TIMEOUT_SECONDS,
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    return r.json()["session_id"], elapsed_ms


def segment_session(session_id: str) -> tuple[dict[str, Any], float]:
    """POST /dicom/segment/{session_id}. Returns (response_json, elapsed_ms)."""
    t0 = time.perf_counter()
    r = requests.post(
        f"{BASE_URL}/dicom/segment/{session_id}",
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    return r.json(), elapsed_ms


def crop_and_analyze(
    session_id: str,
    click_point: list[float],
) -> tuple[dict[str, Any], float]:
    """POST /dicom/crop-and-analyze. Returns (response_json, elapsed_ms)."""
    t0 = time.perf_counter()
    r = requests.post(
        f"{BASE_URL}/dicom/crop-and-analyze",
        json={
            "session_id": session_id,
            "click_point": click_point,
            "crop_radius_mm": CROP_RADIUS_MM,
        },
        timeout=HTTP_TIMEOUT_SECONDS,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    r.raise_for_status()
    return r.json(), elapsed_ms


# ---------------------------------------------------------------------------
# Per-subject driver
# ---------------------------------------------------------------------------

def process_subject(
    subject_id: str,
    group: str,
    stored_click: list[float] | None,
    nii_path: Path,
    lesion_voxel_count: int | str,
    lesion_extent_mm_max: float | str,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Run the full pipeline. Always returns a row dict; never raises."""
    row: dict[str, Any] = {
        "subject_id": subject_id,
        "group": group,
        "click_point_x": "",
        "click_point_y": "",
        "click_point_z": "",
        "lesion_voxel_count": lesion_voxel_count,
        "lesion_extent_mm_max": lesion_extent_mm_max,
        "pipeline_status": "",
        "error_message": "",
        "upload_ms": "", "segment_ms": "", "analyze_ms": "", "total_ms": "",
        "session_id": "",
        "risk_score": "",
        "predicted_class": "",
        "click_point_source": "",
        "mesh_vertex_count": "",
        "mesh_face_count": "",
        "mesh_bounds_x_min": "", "mesh_bounds_x_max": "",
        "mesh_bounds_y_min": "", "mesh_bounds_y_max": "",
        "mesh_bounds_z_min": "", "mesh_bounds_z_max": "",
        "cropped_vertex_count": "",
        "harmonized_vertex_count": "",
        "harmonization_all_in_distribution": "",
        "vertex_count_zscore": "",
        "bbox_diagonal_zscore": "",
        "edge_length_zscore": "",
        "high_risk_count": "",
        "moderate_risk_count": "",
        "low_risk_count": "",
        "raw_response_file": "",
        **{col: "" for col in MORPHOLOGY_COLUMNS},
    }

    raw_path = RAW_RESPONSES_DIR / f"{subject_id}.json"
    row["raw_response_file"] = str(raw_path)

    t_total = time.perf_counter()
    session_id = ""
    stage = "upload"
    try:
        session_id, upload_ms = upload_volume(nii_path)
        row["upload_ms"] = f"{upload_ms:.1f}"
        row["session_id"] = session_id

        stage = "segment"
        segment_resp, segment_ms = segment_session(session_id)
        row["segment_ms"] = f"{segment_ms:.1f}"

        # Mesh metadata from segment response.
        mesh_bounds = segment_resp.get("mesh_bounds_mm")
        if isinstance(mesh_bounds, list) and len(mesh_bounds) == 2:
            try:
                row["mesh_bounds_x_min"] = f"{float(mesh_bounds[0][0]):.3f}"
                row["mesh_bounds_y_min"] = f"{float(mesh_bounds[0][1]):.3f}"
                row["mesh_bounds_z_min"] = f"{float(mesh_bounds[0][2]):.3f}"
                row["mesh_bounds_x_max"] = f"{float(mesh_bounds[1][0]):.3f}"
                row["mesh_bounds_y_max"] = f"{float(mesh_bounds[1][1]):.3f}"
                row["mesh_bounds_z_max"] = f"{float(mesh_bounds[1][2]):.3f}"
            except (TypeError, ValueError, IndexError):
                pass
        vc = segment_resp.get("vertex_count")
        fc = segment_resp.get("face_count")
        if isinstance(vc, int):
            row["mesh_vertex_count"] = vc
        if isinstance(fc, int):
            row["mesh_face_count"] = fc

        # Resolve final click point per policy.
        click_point, click_source = resolve_click_point(group, stored_click, mesh_bounds)
        row["click_point_source"] = click_source
        row["click_point_x"] = f"{click_point[0]:.6f}"
        row["click_point_y"] = f"{click_point[1]:.6f}"
        row["click_point_z"] = f"{click_point[2]:.6f}"
        logger.info(
            "%s click_point=[%.3f, %.3f, %.3f] source=%s",
            subject_id, click_point[0], click_point[1], click_point[2], click_source,
        )
        # Record control override explicitly so batch.log shows that the stored
        # control click from click_points.json was deliberately ignored.
        if group == "control" and stored_click is not None and len(stored_click) == 3:
            logger.info(
                "%s control override: stored_click=[%.3f, %.3f, %.3f] ignored; "
                "using mesh_center=[%.3f, %.3f, %.3f]",
                subject_id,
                float(stored_click[0]), float(stored_click[1]), float(stored_click[2]),
                click_point[0], click_point[1], click_point[2],
            )
        elif group == "patient" and click_source == "mesh_center_fallback_patient" \
                and stored_click is not None and len(stored_click) == 3:
            logger.info(
                "%s patient fallback: stored_click=[%.3f, %.3f, %.3f] outside mesh; "
                "using mesh_center=[%.3f, %.3f, %.3f]",
                subject_id,
                float(stored_click[0]), float(stored_click[1]), float(stored_click[2]),
                click_point[0], click_point[1], click_point[2],
            )

        stage = "analyze"
        # First attempt. If the backend rejects the crop as empty
        # ("Crop too small") AND we used the lesion centroid, retry ONCE
        # with mesh_center. Any other HTTPError, or an empty-crop retry
        # when we already used mesh_center, propagates to the outer
        # handler and marks the subject analyze_failed as before.
        try:
            analyze_resp, analyze_ms = crop_and_analyze(session_id, click_point)
        except requests.HTTPError as first_err:
            body = ""
            if first_err.response is not None:
                body = first_err.response.text or ""
            if "Crop too small" in body and click_source == "lesion_centroid":
                mesh_center: list[float] | None = None
                if isinstance(mesh_bounds, list) and len(mesh_bounds) == 2:
                    try:
                        xmin, ymin, zmin = (float(mesh_bounds[0][0]),
                                            float(mesh_bounds[0][1]),
                                            float(mesh_bounds[0][2]))
                        xmax, ymax, zmax = (float(mesh_bounds[1][0]),
                                            float(mesh_bounds[1][1]),
                                            float(mesh_bounds[1][2]))
                        mesh_center = [(xmin + xmax) / 2.0,
                                       (ymin + ymax) / 2.0,
                                       (zmin + zmax) / 2.0]
                    except (TypeError, ValueError, IndexError):
                        mesh_center = None
                if mesh_center is None:
                    raise
                logger.info(
                    "%s lesion_centroid crop empty; retrying with "
                    "mesh_center=[%.3f, %.3f, %.3f]",
                    subject_id, mesh_center[0], mesh_center[1], mesh_center[2],
                )
                click_point = mesh_center
                click_source = "mesh_center_fallback_empty_crop"
                row["click_point_source"] = click_source
                row["click_point_x"] = f"{click_point[0]:.6f}"
                row["click_point_y"] = f"{click_point[1]:.6f}"
                row["click_point_z"] = f"{click_point[2]:.6f}"
                analyze_resp, analyze_ms = crop_and_analyze(session_id, click_point)
            else:
                raise
        row["analyze_ms"] = f"{analyze_ms:.1f}"

        # Persist the raw response. errors="replace" tolerates mangled UTF-8
        # unit strings (e.g. "mm²" returned as "mmÂ²") without crashing.
        try:
            with raw_path.open("w", encoding="utf-8", errors="replace") as f:
                json.dump(analyze_resp, f, indent=2, default=str, ensure_ascii=False)
        except OSError as write_err:
            logger.warning(
                "Could not persist raw response for %s: %s", subject_id, write_err
            )

        risk = extract_value(analyze_resp, "risk_score")
        if risk is None:
            risk = find_first_key_match(analyze_resp, ["risk_score", "risk", "probability"])
        if isinstance(risk, (int, float)):
            row["risk_score"] = f"{float(risk):.6f}"

        predicted = extract_value(analyze_resp, "risk_level", "predicted_class")
        if predicted is not None:
            row["predicted_class"] = str(predicted)

        morph_cols = extract_morphology(analyze_resp)
        row.update(morph_cols)

        # Crop info.
        crop_info = analyze_resp.get("crop_info")
        if isinstance(crop_info, dict):
            cv = crop_info.get("cropped_vertex_count")
            hv = crop_info.get("harmonized_vertex_count")
            if isinstance(cv, int):
                row["cropped_vertex_count"] = cv
            if isinstance(hv, int):
                row["harmonized_vertex_count"] = hv

        # Harmonization block: all_in_distribution + per-metric z_scores.
        harm = analyze_resp.get("harmonization")
        if isinstance(harm, dict):
            aid = harm.get("all_in_distribution")
            if aid is not None:
                row["harmonization_all_in_distribution"] = str(bool(aid))
            for sub_key, col in (
                ("vertex_count", "vertex_count_zscore"),
                ("bbox_diagonal", "bbox_diagonal_zscore"),
                ("edge_length", "edge_length_zscore"),
            ):
                sub = harm.get(sub_key)
                if isinstance(sub, dict):
                    z = sub.get("z_score")
                    if isinstance(z, (int, float)):
                        row[col] = f"{float(z):.6f}"

        # Clinical report risk-bucket counts.
        clin = analyze_resp.get("clinical_report")
        if isinstance(clin, dict):
            for key in ("high_risk_count", "moderate_risk_count", "low_risk_count"):
                v = clin.get(key)
                if isinstance(v, int):
                    row[key] = v

        row["pipeline_status"] = "success"

    except requests.Timeout as e:
        row["pipeline_status"] = "timeout"
        row["error_message"] = f"timeout at {stage}: {e}"
        _write_failure_response(raw_path, subject_id, stage, str(e), logger)
    except requests.HTTPError as e:
        row["pipeline_status"] = f"{stage}_failed"
        body = ""
        if e.response is not None:
            body = (e.response.text or "")[:500]
        row["error_message"] = f"HTTP {e.response.status_code if e.response else '?'} at {stage}: {body}"
        _write_failure_response(raw_path, subject_id, stage, row["error_message"], logger)
    except requests.RequestException as e:
        row["pipeline_status"] = f"{stage}_failed"
        row["error_message"] = f"{type(e).__name__} at {stage}: {e}"
        _write_failure_response(raw_path, subject_id, stage, str(e), logger)
    except Exception as e:  # last-resort safety net
        row["pipeline_status"] = f"{stage}_failed"
        row["error_message"] = f"unexpected {type(e).__name__} at {stage}: {e}"
        _write_failure_response(raw_path, subject_id, stage, str(e), logger)

    row["total_ms"] = f"{(time.perf_counter() - t_total) * 1000.0:.1f}"
    return row


def _write_failure_response(
    raw_path: Path, subject_id: str, stage: str, message: str, logger: logging.Logger
) -> None:
    try:
        with raw_path.open("w", encoding="utf-8") as f:
            json.dump(
                {"subject_id": subject_id, "failed_stage": stage, "error": message},
                f, indent=2,
            )
    except OSError as e:
        logger.warning("Could not write failure marker for %s: %s", subject_id, e)


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def sanity_check(click_points: dict[str, Any], logger: logging.Logger) -> bool:
    ok = True

    if not CLICK_POINTS_PATH.exists():
        logger.error("click_points.json not found at %s", CLICK_POINTS_PATH)
        return False

    n_p = len(click_points.get("patients", {}))
    n_c = len(click_points.get("controls", {}))
    logger.info("click_points.json: %d patients + %d controls", n_p, n_c)
    if n_p != 25 or n_c != 25:
        logger.error("Expected 25 patients + 25 controls; got %d + %d", n_p, n_c)
        ok = False

    if not ANGIO_DIR.is_dir():
        logger.error("Angio directory not found: %s", ANGIO_DIR)
        return False

    all_ids = list(click_points.get("patients", {}).keys()) + list(
        click_points.get("controls", {}).keys()
    )
    missing = [sid for sid in all_ids if find_angio_file(sid) is None]
    if missing:
        logger.error(
            "Missing angio .nii file for %d subjects: %s",
            len(missing), ", ".join(missing),
        )
        ok = False
    else:
        logger.info("All %d subjects have a matching angio .nii file.", len(all_ids))

    try:
        r = requests.get(f"{BASE_URL}/", timeout=10)
        if r.status_code != 200:
            logger.error("Backend root %s returned %d", BASE_URL, r.status_code)
            ok = False
        else:
            logger.info("Backend reachable at %s (status 200).", BASE_URL)
    except requests.RequestException as e:
        logger.error("Cannot reach backend at %s: %s", BASE_URL, e)
        ok = False

    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    logger = setup_logging()
    logger.info("=" * 78)
    logger.info("Batch evaluation started at %s", datetime.now().isoformat())
    logger.info("=" * 78)

    if not CLICK_POINTS_PATH.exists():
        logger.error("click_points.json not found at %s", CLICK_POINTS_PATH)
        return 1
    click_points = load_click_points()

    if not sanity_check(click_points, logger):
        logger.error("Sanity check failed. Aborting.")
        return 1

    existing = load_existing_results()
    already_success = {
        sid for sid, row in existing.items()
        if row.get("pipeline_status") == "success"
    }
    if already_success:
        logger.info(
            "Resuming: %d subjects already marked success will be skipped.",
            len(already_success),
        )

    patient_ids = sorted(click_points["patients"].keys())
    control_ids = sorted(click_points["controls"].keys())
    work_order = [(sid, "patient") for sid in patient_ids] + [
        (sid, "control") for sid in control_ids
    ]
    total = len(work_order)

    logger.info("Plan: %d subjects (%d patients + %d controls). Crop radius = %.1f mm.",
                total, len(patient_ids), len(control_ids), CROP_RADIUS_MM)

    # Confirm with user before running.
    try:
        answer = input("Proceed with batch run? [y/N] ").strip().lower()
    except EOFError:
        answer = ""
    if answer != "y":
        logger.info("User declined to proceed. Exiting.")
        return 0

    rows_by_id: dict[str, dict[str, Any]] = dict(existing)
    failures_by_stage: dict[str, int] = {}
    successful_runtimes_ms: list[float] = []

    for idx, (subject_id, group) in enumerate(work_order, start=1):
        prefix = f"[{idx:>2}/{total}] {subject_id} ({group})"

        if subject_id in already_success:
            print(f"{prefix} ... skipped (already success)")
            logger.info("%s skipped — already in CSV with status=success", subject_id)
            continue

        if group == "patient":
            entry = click_points["patients"][subject_id]
            click = entry["click_point"]
            lesion_voxels = entry.get("lesion_voxel_count", "")
            lesion_extent_max = (
                max(entry["lesion_extent_mm"])
                if isinstance(entry.get("lesion_extent_mm"), list)
                else ""
            )
            if isinstance(lesion_extent_max, float):
                lesion_extent_max = f"{lesion_extent_max:.4f}"
        else:
            entry = click_points["controls"][subject_id]
            click = entry["click_point"]
            lesion_voxels = ""
            lesion_extent_max = ""

        nii_path = find_angio_file(subject_id)
        if nii_path is None:
            logger.error("%s ... FAILED: no angio .nii file found", subject_id)
            print(f"{prefix} ... FAILED at upload step: no angio file")
            row = {
                "subject_id": subject_id,
                "group": group,
                "click_point_x": f"{click[0]:.6f}",
                "click_point_y": f"{click[1]:.6f}",
                "click_point_z": f"{click[2]:.6f}",
                "lesion_voxel_count": lesion_voxels,
                "lesion_extent_mm_max": lesion_extent_max,
                "pipeline_status": "upload_failed",
                "error_message": "no angio .nii file found for subject",
            }
            rows_by_id[subject_id] = row
            failures_by_stage["upload"] = failures_by_stage.get("upload", 0) + 1
            write_results_csv([rows_by_id[k] for k in sorted(rows_by_id)])
            continue

        row = process_subject(
            subject_id=subject_id,
            group=group,
            stored_click=click,
            nii_path=nii_path,
            lesion_voxel_count=lesion_voxels,
            lesion_extent_mm_max=lesion_extent_max,
            logger=logger,
        )
        rows_by_id[subject_id] = row

        # Persist progress after every subject so a crash loses no work.
        write_results_csv([rows_by_id[k] for k in sorted(rows_by_id)])

        status = row["pipeline_status"]
        total_ms = float(row["total_ms"]) if row["total_ms"] else 0.0
        if status == "success":
            successful_runtimes_ms.append(total_ms)
            risk = row["risk_score"] or "n/a"
            print(f"{prefix} ... success in {total_ms / 1000:.1f}s (risk={risk})")
            logger.info("%s success in %.1fs (risk=%s)", subject_id, total_ms / 1000, risk)
        else:
            stage = status.replace("_failed", "") if "_failed" in status else status
            failures_by_stage[stage] = failures_by_stage.get(stage, 0) + 1
            print(
                f"{prefix} ... FAILED at {stage} step: {row['error_message'][:200]}"
            )
            logger.error("%s FAILED at %s: %s",
                         subject_id, stage, row["error_message"])

    # ---- Final summary ----
    final_rows = [rows_by_id[k] for k in sorted(rows_by_id)]
    n_done = len(final_rows)
    n_success = sum(1 for r in final_rows if r.get("pipeline_status") == "success")
    n_fail = n_done - n_success

    print("\n" + "=" * 78)
    print("BATCH SUMMARY")
    print("=" * 78)
    print(f"Total subjects in CSV : {n_done}")
    print(f"Successful runs       : {n_success}")
    print(f"Failures              : {n_fail}")
    if successful_runtimes_ms:
        mean_s = sum(successful_runtimes_ms) / len(successful_runtimes_ms) / 1000.0
        print(f"Mean runtime (success): {mean_s:.1f}s per subject")
    if failures_by_stage:
        print("Failure breakdown by stage:")
        for stage, count in sorted(failures_by_stage.items()):
            print(f"  {stage:<10s} {count}")
    print(f"\nResults CSV: {CSV_PATH}")
    print(f"Raw responses: {RAW_RESPONSES_DIR}")
    print(f"Log: {LOG_PATH}")

    logger.info(
        "Batch finished: %d successes, %d failures across %d subjects.",
        n_success, n_fail, n_done,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
