"""
Generate click_points.json for the 50-subject AneuXplain prototype evaluation.

Reads lesion mask NIfTI files for 25 patients, computes per-patient click points
in the mesh coordinate space (voxel_index * voxel_spacing, mm, origin at 0,0,0),
then assigns the mean patient centroid as the shared click point for 25 controls.

One-shot, idempotent data-prep script. Does not call the backend.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    sys.stderr.write(
        "nibabel not installed. Run: pip install nibabel --break-system-packages\n"
    )
    sys.exit(1)


LESION_DIR = Path(r"C:\Users\buzok\Desktop\Test-demo\lesion_masks")
OUTPUT_PATH = Path(r"C:\Users\buzok\Desktop\Test-demo\click_points.json")

PATIENT_IDS = [
    "sub-013", "sub-022", "sub-026", "sub-034", "sub-043", "sub-051", "sub-052",
    "sub-062", "sub-066", "sub-074", "sub-075", "sub-076", "sub-081", "sub-099",
    "sub-101", "sub-103", "sub-104", "sub-106", "sub-120", "sub-121", "sub-125",
    "sub-126", "sub-127", "sub-129", "sub-170",
]

CONTROL_IDS = [
    "sub-000", "sub-002", "sub-005", "sub-007", "sub-015", "sub-021", "sub-030",
    "sub-033", "sub-036", "sub-039", "sub-045", "sub-047", "sub-053", "sub-065",
    "sub-069", "sub-073", "sub-078", "sub-079", "sub-092", "sub-105", "sub-107",
    "sub-109", "sub-112", "sub-116", "sub-171",
]


def find_lesion_files(subject_id: str) -> list[Path]:
    """Return all Lesion_*_mask.nii files for a subject, sorted by lesion index."""
    matches = sorted(LESION_DIR.glob(f"{subject_id}_ses-*_desc-Lesion_*_mask.nii"))
    return matches


def analyze_mask(mask_path: Path) -> dict:
    """Load a NIfTI mask, return centroid/voxel-count/extent/spacing/click_point."""
    img = nib.load(str(mask_path))
    data = img.get_fdata()
    spacing = np.asarray(img.header.get_zooms()[:3], dtype=float)

    unique_vals = np.unique(data)
    if not set(unique_vals.tolist()).issubset({0.0, 1.0}):
        print(
            f"  WARNING: {mask_path.name} is not strictly binary "
            f"(unique values: {unique_vals.tolist()}). Using (data > 0)."
        )

    binary = data > 0
    voxel_count = int(binary.sum())
    if voxel_count == 0:
        raise ValueError(f"Mask {mask_path.name} contains zero non-zero voxels.")

    indices = np.argwhere(binary)
    centroid_voxel = indices.mean(axis=0)
    bbox_voxels = indices.max(axis=0) - indices.min(axis=0) + 1

    click_point = centroid_voxel * spacing
    extent_mm = bbox_voxels * spacing

    return {
        "click_point": [float(v) for v in click_point],
        "lesion_voxel_count": voxel_count,
        "lesion_extent_mm": [round(float(v), 4) for v in extent_mm],
        "voxel_spacing_mm": [float(v) for v in spacing],
    }


def process_patient(subject_id: str) -> dict:
    """Process a patient — handle multi-lesion case by selecting larger mask."""
    files = find_lesion_files(subject_id)
    if not files:
        raise FileNotFoundError(
            f"No lesion mask found for {subject_id} in {LESION_DIR}"
        )

    if len(files) == 1:
        result = analyze_mask(files[0])
        result["lesion_file"] = files[0].name
        result["notes"] = None
        return result

    # Multi-lesion case: analyze each, pick the largest by voxel count.
    # Tiebreak (equal counts) is stable — lower Lesion_N index wins because
    # `files` is sorted lexicographically by Path.glob's sort.
    analyses = [(f, analyze_mask(f)) for f in files]
    analyses.sort(key=lambda pair: pair[1]["lesion_voxel_count"], reverse=True)
    chosen_path, chosen = analyses[0]

    def lesion_label(path: Path) -> str:
        m = re.search(r"Lesion_(\d+)", path.name)
        return f"Lesion_{m.group(1)}" if m else path.stem

    chosen_label = lesion_label(chosen_path)
    others = [
        f"{lesion_label(f)} ({a['lesion_voxel_count']} voxels)"
        for f, a in analyses[1:]
    ]

    note = (
        f"multi-lesion patient; chose {chosen_label} "
        f"({chosen['lesion_voxel_count']} voxels) over "
        + ", ".join(others)
    )

    chosen["lesion_file"] = chosen_path.name
    chosen["notes"] = note
    return chosen


def main() -> int:
    if not LESION_DIR.is_dir():
        print(f"ERROR: lesion mask directory not found: {LESION_DIR}", file=sys.stderr)
        return 1

    # Verify every patient has at least one lesion mask before processing.
    missing = [sid for sid in PATIENT_IDS if not find_lesion_files(sid)]
    if missing:
        print(
            "ERROR: missing lesion masks for the following patients: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 1

    print(f"Processing {len(PATIENT_IDS)} patients from {LESION_DIR} ...\n")

    patients_out: dict[str, dict] = {}
    patient_click_points: list[list[float]] = []

    for sid in PATIENT_IDS:
        info = process_patient(sid)
        patients_out[sid] = {
            "click_point": [round(v, 6) for v in info["click_point"]],
            "lesion_file": info["lesion_file"],
            "lesion_voxel_count": info["lesion_voxel_count"],
            "lesion_extent_mm": info["lesion_extent_mm"],
            "voxel_spacing_mm": [round(v, 6) for v in info["voxel_spacing_mm"]],
            "notes": info["notes"],
        }
        patient_click_points.append(info["click_point"])
        if info["notes"]:
            print(f"  {sid}: {info['notes']}")

    pts = np.asarray(patient_click_points, dtype=float)
    mean_click = pts.mean(axis=0)
    shared_control_click = [round(float(v), 6) for v in mean_click]

    controls_out: dict[str, dict] = {
        sid: {
            "click_point": shared_control_click,
            "notes": "shared mean-patient-centroid",
        }
        for sid in CONTROL_IDS
    }

    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "coordinate_system": "voxel_index * voxel_spacing (mm), origin at (0,0,0)",
            "control_strategy": "mean of all 25 patient click points",
            "multi_lesion_policy": "select largest lesion by voxel count",
            "n_patients": len(PATIENT_IDS),
            "n_controls": len(CONTROL_IDS),
        },
        "patients": patients_out,
        "controls": controls_out,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # ---- Summary ----
    print("\n" + "=" * 78)
    print("PATIENT CLICK POINTS")
    print("=" * 78)
    print(f"{'subject_id':<12} {'click_point (mm)':<32} {'voxels':>8} {'max_extent_mm':>14}")
    print("-" * 78)
    for sid in PATIENT_IDS:
        p = patients_out[sid]
        cp = p["click_point"]
        cp_str = f"({cp[0]:.3f}, {cp[1]:.3f}, {cp[2]:.3f})"
        max_ext = max(p["lesion_extent_mm"])
        print(
            f"{sid:<12} {cp_str:<32} {p['lesion_voxel_count']:>8} {max_ext:>14.3f}"
        )

    print("\n" + "=" * 78)
    print("CONTROLS (shared click point)")
    print("=" * 78)
    print(
        f"shared click_point = "
        f"({shared_control_click[0]:.3f}, "
        f"{shared_control_click[1]:.3f}, "
        f"{shared_control_click[2]:.3f}) mm"
    )
    print(f"applied to {len(CONTROL_IDS)} controls:")
    for i in range(0, len(CONTROL_IDS), 5):
        print("  " + "  ".join(CONTROL_IDS[i : i + 5]))

    print("\n" + "=" * 78)
    print("SANITY CHECK")
    print("=" * 78)
    x_min, y_min, z_min = pts.min(axis=0)
    x_max, y_max, z_max = pts.max(axis=0)
    print(
        f"Patient click points ranged from {x_min:.3f} to {x_max:.3f} mm on X axis "
        f"(Y: {y_min:.3f} to {y_max:.3f} mm, Z: {z_min:.3f} to {z_max:.3f} mm). "
        f"Mean patient centroid (= control click): "
        f"({mean_click[0]:.3f}, {mean_click[1]:.3f}, {mean_click[2]:.3f})."
    )

    print(f"\nWrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
