"""
Patient-level train/val/test split utilities for IntrA dataset.

Prevents patient-level leakage: every fragment from a given patient
(AN<id>) is assigned to exactly one of train/val/test.

Usage:
    python -m training.scripts.patient_split --verify
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LABELS_CSV = PROJECT_ROOT / "data" / "combined_labels.csv"
DEFAULT_SPLIT_FILE = PROJECT_ROOT / "data" / "patient_splits.json"


_PATIENT_RE = re.compile(r"^(?:ArteryObj)?(AN\d+)")


def extract_patient_id(filename: str) -> str:
    """
    Extract patient ID from an IntrA filename. Handles:
      ArteryObjAN1-0.obj   -> AN1
      ArteryObjAN11.obj    -> AN11
      ArteryObjAN102-5.obj -> AN102
    Raises ValueError if no AN<number> pattern is found.
    """
    basename = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    m = _PATIENT_RE.match(basename)
    if not m:
        raise ValueError(f"Cannot extract patient ID from: {filename}")
    return m.group(1)


def _self_test_extract_patient_id() -> None:
    """Internal smoke-test for extract_patient_id. Aborts on failure."""
    cases = [
        ("ArteryObjAN1-0.obj", "AN1"),
        ("ArteryObjAN11.obj", "AN11"),
        ("ArteryObjAN102-5.obj", "AN102"),
        ("path/to/ArteryObjAN42-12.obj", "AN42"),
        ("C:\\some\\windows\\path\\ArteryObjAN7-3.obj", "AN7"),
    ]
    failures = []
    for inp, expected in cases:
        try:
            got = extract_patient_id(inp)
        except Exception as e:  # noqa: BLE001
            failures.append(f"  {inp!r} -> EXCEPTION {e!r} (expected {expected!r})")
            continue
        if got != expected:
            failures.append(f"  {inp!r} -> {got!r} (expected {expected!r})")
    if failures:
        raise RuntimeError(
            "extract_patient_id self-test FAILED:\n" + "\n".join(failures)
        )


# Run self-test at import time so any caller is protected.
_self_test_extract_patient_id()


def make_patient_level_splits(
    labels_df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict:
    """
    Build patient-level train/val/test splits.

    Groups rows by extracted patient_id, shuffles unique patients, and
    assigns whole patients to splits according to the given fractions.

    Returns a dict with: seed, train/val/test_patients, train/val/test_indices,
    patient_assignments.
    """
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(
            f"Fractions must sum to 1.0 (got {train_frac + val_frac + test_frac})"
        )

    if "filename" not in labels_df.columns:
        raise ValueError("labels_df must contain a 'filename' column")

    # Stable row index -> patient id
    row_patient_ids: List[str] = [
        extract_patient_id(fn) for fn in labels_df["filename"].tolist()
    ]

    unique_patients = sorted(set(row_patient_ids))
    n_patients = len(unique_patients)

    if not (90 <= n_patients <= 120):
        preview = unique_patients[:10]
        raise RuntimeError(
            f"Unexpected unique patient count: {n_patients} "
            f"(expected in [90, 120]). First IDs: {preview}"
        )

    rng = np.random.default_rng(seed)
    shuffled = list(unique_patients)
    rng.shuffle(shuffled)

    n_train = int(round(n_patients * train_frac))
    n_val = int(round(n_patients * val_frac))
    # Whatever is left goes to test (handles rounding).
    n_test = n_patients - n_train - n_val
    if n_test <= 0:
        raise RuntimeError(
            f"Test split is empty after rounding "
            f"(train={n_train}, val={n_val}, total={n_patients})"
        )

    train_patients = sorted(shuffled[:n_train])
    val_patients = sorted(shuffled[n_train:n_train + n_val])
    test_patients = sorted(shuffled[n_train + n_val:])

    # Mutual exclusivity check
    sets = {
        "train": set(train_patients),
        "val": set(val_patients),
        "test": set(test_patients),
    }
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a < b:
                overlap = sets[a] & sets[b]
                assert not overlap, (
                    f"Patient leakage between {a} and {b}: {sorted(overlap)}"
                )

    patient_assignments: Dict[str, str] = {}
    for p in train_patients:
        patient_assignments[p] = "train"
    for p in val_patients:
        patient_assignments[p] = "val"
    for p in test_patients:
        patient_assignments[p] = "test"

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    for i, pid in enumerate(row_patient_ids):
        bucket = patient_assignments[pid]
        if bucket == "train":
            train_indices.append(i)
        elif bucket == "val":
            val_indices.append(i)
        else:
            test_indices.append(i)

    return {
        "seed": seed,
        "train_patients": train_patients,
        "val_patients": val_patients,
        "test_patients": test_patients,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "patient_assignments": patient_assignments,
    }


def save_splits(splits: dict, path: Path = DEFAULT_SPLIT_FILE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "seed": int(splits["seed"]),
        "train_patients": list(splits["train_patients"]),
        "val_patients": list(splits["val_patients"]),
        "test_patients": list(splits["test_patients"]),
        "train_indices": [int(i) for i in splits["train_indices"]],
        "val_indices": [int(i) for i in splits["val_indices"]],
        "test_indices": [int(i) for i in splits["test_indices"]],
        "patient_assignments": dict(splits["patient_assignments"]),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    return path


def load_splits(path: Path = DEFAULT_SPLIT_FILE) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_or_create_patient_splits(
    labels_df: pd.DataFrame,
    path: Path = DEFAULT_SPLIT_FILE,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict:
    """
    Load splits from `path` if it exists; otherwise compute and persist.

    Once saved, the split is locked. Delete the file manually to regenerate.
    """
    path = Path(path)
    if path.exists():
        splits = load_splits(path)
        # Re-validate row count alignment vs current labels_df
        n_rows = len(labels_df)
        total_indices = (
            len(splits["train_indices"])
            + len(splits["val_indices"])
            + len(splits["test_indices"])
        )
        if total_indices != n_rows:
            raise RuntimeError(
                f"Saved split index count ({total_indices}) does not match "
                f"current labels_df row count ({n_rows}). "
                f"Delete {path} to regenerate."
            )
        return splits

    splits = make_patient_level_splits(
        labels_df,
        seed=seed,
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
    )
    save_splits(splits, path)
    return splits


def _verify(labels_csv: Path = DEFAULT_LABELS_CSV) -> None:
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    labels_df = pd.read_csv(labels_csv)
    splits = load_or_create_patient_splits(labels_df)

    # Determine which score column to use
    score_col = "risk_score" if "risk_score" in labels_df.columns else "label"
    if score_col not in labels_df.columns:
        raise RuntimeError(
            f"Neither 'risk_score' nor 'label' column found in {labels_csv}"
        )

    binary_labels = (labels_df[score_col] >= 0.5).astype(int).tolist()

    def _split_label_counts(indices):
        healthy = sum(1 for i in indices if binary_labels[i] == 0)
        diseased = sum(1 for i in indices if binary_labels[i] == 1)
        return healthy, diseased

    n_total_patients = (
        len(splits["train_patients"])
        + len(splits["val_patients"])
        + len(splits["test_patients"])
    )

    # Mutual exclusivity check
    s_train = set(splits["train_patients"])
    s_val = set(splits["val_patients"])
    s_test = set(splits["test_patients"])
    no_leak = (
        not (s_train & s_val)
        and not (s_train & s_test)
        and not (s_val & s_test)
    )

    h_train, d_train = _split_label_counts(splits["train_indices"])
    h_val, d_val = _split_label_counts(splits["val_indices"])
    h_test, d_test = _split_label_counts(splits["test_indices"])

    print("=" * 60)
    print("PATIENT-LEVEL SPLIT VERIFICATION")
    print("=" * 60)
    print(f"Labels CSV : {labels_csv}")
    print(f"Split file : {DEFAULT_SPLIT_FILE}")
    print(f"Seed       : {splits['seed']}")
    print()
    print(f"Total unique patients: {n_total_patients}")
    print(
        f"Train patients: {len(splits['train_patients']):3d} / "
        f"samples: {len(splits['train_indices'])}"
    )
    print(
        f"Val   patients: {len(splits['val_patients']):3d} / "
        f"samples: {len(splits['val_indices'])}"
    )
    print(
        f"Test  patients: {len(splits['test_patients']):3d} / "
        f"samples: {len(splits['test_indices'])}"
    )
    print(f"No patient appears in more than one split: {no_leak}")
    print()
    print("Healthy / Diseased counts per split (threshold=0.5 on "
          f"'{score_col}'):")
    print(f"  Train: healthy={h_train:4d}  diseased={d_train:4d}")
    print(f"  Val  : healthy={h_val:4d}  diseased={d_val:4d}")
    print(f"  Test : healthy={h_test:4d}  diseased={d_test:4d}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Patient-level split utility")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify (and create on first run) the patient-level split.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=DEFAULT_LABELS_CSV,
        help="Path to combined_labels.csv",
    )
    args = parser.parse_args()

    if args.verify:
        _verify(args.labels_csv)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
