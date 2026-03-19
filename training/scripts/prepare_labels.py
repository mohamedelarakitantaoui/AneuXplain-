"""
Prepare Labels - Generate Geometry-Based Risk Labels

Computes real risk scores from mesh geometry features:
- Discrete Laplacian curvature (surface irregularity)
- Radius skewness (asymmetric bulging / aneurysm protrusion)
- Local bulge detection (localized outward displacement)
- Surface roughness (face normal variance)

Usage:
    python -m training.scripts.prepare_labels
"""

import sys
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import trimesh
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def compute_geometry_features(mesh) -> dict:
    """
    Extract geometry features that correlate with aneurysm risk.

    Returns a dict of raw features (not yet mapped to a score).
    """
    verts = np.array(mesh.vertices, dtype=np.float64)
    faces = np.array(mesh.faces)

    # Normalize to unit scale
    centroid = np.mean(verts, axis=0)
    verts_c = verts - centroid
    scale = np.max(np.linalg.norm(verts_c, axis=1))
    if scale < 1e-8:
        return {}
    verts_n = verts_c / scale

    # --- 1. Discrete Laplacian curvature ---
    adjacency = defaultdict(set)
    for f in faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adjacency[f[i]].add(f[j])

    laplacian = np.zeros(len(verts))
    for i in range(len(verts)):
        nb = list(adjacency[i])
        if nb:
            laplacian[i] = np.linalg.norm(verts_n[i] - np.mean(verts_n[nb], axis=0))

    curvature_mean = float(np.mean(laplacian))
    curvature_std = float(np.std(laplacian))
    curvature_p95 = float(np.percentile(laplacian, 95))

    # --- 2. Radial profile ---
    radii = np.linalg.norm(verts_n, axis=1)
    radius_mean = float(np.mean(radii))
    radius_std = float(np.std(radii))
    radius_cv = radius_std / (radius_mean + 1e-8)
    radius_skew = float(np.mean(
        ((radii - radius_mean) / (radius_std + 1e-8)) ** 3
    ))

    # --- 3. Local bulge detection (KNN) ---
    tree = cKDTree(verts_n)
    _, idxs = tree.query(verts_n, k=11)  # 10 neighbors + self

    bulge = np.zeros(len(verts))
    for i in range(len(verts)):
        local_r = np.mean(radii[idxs[i, 1:]])
        bulge[i] = max(0.0, radii[i] - local_r)

    bulge_mean = float(np.mean(bulge))
    bulge_p95 = float(np.percentile(bulge, 95))
    bulge_max = float(np.max(bulge))

    # Fraction of points with significant outward bulging
    bulge_threshold = 2.0 * bulge_mean if bulge_mean > 0 else 1e-8
    bulge_fraction = float(np.mean(bulge > bulge_threshold))

    # --- 4. Surface roughness (face normal variance) ---
    face_normals = np.array(mesh.face_normals)
    normal_variance = 0.0
    if len(face_normals) > 1:
        vertex_normals = defaultdict(list)
        for fi, f in enumerate(faces):
            for vi in f:
                vertex_normals[vi].append(face_normals[fi])

        angle_devs = []
        for vi in range(len(verts)):
            norms = np.array(vertex_normals.get(vi, [[0, 0, 1]]))
            if len(norms) > 1:
                mean_n = np.mean(norms, axis=0)
                norm_len = np.linalg.norm(mean_n)
                if norm_len > 1e-8:
                    mean_n = mean_n / norm_len
                dots = np.clip(np.dot(norms, mean_n), -1, 1)
                angle_devs.append(float(np.mean(np.arccos(dots))))
        normal_variance = float(np.mean(angle_devs)) if angle_devs else 0.0

    # --- 5. Radius kurtosis (heavy-tailed = outlier bulge) ---
    radius_kurtosis = float(np.mean(
        ((radii - radius_mean) / (radius_std + 1e-8)) ** 4
    )) - 3.0  # excess kurtosis (0 = normal distribution)

    return {
        'curvature_mean': curvature_mean,
        'curvature_std': curvature_std,
        'curvature_p95': curvature_p95,
        'radius_cv': radius_cv,
        'radius_skew': radius_skew,
        'radius_kurtosis': radius_kurtosis,
        'bulge_mean': bulge_mean,
        'bulge_p95': bulge_p95,
        'bulge_max': bulge_max,
        'bulge_fraction': bulge_fraction,
        'normal_variance': normal_variance,
    }


def features_to_risk_score(features: dict) -> float:
    """
    Map raw geometry features to a risk score in [0, 1].

    Uses empirically-calibrated thresholds from IntrA dataset analysis.

    Feature ranking by discriminative power (aneurysm vs vessel):
    1. radius_skew:  aneurysm=+0.30, vessel=+0.17, complete=-0.13  (BEST)
    2. bulge_p95:    somewhat discriminative, resolution-invariant
    3. curvature:    resolution-dependent but useful for segments
    4. normal_variance: NOT discriminative (actually inverted), excluded
    """
    # Component 1: Radius asymmetry / bulge skewness (0-1)
    # Positive skewness = outward protrusion (aneurysm signature)
    skew = features['radius_skew']
    skew_score = np.clip((skew + 0.4) / 1.3, 0, 1)

    # Component 2: Curvature irregularity (0-1)
    # High std/mean ratio = curvature concentrated in specific regions
    curv_mean = features['curvature_mean']
    curv_std = features['curvature_std']
    curv_ratio = curv_std / (curv_mean + 1e-8)
    curv_score = np.clip(curv_ratio / 1.0, 0, 1)

    # Component 3: Localized bulging intensity (0-1)
    bulge_mean = features['bulge_mean']
    bulge_p95 = features['bulge_p95']
    if bulge_mean > 1e-8:
        bulge_concentration = bulge_p95 / bulge_mean
        bulge_score = np.clip(bulge_concentration / 7.0, 0, 1)
    else:
        bulge_score = 0.0

    # Component 4: Bulge extent (0-1)
    bulge_frac = features['bulge_fraction']
    extent_score = np.clip(bulge_frac / 0.35, 0, 1)

    # Component 5: Radius kurtosis (0-1)
    # Excess kurtosis > 0 = heavy-tailed radii = outlier bulge (aneurysm dome)
    # Vessels have negative kurtosis → clipped to 0 → no effect on them
    # Aneurysms with positive kurtosis get a selective boost
    kurtosis = features['radius_kurtosis']
    kurtosis_score = np.clip(kurtosis / 1.5, 0, 1)

    # Weighted combination
    # Base weights identical to v2 (sum = 1.0)
    # Kurtosis added as bonus: only fires for aneurysms (positive kurtosis)
    risk = (
        0.40 * skew_score
        + 0.20 * curv_score
        + 0.25 * bulge_score
        + 0.15 * extent_score
    ) + 0.04 * kurtosis_score

    return float(np.clip(risk, 0.0, 1.0))


def prepare_labels():
    """Generate geometry-based labels CSV from all data sources."""

    print("=" * 60)
    print("PREPARING GEOMETRY-BASED LABELS")
    print("=" * 60)

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        {
            'name': 'complete',
            'folder': PROJECT_ROOT / "IntrA" / "complete",
        },
        {
            'name': 'vessel',
            'folder': PROJECT_ROOT / "IntrA" / "generated" / "vessel" / "obj",
        },
        {
            'name': 'aneurysm',
            'folder': PROJECT_ROOT / "IntrA" / "generated" / "aneurysm" / "obj",
        },
    ]

    records = []
    errors = 0

    for source in sources:
        folder = source['folder']
        if not folder.exists():
            print(f"Skipping {source['name']}: folder not found")
            continue

        files = sorted(folder.glob("*.obj"))
        print(f"\n{source['name']}: Processing {len(files)} files...")

        for i, file_path in enumerate(files):
            if (i + 1) % 100 == 0:
                print(f"  [{i+1}/{len(files)}]")

            try:
                mesh = trimesh.load(str(file_path), force='mesh')
                features = compute_geometry_features(mesh)

                if not features:
                    errors += 1
                    continue

                score = features_to_risk_score(features)

                record = {
                    'filename': file_path.name,
                    'data_folder': str(folder),
                    'source': source['name'],
                    'curvature_score': round(score, 4),
                    # Store individual features for analysis
                    'curvature_mean': round(features['curvature_mean'], 6),
                    'curvature_std': round(features['curvature_std'], 6),
                    'radius_skew': round(features['radius_skew'], 4),
                    'radius_kurtosis': round(features['radius_kurtosis'], 4),
                    'bulge_p95': round(features['bulge_p95'], 6),
                    'normal_variance': round(features['normal_variance'], 4),
                }
                records.append(record)

            except Exception as e:
                print(f"  Error: {file_path.name}: {e}")
                errors += 1

    # Create DataFrame
    df = pd.DataFrame(records)

    # Save full version with all features
    output_path = data_dir / "combined_labels.csv"
    df.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(df)}, Errors: {errors}")
    print(f"\nRisk score distribution by source:")
    print(df.groupby('source').agg({
        'curvature_score': ['count', 'mean', 'std', 'min', 'max']
    }).to_string())

    # Class distribution
    threshold = 0.5
    low_risk = df[df['curvature_score'] < threshold]
    high_risk = df[df['curvature_score'] >= threshold]
    print(f"\nClass split (threshold={threshold}):")
    print(f"  Low risk:  {len(low_risk)} ({100*len(low_risk)/len(df):.1f}%)")
    print(f"  High risk: {len(high_risk)} ({100*len(high_risk)/len(df):.1f}%)")
    print(f"\nSaved to: {output_path}")

    # Create balanced version
    if len(low_risk) > 0 and len(high_risk) > 0:
        if len(low_risk) < len(high_risk):
            factor = max(1, len(high_risk) // len(low_risk))
            balanced_df = pd.concat([pd.concat([low_risk] * factor), high_risk])
        else:
            factor = max(1, len(low_risk) // len(high_risk))
            balanced_df = pd.concat([low_risk, pd.concat([high_risk] * factor)])

        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        balanced_path = data_dir / "balanced_labels.csv"
        balanced_df.to_csv(balanced_path, index=False)
        print(f"Balanced version: {balanced_path} ({len(balanced_df)} samples)")

    return df


if __name__ == "__main__":
    prepare_labels()
