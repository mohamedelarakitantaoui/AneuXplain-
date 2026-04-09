"""
calibrate_heatmap_reference.py

Compute a global reference value for heatmap normalization by running the
gradient pipeline over the IntrA dataset.  Gradients are computed w.r.t.
the raw LOGIT (not the sigmoid probability) so that sensitivity is not
suppressed by sigmoid saturation at high/low confidence levels.

The output is a single float (HEATMAP_GLOBAL_REF) that should be pasted
into backend/app/main.py.

Usage:
    python -m scripts.calibrate_heatmap_reference          # from project root
    python scripts/calibrate_heatmap_reference.py           # also works
"""

import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import trimesh
from backend.app.engine import CounterfactualEngine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INTRA_DIRS = [
    PROJECT_ROOT / "IntrA" / "complete",           # full arteries
    PROJECT_ROOT / "IntrA" / "annotated" / "obj",  # annotated full meshes
]
MODELS_DIR = PROJECT_ROOT / "models"
NUM_POINTS = 2048
# Which percentile of the *high-risk* per-vertex gradient magnitudes to use
GLOBAL_PERCENTILE = 95


def gather_mesh_paths(max_meshes: int = 200) -> list[Path]:
    """Collect .obj mesh paths from IntrA directories."""
    paths: list[Path] = []
    for d in INTRA_DIRS:
        if d.exists():
            paths.extend(sorted(d.glob("*.obj")))
    # Deduplicate by stem (annotated and complete may overlap)
    seen = set()
    unique = []
    for p in paths:
        if p.stem not in seen:
            seen.add(p.stem)
            unique.append(p)
    return unique[:max_meshes]


def compute_gradient_magnitudes(engine: CounterfactualEngine, mesh_path: Path):
    """
    Return (risk_score, raw_grad_magnitudes_array) for one mesh.
    Gradients are computed w.r.t. the raw logit to avoid sigmoid saturation.
    """
    points = engine._load_mesh_as_points(str(mesh_path))
    tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
    tensor = tensor.transpose(2, 1).to(engine.device)
    tensor.requires_grad_(True)

    # Forward pass — get LOGIT for gradient, probability for reporting
    if engine.is_v2_model:
        logit = engine.risk_predictor(tensor, return_logits=True)
        risk_score = torch.sigmoid(logit).item()
    else:
        # V1 model doesn't separate logit/prob — use raw output
        logit = engine.risk_predictor(tensor)
        risk_score = logit.item()

    # Backward from the logit (NOT the probability)
    logit.backward()

    grad = tensor.grad.squeeze(0)            # (3, N)
    grad_mag = torch.norm(grad, dim=0)       # (N,)
    return risk_score, grad_mag.detach().cpu().numpy()


def main():
    # Load engine
    engine = CounterfactualEngine(
        models_dir=str(MODELS_DIR),
        output_dir=str(PROJECT_ROOT / "backend" / "outputs"),
    )

    risk_predictor_path = MODELS_DIR / "risk_predictor_v2.pth"
    if not risk_predictor_path.exists():
        for alt in ["risk_predictor.pth", "risk_predictor_best_gap.pth"]:
            alt_path = MODELS_DIR / alt
            if alt_path.exists():
                risk_predictor_path = alt_path
                break

    engine.load_models(risk_predictor_path=str(risk_predictor_path))
    print(f"Model loaded on {engine.device}")
    print(f"NOTE: Gradients computed w.r.t. LOGIT (not sigmoid probability)\n")

    mesh_paths = gather_mesh_paths()
    print(f"Found {len(mesh_paths)} meshes to process\n")

    all_grad_mags = []       # every per-vertex magnitude across ALL meshes
    high_risk_mags = []      # per-vertex magnitudes from HIGH-risk meshes only
    results = []             # (name, risk, min, max, mean, p95) per mesh

    for i, mp in enumerate(mesh_paths):
        try:
            risk, mags = compute_gradient_magnitudes(engine, mp)
        except Exception as e:
            print(f"  [{i+1}/{len(mesh_paths)}] SKIP {mp.name}: {e}")
            continue

        mn, mx, mean, p95 = mags.min(), mags.max(), mags.mean(), np.percentile(mags, 95)
        results.append((mp.name, risk, mn, mx, mean, p95))
        all_grad_mags.append(mags)

        if risk >= 0.5:
            high_risk_mags.append(mags)

        tag = "HIGH" if risk >= 0.5 else "low "
        print(f"  [{i+1:3d}/{len(mesh_paths)}] {tag} risk={risk:.3f}  "
              f"grad min={mn:.6f} max={mx:.6f} mean={mean:.6f} p95={p95:.6f}  {mp.name}")

    if not all_grad_mags:
        print("\nNo meshes processed. Check paths.")
        return

    all_concat = np.concatenate(all_grad_mags)

    print(f"\n{'='*70}")
    print("GLOBAL STATISTICS (all meshes, logit-space gradients)")
    print(f"  Total vertices processed: {len(all_concat)}")
    print(f"  Min:  {all_concat.min():.8f}")
    print(f"  Max:  {all_concat.max():.8f}")
    print(f"  Mean: {all_concat.mean():.8f}")
    print(f"  P90:  {np.percentile(all_concat, 90):.8f}")
    print(f"  P95:  {np.percentile(all_concat, 95):.8f}")
    print(f"  P99:  {np.percentile(all_concat, 99):.8f}")

    if high_risk_mags:
        hr_concat = np.concatenate(high_risk_mags)
        print(f"\nHIGH-RISK STATISTICS (risk >= 0.5, {len(high_risk_mags)} meshes)")
        print(f"  Total vertices: {len(hr_concat)}")
        print(f"  Min:  {hr_concat.min():.8f}")
        print(f"  Max:  {hr_concat.max():.8f}")
        print(f"  Mean: {hr_concat.mean():.8f}")
        print(f"  P85:  {np.percentile(hr_concat, 85):.8f}")
        print(f"  P90:  {np.percentile(hr_concat, 90):.8f}")
        print(f"  P95:  {np.percentile(hr_concat, 95):.8f}")
        print(f"  P99:  {np.percentile(hr_concat, 99):.8f}")

        ref_value = float(np.percentile(hr_concat, GLOBAL_PERCENTILE))
    else:
        print("\nNo high-risk meshes found; using all-mesh P95 as fallback.")
        ref_value = float(np.percentile(all_concat, GLOBAL_PERCENTILE))

    print(f"\n{'='*70}")
    print(f">>> HEATMAP_GLOBAL_REF = {ref_value:.8f}")
    print(f"    (P{GLOBAL_PERCENTILE} of high-risk per-vertex logit-gradient magnitudes)")
    print(f"{'='*70}")

    # --- Validation preview on specific test cases ---
    test_cases = {
        "AN26_full": PROJECT_ROOT / "IntrA" / "annotated" / "obj" / "AN26_full.obj",
        "ArteryObjAN40-10": PROJECT_ROOT / "IntrA" / "generated" / "aneurysm" / "obj" / "ArteryObjAN40-10.obj",
    }
    print("\n--- Validation Preview (using computed HEATMAP_GLOBAL_REF) ---")
    for name, path in test_cases.items():
        if not path.exists():
            print(f"  {name}: file not found at {path}")
            continue
        risk, mags = compute_gradient_magnitudes(engine, path)
        normalized = np.clip(mags / ref_value, 0, 1)
        print(f"\n  {name} (risk={risk:.4f}):")
        print(f"    Raw  grad: min={mags.min():.6f}  max={mags.max():.6f}  "
              f"mean={mags.mean():.6f}  p95={np.percentile(mags, 95):.6f}")
        print(f"    Norm heat: min={normalized.min():.4f}  max={normalized.max():.4f}  "
              f"mean={normalized.mean():.4f}  p95={np.percentile(normalized, 95):.4f}")
        pct_above_05 = (normalized > 0.5).sum() / len(normalized) * 100
        pct_above_02 = (normalized > 0.2).sum() / len(normalized) * 100
        print(f"    Vertices > 0.5 (warm+hot): {pct_above_05:.1f}%")
        print(f"    Vertices > 0.2 (any color): {pct_above_02:.1f}%")


if __name__ == "__main__":
    main()
