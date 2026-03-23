"""
End-to-end validation of the NeuroTwin pipeline.

Tests:
  1. Risk predictor separates aneurysm vs vessel
  2. CVAE healing reduces risk on a sick artery
  3. CVAE reverse test: conditioning a healthy artery on high risk

Run from project root:  python test_pipeline.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import trimesh

from training.models import RiskPredictorV2, ConditionalVAE

# -- Config --------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
NUM_POINTS = 2048
LATENT_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ANEURYSM_FILE = PROJECT_ROOT / "IntrA" / "generated" / "aneurysm" / "obj" / "ArteryObjAN117-2.obj"
VESSEL_FILE   = PROJECT_ROOT / "IntrA" / "generated" / "vessel"   / "obj" / "ArteryObjAN1-0.obj"


def load_mesh_as_tensor(path: str) -> torch.Tensor:
    """Load mesh -> sample 2048 points -> normalize -> tensor (1, 3, N)."""
    mesh = trimesh.load(str(path), force="mesh")
    points = np.array(trimesh.sample.sample_surface(mesh, count=NUM_POINTS)[0], dtype=np.float32)
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist
    tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0).transpose(2, 1)  # (1,3,N)
    return tensor.to(DEVICE)


def predict_risk(model, tensor: torch.Tensor) -> float:
    """Run risk predictor and return probability."""
    with torch.no_grad():
        logits = model(tensor, return_logits=True)
        return torch.sigmoid(logits).item()


def main():
    print("=" * 60)
    print("NEUROTWIN END-TO-END VALIDATION")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # -- Load models -----------------------------------------
    print("\nLoading models...")

    risk_model = RiskPredictorV2(latent_dim=LATENT_DIM).to(DEVICE)
    risk_model.load_state_dict(
        torch.load(str(MODELS_DIR / "risk_predictor_v2.pth"), map_location=DEVICE, weights_only=True)
    )
    risk_model.eval()
    print("  Risk predictor loaded")

    cvae = ConditionalVAE(latent_dim=LATENT_DIM, num_points=NUM_POINTS).to(DEVICE)
    cvae.load_state_dict(
        torch.load(str(MODELS_DIR / "cvae.pth"), map_location=DEVICE, weights_only=True)
    )
    cvae.eval()
    print("  CVAE loaded")

    # -- Test 1: Risk predictor separation -------------------
    print(f"\n{'-'*60}")
    print("TEST 1: Risk Predictor Separation")
    print(f"{'-'*60}")

    aneurysm_tensor = load_mesh_as_tensor(ANEURYSM_FILE)
    vessel_tensor = load_mesh_as_tensor(VESSEL_FILE)

    aneurysm_risk = predict_risk(risk_model, aneurysm_tensor)
    vessel_risk = predict_risk(risk_model, vessel_tensor)

    print(f"  Aneurysm ({ANEURYSM_FILE.name}): {aneurysm_risk:.4f}  {'PASS' if aneurysm_risk > 0.6 else 'FAIL'} (target: >0.6)")
    print(f"  Vessel   ({VESSEL_FILE.name}):   {vessel_risk:.4f}  {'PASS' if vessel_risk < 0.4 else 'FAIL'} (target: <0.4)")
    print(f"  Gap: {aneurysm_risk - vessel_risk:.4f}")

    # -- Test 2: CVAE healing (aneurysm → healthy) ----------
    print(f"\n{'-'*60}")
    print("TEST 2: CVAE Healing (aneurysm -> target_risk=0.05)")
    print(f"{'-'*60}")

    with torch.no_grad():
        healed_tensor = cvae.generate_healthy(aneurysm_tensor, target_risk=0.05)
        healed_risk = predict_risk(risk_model, healed_tensor)

    reduction = aneurysm_risk - healed_risk
    reduction_pct = 100 * reduction / aneurysm_risk if aneurysm_risk > 0 else 0
    print(f"  Before healing: {aneurysm_risk:.4f}")
    print(f"  After healing:  {healed_risk:.4f}")
    print(f"  Risk reduction: {reduction:.4f} ({reduction_pct:.1f}%)")
    print(f"  {'PASS' if healed_risk < aneurysm_risk else 'FAIL'} — healed risk should be lower than original")

    # -- Test 3: Reverse test (vessel → sick) ----------------
    print(f"\n{'-'*60}")
    print("TEST 3: Reverse Test (vessel -> target_risk=0.9)")
    print(f"{'-'*60}")

    with torch.no_grad():
        mu, _ = cvae.encode(vessel_tensor)
        target = torch.full((1, 1), 0.9, device=DEVICE)
        z_cond = torch.cat([mu, target], dim=1)
        sick_tensor = cvae.decoder(z_cond)
        sick_risk = predict_risk(risk_model, sick_tensor)

    increase = sick_risk - vessel_risk
    print(f"  Before (healthy): {vessel_risk:.4f}")
    print(f"  After (sick):     {sick_risk:.4f}")
    print(f"  Risk increase:    {increase:.4f}")
    print(f"  {'PASS' if sick_risk > vessel_risk else 'FAIL'} — sick version should score higher")

    # -- Summary table ---------------------------------------
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Test':<40} {'Score':>8} {'Result':>8}")
    print(f"{'-'*56}")
    print(f"{'Aneurysm risk score':<40} {aneurysm_risk:>8.4f} {'PASS' if aneurysm_risk > 0.6 else 'FAIL':>8}")
    print(f"{'Vessel risk score':<40} {vessel_risk:>8.4f} {'PASS' if vessel_risk < 0.4 else 'FAIL':>8}")
    print(f"{'Separation gap':<40} {aneurysm_risk - vessel_risk:>8.4f} {'PASS' if aneurysm_risk - vessel_risk > 0.3 else 'FAIL':>8}")
    print(f"{'Healed aneurysm risk':<40} {healed_risk:>8.4f} {'PASS' if healed_risk < aneurysm_risk else 'FAIL':>8}")
    print(f"{'Risk reduction from healing':<40} {reduction:>8.4f} {'PASS' if reduction > 0.1 else 'FAIL':>8}")
    print(f"{'Sickened vessel risk':<40} {sick_risk:>8.4f} {'PASS' if sick_risk > vessel_risk else 'FAIL':>8}")
    print(f"{'Risk increase from sickening':<40} {increase:>8.4f} {'PASS' if increase > 0.05 else 'FAIL':>8}")
    print(f"{'-'*56}")

    total = 7
    passed = sum([
        aneurysm_risk > 0.6,
        vessel_risk < 0.4,
        aneurysm_risk - vessel_risk > 0.3,
        healed_risk < aneurysm_risk,
        reduction > 0.1,
        sick_risk > vessel_risk,
        increase > 0.05,
    ])
    print(f"\n  {passed}/{total} tests passed")


if __name__ == "__main__":
    main()
