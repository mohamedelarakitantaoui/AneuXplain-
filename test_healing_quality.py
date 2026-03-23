"""Diagnose healing quality: is the CVAE doing real work or just scaling?"""
import sys
sys.path.insert(0, "backend")

import numpy as np
import torch

print("Loading...", flush=True)
from app.engine import CounterfactualEngine

engine = CounterfactualEngine(models_dir="models", output_dir="outputs")
engine.load_models()

# Test on a few arteries
test_files = [
    "IntrA/generated/aneurysm/obj/ArteryObjAN117-2.obj",
    "IntrA/generated/aneurysm/obj/ArteryObjAN117-5.obj",
    "IntrA/generated/vessel/obj/ArteryObjAN1-0.obj",
]

for path in test_files:
    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"{'='*60}")

    # Load and encode
    original_points = engine._load_mesh_as_points(path)
    original_tensor = engine._points_to_tensor(original_points)

    with torch.no_grad():
        # Original risk
        risk_orig = engine.risk_predictor(original_tensor, return_logits=False).item()

        # CVAE output (raw)
        healed_tensor = engine.cvae.generate_healthy(original_tensor, target_risk=0.15)
        healed_raw = healed_tensor.squeeze(0).transpose(0, 1).cpu().numpy()

        # Healed risk
        risk_healed = engine.risk_predictor(healed_tensor, return_logits=False).item()

        # Also check autoencoder reconstruction quality
        ae_recon = engine.autoencoder(original_tensor)
        ae_points = ae_recon.squeeze(0).transpose(0, 1).cpu().numpy()
        risk_ae = engine.risk_predictor(ae_recon, return_logits=False).item()

    # --- Displacement analysis ---
    disp = healed_raw - original_points
    disp_norms = np.linalg.norm(disp, axis=1)

    # Radial analysis: is displacement inward or outward?
    centroid = np.mean(original_points, axis=0)
    radial_dirs = original_points - centroid
    radial_dirs = radial_dirs / (np.linalg.norm(radial_dirs, axis=1, keepdims=True) + 1e-8)
    radial_component = np.sum(disp * radial_dirs, axis=1)  # positive = outward

    # AE recon error (baseline)
    ae_disp = ae_points - original_points
    ae_disp_norms = np.linalg.norm(ae_disp, axis=1)

    print(f"\n  Risk: {risk_orig:.4f} -> {risk_healed:.4f} (reduction: {risk_orig - risk_healed:.4f})")
    print(f"  AE reconstruction risk: {risk_ae:.4f}")
    print(f"\n  --- Raw CVAE displacement (before clamping) ---")
    print(f"  Mean displacement: {np.mean(disp_norms):.6f}")
    print(f"  Max displacement:  {np.max(disp_norms):.6f}")
    print(f"  Std displacement:  {np.std(disp_norms):.6f}")
    print(f"  Radial component:  mean={np.mean(radial_component):.6f} (+ = outward)")
    print(f"  Points moving outward: {np.sum(radial_component > 0)}/{len(radial_component)}")
    print(f"  Points moving inward:  {np.sum(radial_component < 0)}/{len(radial_component)}")
    print(f"\n  --- AE reconstruction error (baseline) ---")
    print(f"  Mean AE error: {np.mean(ae_disp_norms):.6f}")
    print(f"  Max AE error:  {np.max(ae_disp_norms):.6f}")

    # Is the displacement uniform (zoom) or localized (real healing)?
    # Check: coefficient of variation of displacement norms
    cv = np.std(disp_norms) / (np.mean(disp_norms) + 1e-8)
    print(f"\n  --- Displacement pattern ---")
    print(f"  CV of displacement: {cv:.4f} (low=uniform/zoom, high=localized)")

    # Top 10% vs bottom 10% displacement
    p90 = np.percentile(disp_norms, 90)
    p10 = np.percentile(disp_norms, 10)
    print(f"  P90/P10 ratio: {p90/(p10+1e-8):.2f} (close to 1 = zoom, >>1 = localized)")

    # Is CVAE output different from AE reconstruction?
    cvae_vs_ae = np.linalg.norm(healed_raw - ae_points, axis=1)
    print(f"\n  --- CVAE vs AE difference ---")
    print(f"  Mean CVAE-AE distance: {np.mean(cvae_vs_ae):.6f}")
    print(f"  (if ~0, CVAE is just reconstructing, not healing)")

print(f"\n{'='*60}")
print("INTERPRETATION:")
print("  - If 'CVAE-AE distance' is near 0: CVAE = just autoencoder, risk conditioning not working")
print("  - If displacement CV is low (<0.5): uniform scaling (zoom), not real healing")
print("  - If radial component is mostly positive: outward expansion (zoom)")
print("  - If radial component is mixed: actual shape modification")
print(f"{'='*60}")
