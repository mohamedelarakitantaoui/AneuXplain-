"""
validate_heatmap.py

Validate the heatmap fix on the two test cases:
  - AN26_full (low risk) — should be mostly blue
  - ArteryObjAN40-10 (high risk) — should show concentrated hot spots

Replicates the exact backend pipeline (logit gradients + global ref + risk scaling).
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from backend.app.engine import CounterfactualEngine

MODELS_DIR = PROJECT_ROOT / "models"
HEATMAP_GLOBAL_REF = 0.0672  # must match backend/app/main.py

TEST_CASES = {
    "AN26_full (LOW risk)": PROJECT_ROOT / "IntrA" / "annotated" / "obj" / "AN26_full.obj",
    "ArteryObjAN40-10 (CRITICAL risk)": PROJECT_ROOT / "IntrA" / "generated" / "aneurysm" / "obj" / "ArteryObjAN40-10.obj",
}


def main():
    engine = CounterfactualEngine(models_dir=str(MODELS_DIR), output_dir=str(PROJECT_ROOT / "backend" / "outputs"))
    rp = MODELS_DIR / "risk_predictor_v2.pth"
    if not rp.exists():
        for alt in ["risk_predictor.pth", "risk_predictor_best_gap.pth"]:
            if (MODELS_DIR / alt).exists():
                rp = MODELS_DIR / alt
                break
    engine.load_models(risk_predictor_path=str(rp))
    print(f"HEATMAP_GLOBAL_REF = {HEATMAP_GLOBAL_REF}\n")

    for name, path in TEST_CASES.items():
        if not path.exists():
            print(f"  SKIP {name}: not found")
            continue

        points = engine._load_mesh_as_points(str(path))
        tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
        tensor = tensor.transpose(2, 1).to(engine.device)
        tensor.requires_grad_(True)

        if engine.is_v2_model:
            logit = engine.risk_predictor(tensor, return_logits=True)
            risk_score = torch.sigmoid(logit).item()
        else:
            logit = engine.risk_predictor(tensor)
            risk_score = logit.item()

        logit.backward()
        grad_mag = torch.norm(tensor.grad.squeeze(0), dim=0).detach().cpu().numpy()

        # Exact same pipeline as the fixed /heatmap endpoint
        heatmap = np.clip(grad_mag / HEATMAP_GLOBAL_REF, 0.0, 1.0) * risk_score

        print(f"  {name}")
        print(f"    Risk score: {risk_score:.4f}")
        print(f"    Raw logit-grad: min={grad_mag.min():.6f}  max={grad_mag.max():.6f}  "
              f"mean={grad_mag.mean():.6f}  p95={np.percentile(grad_mag, 95):.6f}")
        print(f"    Final heatmap:  min={heatmap.min():.4f}  max={heatmap.max():.4f}  "
              f"mean={heatmap.mean():.4f}  p95={np.percentile(heatmap, 95):.4f}")
        pct_05 = (heatmap > 0.5).sum() / len(heatmap) * 100
        pct_02 = (heatmap > 0.2).sum() / len(heatmap) * 100
        pct_01 = (heatmap > 0.1).sum() / len(heatmap) * 100
        print(f"    Vertices > 0.5 (hot):  {pct_05:.1f}%")
        print(f"    Vertices > 0.2 (warm): {pct_02:.1f}%")
        print(f"    Vertices > 0.1 (any):  {pct_01:.1f}%")
        print()


if __name__ == "__main__":
    main()
