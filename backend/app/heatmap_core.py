"""
Shared heatmap helper used by both /heatmap (mesh upload) and
/dicom/cropped-heatmap/{session_id} (DICOM pipeline). Given a path to
a mesh file on disk and a loaded engine, returns the same dict the
/heatmap endpoint serializes: ``{"heatmap": List[float], "risk_score": float}``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np
import torch

# Global reference for heatmap normalization. Calibrated by
# scripts/calibrate_heatmap_reference.py — keep in sync with main.py.
HEATMAP_GLOBAL_REF = 0.0672

logger = logging.getLogger("aneuxplain.heatmap")


def compute_heatmap_for_mesh(mesh_path: str, engine) -> Dict[str, Any]:
    """
    Compute a gradient-based spatial risk heatmap for a mesh on disk.

    Mirrors the logic previously inlined in main.py's /heatmap handler so
    both the upload path and the DICOM session path can share it.
    """
    points = engine._load_mesh_as_points(mesh_path)
    tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
    tensor = tensor.transpose(2, 1)  # (1, 3, N)
    tensor = tensor.to(engine.device)
    tensor.requires_grad_(True)

    assert engine.risk_predictor is not None
    if engine.is_v2_model:
        logit = engine.risk_predictor(tensor, return_logits=True)
        risk_score = torch.sigmoid(logit).item()
    else:
        logit = engine.risk_predictor(tensor)
        risk_score = logit.item()

    logit.backward()
    assert tensor.grad is not None, "Gradient computation failed"
    grad = tensor.grad  # (1, 3, N)

    grad_mag = torch.norm(grad.squeeze(0), dim=0)  # (N,)
    grad_mag_np = grad_mag.detach().cpu().numpy()

    logger.info(
        "Heatmap grads (logit-space): min=%.6f max=%.6f mean=%.6f p95=%.6f | risk=%.4f",
        grad_mag_np.min(),
        grad_mag_np.max(),
        grad_mag_np.mean(),
        float(np.percentile(grad_mag_np, 95)),
        risk_score,
    )

    heatmap_np = np.clip(grad_mag_np / HEATMAP_GLOBAL_REF, 0.0, 1.0) * risk_score
    return {
        "heatmap": heatmap_np.tolist(),
        "risk_score": round(float(risk_score), 4),
    }
