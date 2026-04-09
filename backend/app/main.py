"""
main.py - FastAPI Backend for Artery Risk Analysis

REST API endpoints for analyzing artery meshes, morphological assessment,
gradient-based heatmaps, and counterfactual "healed" versions.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .engine import CounterfactualEngine

# Backend-sibling modules (one level up from app/)
import sys
_backend_dir = Path(__file__).parent.parent
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))
from morphology_analyzer import MorphologyAnalyzer
from clinical_explainer import ClinicalExplainer

logger = logging.getLogger("aneuxplain")


# ============================================
# Configuration
# ============================================
BACKEND_DIR = Path(__file__).parent.parent  # backend/
PROJECT_ROOT = BACKEND_DIR.parent  # project root
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = BACKEND_DIR / "outputs"
TEMP_DIR = BACKEND_DIR / "temp"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Global reference for heatmap normalization.
# P95 of per-vertex logit-gradient magnitudes across high-risk IntrA meshes.
# Calibrated by scripts/calibrate_heatmap_reference.py — re-run if model changes.
HEATMAP_GLOBAL_REF = 0.0672

# Global instances
engine: Optional[CounterfactualEngine] = None
morphology_analyzer: Optional[MorphologyAnalyzer] = None
clinical_explainer: Optional[ClinicalExplainer] = None


# ============================================
# Lifespan (Startup/Shutdown)
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global engine, morphology_analyzer, clinical_explainer

    print("Starting Artery Analysis API...")
    print(f"   Models directory: {MODELS_DIR}")
    print(f"   Output directory: {OUTPUT_DIR}")

    # Initialize engine
    engine = CounterfactualEngine(
        models_dir=str(MODELS_DIR),
        output_dir=str(OUTPUT_DIR)
    )

    # Initialize morphology + clinical modules (lightweight, no model loading)
    morphology_analyzer = MorphologyAnalyzer()
    clinical_explainer = ClinicalExplainer()
    print("   [OK] MorphologyAnalyzer + ClinicalExplainer initialized")

    # Try to load models
    try:
        # Look for risk predictor weights - prefer V2
        risk_predictor_path = MODELS_DIR / "risk_predictor_v2.pth"
        if not risk_predictor_path.exists():
            # Fall back to other options
            for alt_name in ["risk_predictor.pth", "risk_predictor_best_gap.pth"]:
                alt_path = MODELS_DIR / alt_name
                if alt_path.exists():
                    risk_predictor_path = alt_path
                    break

        if risk_predictor_path.exists():
            engine.load_models(risk_predictor_path=str(risk_predictor_path))
            print("   [OK] Models loaded successfully")
        else:
            print(f"   [WARN] No model weights found in {MODELS_DIR}")
            print(f"     Copy your .pth files to: {MODELS_DIR}")
    except Exception as e:
        print(f"   [ERR] Error loading models: {e}")

    device_info = engine.get_device_info()
    print(f"   Device: {device_info['device']}")

    yield

    # Cleanup on shutdown
    print("Shutting down...")
    # Clean temp files
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)


# ============================================
# FastAPI App
# ============================================
app = FastAPI(
    title="Artery Risk Analysis API",
    description="Analyze 3D artery meshes for aneurysm risk and generate counterfactual 'healed' versions.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (allow frontend connections)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Response Models
# ============================================
class RiskResponse(BaseModel):
    """Response model for risk analysis (with morphology + clinical report)."""
    risk_score: float
    risk_level: str
    interpretation: str
    morphology: Optional[dict] = None
    clinical_report: Optional[dict] = None


class MorphologyResponse(BaseModel):
    """Response model for standalone morphology analysis."""
    morphology: dict
    clinical_report: dict


class HeatmapResponse(BaseModel):
    """Response model for gradient-based spatial heatmap."""
    heatmap: List[float]
    risk_score: float


class HealResponse(BaseModel):
    """Response model for healing operation."""
    initial_risk: float
    final_risk: float
    risk_reduction: float
    risk_reduction_pct: float
    success: bool
    steps_taken: int
    mean_movement: float
    message: str

    # Healed file ID for downloading via /heal/{file_id}
    healed_file_id: Optional[str] = None

    # Intermediate risk scores at discrete morph positions (t=0, 0.25, 0.5, 0.75, 1.0)
    # Enables accurate non-linear risk interpolation in the frontend morph slider
    intermediate_risks: Optional[dict] = None

    # Scientific Geometric Deltas (Phase 5: Interpretation)
    max_displacement_mm: Optional[float] = None
    mean_displacement_mm: Optional[float] = None
    displacement_std_mm: Optional[float] = None
    volume_change_pct: Optional[float] = None
    surface_area_change_pct: Optional[float] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    models_loaded: bool
    device: str


# ============================================
# Helper Functions
# ============================================
async def save_upload_to_temp(upload: UploadFile) -> str:
    """Save an uploaded file to a temporary location."""
    # Create unique temp file
    suffix = Path(upload.filename).suffix if upload.filename else ".obj"
    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
        dir=str(TEMP_DIR)
    )
    
    try:
        content = await upload.read()
        temp_file.write(content)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        temp_file.close()
        os.unlink(temp_file.name)
        raise e


def get_risk_interpretation(risk_score: float) -> tuple:
    """Get risk level and interpretation from score."""
    if risk_score < 0.3:
        return "LOW", "This artery appears healthy with low aneurysm risk."
    elif risk_score < 0.5:
        return "MODERATE", "This artery shows some concerning features. Monitoring recommended."
    elif risk_score < 0.7:
        return "HIGH", "This artery shows significant risk factors. Medical consultation advised."
    else:
        return "CRITICAL", "This artery shows very high risk indicators. Immediate medical attention recommended."


ALLOWED_MESH_EXTENSIONS = {".obj", ".ply", ".stl", ".off"}


def _validate_mesh_extension(filename: Optional[str]) -> None:
    """Raise 400 if the uploaded file has an unsupported extension."""
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_MESH_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {suffix}. Use .obj, .ply, .stl, or .off"
            )


# ============================================
# API Endpoints
# ============================================
@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Artery Risk Analysis API",
        "version": "2.0.0",
        "endpoints": {
            "/health": "Health check",
            "/analyze": "POST - Risk + morphology + clinical report",
            "/morphology": "POST - Standalone morphology analysis",
            "/heatmap": "POST - Gradient-based spatial heatmap",
            "/heal": "POST - Generate healed counterfactual",
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    device_info = engine.get_device_info()
    
    return HealthResponse(
        status="healthy",
        models_loaded=device_info["models_loaded"],
        device=device_info["device"]
    )


@app.post("/analyze", response_model=RiskResponse)
async def analyze_artery(file: UploadFile = File(...)):
    """
    Analyze an artery mesh: risk prediction + morphological analysis + clinical report.

    - **file**: 3D mesh file (.obj, .ply, .stl)

    Returns the risk score, risk level, interpretation, full morphological
    measurements with spatial vertex data, and a clinical explanation report.
    If morphology analysis fails, risk fields are still returned with
    morphology and clinical_report set to null.
    """
    if engine is None or not engine._models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Ensure .pth files are in saved_models/"
        )

    _validate_mesh_extension(file.filename)

    temp_path = None
    try:
        temp_path = await save_upload_to_temp(file)

        # --- Risk prediction (core — must succeed) ---
        risk_score = engine.predict_risk(temp_path)
        risk_level, interpretation = get_risk_interpretation(risk_score)

        # --- Morphology + clinical report (best-effort) ---
        morphology_data = None
        clinical_report_data = None
        try:
            morph_result = morphology_analyzer.analyze(temp_path)
            morphology_data = morph_result
            clinical_report_data = clinical_explainer.explain(
                morph_result, risk_score, include_spatial=True
            )
        except Exception as morph_err:
            logger.warning("Morphology analysis failed: %s", morph_err, exc_info=True)

        return RiskResponse(
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            interpretation=interpretation,
            morphology=morphology_data,
            clinical_report=clinical_report_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/morphology", response_model=MorphologyResponse)
async def morphology_analysis(file: UploadFile = File(...)):
    """
    Standalone morphological analysis + clinical report (no risk prediction).

    - **file**: 3D mesh file (.obj, .ply, .stl)

    Useful for debugging and testing the morphology module independently.
    """
    if morphology_analyzer is None or clinical_explainer is None:
        raise HTTPException(status_code=503, detail="Morphology modules not initialized")

    _validate_mesh_extension(file.filename)

    temp_path = None
    try:
        temp_path = await save_upload_to_temp(file)

        morph_result = morphology_analyzer.analyze(temp_path)

        # Run clinical report without a model risk score — use morphology-only
        # heuristic: fraction of HIGH-risk parameters as a proxy score
        clinical_report_data = clinical_explainer.explain(
            morph_result, risk_score=0.0, include_spatial=True
        )
        # Derive a simple proxy score from the parameter risk counts
        total = (
            clinical_report_data["high_risk_count"]
            + clinical_report_data["moderate_risk_count"]
            + clinical_report_data["low_risk_count"]
        )
        if total > 0:
            proxy_score = (
                clinical_report_data["high_risk_count"]
                + 0.5 * clinical_report_data["moderate_risk_count"]
            ) / total
        else:
            proxy_score = 0.0
        # Re-run with the proxy score so the summary text is coherent
        clinical_report_data = clinical_explainer.explain(
            morph_result, risk_score=proxy_score, include_spatial=True
        )

        return MorphologyResponse(
            morphology=morph_result,
            clinical_report=clinical_report_data,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Morphology analysis failed: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/heatmap", response_model=HeatmapResponse)
async def gradient_heatmap(file: UploadFile = File(...)):
    """
    Compute a gradient-based spatial risk heatmap for an artery mesh.

    - **file**: 3D mesh file (.obj, .ply, .stl)

    For each of the 2048 sampled surface points, computes the gradient of
    the risk score with respect to that point's coordinates. The magnitude
    of each gradient vector indicates how sensitive the risk prediction is
    to perturbations at that location — high values mark the regions most
    responsible for the predicted risk.

    Returns a list of 2048 normalized gradient magnitudes in [0, 1].
    """
    if engine is None or not engine._models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Ensure .pth files are in saved_models/"
        )

    _validate_mesh_extension(file.filename)

    temp_path = None
    try:
        temp_path = await save_upload_to_temp(file)

        # Load and preprocess (same pipeline as predict_risk)
        points = engine._load_mesh_as_points(temp_path)
        tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
        tensor = tensor.transpose(2, 1)  # (1, 3, N)
        tensor = tensor.to(engine.device)
        tensor.requires_grad_(True)

        # Forward pass — backprop from LOGIT to avoid sigmoid saturation
        # (high-confidence predictions would otherwise have near-zero gradients)
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

        # Per-point gradient magnitude: ||grad_i|| for each of N points
        grad_mag = torch.norm(grad.squeeze(0), dim=0)  # (N,)
        grad_mag_np = grad_mag.detach().cpu().numpy()

        # Log raw gradient stats for calibration diagnostics
        logger.info(
            "Heatmap grads (logit-space): min=%.6f max=%.6f mean=%.6f p95=%.6f | risk=%.4f",
            grad_mag_np.min(), grad_mag_np.max(),
            grad_mag_np.mean(), float(np.percentile(grad_mag_np, 95)),
            risk_score,
        )

        # Global normalization: divide by calibrated reference, clip to [0,1].
        # Then scale by risk_score so low-risk meshes appear cool (blue) and
        # high-risk meshes retain their hot spots.
        heatmap_np = np.clip(grad_mag_np / HEATMAP_GLOBAL_REF, 0.0, 1.0) * risk_score
        heatmap = heatmap_np.tolist()

        return HeatmapResponse(
            heatmap=heatmap,
            risk_score=round(risk_score, 4),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.warning("Heatmap computation failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Heatmap computation failed: {str(e)}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/heal")
async def heal_artery(
    file: UploadFile = File(...),
    num_steps: int = Query(default=200, ge=50, le=1000, description="Optimization steps"),
    target_risk: float = Query(default=0.1, ge=0.01, le=0.5, description="Target risk score"),
    return_file: bool = Query(default=False, description="Return the healed mesh file directly")
):
    """
    Generate a counterfactual "healed" version of an artery.
    
    - **file**: 3D mesh file (.obj, .ply, .stl)
    - **num_steps**: Number of optimization steps (default: 300)
    - **target_risk**: Target risk score to achieve (default: 0.3)
    - **return_file**: If true, returns the .obj file; otherwise returns JSON
    
    This optimization finds minimal changes to the artery geometry
    that would reduce its predicted risk score.
    """
    if engine is None or not engine._models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Ensure .pth files are in saved_models/"
        )
    
    _validate_mesh_extension(file.filename)

    # Save uploaded file
    temp_path = None
    try:
        temp_path = await save_upload_to_temp(file)

        # Heal
        healed_path, result = engine.heal_artery(
            mesh_path=temp_path,
            num_steps=num_steps,
            target_risk=target_risk
        )
        
        if return_file:
            # Return the healed mesh file
            return FileResponse(
                path=healed_path,
                media_type="application/octet-stream",
                filename="healed_artery.obj"
            )
        else:
            # Calculate scientific geometric deltas (Phase 5: Interpretation)
            geometric_deltas = engine.calculate_geometric_deltas(
                original_mesh_path=temp_path,
                healed_mesh_path=healed_path,
                original_risk=result["initial_risk"],
                final_risk=result["final_risk"]
            )
            
            # Extract file ID from healed path (e.g. "healed_a1b2c3d4.obj" -> "a1b2c3d4")
            healed_file_id = Path(healed_path).stem.replace("healed_", "")

            # Return JSON with results
            message = (
                "Successfully transformed artery from HIGH to LOW risk!"
                if result["success"]
                else "Risk was reduced but target not fully achieved."
            )

            return HealResponse(
                initial_risk=round(result["initial_risk"], 4),
                final_risk=round(result["final_risk"], 4),
                risk_reduction=round(result["risk_reduction"], 4),
                risk_reduction_pct=round(result["risk_reduction_pct"], 1),
                success=result["success"],
                steps_taken=result["steps_taken"],
                mean_movement=round(result["mean_movement"], 6),
                message=message,
                healed_file_id=healed_file_id,
                # Intermediate risk scores for morph slider
                intermediate_risks=result.get("intermediate_risks"),
                # Scientific Geometric Deltas
                max_displacement_mm=geometric_deltas.get("max_displacement_mm"),
                mean_displacement_mm=geometric_deltas.get("mean_displacement_mm"),
                displacement_std_mm=geometric_deltas.get("displacement_std_mm"),
                volume_change_pct=geometric_deltas.get("volume_change_pct"),
                surface_area_change_pct=geometric_deltas.get("surface_area_change_pct"),
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Healing failed: {str(e)}")
    
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.get("/heal/{file_id}")
async def download_healed_artery(file_id: str):
    """
    Download a previously generated healed artery mesh.
    
    - **file_id**: The 8-character ID from the heal operation
    """
    # Find file matching the ID
    matches = list(OUTPUT_DIR.glob(f"healed_{file_id}*.obj"))
    
    if not matches:
        raise HTTPException(status_code=404, detail=f"Healed file not found: {file_id}")
    
    return FileResponse(
        path=str(matches[0]),
        media_type="application/octet-stream",
        filename=matches[0].name
    )


# ============================================
# Run with: uvicorn backend.app.main:app --reload
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
