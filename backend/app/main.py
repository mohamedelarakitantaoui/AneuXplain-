"""
main.py - FastAPI Backend for Artery Risk Analysis

REST API endpoints for analyzing artery meshes and generating
counterfactual "healed" versions.
"""

import os
import shutil
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .engine import CounterfactualEngine


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

# Global engine instance
engine: Optional[CounterfactualEngine] = None


# ============================================
# Lifespan (Startup/Shutdown)
# ============================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global engine
    
    print("Starting Artery Analysis API...")
    print(f"   Models directory: {MODELS_DIR}")
    print(f"   Output directory: {OUTPUT_DIR}")
    
    # Initialize engine
    engine = CounterfactualEngine(
        models_dir=str(MODELS_DIR),
        output_dir=str(OUTPUT_DIR)
    )
    
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
    """Response model for risk analysis."""
    risk_score: float
    risk_level: str
    interpretation: str


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


# ============================================
# API Endpoints
# ============================================
@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Artery Risk Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/analyze": "POST - Analyze artery mesh for risk",
            "/heal": "POST - Generate healed counterfactual"
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
    Analyze an artery mesh and return its risk score.
    
    - **file**: 3D mesh file (.obj, .ply, .stl)
    
    Returns the risk score (0-1), risk level, and interpretation.
    """
    if engine is None or not engine._models_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Ensure .pth files are in saved_models/"
        )
    
    # Validate file type
    if file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in [".obj", ".ply", ".stl", ".off"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {suffix}. Use .obj, .ply, .stl, or .off"
            )
    
    # Save uploaded file
    temp_path = None
    try:
        temp_path = await save_upload_to_temp(file)
        
        # Analyze
        risk_score = engine.predict_risk(temp_path)
        risk_level, interpretation = get_risk_interpretation(risk_score)
        
        return RiskResponse(
            risk_score=round(risk_score, 4),
            risk_level=risk_level,
            interpretation=interpretation
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    finally:
        # Cleanup temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@app.post("/heal")
async def heal_artery(
    file: UploadFile = File(...),
    num_steps: int = Query(default=300, ge=50, le=1000, description="Optimization steps"),
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
    
    # Validate file type
    if file.filename:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in [".obj", ".ply", ".stl", ".off"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {suffix}. Use .obj, .ply, .stl, or .off"
            )
    
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
