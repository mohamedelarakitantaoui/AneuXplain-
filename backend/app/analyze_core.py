"""
Shared analysis helper used by both /analyze (mesh upload) and
/dicom/crop-and-analyze (DICOM pipeline). Takes a path to a mesh file on
disk and returns the same dict the /analyze endpoint serializes.

The route layer owns HTTP concerns (upload, temp files, status codes);
this module owns the inference + morphology + clinical report pipeline.
"""

from __future__ import annotations

from typing import Any, Dict


def get_risk_interpretation(risk_score: float) -> tuple[str, str]:
    if risk_score < 0.3:
        return "LOW", "This artery appears healthy with low aneurysm risk."
    elif risk_score < 0.5:
        return "MODERATE", "This artery shows some concerning features. Monitoring recommended."
    elif risk_score < 0.7:
        return "HIGH", "This artery shows significant risk factors. Medical consultation advised."
    else:
        return "CRITICAL", "This artery shows very high risk indicators. Immediate medical attention recommended."


def analyze_mesh_file(
    mesh_path: str,
    engine,
    morphology_analyzer,
    clinical_explainer,
) -> Dict[str, Any]:
    """
    Run risk + morphology + clinical report on a mesh file on disk.

    Raises RuntimeError if the risk predictor itself fails. Morphology /
    clinical report failures are logged by callers and returned as None.
    """
    import logging

    logger = logging.getLogger("aneuxplain")

    risk_score = engine.predict_risk(mesh_path)
    risk_level, interpretation = get_risk_interpretation(risk_score)

    morphology_data = None
    clinical_report_data = None
    try:
        morph_result = morphology_analyzer.analyze(mesh_path)
        morphology_data = morph_result
        clinical_report_data = clinical_explainer.explain(
            morph_result, risk_score, include_spatial=True
        )
    except Exception as morph_err:
        logger.warning("Morphology analysis failed: %s", morph_err, exc_info=True)

    return {
        "risk_score": round(float(risk_score), 4),
        "risk_level": risk_level,
        "interpretation": interpretation,
        "morphology": morphology_data,
        "clinical_report": clinical_report_data,
    }
