"""
clinical_explainer.py - Clinical Explanation Engine for Aneurysm Risk Reports

Takes morphological measurements (from MorphologyAnalyzer) and a predicted risk
score (from RiskPredictorV2) and produces a structured clinical report in the
style of a blood-test panel — each parameter classified as LOW / MODERATE / HIGH
with biomechanical explanations grounded in published neurosurgery literature.
"""

import json

# ============================================================================
# Configurable clinical thresholds
#
# Each entry defines the LOW/MODERATE/HIGH boundaries for one morphological
# parameter.  Update these as the clinical grounding research evolves.
#
# Format per parameter:
#   "name"  — must match MorphologyAnalyzer measurement names exactly
#   "unit"  — display unit
#   "low"   — (operator, value) pair that defines the LOW-risk region
#   "high"  — (operator, value) pair that defines the HIGH-risk region
#              everything between is MODERATE
#   "invert" — if True, the scale is reversed (smaller value = higher risk)
#   "normal_range" — human-readable string shown in the report
#   "source" — literature reference
# ============================================================================
CLINICAL_THRESHOLDS = {
    "Aspect Ratio": {
        "unit": "",
        "low_upper": 1.0,
        "high_lower": 1.6,
        "invert": False,
        "normal_range": "< 1.6",
        "source": "ISUIA, Ujiie et al. 2001",
    },
    "Dome-to-Neck Ratio": {
        "unit": "",
        "low_upper": 1.0,
        "high_lower": 2.0,
        "invert": False,
        "normal_range": "< 2.0",
        "source": "Multiple clinical studies",
    },
    "Irregularity Index": {
        "unit": "",
        "low_upper": 0.1,
        "high_lower": 0.25,
        "invert": False,
        "normal_range": "< 0.25",
        "source": "Based on surface deviation norms",
    },
    "Neck Width": {
        "unit": "mm",
        "low_upper": 4.0,   # > 4.0 mm is LOW risk (wide neck = stable)
        "high_lower": 2.0,  # < 2.0 mm is HIGH risk (narrow neck)
        "invert": True,
        "normal_range": "> 4.0 mm",
        "source": "Endovascular treatment planning",
    },
    "Dome Height": {
        "unit": "mm",
        "low_upper": 5.0,
        "high_lower": 10.0,
        "invert": False,
        "normal_range": "< 10.0 mm",
        "source": "Size-based rupture risk studies",
    },
    "Volume": {
        "unit": "mm\u00b3",
        "low_upper": 100.0,
        "high_lower": 500.0,
        "invert": False,
        "normal_range": "< 500 mm\u00b3",
        "source": "Volumetric risk assessment",
    },
    "Size Ratio": {
        "unit": "",
        "low_upper": 2.0,
        "high_lower": 4.0,
        "invert": False,
        "normal_range": "< 4.0",
        "source": "Kashiwazaki et al.",
        "requires_parent_vessel": True,
    },
}

# Risk level colors
RISK_COLORS = {
    "LOW": "#2ECC71",       # green
    "MODERATE": "#F39C12",  # amber
    "HIGH": "#E74C3C",      # red
}

# Overall risk score thresholds
OVERALL_RISK_THRESHOLDS = {"low_upper": 0.33, "high_lower": 0.66}

# ============================================================================
# Clinical explanation templates
#
# Each parameter has three templates (one per risk level).  Placeholders:
#   {value}     — the measured numeric value
#   {threshold} — the threshold it exceeds or falls within
# ============================================================================
EXPLANATIONS = {
    "Aspect Ratio": {
        "LOW": (
            "The aspect ratio of {value:.2f} is below 1.0, indicating a wide, "
            "shallow dome relative to the neck. This geometry promotes efficient "
            "blood flow washout from the dome, reducing hemodynamic wall stress."
        ),
        "MODERATE": (
            "The aspect ratio of {value:.2f} falls in the moderate range "
            "(1.0\u20131.6). Blood flow recirculation within the dome is increasing, "
            "which may elevate localized wall shear stress but remains below the "
            "critical threshold identified in rupture studies."
        ),
        "HIGH": (
            "The aspect ratio of {value:.2f} exceeds the critical threshold of "
            "1.6. A high aspect ratio indicates a tall, narrow-necked aneurysm "
            "which creates unfavorable hemodynamic conditions \u2014 blood flow "
            "entering the dome has difficulty exiting, increasing wall stress "
            "and rupture risk."
        ),
    },
    "Dome-to-Neck Ratio": {
        "LOW": (
            "The dome-to-neck ratio of {value:.2f} is below 1.0, meaning the "
            "dome is narrower than the neck. This is a favorable geometry with "
            "unobstructed outflow and lower risk of flow stagnation."
        ),
        "MODERATE": (
            "The dome-to-neck ratio of {value:.2f} falls in the moderate range "
            "(1.0\u20132.0). The dome is wider than the neck, creating a partial "
            "bottleneck that slows blood outflow and may promote thrombus "
            "formation in the dome apex."
        ),
        "HIGH": (
            "The dome-to-neck ratio of {value:.2f} exceeds 2.0, indicating a "
            "significantly wider dome than neck. This bottleneck geometry traps "
            "recirculating blood flow, increases residence time, and creates "
            "high oscillatory shear on the dome wall \u2014 conditions strongly "
            "associated with wall degradation and rupture."
        ),
    },
    "Irregularity Index": {
        "LOW": (
            "The irregularity index of {value:.3f} indicates a smooth dome "
            "surface that closely conforms to an ellipsoidal shape. Smooth walls "
            "distribute hemodynamic forces evenly, reducing focal stress "
            "concentrations."
        ),
        "MODERATE": (
            "The irregularity index of {value:.3f} shows moderate surface "
            "deviation from a smooth ellipsoid (0.1\u20130.25). Localized blebs or "
            "undulations are present, which can create focal regions of elevated "
            "wall shear stress and may indicate early wall remodeling."
        ),
        "HIGH": (
            "The irregularity index of {value:.3f} exceeds 0.25, indicating "
            "significant surface irregularity with daughter blebs or lobulations. "
            "These focal wall protrusions are thin-walled regions where "
            "hemodynamic forces concentrate, and are among the strongest "
            "morphological predictors of imminent rupture."
        ),
    },
    "Neck Width": {
        "LOW": (
            "The neck width of {value:.1f} mm exceeds 4.0 mm, indicating a "
            "wide, stable connection to the parent vessel. A wide neck allows "
            "efficient blood flow exchange and distributes mechanical load over "
            "a larger area."
        ),
        "MODERATE": (
            "The neck width of {value:.1f} mm is in the moderate range "
            "(2.0\u20134.0 mm). While structurally adequate, the narrower neck "
            "concentrates hemodynamic forces at the neck\u2013dome junction, a "
            "common site of rupture initiation."
        ),
        "HIGH": (
            "The neck width of {value:.1f} mm is below 2.0 mm, indicating a "
            "very narrow neck. This concentrates all inflow and outflow through "
            "a small orifice, creating high-velocity jets and intense shear "
            "stress at the neck that accelerates wall degradation."
        ),
    },
    "Dome Height": {
        "LOW": (
            "The dome height of {value:.1f} mm is below 5.0 mm, placing this "
            "aneurysm in the small-size category. Smaller aneurysms generally "
            "experience lower wall tension per Laplace\u2019s law and have lower "
            "annual rupture rates."
        ),
        "MODERATE": (
            "The dome height of {value:.1f} mm falls in the moderate range "
            "(5.0\u201310.0 mm). Wall tension increases with dome size per "
            "Laplace\u2019s law. Aneurysms in this size range warrant monitoring and "
            "consideration of patient-specific factors."
        ),
        "HIGH": (
            "The dome height of {value:.1f} mm exceeds 10.0 mm. Large aneurysms "
            "experience significantly higher wall tension, and the probability "
            "of containing thin-walled regions increases with size. Clinical "
            "studies consistently show elevated rupture rates above this "
            "threshold."
        ),
    },
    "Volume": {
        "LOW": (
            "The dome volume of {value:.1f} mm\u00b3 is below 100 mm\u00b3, consistent "
            "with a small aneurysm. Lower volume means less total wall area "
            "under hemodynamic stress and a lower probability of harboring "
            "focal weak points."
        ),
        "MODERATE": (
            "The dome volume of {value:.1f} mm\u00b3 is in the moderate range "
            "(100\u2013500 mm\u00b3). The increased volume extends the surface area "
            "exposed to hemodynamic forces, raising the likelihood of focal "
            "thinning and wall remodeling."
        ),
        "HIGH": (
            "The dome volume of {value:.1f} mm\u00b3 exceeds 500 mm\u00b3, indicating a "
            "large aneurysm. Large volumes create extensive wall surface under "
            "stress, higher blood residence time within the dome, and greater "
            "probability of containing critically thin wall regions."
        ),
    },
    "Size Ratio": {
        "LOW": (
            "The size ratio of {value:.2f} is below 2.0, meaning the aneurysm "
            "is less than twice the diameter of the parent vessel. This is a "
            "proportionate size relationship with favorable hemodynamic coupling."
        ),
        "MODERATE": (
            "The size ratio of {value:.2f} falls in the moderate range "
            "(2.0\u20134.0). The aneurysm is disproportionately larger than its "
            "parent vessel, which alters local flow dynamics and increases the "
            "fraction of cardiac output entering the dome."
        ),
        "HIGH": (
            "The size ratio of {value:.2f} exceeds 4.0, indicating the "
            "aneurysm is more than four times the parent vessel diameter. This "
            "extreme size disproportion creates severe flow disturbance at the "
            "neck and is a strong independent predictor of rupture per "
            "Kashiwazaki et al."
        ),
    },
}

CLINICAL_SIGNIFICANCE = {
    "Aspect Ratio": (
        "Aspect ratio is one of the strongest independent predictors of "
        "rupture, identified in the ISUIA study and validated across multiple "
        "cohorts (Ujiie et al. 2001)."
    ),
    "Dome-to-Neck Ratio": (
        "The dome-to-neck ratio reflects the degree of flow confinement "
        "within the dome. It is widely used in treatment planning to assess "
        "both rupture risk and suitability for coil embolization."
    ),
    "Irregularity Index": (
        "Surface irregularity \u2014 particularly the presence of daughter blebs "
        "\u2014 is recognized as one of the most reliable morphological markers of "
        "rupture risk, independent of aneurysm size."
    ),
    "Neck Width": (
        "Neck width influences both rupture biomechanics and treatment "
        "feasibility. Narrow necks concentrate hemodynamic stress but may "
        "paradoxically facilitate coil embolization."
    ),
    "Dome Height": (
        "Dome height is a primary size metric used in clinical decision-making. "
        "The ISUIA and UCAS Japan studies established size-based rupture risk "
        "stratification thresholds."
    ),
    "Volume": (
        "Volumetric assessment captures the three-dimensional extent of the "
        "aneurysm, providing a more complete size measure than any single "
        "linear dimension."
    ),
    "Size Ratio": (
        "The size ratio normalizes aneurysm size to the parent vessel, "
        "controlling for anatomical location. Kashiwazaki et al. demonstrated "
        "its independent predictive value for rupture."
    ),
}


# ============================================================================
# ClinicalExplainer
# ============================================================================
class ClinicalExplainer:
    """
    Produces a structured clinical report from morphological measurements
    and a model-predicted risk score.

    Usage:
        explainer = ClinicalExplainer()
        report = explainer.explain(morphology_result, risk_score=0.78)
    """

    def __init__(self, thresholds=None):
        """
        Args:
            thresholds: Optional dict to override CLINICAL_THRESHOLDS.
                        Keys must match parameter names from MorphologyAnalyzer.
        """
        self.thresholds = thresholds if thresholds is not None else CLINICAL_THRESHOLDS

    def explain(self, morphology_result, risk_score, include_spatial=True):
        """
        Generate a clinical explanation report.

        Args:
            morphology_result: Dict returned by MorphologyAnalyzer.analyze().
            risk_score: Float in [0, 1] from RiskPredictorV2.
            include_spatial: If True, pass through spatial vertex data for
                             frontend rendering of highlighted regions.

        Returns:
            Dict with keys: parameters, summary, risk_score, risk_level,
            high_risk_count, moderate_risk_count, low_risk_count.
        """
        measurements = morphology_result.get("measurements", [])
        measurement_map = {m["name"]: m for m in measurements}

        parameter_reports = []
        for name, thresh in self.thresholds.items():
            report = self._evaluate_parameter(
                name, thresh, measurement_map, include_spatial
            )
            if report is not None:
                parameter_reports.append(report)

        counts = {"HIGH": 0, "MODERATE": 0, "LOW": 0}
        for p in parameter_reports:
            if p["risk_level"] in counts:
                counts[p["risk_level"]] += 1

        overall_level = self._classify_overall_risk(risk_score)
        summary = self._generate_summary(
            parameter_reports, counts, risk_score, overall_level
        )

        return {
            "parameters": parameter_reports,
            "summary": summary,
            "risk_score": round(risk_score, 4),
            "risk_level": overall_level,
            "high_risk_count": counts["HIGH"],
            "moderate_risk_count": counts["MODERATE"],
            "low_risk_count": counts["LOW"],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _evaluate_parameter(self, name, thresh, measurement_map, include_spatial):
        """Classify a single parameter and build its report entry."""
        m = measurement_map.get(name)

        # Size Ratio requires parent vessel data we may not have
        if thresh.get("requires_parent_vessel") and (m is None or m.get("value") is None):
            return {
                "parameter": name,
                "value": None,
                "unit": thresh["unit"],
                "risk_level": "N/A",
                "risk_color": "#95A5A6",
                "normal_range": thresh["normal_range"],
                "explanation": "N/A \u2014 requires parent vessel reference diameter, "
                               "which is not available from the isolated dome mesh.",
                "clinical_significance": CLINICAL_SIGNIFICANCE.get(name, ""),
            }

        if m is None or m.get("status") != "computed" or m.get("value") is None:
            return None

        value = m["value"]
        risk_level = self._classify_value(value, thresh)
        explanation = EXPLANATIONS.get(name, {}).get(risk_level, "")
        if explanation:
            explanation = explanation.format(value=value, threshold="")

        entry = {
            "parameter": name,
            "value": round(value, 4),
            "unit": thresh["unit"],
            "risk_level": risk_level,
            "risk_color": RISK_COLORS[risk_level],
            "normal_range": thresh["normal_range"],
            "explanation": explanation,
            "clinical_significance": CLINICAL_SIGNIFICANCE.get(name, ""),
        }

        if include_spatial and "spatial" in m:
            entry["spatial"] = m["spatial"]

        return entry

    @staticmethod
    def _classify_value(value, thresh):
        """Classify a value as LOW, MODERATE, or HIGH given threshold config."""
        if thresh.get("invert", False):
            # Inverted scale: high value = low risk, low value = high risk
            if value > thresh["low_upper"]:
                return "LOW"
            elif value < thresh["high_lower"]:
                return "HIGH"
            else:
                return "MODERATE"
        else:
            # Normal scale: low value = low risk, high value = high risk
            if value < thresh["low_upper"]:
                return "LOW"
            elif value > thresh["high_lower"]:
                return "HIGH"
            else:
                return "MODERATE"

    @staticmethod
    def _classify_overall_risk(risk_score):
        """Map a 0-1 risk score to LOW/MODERATE/HIGH."""
        if risk_score < OVERALL_RISK_THRESHOLDS["low_upper"]:
            return "LOW"
        elif risk_score > OVERALL_RISK_THRESHOLDS["high_lower"]:
            return "HIGH"
        else:
            return "MODERATE"

    @staticmethod
    def _generate_summary(parameter_reports, counts, risk_score, overall_level):
        """Synthesize an overall clinical summary paragraph."""
        evaluated = [p for p in parameter_reports if p["risk_level"] != "N/A"]
        high_params = [p for p in evaluated if p["risk_level"] == "HIGH"]
        moderate_params = [p for p in evaluated if p["risk_level"] == "MODERATE"]

        # Identify the most concerning parameters (HIGH first, then MODERATE)
        concerning = high_params + moderate_params
        top_concerns = concerning[:3]

        total = counts["HIGH"] + counts["MODERATE"] + counts["LOW"]
        risk_pct = round(risk_score * 100)

        # Build the summary
        parts = []

        # Tier counts
        tier_parts = []
        if counts["HIGH"] > 0:
            tier_parts.append(f"{counts['HIGH']} high-risk")
        if counts["MODERATE"] > 0:
            tier_parts.append(f"{counts['MODERATE']} moderate-risk")
        if counts["LOW"] > 0:
            tier_parts.append(f"{counts['LOW']} low-risk")

        if total > 0:
            parts.append(
                f"This aneurysm presents {' and '.join(tier_parts)} "
                f"morphological feature{'s' if total != 1 else ''} "
                f"out of {total} evaluated."
            )

        # Top concerns
        if len(top_concerns) >= 2:
            concern_strs = []
            for p in top_concerns:
                if p["value"] is not None:
                    concern_strs.append(
                        f"{p['risk_level'].lower()} {p['parameter'].lower()} "
                        f"({p['value']:.2f}{(' ' + p['unit']) if p['unit'] else ''})"
                    )
            if concern_strs:
                parts.append(
                    f"The combination of {' and '.join(concern_strs)} is "
                    f"particularly concerning, as these factors independently "
                    f"and synergistically elevate rupture risk."
                )
        elif len(top_concerns) == 1:
            p = top_concerns[0]
            if p["value"] is not None:
                parts.append(
                    f"The {p['risk_level'].lower()} {p['parameter'].lower()} "
                    f"({p['value']:.2f}{(' ' + p['unit']) if p['unit'] else ''}) "
                    f"is the primary morphological concern."
                )

        # Model correlation
        if counts["HIGH"] >= 2 and risk_score > 0.5:
            parts.append(
                f"The model's predicted risk of {risk_pct}% is consistent "
                f"with these morphological findings."
            )
        elif counts["HIGH"] == 0 and risk_score < 0.4:
            parts.append(
                f"The model's predicted risk of {risk_pct}% aligns with the "
                f"favorable morphological profile."
            )
        else:
            parts.append(
                f"The model's predicted risk of {risk_pct}% should be "
                f"interpreted alongside these morphological features for "
                f"comprehensive clinical assessment."
            )

        return " ".join(parts)


# ============================================================================
# Standalone test
# ============================================================================
if __name__ == "__main__":
    # Mock measurement dict simulating MorphologyAnalyzer.analyze() output
    mock_morphology = {
        "measurements": [
            {
                "name": "Neck Width",
                "value": 3.2,
                "unit": "mm",
                "status": "computed",
                "spatial": {
                    "vertex_indices": [10, 245],
                    "type": "line",
                    "color": "#00BFFF",
                },
            },
            {
                "name": "Dome Height",
                "value": 11.4,
                "unit": "mm",
                "status": "computed",
                "spatial": {
                    "vertex_indices": [512],
                    "plane_vertices": [10, 245],
                    "type": "line",
                    "color": "#FF6B35",
                },
            },
            {
                "name": "Aspect Ratio",
                "value": 1.82,
                "unit": "",
                "status": "computed",
                "spatial": {"vertex_indices": [], "type": "point", "color": "#FFD700"},
            },
            {
                "name": "Max Dome Diameter",
                "value": 8.6,
                "unit": "mm",
                "status": "computed",
                "spatial": {
                    "vertex_indices": [300, 420],
                    "type": "line",
                    "color": "#FF1493",
                },
            },
            {
                "name": "Dome-to-Neck Ratio",
                "value": 2.69,
                "unit": "",
                "status": "computed",
                "spatial": {"vertex_indices": [], "type": "point", "color": "#8A2BE2"},
            },
            {
                "name": "Irregularity Index",
                "value": 0.31,
                "unit": "",
                "status": "computed",
                "spatial": {
                    "vertex_indices": [101, 102, 103, 104, 105],
                    "type": "region",
                    "color": "#FF4444",
                },
            },
            {
                "name": "Volume",
                "value": 342.7,
                "unit": "mm\u00b3",
                "status": "computed",
                "spatial": {
                    "vertex_indices": list(range(50)),
                    "type": "region",
                    "color": "#32CD32",
                },
            },
            {
                "name": "Surface Area",
                "value": 285.3,
                "unit": "mm\u00b2",
                "status": "computed",
                "spatial": {
                    "vertex_indices": list(range(50)),
                    "type": "region",
                    "color": "#1E90FF",
                },
            },
        ],
        "neck_plane": {
            "point": [0.0, 0.0, 0.0],
            "normal": [0.0, 0.0, 1.0],
        },
        "dome_vertices": list(range(200)),
        "neck_vertices": [10, 245, 246, 247],
    }

    mock_risk_score = 0.78

    explainer = ClinicalExplainer()
    report = explainer.explain(
        mock_morphology, risk_score=mock_risk_score, include_spatial=True
    )

    print(json.dumps(report, indent=2))
