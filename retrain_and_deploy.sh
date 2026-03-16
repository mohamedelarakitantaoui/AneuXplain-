#!/bin/bash
# =============================================================
# Retrain and Deploy Pipeline
# =============================================================
# Run this after fixing training labels or model architecture.
# It regenerates labels, retrains the risk predictor, and
# deploys the new weights to the backend.
#
# Usage:
#   chmod +x retrain_and_deploy.sh
#   ./retrain_and_deploy.sh
# =============================================================

set -e  # Exit on error

echo "============================================================"
echo "  RETRAIN AND DEPLOY PIPELINE"
echo "============================================================"

# Step 1: Prepare labels
echo ""
echo "[1/3] Preparing labels..."
echo "------------------------------------------------------------"
python -m training.scripts.prepare_labels

# Step 2: Retrain risk predictor
echo ""
echo "[2/3] Training risk predictor..."
echo "------------------------------------------------------------"
python -m training.scripts.train_risk_predictor

# Step 3: Done — model auto-deployed by train_risk_predictor.py
echo ""
echo "[3/3] Deployment complete!"
echo "------------------------------------------------------------"
echo "  Models deployed to:"
echo "    - backend/saved_models/risk_predictor.pth"
echo "    - models/risk_predictor.pth"
echo ""
echo "  You can now run the app:"
echo "    cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000"
echo "============================================================"
