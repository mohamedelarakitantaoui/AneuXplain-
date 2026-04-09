# AneuXplain - Explainable AI for Intracranial Aneurysm Rupture Risk Prediction

A medical AI system that predicts aneurysm risk from 3D intracranial artery meshes and generates counterfactual "healed" geometries, showing what minimal geometric changes would theoretically reduce the risk. Built as a capstone project combining deep learning on 3D point clouds with an interactive medical workstation UI.

---

## Table of Contents

- [Project Overview](#project-overview)
- [How It Works (End-to-End)](#how-it-works-end-to-end)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Machine Learning Pipeline](#machine-learning-pipeline)
  - [1. Data & Labeling](#1-data--labeling)
  - [2. Risk Predictor (RiskPredictorV2)](#2-risk-predictor-riskpredictorv2)
  - [3. Autoencoder](#3-autoencoder)
  - [4. Conditional VAE (CVAE)](#4-conditional-vae-cvae)
- [Backend API](#backend-api)
  - [CounterfactualEngine](#counterfactualengine)
  - [API Endpoints](#api-endpoints)
  - [Healing Algorithm](#healing-algorithm)
- [Frontend UI](#frontend-ui)
  - [3D Viewer](#3d-viewer)
  - [View Modes](#view-modes)
  - [Medical Workstation Tools](#medical-workstation-tools)
  - [Risk Analysis HUD](#risk-analysis-hud)
- [Setup & Running](#setup--running)
- [Training Models](#training-models)

---

## Project Overview

**Domain:** Medical AI / Neurovascular Analysis / Explainable AI

**Problem:** Given a 3D mesh of an intracranial artery, can we:
1. Predict how likely the artery is to develop an aneurysm (risk score 0-1)?
2. Generate a hypothetical "healed" version showing what a lower-risk geometry would look like?
3. Quantify exactly what changed (displacement in mm, volume change, etc.)?

**Key Innovation:** Counterfactual geometry generation - the system doesn't just classify risk, it produces an interpretable "what-if" output: a deformed mesh that preserves the original topology while showing the minimal geometric changes needed to lower risk.

**Named "AneuXplain"** in the UI - explainable AI applied to neurovascular anatomy.

---

## How It Works (End-to-End)

```
User uploads .obj artery mesh
        |
        v
[1. ANALYZE] ──> Sample 2048 points from surface
                  ──> Normalize to unit sphere
                  ──> Feed into RiskPredictorV2 (PointNet encoder + MLP)
                  ──> Output: risk score [0, 1]
                       0.0 = healthy, 1.0 = aneurysm likely
        |
        v
[2. HEAL] ──> If CVAE available:
              |   Encode artery ──> latent vector z
              |   Concatenate z with target_risk (e.g. 0.1)
              |   Decode ──> healed point cloud (one forward pass)
              |
              ──> Else if Autoencoder available:
              |   Encode ──> z_original
              |   Gradient descent on z to minimize risk
              |   While risk > target: z = z - lr * d(risk)/dz
              |   Decode best z ──> healed point cloud
              |
              ──> Transfer displacements to original mesh:
                   Build KD-tree of sampled points
                   For each mesh vertex: find 6 nearest sampled points
                   Apply Gaussian-weighted displacement interpolation
                   Cap max displacement at 8% of unit sphere
                   ──> Output: deformed mesh (.obj) with preserved faces
        |
        v
[3. VISUALIZE] ──> 3D canvas with Three.js
                    3 view modes: Anatomical, Heatmap, Morph
                    Risk HUD with live score updates
                    Measurement tools, clipping planes
        |
        v
[4. EXPORT] ──> JSON report with all metrics
                Healed .obj file download
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      FRONTEND                           │
│   React 19 + Three.js + React Three Fiber               │
│   ┌──────────┐ ┌──────────────┐ ┌───────────────────┐  │
│   │ PACS     │ │ 3D Canvas    │ │ Risk Analysis HUD │  │
│   │ Toolbar  │ │ (WebGL)      │ │ (Expandable)      │  │
│   │          │ │              │ │                   │  │
│   │ Upload   │ │ OBJ Loader   │ │ Rupture %         │  │
│   │ Analyze  │ │ Morph Targets│ │ Risk Level        │  │
│   │ Slice    │ │ Heatmap      │ │ Geometric Deltas  │  │
│   │ Measure  │ │ Clipping     │ │ Counterfactual    │  │
│   │ Heatmap  │ │ Measurement  │ │ Analysis          │  │
│   └──────────┘ └──────────────┘ └───────────────────┘  │
│                 ┌──────────────────┐                     │
│                 │ Morph Control Bar│                     │
│                 │ Pre-Op ◄━━► Post-Op                   │
│                 └──────────────────┘                     │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP (localhost:8000)
                           │ POST /analyze, POST /heal
┌──────────────────────────┴──────────────────────────────┐
│                      BACKEND                            │
│   FastAPI + PyTorch + Trimesh                            │
│   ┌─────────────────────────────────────────────────┐   │
│   │            CounterfactualEngine                  │   │
│   │                                                  │   │
│   │  ┌──────────────┐  ┌────────────┐  ┌─────────┐ │   │
│   │  │ RiskPredictor │  │ Autoencoder│  │  CVAE   │ │   │
│   │  │ V2 (3.3 MB)  │  │ (30 MB)    │  │ (31 MB) │ │   │
│   │  │              │  │            │  │         │ │   │
│   │  │ PointNet     │  │ Encode/    │  │ One-pass│ │   │
│   │  │ Encoder      │  │ Decode     │  │ Healing │ │   │
│   │  │ + MLP Head   │  │ + Latent   │  │ Cond.   │ │   │
│   │  │              │  │ Optim.     │  │ on Risk │ │   │
│   │  └──────────────┘  └────────────┘  └─────────┘ │   │
│   │                                                  │   │
│   │  Mesh Loading ──> Point Sampling ──> Prediction  │   │
│   │  Displacement Transfer ──> KD-Tree Interpolation │   │
│   │  Geometric Deltas ──> Scientific Metrics         │   │
│   └─────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────┴──────────────────────────────┐
│                    DATA & MODELS                        │
│                                                          │
│  IntrA/complete/         ~500 real artery meshes (.obj)  │
│  IntrA/generated/        AI-generated aneurysm + vessel  │
│  data/combined_labels.csv  2012 arteries with risk scores│
│  models/*.pth             Pre-trained weights            │
└─────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Backend
| Technology | Purpose |
|---|---|
| **Python 3.10+** | Runtime |
| **FastAPI** | REST API framework |
| **Uvicorn** | ASGI server |
| **PyTorch 2.0+** | Deep learning (all models) |
| **Trimesh** | 3D mesh loading, surface sampling, export |
| **SciPy** | KD-tree for spatial nearest-neighbor queries |
| **NumPy** | Numerical computation |
| **Pydantic** | Request/response validation |

### Frontend
| Technology | Purpose |
|---|---|
| **React 19** | UI framework |
| **Three.js 0.182** | WebGL 3D rendering engine |
| **React Three Fiber (r3f)** | Declarative Three.js in React |
| **@react-three/drei** | Camera controls, gizmos, grid, shadows |
| **TailwindCSS 4** | Utility-first CSS |
| **Framer Motion** | Animations and transitions |
| **Lucide React** | Icon library |
| **Vite (rolldown-vite)** | Build tool |

### ML / Training
| Technology | Purpose |
|---|---|
| **PyTorch** | Model training and inference |
| **Pandas** | CSV label loading |
| **Matplotlib** | Training visualization |
| **Trimesh** | Point cloud sampling from meshes |

---

## Project Structure

```
CapstoneProject/
│
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI server, endpoints, lifespan
│   │   ├── engine.py            # CounterfactualEngine (core ML logic)
│   │   ├── architecture.py      # V1 model definitions (legacy)
│   │   └── architecture_v2.py   # V2 models: RiskPredictorV2, CVAE, Autoencoder
│   ├── outputs/                 # Generated healed meshes (.obj)
│   ├── temp/                    # Temporary uploaded files
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main app: state, handlers, layout
│   │   ├── main.jsx             # React entry point
│   │   └── components/
│   │       ├── ArteryViewer.jsx  # 3D canvas, camera, lighting, view modes
│   │       ├── Overlay.jsx       # PACSToolbar, MorphControlBar, indicators
│   │       ├── RiskAnalysisHUD.jsx # Expandable risk score card
│   │       ├── MorphArtery.jsx   # GPU morph targets (original <-> healed)
│   │       ├── MorphingMesh.jsx  # CPU vertex interpolation + heatmap
│   │       └── DifferenceMesh.jsx # Displacement heatmap visualization
│   ├── package.json
│   └── vite.config.js
│
├── training/
│   ├── models/
│   │   ├── __init__.py          # Re-exports all architectures + datasets
│   │   ├── architectures.py     # All neural network definitions
│   │   └── datasets.py          # PointCloudDataset, LabeledArteryDataset
│   └── scripts/
│       ├── train_risk_predictor.py  # Train RiskPredictorV2
│       ├── train_autoencoder.py     # Train Autoencoder (unsupervised)
│       ├── train_cvae.py            # Train Conditional VAE
│       └── prepare_labels.py        # Generate risk labels from geometry
│
├── data/
│   ├── combined_labels.csv      # 2012 arteries with computed risk scores
│   └── geometry_labels.csv      # Raw geometric feature values
│
├── IntrA/                       # IntrA Dataset (3D artery meshes)
│   ├── complete/                # ~500 real intracranial arteries (.obj)
│   ├── annotated/               # Annotated clinical subset
│   └── generated/
│       ├── aneurysm/obj/        # Synthetic aneurysm meshes
│       └── vessel/obj/          # Synthetic healthy vessel meshes
│
├── models/                      # Pre-trained weights (deployed)
│   ├── risk_predictor_v2.pth    # Risk prediction model (~3.3 MB)
│   ├── autoencoder_v2.pth       # Point cloud autoencoder (~30 MB)
│   └── cvae.pth                 # Conditional VAE (~31 MB)
│
└── README.md
```

---

## Machine Learning Pipeline

### 1. Data & Labeling

**Dataset:** IntrA (Intracranial Artery) - a collection of 3D surface meshes of intracranial arteries in `.obj` format.

**Sources:**
- `IntrA/complete/` - ~500 real clinical artery meshes
- `IntrA/generated/aneurysm/` - Synthetic aneurysm-like meshes
- `IntrA/generated/vessel/` - Synthetic healthy vessels

**Label Generation** (`training/scripts/prepare_labels.py`):

Risk labels are computed purely from geometric features (no manual annotation):

| Feature | What It Measures |
|---|---|
| `curvature_score` | Mean surface curvature magnitude |
| `curvature_mean` | Average curvature value |
| `curvature_std` | Curvature variability across surface |
| `radius_skew` | Asymmetry of vessel radius distribution |
| `radius_kurtosis` | Peakedness of radius distribution |
| `bulge_p95` | 95th percentile of local bulge detection |
| `normal_variance` | How irregular the surface normals are |

These features are combined into a normalized `risk_score` in [0, 1]:
- **< 0.3** = LOW risk (healthy geometry)
- **0.3 - 0.5** = MODERATE risk
- **0.5 - 0.7** = HIGH risk
- **> 0.7** = CRITICAL risk

**Output:** `data/combined_labels.csv` - 2012 labeled arteries.

### 2. Risk Predictor (RiskPredictorV2)

**Purpose:** Given a 3D artery mesh, predict its aneurysm risk score.

**Architecture:**
```
Input: Point Cloud (B, 3, 2048)
    |
    v
PointNet Encoder:
    Conv1d(3, 64)    + BN + ReLU
    Conv1d(64, 128)  + BN + ReLU
    Conv1d(128, 256) + BN + ReLU
    Conv1d(256, 512) + BN + ReLU
    Conv1d(512, 1024)+ BN + ReLU
    MaxPool (global)
    Linear(1024, 128) ──> Latent Vector (B, 128)
    |
    v
MLP Head:
    Linear(128, 64)  + BN + ReLU + Dropout(0.3)
    Linear(64, 32)   + BN + ReLU + Dropout(0.3)
    Linear(32, 16)   + BN + ReLU + Dropout(0.15)
    Linear(16, 1)    ──> Raw Logit
    |
    v
sigmoid() ──> Risk Probability [0, 1]
```

**Key Design Choices:**
- Outputs **logits** by default (use `return_logits=False` for probabilities)
- BatchNorm after every linear layer for stable training
- 3-layer MLP head (V1 had only 2 layers)
- Designed for **continuous regression**, not binary classification

**Training:**
- **Loss:** SmoothL1Loss (robust to outliers, better than MSE for continuous targets)
- **Optimizer:** Adam (lr=0.0005, weight_decay=1e-4)
- **Scheduler:** Cosine Annealing over 100 epochs
- **Class Balancing:** WeightedRandomSampler (50/50 low-risk vs high-risk batches)
- **Augmentation:** Random Z-rotation, Gaussian jitter (sigma=0.01), random scaling (0.9-1.1)
- **Split:** 80% train / 20% validation (seed=42)
- **Gradient Clipping:** max_norm=1.0

**Metrics tracked:** Loss, Accuracy (threshold=0.5), MAE, Prediction Gap (mean high-risk score - mean low-risk score).

### 3. Autoencoder

**Purpose:** Learn a compressed representation of artery geometry for latent-space optimization (fallback healing method).

**Architecture:**
```
Encoder: Point Cloud (B, 3, 2048) ──> Latent (B, 128)
    Same PointNet encoder as RiskPredictorV2

Decoder: Latent (B, 128) ──> Reconstructed Cloud (B, 3, 2048)
    Linear(128, 256)  + BN + ReLU
    Linear(256, 512)  + BN + ReLU
    Linear(512, 1024) + BN + ReLU
    Linear(1024, 6144) ──> tanh()
    Reshape to (B, 3, 2048)
```

**Training:**
- **Loss:** Chamfer Distance (permutation-invariant point cloud comparison)
- Unsupervised - no labels needed, just reconstruct the input

### 4. Conditional VAE (CVAE)

**Purpose:** One-pass healing - generate a low-risk version of any artery in a single forward pass.

**Architecture:**
```
Encoder: Point Cloud (B, 3, 2048) ──> (mu, logvar) each (B, 128)
    PointNet Encoder outputs 256-dim, split into mu + logvar

Reparameterization:
    z = mu + std * epsilon,  epsilon ~ N(0, I)

Decoder: (z concat risk_label) = (B, 129) ──> Point Cloud (B, 3, 2048)
    Same decoder architecture but input dim = 129 (latent + risk condition)

Healing (generate_healthy):
    mu, _ = encode(sick_artery)
    z_cond = [mu, target_risk=0.1]   # Use mean, skip sampling
    healed = decode(z_cond)
```

**Key Concept:** The decoder is **conditioned on risk**. During training, it learns that `risk=0.1` means "produce healthy geometry" and `risk=0.9` means "produce aneurysm-like geometry". At inference, we encode a sick artery, set `target_risk=0.1`, and decode - producing a healed version.

**Training Loss:**
```
Total Loss = Chamfer(reconstruction, input)
           + beta * KL(q(z|x) || N(0,I))
           + gamma * MSE(predicted_risk_of_reconstruction, label)
```

- **Chamfer:** Ensures reconstruction quality
- **KL Divergence:** Regularizes the latent space (annealed: 0 -> beta_max=0.01 over 20 epochs)
- **Risk Consistency:** The reconstructed artery should have the same risk as the conditioning label. Uses a **frozen** RiskPredictorV2 as evaluation function.

**Hyperparameters:** beta_max=0.01, beta_warmup=20 epochs, gamma=0.5, lr=0.0005, 100 epochs.

---

## Backend API

### CounterfactualEngine

The core class in `backend/app/engine.py`. Encapsulates all ML logic.

**Initialization:**
```python
engine = CounterfactualEngine(
    models_dir="models",    # Directory with .pth files
    output_dir="outputs",   # Where healed meshes are saved
    num_points=2048,        # Points sampled from each mesh
    latent_dim=128          # Latent dimension of all models
)
engine.load_models()  # Auto-detects V1 vs V2 models
```

**Model Loading Priority:**
1. Risk Predictor: `risk_predictor_v2.pth` > `risk_predictor.pth`
2. Autoencoder: `autoencoder_v2.pth` > `artery_autoencoder.pth`
3. CVAE: `cvae.pth` (optional, enables one-pass healing)

**Auto-detection:** V2 models are detected by checking for `mlp_head.8.weight` key in the state dict (V2 has deeper MLP head with more layers).

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info and endpoint list |
| GET | `/health` | Health check, device info, model status |
| POST | `/analyze` | Upload mesh -> receive risk score, level, interpretation |
| POST | `/heal` | Upload mesh -> receive healed mesh + metrics |
| GET | `/heal/{file_id}` | Download a previously generated healed mesh |

**`POST /analyze`**
- Input: Multipart file upload (`.obj`, `.ply`, `.stl`, `.off`)
- Output:
```json
{
  "risk_score": 0.7234,
  "risk_level": "HIGH",
  "interpretation": "This artery shows significant risk factors. Medical consultation advised."
}
```

**`POST /heal`**
- Input: Multipart file upload + query params
  - `num_steps` (50-1000, default 300): Optimization steps (latent method only)
  - `target_risk` (0.1-0.5, default 0.3): Desired risk level
  - `return_file` (bool, default false): Return `.obj` file vs JSON
- Output (JSON mode):
```json
{
  "initial_risk": 0.7234,
  "final_risk": 0.2100,
  "risk_reduction": 0.5134,
  "risk_reduction_pct": 71.0,
  "success": true,
  "steps_taken": 1,
  "mean_movement": 0.023456,
  "message": "Successfully transformed artery from HIGH to LOW risk!",
  "max_displacement_mm": 0.412,
  "mean_displacement_mm": 0.089,
  "displacement_std_mm": 0.067,
  "volume_change_pct": null,
  "surface_area_change_pct": null
}
```

### Healing Algorithm

**Priority: CVAE > Autoencoder Latent Optimization**

#### Method 1: CVAE (One-Pass)
```
1. Encode sick artery: mu, _ = cvae.encode(points)
2. Set target: risk_label = 0.1
3. Decode: healed = cvae.decode(mu, risk_label)
4. Done in 1 forward pass (~100ms)
```

#### Method 2: Latent-Space Gradient Descent
```
1. Encode: z_original = autoencoder.encode(points)
2. z_opt = z_original (detached, requires_grad=True)
3. For each step:
   a. Decode: recon = autoencoder.decode(z_opt)
   b. Compute: risk = risk_predictor(recon)
   c. Loss = risk + lambda * ||z_opt - z_original||²
   d. Backprop through frozen models, update z_opt
   e. Track best_z (lowest risk)
   f. Early stop if risk < target_risk
4. Final decode: healed = autoencoder.decode(best_z)
```

#### Displacement Transfer (Both Methods)
The healing produces a point cloud, but we need a mesh with faces. The transfer algorithm:

```
1. Load original mesh with full face topology
2. Normalize mesh vertices (same as preprocessing)
3. Build KD-tree of the 2048 sampled original points
4. Compute displacement field: delta = healed_points - original_points
5. Cap max displacement at 8% of unit sphere radius
6. For each mesh vertex:
   a. Find 6 nearest sampled points (KD-tree query)
   b. Compute Gaussian kernel weights: w = exp(-d²/2σ²)
     where σ = mean nearest-neighbor distance in the sampled points
   c. Weighted average displacement from neighbors
   d. Apply: new_vertex = vertex + weighted_displacement
7. Denormalize back to original scale
8. Export as .obj (preserving original face connectivity)
```

---

## Frontend UI

### 3D Viewer (`ArteryViewer.jsx`)

**Rendering Setup:**
- **Camera:** Orthographic (no perspective distortion - medical precision)
- **Lighting:** Ambient (0.5) + 2 Directional + 2 Point lights (cyan/indigo tints)
- **Environment:** HDR city environment map for realistic reflections
- **Grid:** Infinite faded grid for spatial reference
- **Shadows:** Contact shadows for grounding
- **Controls:** OrbitControls with damping, zoom-to-cursor, bounded distance

**Model Loading:**
- Uses Three.js `OBJLoader` to load uploaded meshes
- Auto-centers and scales to fit viewport (normalized to 2 units)
- Applies custom materials with clipping plane support

### View Modes

Three view modes available after healing:

1. **Anatomical View** (default)
   - Original mesh: Red (solid or wireframe)
   - Healed mesh: Green (50% transparent)
   - Overlaid to show where the aneurysm bulges beyond the healed shape
   - Toggle wireframe for the original to see through

2. **Heatmap View** (`DifferenceMesh.jsx`)
   - Single mesh colored by displacement magnitude
   - Color gradient: Blue (no change) -> Cyan -> Green -> Yellow -> Red (high change)
   - Uses spatial hash grid for O(1) nearest-neighbor lookup
   - Blue = healthy tissue that stayed the same, Red = aneurysm site that shrank

3. **Morph View** (`MorphArtery.jsx`)
   - GPU-accelerated morph targets (Three.js `morphTargetInfluences`)
   - Single mesh smoothly interpolates between original and healed
   - Morph delta stored as relative offset: `morphDelta = healed - original`
   - Color transitions: red (t=0) -> yellow -> green (t=1) via HSL interpolation
   - Handles different vertex counts via nearest-neighbor correspondence

### Medical Workstation Tools

**PACS Toolbar** (left sidebar):
- **Upload:** File picker for `.obj` meshes
- **Analyze:** Run risk prediction on loaded mesh
- **Slice:** Toggle clipping plane with adjustable Y position (-2 to 2)
- **Measure:** Click two points on mesh surface to measure distance (mm)
- **Heatmap:** Toggle displacement heatmap coloring
- **Export:** Download JSON report with all metrics

**Morph Control Bar** (bottom):
- Pre-Op / Post-Op slider (t = 0.0 to 1.0)
- Status labels: ORIGINAL / MORPHING / HEALED
- "Generate Counterfactual" button to trigger healing
- Color gradient from red (original) to green (healed)

**Measurement Tool:**
- Click two points on the 3D mesh surface
- Displays distance in mm with a dashed line
- Uses Three.js raycasting for precise point picking
- Cursor changes to crosshair in measurement mode

**Clipping Plane:**
- Adjustable Y-axis clipping plane for cross-section views
- Semi-transparent cyan plane visualization
- Vertical slider in toolbar for position control

### Risk Analysis HUD (`RiskAnalysisHUD.jsx`)

Expandable floating card in top-right corner:

**Collapsed State:**
- Large percentage display with risk-colored text
- Risk level badge (LOW/MODERATE/HIGH/CRITICAL)
- Color-coded glow shadow matching risk level
- Pulsing border animation for CRITICAL risk

**Expanded State (click to expand):**
- Peak Wall Stress estimate (kPa)
- Geometry Complexity Index
- Confidence Interval
- Neck-to-Dome Ratio
- Counterfactual Analysis section (after healing):
  - Peak Displacement (mm)
  - Mean Displacement (mm)
  - Risk Reduction (%)

**Risk Interpolation:**
When using the morph slider, the displayed risk score interpolates linearly between original and healed risk, with "(Est.)" indicator shown during interpolation.

---

## Setup & Running

### Prerequisites
- Python 3.10+
- Node.js 18+
- CUDA-capable GPU (optional, falls back to CPU)

### Backend Setup
```bash
cd backend
pip install -r requirements.txt

# Start API server
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

The server automatically loads models from `models/` on startup.

### Frontend Setup
```bash
cd frontend
npm install

# Start dev server
npm run dev
```

Opens at `http://localhost:5173`, connects to API at `http://localhost:8000`.

### Required Model Files
Place these in the `models/` directory:
- `risk_predictor_v2.pth` (required) - Risk prediction
- `cvae.pth` (recommended) - One-pass healing
- `autoencoder_v2.pth` (optional) - Fallback healing via latent optimization

---

## Training Models

All training scripts are run from the project root:

### 1. Generate Labels (if needed)
```bash
python -m training.scripts.prepare_labels
```
Computes geometric features for all meshes and outputs `data/combined_labels.csv`.

### 2. Train Risk Predictor
```bash
python -m training.scripts.train_risk_predictor
```
Config: 2048 points, latent_dim=128, batch_size=16, lr=0.0005, 100 epochs.
Best model auto-deployed to `models/risk_predictor_v2.pth`.

### 3. Train Autoencoder
```bash
python -m training.scripts.train_autoencoder
```
Unsupervised training using Chamfer Distance loss.

### 4. Train CVAE (requires trained risk predictor)
```bash
python -m training.scripts.train_cvae
```
Config: Same base + beta_max=0.01, beta_warmup=20, gamma=0.5.
Uses frozen risk predictor for risk consistency loss.
Best model auto-deployed to `models/cvae.pth`.

### Training Output
- Checkpoints saved to `training/checkpoints/`
- Training plots saved as PNG (loss, accuracy, MAE, gap curves)
- Best models automatically copied to `models/` for backend use
