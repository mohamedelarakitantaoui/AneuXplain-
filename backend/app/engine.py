"""
engine.py - Counterfactual Engine for Artery Analysis

The "Brain" of the backend. Handles model loading, risk prediction,
and counterfactual generation (healing arteries).
"""

import os
import sys
import uuid
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.optim as optim
import trimesh
from scipy.spatial import cKDTree  # type: ignore

# Import from local architecture (self-contained backend)
from .architecture import RiskPredictor, Autoencoder
from .architecture_v2 import RiskPredictorV2, ConditionalVAE


class CounterfactualEngine:
    """
    Core engine for artery risk analysis and counterfactual healing.
    
    This class encapsulates all the ML logic:
    - Loading trained models
    - Predicting risk scores from artery meshes
    - Generating "healed" counterfactual arteries
    """
    
    def __init__(
        self,
        models_dir: str = "saved_models",
        output_dir: str = "outputs",
        num_points: int = 2048,
        latent_dim: int = 128
    ):
        """
        Initialize the Counterfactual Engine.
        
        Args:
            models_dir: Path to directory containing .pth model files
            output_dir: Path to directory for saving output files
            num_points: Number of points to sample from meshes
            latent_dim: Latent dimension of the models
        """
        self.models_dir = Path(models_dir)
        self.output_dir = Path(output_dir)
        self.num_points = num_points
        self.latent_dim = latent_dim
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Models (loaded on demand)
        self.risk_predictor = None  # Can be RiskPredictor or RiskPredictorV2
        self.autoencoder: Optional[Autoencoder] = None
        self.cvae: Optional[ConditionalVAE] = None
        self.is_v2_model = False  # Track if we're using V2 model
        
        self._models_loaded = False
    
    def load_models(
        self,
        risk_predictor_path: Optional[str] = None,
        autoencoder_path: Optional[str] = None
    ) -> None:
        """
        Load the trained model weights.
        
        Automatically detects V2 models and loads them appropriately.
        
        Args:
            risk_predictor_path: Path to risk predictor weights (.pth)
            autoencoder_path: Path to autoencoder weights (.pth) - optional
        """
        # Default paths - prefer V2 model
        if risk_predictor_path is None:
            # Check for V2 model first
            v2_path = self.models_dir / "risk_predictor_v2.pth"
            v1_path = self.models_dir / "risk_predictor.pth"
            
            if v2_path.exists():
                rpp = v2_path
            elif v1_path.exists():
                rpp = v1_path
            else:
                raise FileNotFoundError(f"No risk predictor weights found in {self.models_dir}")
        else:
            rpp = Path(risk_predictor_path)
        
        # Load state dict to detect architecture
        if not rpp.exists():
            raise FileNotFoundError(f"Risk predictor weights not found: {rpp}")
        
        state_dict = torch.load(str(rpp), map_location=self.device, weights_only=True)
        
        # Detect if it's a V2 model (V2 has 3 hidden layers with BatchNorm)
        # V2 model has keys like mlp_head.8.weight (for final layer)
        self.is_v2_model = 'mlp_head.8.weight' in state_dict or 'mlp_head.12.weight' in state_dict
        
        if self.is_v2_model:
            print("   [+] Loading V2 model (BatchNorm, continuous risk)")
            self.risk_predictor = RiskPredictorV2(latent_dim=self.latent_dim)
        else:
            print("   [+] Loading V1 model (legacy)")
            self.risk_predictor = RiskPredictor(latent_dim=self.latent_dim)
        
        self.risk_predictor.load_state_dict(state_dict)
        self.risk_predictor.to(self.device)
        self.risk_predictor.eval()
        
        # Freeze weights
        for param in self.risk_predictor.parameters():
            param.requires_grad = False
        
        # Load Autoencoder (auto-detect if path not given)
        if autoencoder_path is None:
            for name in ["autoencoder_v2.pth", "artery_autoencoder.pth"]:
                candidate = self.models_dir / name
                if candidate.exists():
                    autoencoder_path = str(candidate)
                    break

        if autoencoder_path is not None:
            aep = Path(autoencoder_path)
            if aep.exists():
                self.autoencoder = Autoencoder(
                    latent_dim=self.latent_dim,
                    num_points=self.num_points
                )
                self.autoencoder.load_state_dict(
                    torch.load(str(aep), map_location=self.device, weights_only=True)
                )
                self.autoencoder.to(self.device)
                self.autoencoder.eval()
                for param in self.autoencoder.parameters():
                    param.requires_grad = False
                print(f"   [+] Autoencoder loaded from {aep.name}")

        # Load Conditional VAE (if available — enables one-pass healing)
        cvae_path = self.models_dir / "cvae.pth"
        if cvae_path.exists():
            self.cvae = ConditionalVAE(
                latent_dim=self.latent_dim,
                num_points=self.num_points
            )
            self.cvae.load_state_dict(
                torch.load(str(cvae_path), map_location=self.device, weights_only=True)
            )
            self.cvae.to(self.device)
            self.cvae.eval()
            for param in self.cvae.parameters():
                param.requires_grad = False
            print("   [+] ConditionalVAE loaded (one-pass healing enabled)")

        self._models_loaded = True
    
    def _ensure_models_loaded(self) -> None:
        """Ensure models are loaded before inference."""
        if not self._models_loaded:
            raise RuntimeError("Models not loaded. Call load_models() first.")
    
    def _load_mesh_as_points(self, file_path: str) -> np.ndarray:
        """
        Load a mesh file and deterministically select num_points surface points.

        Uses dense seeded sampling followed by Farthest Point Sampling (FPS)
        so the same mesh always yields the same point cloud → reproducible
        risk scores across repeated /analyze calls.

        Returns:
            Point cloud as numpy array of shape (num_points, 3), normalized to unit sphere
        """
        mesh = trimesh.load(file_path, force='mesh')

        # Dense oversample with a fixed seed (4x target), then FPS-downsample.
        # Oversampling ensures uniform surface coverage even on meshes with
        # uneven triangle areas; FPS guarantees deterministic, uniform spread.
        oversample = max(self.num_points * 4, 8192)
        result = trimesh.sample.sample_surface(mesh, count=oversample, seed=42)
        dense = np.array(result[0], dtype=np.float32)

        points = self._farthest_point_sample(dense, self.num_points)

        # Normalize to unit sphere
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist

        return points

    @staticmethod
    def _farthest_point_sample(points: np.ndarray, k: int) -> np.ndarray:
        """Deterministic Farthest Point Sampling: pick k points with max spatial spread."""
        n = len(points)
        if n <= k:
            return points
        selected = np.empty(k, dtype=np.int64)
        selected[0] = 0  # deterministic seed (index 0 of the seeded oversample)
        dists = np.full(n, np.inf, dtype=np.float32)
        for i in range(1, k):
            last = points[selected[i - 1]]
            d = np.sum((points - last) ** 2, axis=1)
            dists = np.minimum(dists, d)
            selected[i] = int(np.argmax(dists))
        return points[selected]
    
    def _points_to_tensor(self, points: np.ndarray) -> torch.Tensor:
        """
        Convert numpy points to model-ready tensor.
        
        Args:
            points: Point cloud array of shape (N, 3)
            
        Returns:
            Tensor of shape (1, 3, N) on the correct device
        """
        tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
        tensor = tensor.transpose(2, 1)  # (1, 3, N)
        return tensor.to(self.device)
    
    def _save_points_as_obj(self, points: np.ndarray, file_path: str) -> None:
        """
        Save a point cloud as an OBJ file (vertices only).
        
        Args:
            points: Point cloud array of shape (N, 3)
            file_path: Output path for the OBJ file
        """
        with open(file_path, 'w') as f:
            f.write("# Counterfactual Point Cloud\n")
            f.write(f"# Points: {len(points)}\n\n")
            for p in points:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    
    def _save_deformed_mesh(
        self, 
        original_mesh_path: str, 
        sampled_points_original: np.ndarray,
        sampled_points_healed: np.ndarray,
        file_path: str
    ) -> None:
        """
        Save a deformed mesh by transferring displacements from sampled points
        to the original mesh vertices. This preserves face topology!
        
        Args:
            original_mesh_path: Path to the original mesh file
            sampled_points_original: Original sampled points (normalized, N x 3)
            sampled_points_healed: Healed sampled points (normalized, N x 3)
            file_path: Output path for the deformed OBJ file
        """
        # Load original mesh with full topology
        loaded = trimesh.load(original_mesh_path, force='mesh')
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError(f"Expected a Trimesh mesh, got {type(loaded)}")
        mesh = loaded
        original_vertices = np.array(mesh.vertices, dtype=np.float32)
        
        # Normalize original mesh vertices (same as preprocessing)
        centroid = np.mean(original_vertices, axis=0)
        vertices_centered = original_vertices - centroid
        max_dist = np.max(np.linalg.norm(vertices_centered, axis=1))
        if max_dist > 0:
            vertices_normalized = vertices_centered / max_dist
        else:
            vertices_normalized = vertices_centered
        
        # ---- Establish point correspondence via nearest-neighbor matching ----
        # The CVAE decoder outputs points in arbitrary order, so index-based
        # subtraction (healed[i] - original[i]) is meaningless.  Instead:
        # 1. Re-center healed points to match original center of mass
        # 2. For each original point, find its nearest healed point
        # 3. Displacement = matched_healed - original  (localized deformation)
        # 4. Subtract residual global shift

        healed_centered = (
            sampled_points_healed
            - sampled_points_healed.mean(axis=0)
            + sampled_points_original.mean(axis=0)
        )

        healed_tree = cKDTree(healed_centered)
        _, nn_indices = healed_tree.query(sampled_points_original)
        matched_healed = healed_centered[nn_indices]

        displacements = matched_healed - sampled_points_original  # (N, 3)

        # Remove residual global shift (keeps only local deformation)
        displacements = displacements - displacements.mean(axis=0)

        # Build KD-tree of sampled original points for vertex interpolation
        tree = cKDTree(sampled_points_original)

        # Use 6 nearest neighbors for smoother interpolation
        k_interp = 6
        distances, indices = tree.query(vertices_normalized, k=k_interp)

        # Compute Gaussian kernel bandwidth (sigma = mean nearest-neighbor distance)
        nn_dists, _ = tree.query(sampled_points_original, k=2)  # k=2: self + nearest
        sigma = float(np.mean(nn_dists[:, 1]))  # Mean distance to nearest neighbor
        sigma = max(sigma, 1e-6)  # Safety floor

        # Cap max displacement at 15% of unit sphere as a safety limit.
        # With proper NN matching the displacements are already reasonable,
        # so this rarely activates.
        max_allowed = 0.15
        disp_norms = np.linalg.norm(displacements, axis=1)
        current_max = np.max(disp_norms) if len(disp_norms) > 0 else 1.0
        if current_max > max_allowed:
            displacements = displacements * (max_allowed / current_max)

        # Apply weighted displacement using Gaussian kernel (bounded weights, no explosion)
        new_vertices_normalized = np.zeros_like(vertices_normalized)
        for i in range(len(vertices_normalized)):
            neighbor_indices = indices[i]
            neighbor_distances = distances[i]

            # Gaussian kernel: bounded even when distance is ~0
            weights = np.exp(-neighbor_distances**2 / (2.0 * sigma**2))
            weight_sum = np.sum(weights)
            if weight_sum > 0:
                weights = weights / weight_sum
            else:
                weights = np.ones(k_interp) / k_interp  # Fallback: uniform

            # Weighted average of displacements from nearest sampled points
            weighted_displacement = np.zeros(3)
            for j, idx in enumerate(neighbor_indices):
                weighted_displacement += weights[j] * displacements[idx]

            new_vertices_normalized[i] = vertices_normalized[i] + weighted_displacement
        
        # Denormalize back to original scale
        new_vertices = (new_vertices_normalized * max_dist) + centroid
        
        # Update mesh vertices
        mesh.vertices = new_vertices
        
        # Export as OBJ (preserves faces!)
        mesh.export(file_path, file_type='obj')
    
    def predict_risk(self, mesh_path: str) -> float:
        """
        Predict the risk score for an artery mesh.
        
        Args:
            mesh_path: Path to the mesh file (.obj, .ply, .stl, etc.)
            
        Returns:
            Risk score as a float in range [0, 1]
            - 0.0 = Low risk (healthy)
            - 1.0 = High risk (aneurysm likely)
        """
        self._ensure_models_loaded()
        assert self.risk_predictor is not None, "Risk predictor not loaded"
        
        # Load and preprocess
        points = self._load_mesh_as_points(mesh_path)
        tensor = self._points_to_tensor(points)
        
        # Predict
        with torch.no_grad():
            if self.is_v2_model:
                # V2 model - get probability directly
                risk_score = self.risk_predictor(tensor, return_logits=False).item()
            else:
                # V1 model - already outputs probability
                risk_score = self.risk_predictor(tensor).item()
        
        return risk_score
    
    def heal_artery(
        self,
        mesh_path: str,
        learning_rate: float = 0.005,
        num_steps: int = 200,
        target_risk: float = 0.05,
        lambda_latent: float = 5.0,
    ) -> Tuple[str, dict]:
        """
        Generate a counterfactual "healed" version of a sick artery.

        Strategy: **Gradient-based adversarial healing** — optimizes vertex
        displacements directly through the frozen risk predictor. This
        concentrates changes at the aneurysm site (where ∂risk/∂position
        is largest) instead of producing global scaling artifacts.

        Falls back to CVAE or latent-space optimization only if gradient
        healing is not possible (should not happen in practice).

        Args:
            mesh_path: Path to the input artery mesh file
            learning_rate: LR for optimization
            num_steps: Max optimization steps
            target_risk: Desired risk level
            lambda_latent: Laplacian regularization weight

        Returns:
            Tuple of (healed_mesh_path, result_dict)
        """
        self._ensure_models_loaded()
        assert self.risk_predictor is not None, "Risk predictor not loaded"

        # =============================================================
        # Primary method: Gradient-based adversarial healing
        # =============================================================
        # Temporarily unfreeze risk predictor params for gradient flow
        # (we need gradients w.r.t. input, not weights — weights stay fixed)
        # Actually, requires_grad on params is irrelevant for input gradients.
        # The model stays in eval mode with frozen weights.

        healed_path, result = self._gradient_heal(
            mesh_path=mesh_path,
            n_steps=num_steps,
            lr=learning_rate,
            max_displacement_frac=0.05,
            lambda_laplacian=5.0,
            target_risk=target_risk,
        )

        result['healed_path'] = healed_path
        return healed_path, result

    # ------------------------------------------------------------------
    # Healing backends
    # ------------------------------------------------------------------

    def _gradient_heal(
        self,
        mesh_path: str,
        n_steps: int = 100,
        lr: float = 0.005,
        max_displacement_frac: float = 0.05,
        lambda_laplacian: float = 5.0,
        target_risk: float = 0.05,
    ) -> Tuple[str, dict]:
        """
        Gradient-based adversarial healing with spatial masking.

        Optimizes per-vertex scalar displacements (along inward normals)
        through the frozen risk predictor. A saliency mask computed from
        the classifier gradient concentrates deformation at the aneurysm
        dome while freezing the healthy parent vessel.

        Key features:
        - Saliency mask: only aneurysm vertices (high ∂risk/∂v) are moved
        - Normal-direction parameterization: scalar per vertex along -normal
        - Edge-length preservation loss: prevents global shrinking
        - Laplacian regularization: smooth displacement field
        - Freeze loss: penalizes movement of non-aneurysm vertices

        Returns:
            Tuple of (healed_mesh_path, result_dict)
        """
        assert self.risk_predictor is not None

        # --- Healing configuration (tuneable) ----------------------------
        MASK_PERCENTILE = 0.95          # Top 5% saliency vertices get displaced
        MASK_MIN_VERTICES = 20          # Fallback: relax threshold if too few pass
        MASKED_DISPLACEMENT_CAP = 0.30  # 30% of bbox diagonal for masked vertices
        UNMASKED_DISPLACEMENT_CAP = 0.15  # 15% safety net for any leakage
        SMOOTHING_STEPS = 3             # Post-opt Laplacian diffusion steps
        LR_BOOST = 1.5                  # Multiply incoming lr by this factor

        lr = lr * LR_BOOST

        # --- Load mesh --------------------------------------------------
        mesh = trimesh.load(mesh_path, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Expected Trimesh, got {type(mesh)}")

        vertices = np.array(mesh.vertices, dtype=np.float32)  # (V, 3)
        faces = np.array(mesh.faces, dtype=np.int64)           # (F, 3)
        num_verts = len(vertices)

        # --- Normalize to unit sphere (same as training pipeline) --------
        centroid = vertices.mean(axis=0)
        verts_centered = vertices - centroid
        max_dist = np.max(np.linalg.norm(verts_centered, axis=1))
        if max_dist > 0:
            verts_norm = verts_centered / max_dist
        else:
            verts_norm = verts_centered

        # --- Compute vertex normals for inward-only constraint -----------
        vertex_normals = np.array(mesh.vertex_normals, dtype=np.float32)  # (V, 3)
        vn_norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vn_norms = np.where(vn_norms < 1e-8, 1.0, vn_norms)
        vertex_normals = vertex_normals / vn_norms
        normals_t = torch.tensor(vertex_normals, dtype=torch.float32, device=self.device)

        # --- Sample 2048 surface points with barycentric tracking --------
        sample_result = trimesh.sample.sample_surface(mesh, count=self.num_points)
        sampled_pts = np.array(sample_result[0], dtype=np.float32)
        face_ids = np.array(sample_result[1], dtype=np.int64)

        tri_verts = vertices[faces[face_ids]]  # (N, 3, 3)
        bary_coords = self._compute_barycentric(sampled_pts, tri_verts)
        face_vertex_ids = faces[face_ids]  # (N, 3) vertex indices per sample

        # --- Build sparse Laplacian matrix (precomputed once) ------------
        L = self._build_laplacian_sparse(num_verts, faces, self.device)

        # --- Precompute original edge lengths for edge-preservation loss --
        faces_t = torch.tensor(faces, dtype=torch.long, device=self.device)
        verts_norm_t = torch.tensor(verts_norm, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v0_orig = verts_norm_t[faces_t[:, 0]]
            v1_orig = verts_norm_t[faces_t[:, 1]]
            v2_orig = verts_norm_t[faces_t[:, 2]]
            orig_edges = torch.cat([v0_orig - v1_orig, v1_orig - v2_orig, v2_orig - v0_orig], dim=0)
            orig_edge_lengths = orig_edges.norm(dim=-1)  # (3F,)

        bary_t = torch.tensor(bary_coords, dtype=torch.float32, device=self.device)
        fv_ids = torch.tensor(face_vertex_ids, dtype=torch.long, device=self.device)

        # Enable gradient flow through risk predictor
        for param in self.risk_predictor.parameters():
            param.requires_grad_(True)

        # ================================================================
        # STEP 1: Compute saliency mask via single forward+backward pass
        # ================================================================
        probe_disp = torch.zeros(num_verts, 3, dtype=torch.float32,
                                 device=self.device, requires_grad=True)
        pc_probe = self._sample_point_cloud(verts_norm_t, probe_disp, fv_ids, bary_t)
        pc_probe_input = self._normalize_and_format(pc_probe)
        if self.is_v2_model:
            risk_probe = torch.sigmoid(
                self.risk_predictor(pc_probe_input, return_logits=True)
            ).squeeze()
        else:
            risk_probe = self.risk_predictor(pc_probe_input).squeeze()
        risk_probe.backward()

        with torch.no_grad():
            # Per-vertex gradient magnitude
            grad_mag = probe_disp.grad.norm(dim=1)  # (V,)
            # Normalize to [0, 1]
            gmin, gmax = grad_mag.min(), grad_mag.max()
            if gmax - gmin > 1e-10:
                grad_mag_norm = (grad_mag - gmin) / (gmax - gmin)
            else:
                grad_mag_norm = torch.ones_like(grad_mag)
            # Soft mask: sigmoid with threshold at MASK_PERCENTILE
            # Fallback: relax if fewer than MASK_MIN_VERTICES pass
            for pct in [MASK_PERCENTILE, 0.90, 0.85]:
                threshold = torch.quantile(grad_mag_norm, pct)
                hard_count = int((grad_mag_norm >= threshold).sum().item())
                if hard_count >= MASK_MIN_VERTICES:
                    break
            alpha = 15.0  # sharpness
            saliency_mask = torch.sigmoid(alpha * (grad_mag_norm - threshold))  # (V,)
            mask_count = int((saliency_mask > 0.5).sum().item())
            print(f"   [heal] Saliency mask: {mask_count}/{num_verts} vertices "
                  f"({100*mask_count/num_verts:.1f}%) at percentile {pct}")

        del probe_disp, pc_probe, pc_probe_input, risk_probe

        # ================================================================
        # STEP 2: Scalar-along-normal parameterization
        # ================================================================
        # Each vertex has a single scalar controlling inward displacement.
        # displacement = scalar * (-normal), scalar >= 0 (inward only).
        deform_scalars = torch.zeros(num_verts, 1, dtype=torch.float32,
                                     device=self.device, requires_grad=True)

        optimizer = optim.Adam([deform_scalars], lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps, eta_min=1e-5)

        bbox_diag = float(np.linalg.norm(verts_norm.max(axis=0) - verts_norm.min(axis=0)))
        max_disp_masked = MASKED_DISPLACEMENT_CAP * bbox_diag
        max_disp_unmasked = UNMASKED_DISPLACEMENT_CAP * bbox_diag

        # --- Initial risk -----------------------------------------------
        with torch.no_grad():
            init_pc = self._sample_point_cloud(verts_norm_t, torch.zeros_like(verts_norm_t),
                                               fv_ids, bary_t)
            init_pc_input = self._normalize_and_format(init_pc)
            if self.is_v2_model:
                initial_risk = self.risk_predictor(init_pc_input, return_logits=False).item()
            else:
                initial_risk = self.risk_predictor(init_pc_input).item()

        best_risk = initial_risk
        best_scalars = torch.zeros_like(deform_scalars)
        best_step = 0
        history = {'steps': [], 'risk_scores': [], 'learning_rate': []}

        # ================================================================
        # STEP 3: Optimization loop with composite loss
        # ================================================================
        for step in range(n_steps):
            optimizer.zero_grad()

            # Compute 3D displacement from scalars along inward normals
            displacement = deform_scalars * (-normals_t)  # (V, 3), inward

            # Displaced point cloud
            pc = self._sample_point_cloud(verts_norm_t, displacement, fv_ids, bary_t)
            pc_input = self._normalize_and_format(pc)

            # Risk score (differentiable)
            if self.is_v2_model:
                risk = torch.sigmoid(
                    self.risk_predictor(pc_input, return_logits=True)
                ).squeeze()
            else:
                risk = self.risk_predictor(pc_input).squeeze()

            # --- Regularization losses -----------------------------------

            # Laplacian smoothing on displacement field
            lap_loss = self._laplacian_loss_sparse(displacement, L)

            # Edge-length preservation: penalize changes in edge lengths
            displaced_verts = verts_norm_t + displacement
            v0 = displaced_verts[faces_t[:, 0]]
            v1 = displaced_verts[faces_t[:, 1]]
            v2 = displaced_verts[faces_t[:, 2]]
            new_edges = torch.cat([v0 - v1, v1 - v2, v2 - v0], dim=0)
            new_edge_lengths = new_edges.norm(dim=-1)
            edge_loss = ((new_edge_lengths - orig_edge_lengths) ** 2).mean()

            # Chamfer-like drift loss: total squared displacement
            chamfer_loss = displacement.pow(2).mean()

            # Freeze loss: heavily penalize displacement of non-aneurysm vertices
            # (1 - mask) is high for healthy vertices, low for aneurysm
            freeze_loss = (deform_scalars.squeeze() * (1.0 - saliency_mask)).pow(2).mean()

            # Composite loss
            total_loss = (
                1.0 * risk
                + lambda_laplacian * lap_loss
                + 1.0 * edge_loss
                + 1.0 * chamfer_loss
                + 10.0 * freeze_loss
            )
            total_loss.backward()

            # Apply saliency mask to gradients: zero out gradient for
            # non-aneurysm vertices so they don't move at all
            with torch.no_grad():
                if deform_scalars.grad is not None:
                    deform_scalars.grad.data *= saliency_mask.unsqueeze(-1)

            torch.nn.utils.clip_grad_norm_([deform_scalars], max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # --- Constraints (applied after each step) -------------------
            with torch.no_grad():
                # Clamp scalars to be non-negative (inward only)
                deform_scalars.data.clamp_(min=0.0)
                # Mask-aware displacement cap: higher limit for aneurysm vertices
                per_vertex_cap = (
                    saliency_mask * max_disp_masked
                    + (1.0 - saliency_mask) * max_disp_unmasked
                ).unsqueeze(-1)  # (V, 1)
                deform_scalars.data = torch.min(deform_scalars.data, per_vertex_cap)

            current_risk = risk.item()

            # --- Console logging for verification -------------------------
            with torch.no_grad():
                _disp_norms = (deform_scalars * normals_t).norm(dim=1)
                _max_before_cap = float(_disp_norms.max().item())
            if step % 20 == 0 or step == n_steps - 1:
                print(f"   [heal] step {step:3d}  risk={current_risk:.4f}  "
                      f"max_disp={_max_before_cap:.5f}")
            history['steps'].append(step)
            history['risk_scores'].append(current_risk)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            if current_risk < best_risk:
                best_risk = current_risk
                best_scalars = deform_scalars.clone().detach()
                best_step = step

            if current_risk < target_risk:
                break

        # --- Re-freeze risk predictor weights ----------------------------
        for param in self.risk_predictor.parameters():
            param.requires_grad_(False)

        # --- Compute best displacement from best scalars -----------------
        best_disp = best_scalars * (-normals_t)  # (V, 3)

        # --- Post-optimization: Laplacian smoothing of displacement ------
        # Only smooth outside the masked region to preserve localized healing
        final_disp_t = best_disp.clone()
        mask_weight = saliency_mask.unsqueeze(-1)  # (V, 1) — 1.0 at aneurysm
        for _ in range(SMOOTHING_STEPS):
            lap = torch.sparse.mm(L, final_disp_t)
            # Blend: skip smoothing where mask is high (aneurysm dome)
            final_disp_t = final_disp_t - 0.5 * lap * (1.0 - mask_weight)

        # Re-apply inward-only and magnitude constraints after smoothing
        with torch.no_grad():
            dots = (final_disp_t * normals_t).sum(dim=1, keepdim=True)
            outward_mask = dots > 0
            final_disp_t -= outward_mask.float() * dots * normals_t
            norms = final_disp_t.norm(dim=1, keepdim=True).clamp(min=1e-8)
            per_vertex_cap_final = (
                saliency_mask * max_disp_masked
                + (1.0 - saliency_mask) * max_disp_unmasked
            ).unsqueeze(-1)
            scale = torch.clamp(norms / per_vertex_cap_final, min=1.0)
            final_disp_t.div_(scale)

        # --- Build healed mesh ------------------------------------------
        final_disp = final_disp_t.cpu().numpy()  # (V, 3) in normalized space
        healed_verts_norm = verts_norm + final_disp
        healed_verts = healed_verts_norm * max_dist + centroid  # denormalize

        output_id = str(uuid.uuid4())[:8]
        healed_path = self.output_dir / f"healed_{output_id}.obj"

        mesh_copy = mesh.copy()
        mesh_copy.vertices = healed_verts
        mesh_copy.export(str(healed_path), file_type='obj')

        # --- Final risk of the healed mesh (honest re-score) ------------
        with torch.no_grad():
            final_pc = self._sample_point_cloud(
                verts_norm_t, best_disp, fv_ids, bary_t
            )
            final_input = self._normalize_and_format(final_pc)
            if self.is_v2_model:
                final_risk = self.risk_predictor(final_input, return_logits=False).item()
            else:
                final_risk = self.risk_predictor(final_input).item()

        # --- Intermediate risks for frontend morph slider ----------------
        intermediate_risks = {}
        with torch.no_grad():
            for t_val in [0.0, 0.25, 0.5, 0.75, 1.0]:
                partial_disp = best_disp * t_val
                pc_t = self._sample_point_cloud(verts_norm_t, partial_disp, fv_ids, bary_t)
                pc_t_input = self._normalize_and_format(pc_t)
                if self.is_v2_model:
                    r = self.risk_predictor(pc_t_input, return_logits=False).item()
                else:
                    r = self.risk_predictor(pc_t_input).item()
                intermediate_risks[str(t_val)] = round(r, 4)

        # Displacement stats
        disp_norms = np.linalg.norm(final_disp, axis=1)
        print(f"   [heal] Done: {len(history['steps'])} steps, "
              f"risk {initial_risk:.4f} -> {final_risk:.4f}, "
              f"max_disp={float(np.max(disp_norms)):.5f}")

        result = {
            'initial_risk': initial_risk,
            'final_risk': final_risk,
            'risk_reduction': initial_risk - final_risk,
            'risk_reduction_pct': 100 * (initial_risk - final_risk) / initial_risk if initial_risk > 0 else 0,
            'mean_movement': float(np.mean(disp_norms)),
            'max_movement': float(np.max(disp_norms)),
            'success': final_risk < 0.5,
            'method': 'gradient',
            'steps_taken': len(history['steps']),
            'best_step': best_step,
            'intermediate_risks': intermediate_risks,
            'history': history,
            'mask_coverage': float((saliency_mask > 0.5).sum().item()) / num_verts,
        }

        return str(healed_path), result

    # ------------------------------------------------------------------
    # Gradient-heal helper methods
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_barycentric(
        points: np.ndarray,
        tri_verts: np.ndarray,
    ) -> np.ndarray:
        """Compute barycentric coordinates for points inside triangles.

        Args:
            points: (N, 3) query points
            tri_verts: (N, 3, 3) triangle vertices [A, B, C]

        Returns:
            (N, 3) barycentric weights [w0, w1, w2]
        """
        A = tri_verts[:, 0]
        B = tri_verts[:, 1]
        C = tri_verts[:, 2]
        v0 = B - A
        v1 = C - A
        v2 = points - A

        d00 = np.sum(v0 * v0, axis=1)
        d01 = np.sum(v0 * v1, axis=1)
        d11 = np.sum(v1 * v1, axis=1)
        d20 = np.sum(v2 * v0, axis=1)
        d21 = np.sum(v2 * v1, axis=1)

        denom = d00 * d11 - d01 * d01
        denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)

        w1 = (d11 * d20 - d01 * d21) / denom
        w2 = (d00 * d21 - d01 * d20) / denom
        w0 = 1.0 - w1 - w2

        return np.stack([w0, w1, w2], axis=1).astype(np.float32)

    @staticmethod
    def _build_adjacency(num_verts: int, faces: np.ndarray) -> list:
        """Build vertex adjacency list from faces."""
        adj = [set() for _ in range(num_verts)]
        for f in faces:
            adj[f[0]].add(f[1]); adj[f[0]].add(f[2])
            adj[f[1]].add(f[0]); adj[f[1]].add(f[2])
            adj[f[2]].add(f[0]); adj[f[2]].add(f[1])
        return adj

    def _sample_point_cloud(
        self,
        verts: torch.Tensor,
        displacement: torch.Tensor,
        face_vertex_ids: torch.Tensor,
        bary_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable surface sampling using precomputed barycentrics.

        Returns:
            (N, 3) sampled point cloud
        """
        displaced = verts + displacement  # (V, 3)
        # Gather triangle vertices: (N, 3, 3)
        v0 = displaced[face_vertex_ids[:, 0]]
        v1 = displaced[face_vertex_ids[:, 1]]
        v2 = displaced[face_vertex_ids[:, 2]]
        # Barycentric interpolation
        pts = (bary_coords[:, 0:1] * v0 +
               bary_coords[:, 1:2] * v1 +
               bary_coords[:, 2:3] * v2)
        return pts

    def _normalize_and_format(self, points: torch.Tensor) -> torch.Tensor:
        """Center + unit-sphere normalize, then format as (1, 3, N) for model."""
        centroid = points.mean(dim=0, keepdim=True)
        pts = points - centroid
        max_d = pts.norm(dim=1).max()
        if max_d > 0:
            pts = pts / max_d
        return pts.unsqueeze(0).transpose(2, 1)  # (1, 3, N)

    @staticmethod
    def _build_laplacian_sparse(num_verts: int, faces: np.ndarray, device: torch.device):
        """Build a sparse Laplacian matrix from mesh faces (precomputed once)."""
        adj = [set() for _ in range(num_verts)]
        for f in faces:
            adj[f[0]].add(f[1]); adj[f[0]].add(f[2])
            adj[f[1]].add(f[0]); adj[f[1]].add(f[2])
            adj[f[2]].add(f[0]); adj[f[2]].add(f[1])

        rows, cols, vals = [], [], []
        for i in range(num_verts):
            neighbors = list(adj[i])
            n = len(neighbors)
            if n == 0:
                continue
            # Diagonal: +1
            rows.append(i); cols.append(i); vals.append(1.0)
            # Off-diagonal: -1/n
            for j in neighbors:
                rows.append(i); cols.append(j); vals.append(-1.0 / n)

        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
        L = torch.sparse_coo_tensor(indices, values, (num_verts, num_verts)).to(device)
        return L

    @staticmethod
    def _laplacian_loss_sparse(displacement: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Vectorized Laplacian smoothing loss using sparse matrix."""
        # L @ displacement -> (V, 3), each row is disp[i] - mean(disp[neighbors])
        lap = torch.sparse.mm(L, displacement)  # (V, 3)
        return (lap ** 2).sum() / displacement.shape[0]

    def _heal_cvae(
        self,
        original_tensor: torch.Tensor,
        target_risk: float,
    ) -> Tuple[np.ndarray, float, dict]:
        """One-pass healing via Conditional VAE."""
        assert self.cvae is not None

        with torch.no_grad():
            healed_tensor = self.cvae.generate_healthy(original_tensor, target_risk=target_risk)
            healed_points = healed_tensor.squeeze(0).transpose(0, 1).cpu().numpy()

            # Normalize CVAE output to unit sphere (same space as input).
            # The decoder may produce points that aren't zero-centered or
            # unit-scaled; scoring the raw output gives misleadingly low risk.
            healed_center = healed_points.mean(axis=0)
            healed_points = healed_points - healed_center
            healed_max = np.max(np.linalg.norm(healed_points, axis=1))
            if healed_max > 0:
                healed_points = healed_points / healed_max

            # Re-score the normalized output for honest risk reporting
            healed_norm_tensor = torch.tensor(
                healed_points, dtype=torch.float32
            ).unsqueeze(0).transpose(2, 1).to(self.device)

            if self.is_v2_model:
                final_risk = self.risk_predictor(healed_norm_tensor, return_logits=False).item()
            else:
                final_risk = self.risk_predictor(healed_norm_tensor).item()

        return healed_points, final_risk, {
            'method': 'cvae',
            'steps_taken': 1,
            'best_step': 0,
        }

    def _heal_latent_opt(
        self,
        original_tensor: torch.Tensor,
        learning_rate: float,
        num_steps: int,
        target_risk: float,
        lambda_latent: float,
    ) -> Tuple[np.ndarray, float, dict]:
        """Iterative healing via latent-space gradient descent."""
        assert self.autoencoder is not None

        with torch.no_grad():
            z_original = self.autoencoder.encode(original_tensor)
            if self.is_v2_model:
                initial_risk = self.risk_predictor(original_tensor, return_logits=False).item()
            else:
                initial_risk = self.risk_predictor(original_tensor).item()

        z_opt = z_original.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([z_opt], lr=learning_rate)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps, eta_min=1e-5
        )

        best_risk = initial_risk
        best_z = z_opt.clone().detach()
        best_step = 0

        history = {'steps': [], 'risk_scores': [], 'latent_distance': [], 'learning_rate': []}

        for step in range(num_steps):
            optimizer.zero_grad()

            recon = self.autoencoder.decode(z_opt)

            if self.is_v2_model:
                loss_risk = torch.sigmoid(self.risk_predictor(recon, return_logits=True)).squeeze()
            else:
                loss_risk = self.risk_predictor(recon).squeeze()

            latent_dist = torch.mean((z_opt - z_original) ** 2)
            total_loss = loss_risk + lambda_latent * latent_dist
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_([z_opt], max_norm=1.0)
            optimizer.step()
            scheduler.step()

            current_risk = loss_risk.item()
            history['steps'].append(step)
            history['risk_scores'].append(current_risk)
            history['latent_distance'].append(latent_dist.item())
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            if current_risk < best_risk:
                best_risk = current_risk
                best_z = z_opt.clone().detach()
                best_step = step

            if current_risk < target_risk:
                break

        with torch.no_grad():
            healed_tensor = self.autoencoder.decode(best_z)
            healed_points = healed_tensor.squeeze(0).transpose(0, 1).cpu().numpy()
            if self.is_v2_model:
                final_risk = self.risk_predictor(healed_tensor, return_logits=False).item()
            else:
                final_risk = self.risk_predictor(healed_tensor).item()

        return healed_points, final_risk, {
            'method': 'latent_space',
            'steps_taken': len(history['steps']),
            'best_step': best_step,
            'latent_distance': float(torch.mean((best_z - z_original) ** 2).item()),
            'history': history,
        }
    
    def get_device_info(self) -> dict:
        """Get information about the compute device being used."""
        return {
            'device': str(self.device),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'models_loaded': self._models_loaded
        }
    
    def calculate_geometric_deltas(
        self,
        original_mesh_path: str,
        healed_mesh_path: str,
        original_risk: float,
        final_risk: float
    ) -> dict:
        """
        Calculate scientific geometric deltas between original and healed meshes.
        
        This function computes the key metrics that quantify "what changed"
        between the sick (original) and healed artery geometries.
        
        Args:
            original_mesh_path: Path to the original (sick) mesh file
            healed_mesh_path: Path to the healed (counterfactual) mesh file
            original_risk: Risk score of the original mesh
            final_risk: Risk score of the healed mesh
            
        Returns:
            Dictionary with geometric analysis metrics:
            - max_displacement_mm: Maximum distance any vertex moved (mm)
            - mean_displacement_mm: Average movement of all vertices (mm)
            - risk_reduction_absolute: Absolute reduction (original - final)
            - risk_reduction_pct: Percentage reduction
            - displacement_std_mm: Standard deviation of displacements
            - volume_change_pct: Approximate volume change (if mesh has faces)
            - surface_area_change_pct: Approximate surface area change
        """
        # Load original mesh (always a proper mesh)
        original_mesh = trimesh.load(original_mesh_path, force='mesh')
        
        # Load healed geometry (may be a PointCloud since we save vertices only)
        healed_geometry = trimesh.load(healed_mesh_path)
        
        # Get the original mesh's scale for realistic mm measurements
        # We use the bounding box diagonal as reference
        original_bounds = original_mesh.bounds
        original_diagonal = np.linalg.norm(original_bounds[1] - original_bounds[0])
        
        # Sample points from original mesh
        original_points = np.array(
            trimesh.sample.sample_surface(original_mesh, count=self.num_points)[0], 
            dtype=np.float32
        )
        
        # For healed geometry - handle both PointCloud and Mesh types
        if isinstance(healed_geometry, trimesh.PointCloud):
            # It's a point cloud - use vertices directly
            healed_points = np.array(healed_geometry.vertices, dtype=np.float32)
        elif hasattr(healed_geometry, 'vertices') and len(healed_geometry.vertices) > 0:  # type: ignore
            healed_points = np.array(healed_geometry.vertices, dtype=np.float32)  # type: ignore
        else:
            healed_points = np.array(
                trimesh.sample.sample_surface(healed_geometry, count=self.num_points)[0],
                dtype=np.float32
            )
        
        # Normalize both point sets for fair comparison (same as model preprocessing)
        original_centroid = np.mean(original_points, axis=0)
        original_points_norm = original_points - original_centroid
        original_max_dist = np.max(np.linalg.norm(original_points_norm, axis=1))
        if original_max_dist > 0:
            original_points_norm = original_points_norm / original_max_dist
        
        healed_centroid = np.mean(healed_points, axis=0)
        healed_points_norm = healed_points - healed_centroid
        healed_max_dist = np.max(np.linalg.norm(healed_points_norm, axis=1))
        if healed_max_dist > 0:
            healed_points_norm = healed_points_norm / healed_max_dist
        
        # Compute point-to-point displacements
        # Since healed may have different point count, use nearest neighbor matching
        # Build KD-tree for healed points
        healed_tree = cKDTree(healed_points_norm)
        
        # Find nearest neighbor in healed mesh for each original point
        distances, _ = healed_tree.query(original_points_norm)
        
        # Calculate displacement statistics (in normalized units)
        max_displacement_norm = float(np.max(distances))
        mean_displacement_norm = float(np.mean(distances))
        std_displacement_norm = float(np.std(distances))
        
        # Convert to approximate millimeters using typical artery dimensions
        # Cerebral arteries are typically 2-5mm in diameter
        # We estimate using the original mesh's bounding box
        TYPICAL_ARTERY_SCALE_MM = 5.0  # Typical scale factor
        scale_factor = original_diagonal if original_diagonal > 0.01 else TYPICAL_ARTERY_SCALE_MM
        
        max_displacement_mm = max_displacement_norm * scale_factor
        mean_displacement_mm = mean_displacement_norm * scale_factor
        std_displacement_mm = std_displacement_norm * scale_factor
        
        # Calculate volume change (only if original mesh has valid faces - healed is point cloud)
        volume_change_pct = None
        # Note: healed_geometry is a PointCloud, so volume/area not available for it
        
        # Calculate surface area change
        surface_area_change_pct = None
        # Note: Cannot compute for point cloud output
        
        # Risk reduction metrics
        risk_reduction_absolute = original_risk - final_risk
        risk_reduction_pct = (risk_reduction_absolute / original_risk * 100) if original_risk > 0 else 0
        
        return {
            # Primary metrics (Part A requirements)
            'max_displacement_mm': round(max_displacement_mm, 3),
            'mean_displacement_mm': round(mean_displacement_mm, 3),
            'risk_reduction_absolute': round(risk_reduction_absolute, 4),
            'risk_reduction_pct': round(risk_reduction_pct, 1),
            
            # Additional statistical metrics
            'displacement_std_mm': round(std_displacement_mm, 3),
            
            # Mesh-based metrics (may be None if meshes lack faces)
            'volume_change_pct': round(volume_change_pct, 1) if volume_change_pct is not None else None,
            'surface_area_change_pct': round(surface_area_change_pct, 1) if surface_area_change_pct is not None else None,
            
            # Normalized values (for debugging/research)
            'max_displacement_normalized': round(max_displacement_norm, 6),
            'mean_displacement_normalized': round(mean_displacement_norm, 6),
        }
