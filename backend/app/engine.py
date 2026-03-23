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
        Load a mesh file and sample points from its surface.
        
        Args:
            file_path: Path to the mesh file (.obj, .ply, .stl, etc.)
            
        Returns:
            Point cloud as numpy array of shape (N, 3), normalized to unit sphere
        """
        mesh = trimesh.load(file_path, force='mesh')
        result = trimesh.sample.sample_surface(mesh, count=self.num_points)
        points = np.array(result[0], dtype=np.float32)
        
        # Normalize to unit sphere
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        
        return points
    
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
        learning_rate: float = 0.01,
        num_steps: int = 300,
        target_risk: float = 0.05,
        lambda_latent: float = 0.1,
    ) -> Tuple[str, dict]:
        """
        Generate a counterfactual "healed" version of a sick artery.

        Strategy (in priority order):
        1. **CVAE** — single forward pass conditioned on target_risk.
        2. **Latent-space optimization** — gradient descent on z through
           the autoencoder decoder + frozen risk predictor.

        Args:
            mesh_path: Path to the input artery mesh file
            learning_rate: LR for latent-space fallback
            num_steps: Max steps for latent-space fallback
            target_risk: Desired risk level
            lambda_latent: Latent regularization for fallback

        Returns:
            Tuple of (healed_mesh_path, result_dict)
        """
        self._ensure_models_loaded()
        assert self.risk_predictor is not None, "Risk predictor not loaded"

        # Generate unique output filename
        output_id = str(uuid.uuid4())[:8]
        healed_path = self.output_dir / f"healed_{output_id}.obj"

        # Load original artery
        original_points = self._load_mesh_as_points(mesh_path)
        original_tensor = self._points_to_tensor(original_points)

        # Initial risk
        with torch.no_grad():
            if self.is_v2_model:
                initial_risk = self.risk_predictor(original_tensor, return_logits=False).item()
            else:
                initial_risk = self.risk_predictor(original_tensor).item()

        # =============================================================
        # Choose healing method
        # =============================================================
        if self.cvae is not None:
            healed_points, final_risk, method_result = self._heal_cvae(
                original_tensor, target_risk
            )
        elif self.autoencoder is not None:
            healed_points, final_risk, method_result = self._heal_latent_opt(
                original_tensor, learning_rate, num_steps, target_risk, lambda_latent
            )
        else:
            raise RuntimeError(
                "Healing requires a CVAE or Autoencoder. "
                "Place cvae.pth or autoencoder_v2.pth in models/."
            )

        # Transfer deformation to full mesh topology (preserves faces)
        self._save_deformed_mesh(
            original_mesh_path=mesh_path,
            sampled_points_original=original_points,
            sampled_points_healed=healed_points,
            file_path=str(healed_path)
        )

        # Displacement statistics (NN-matched, consistent with mesh deformation)
        healed_centered = (
            healed_points
            - healed_points.mean(axis=0)
            + original_points.mean(axis=0)
        )
        _ht = cKDTree(healed_centered)
        _, _nn_idx = _ht.query(original_points)
        displacement = healed_centered[_nn_idx] - original_points
        displacement = displacement - displacement.mean(axis=0)
        point_movements = np.linalg.norm(displacement, axis=1)

        result = {
            'initial_risk': initial_risk,
            'final_risk': final_risk,
            'risk_reduction': initial_risk - final_risk,
            'risk_reduction_pct': 100 * (initial_risk - final_risk) / initial_risk if initial_risk > 0 else 0,
            'mean_movement': float(np.mean(point_movements)),
            'max_movement': float(np.max(point_movements)),
            'healed_path': str(healed_path),
            'success': final_risk < 0.5,
        }
        result.update(method_result)

        return str(healed_path), result

    # ------------------------------------------------------------------
    # Healing backends
    # ------------------------------------------------------------------

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
