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
        self.risk_predictor: Optional[RiskPredictor] = None
        self.autoencoder: Optional[Autoencoder] = None
        
        self._models_loaded = False
    
    def load_models(
        self,
        risk_predictor_path: Optional[str] = None,
        autoencoder_path: Optional[str] = None
    ) -> None:
        """
        Load the trained model weights.
        
        Args:
            risk_predictor_path: Path to risk predictor weights (.pth)
            autoencoder_path: Path to autoencoder weights (.pth) - optional
        """
        # Default paths
        if risk_predictor_path is None:
            rpp = self.models_dir / "risk_predictor.pth"
        else:
            rpp = Path(risk_predictor_path)
        
        # Load Risk Predictor
        if not rpp.exists():
            raise FileNotFoundError(f"Risk predictor weights not found: {rpp}")
        
        self.risk_predictor = RiskPredictor(latent_dim=self.latent_dim)
        self.risk_predictor.load_state_dict(
            torch.load(str(rpp), map_location=self.device, weights_only=True)
        )
        self.risk_predictor.to(self.device)
        self.risk_predictor.eval()
        
        # Freeze weights
        for param in self.risk_predictor.parameters():
            param.requires_grad = False
        
        # Optionally load Autoencoder
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
        
        # Build KD-tree of sampled original points for nearest neighbor lookup
        tree = cKDTree(sampled_points_original)
        
        # For each mesh vertex, find the closest sampled point and apply its displacement
        distances, indices = tree.query(vertices_normalized, k=3)  # Use 3 nearest neighbors
        
        # Compute displacement field from sampled points
        displacements = sampled_points_healed - sampled_points_original  # (N, 3)
        
        # Apply weighted displacement to each vertex using inverse distance weighting
        new_vertices_normalized = np.zeros_like(vertices_normalized)
        for i in range(len(vertices_normalized)):
            neighbor_indices = indices[i]
            neighbor_distances = distances[i]
            
            # Avoid division by zero
            weights = 1.0 / (neighbor_distances + 1e-8)
            weights = weights / np.sum(weights)
            
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
            risk_score = self.risk_predictor(tensor).item()
        
        return risk_score
    
    def heal_artery(
        self,
        mesh_path: str,
        learning_rate: float = 0.015,
        num_steps: int = 500,
        lambda_smooth: float = 0.1,
        target_risk: float = 0.15,
        save_intermediate: bool = False,
    ) -> Tuple[str, dict]:
        """
        Generate a counterfactual "healed" version of a sick artery.
        
        Uses a DYNAMIC WEIGHT SCHEDULE for the distance regularization:
        - Steps 0-149: lambda_distance = 0.0 (NO constraint - let it deform wildly)
        - Steps 150-299: lambda_distance = 0.5 (weak constraint - start pulling back)
        - Steps 300-500: lambda_distance = 2.0 (tighten up - preserve topology)
        
        Also includes LAPLACIAN SMOOTHING LOSS to prevent crumpled/noisy geometry.
        
        CRITICAL: Returns the mesh from the step with LOWEST RISK, not the last step.
        
        Args:
            mesh_path: Path to the input artery mesh file
            learning_rate: Optimization learning rate
            num_steps: Number of optimization steps (default: 500)
            lambda_smooth: Weight for Laplacian smoothing loss
            target_risk: Stop early if risk drops below this value
            save_intermediate: Whether to save intermediate results
            
        Returns:
            Tuple of (healed_mesh_path, result_dict)
            - healed_mesh_path: Path to the output healed OBJ file
            - result_dict: Dictionary with optimization details
        """
        self._ensure_models_loaded()
        assert self.risk_predictor is not None, "Risk predictor not loaded"
        
        # Generate unique output filename
        output_id = str(uuid.uuid4())[:8]
        healed_path = self.output_dir / f"healed_{output_id}.obj"
        
        # Load original artery
        original_points = self._load_mesh_as_points(mesh_path)
        original_tensor = self._points_to_tensor(original_points)
        
        # Get initial risk score
        with torch.no_grad():
            initial_risk = self.risk_predictor(original_tensor).item()
        
        # ================================================================
        # BUILD LAPLACIAN MATRIX FOR SMOOTHING
        # ================================================================
        # For point clouds, we approximate Laplacian using k-nearest neighbors
        # This ensures smooth deformations (no crumpled paper effect)
        k_neighbors = 8
        points_np = original_points  # (N, 3)
        tree = cKDTree(points_np)
        _, neighbor_indices = tree.query(points_np, k=k_neighbors + 1)  # +1 because first is self
        neighbor_indices = neighbor_indices[:, 1:]  # Remove self (N, k)
        
        # Convert to tensor for GPU computation
        neighbor_indices_tensor = torch.tensor(neighbor_indices, dtype=torch.long, device=self.device)
        
        # ================================================================
        # SETUP OPTIMIZATION
        # ================================================================
        # We optimize a deformation field (delta) rather than absolute positions
        points_delta = torch.zeros_like(original_tensor, requires_grad=True)
        optimizer = optim.Adam([points_delta], lr=learning_rate, betas=(0.9, 0.999))
        
        # Cosine annealing for smooth LR decay
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps, eta_min=1e-5
        )
        
        # Optimization history
        history = {
            'steps': [],
            'risk_scores': [],
            'movements': [],
            'lambda_distance': [],
            'lambda_smooth': [],
            'learning_rate': []
        }
        
        # Track the BEST result (lowest risk)
        best_risk = initial_risk
        best_delta = points_delta.clone().detach()
        best_step = 0
        
        # ================================================================
        # OPTIMIZATION LOOP WITH DYNAMIC WEIGHT SCHEDULE
        # ================================================================
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # ============================================================
            # DYNAMIC LAMBDA SCHEDULE (the key to handling bifurcations!)
            # ============================================================
            if step < 150:
                # Phase 1: WILD DEFORMATION - disable distance penalty completely
                # Let the optimizer find the risk-reducing direction freely
                lambda_dist = 0.0
            elif step < 300:
                # Phase 2: GENTLE CONSTRAINT - start pulling back
                lambda_dist = 0.5
            else:
                # Phase 3: TIGHTEN UP - enforce topology preservation
                lambda_dist = 2.0
            
            # Apply deformation to get current points
            current_points = original_tensor + points_delta
            
            # ============================================================
            # LOSS 1: RISK SCORE (primary objective)
            # ============================================================
            risk_score = self.risk_predictor(current_points)
            loss_risk = risk_score.squeeze()
            
            # ============================================================
            # LOSS 2: DISTANCE REGULARIZATION (dynamic weight)
            # ============================================================
            movement_sq = torch.mean(points_delta ** 2)
            loss_distance = lambda_dist * movement_sq
            
            # ============================================================
            # LOSS 3: LAPLACIAN SMOOTHING (prevents crumpled geometry)
            # ============================================================
            # Compute Laplacian: for each point, difference from neighbor average
            # points_delta shape: (1, 3, N)
            delta_transposed = points_delta.squeeze(0).transpose(0, 1)  # (N, 3)
            
            # Gather neighbor deltas: (N, k, 3)
            neighbor_deltas = delta_transposed[neighbor_indices_tensor]  # (N, k, 3)
            
            # Average of neighbors: (N, 3)
            neighbor_avg = torch.mean(neighbor_deltas, dim=1)
            
            # Laplacian: difference between point and its neighbor average
            laplacian = delta_transposed - neighbor_avg  # (N, 3)
            
            # Smoothing loss: penalize large Laplacian (non-smooth deformations)
            loss_smooth = lambda_smooth * torch.mean(laplacian ** 2)
            
            # ============================================================
            # LOSS 4: INWARD BIAS (prevents inflation, promotes deflation)
            # ============================================================
            # For aneurysms, we want to SHRINK bulges, not inflate
            # Penalize outward (positive radial) movement more than inward
            
            # Compute centroid of original points
            # original_tensor shape: (1, 3, N)
            centroid = torch.mean(original_tensor, dim=2, keepdim=True)  # (1, 3, 1)
            
            # Radial direction from centroid to each point (normalized)
            radial_vec = original_tensor - centroid  # (1, 3, N)
            radial_norm = torch.norm(radial_vec, dim=1, keepdim=True) + 1e-8  # (1, 1, N)
            radial_unit = radial_vec / radial_norm  # (1, 3, N) - unit vectors pointing outward
            
            # Project delta onto radial direction: positive = outward, negative = inward
            radial_movement = torch.sum(points_delta * radial_unit, dim=1)  # (1, N)
            
            # Penalize ONLY outward movement (positive values)
            # ReLU keeps only positive values, so only outward movement is penalized
            lambda_inward = 0.3  # Weight for inward bias
            loss_inward = lambda_inward * torch.mean(torch.relu(radial_movement) ** 2)
            
            # ============================================================
            # TOTAL LOSS
            # ============================================================
            total_loss = loss_risk + loss_distance + loss_smooth + loss_inward
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping (allow larger gradients in wild phase)
            max_grad_norm = 5.0 if step < 150 else 2.0
            torch.nn.utils.clip_grad_norm_([points_delta], max_norm=max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            # Track progress
            current_risk = risk_score.item()
            current_movement = torch.sqrt(movement_sq).item()
            
            history['steps'].append(step)
            history['risk_scores'].append(current_risk)
            history['movements'].append(current_movement)
            history['lambda_distance'].append(lambda_dist)
            history['lambda_smooth'].append(lambda_smooth)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # ============================================================
            # SAVE BEST (lowest risk, not last step!)
            # ============================================================
            if current_risk < best_risk:
                best_risk = current_risk
                best_delta = points_delta.clone().detach()
                best_step = step
            
            # Save intermediate (optional)
            if save_intermediate and step % 100 == 0:
                with torch.no_grad():
                    intermediate = (original_tensor + points_delta).squeeze(0).transpose(0, 1).cpu().numpy()
                intermediate_path = self.output_dir / f"step_{step:04d}_{output_id}.obj"
                self._save_points_as_obj(intermediate, str(intermediate_path))
            
            # Early stopping if we hit target
            if current_risk < target_risk:
                break
        
        # ================================================================
        # GENERATE FINAL HEALED ARTERY FROM BEST STEP
        # ================================================================
        with torch.no_grad():
            healed_tensor = original_tensor + best_delta
            healed_points = healed_tensor.squeeze(0).transpose(0, 1).cpu().numpy()
            final_risk = self.risk_predictor(healed_tensor).item()
        
        # Save output as a proper mesh with faces (not just points!)
        # This transfers the learned deformation to the full mesh topology
        self._save_deformed_mesh(
            original_mesh_path=mesh_path,
            sampled_points_original=original_points,
            sampled_points_healed=healed_points,
            file_path=str(healed_path)
        )
        
        # Compute statistics
        delta_np = best_delta.squeeze(0).transpose(0, 1).cpu().numpy()
        point_movements = np.linalg.norm(delta_np, axis=1)
        
        result = {
            'initial_risk': initial_risk,
            'final_risk': final_risk,
            'risk_reduction': initial_risk - final_risk,
            'risk_reduction_pct': 100 * (initial_risk - final_risk) / initial_risk if initial_risk > 0 else 0,
            'steps_taken': len(history['steps']),
            'best_step': best_step,
            'mean_movement': float(np.mean(point_movements)),
            'max_movement': float(np.max(point_movements)),
            'healed_path': str(healed_path),
            'success': final_risk < 0.5,
            'schedule_used': 'dynamic_3phase',
            'laplacian_smoothing': True,
            'inward_bias': True,  # NEW: Inward bias constraint enabled
            'history': history
        }
        
        return str(healed_path), result
    
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
