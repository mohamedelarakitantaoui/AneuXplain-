"""
Generate Counterfactual - Heal a Sick Artery

Optimizes a point cloud to reduce its predicted risk score,
generating a "healed" version of the artery.

Usage:
    python -m training.scripts.generate_counterfactual --input path/to/artery.obj
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.optim as optim
import trimesh
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from training.models import RiskPredictor


def load_mesh_as_points(file_path: str, num_points: int = 2048) -> np.ndarray:
    """Load a mesh file and sample points from its surface."""
    mesh = trimesh.load(file_path, force='mesh')
    result = trimesh.sample.sample_surface(mesh, count=num_points)
    points = np.array(result[0], dtype=np.float32)
    
    # Normalize to unit sphere
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist > 0:
        points = points / max_dist
    
    return points


def save_points_as_obj(points: np.ndarray, file_path: str):
    """Save a point cloud as an OBJ file."""
    with open(file_path, 'w') as f:
        f.write("# Counterfactual Point Cloud\n")
        f.write(f"# Points: {len(points)}\n\n")
        for p in points:
            f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    print(f"Saved: {file_path}")


def generate_counterfactual(config: dict):
    """Generate a counterfactual healed artery."""
    
    print("=" * 60)
    print("COUNTERFACTUAL GENERATION")
    print("=" * 60)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load Risk Predictor
    print("\nLoading risk predictor...")
    model_path = config.get('model_path') or (PROJECT_ROOT / "training" / "saved_models" / "risk_predictor.pth")
    
    if not Path(model_path).exists():
        # Try alternative paths
        alt_paths = [
            PROJECT_ROOT / "models" / "risk_predictor_best_gap.pth",
            PROJECT_ROOT / "models" / "risk_predictor.pth",
        ]
        for alt in alt_paths:
            if alt.exists():
                model_path = alt
                break
    
    risk_predictor = RiskPredictor(latent_dim=config['latent_dim'])
    risk_predictor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    risk_predictor.to(device)
    risk_predictor.eval()
    
    # Freeze weights
    for param in risk_predictor.parameters():
        param.requires_grad = False
    
    print(f"Loaded: {model_path}")
    
    # Load patient artery
    print(f"\nLoading patient: {config['input_file']}")
    original_points = load_mesh_as_points(config['input_file'], config['num_points'])
    
    original_tensor = torch.tensor(original_points, dtype=torch.float32).unsqueeze(0).to(device)
    original_tensor = original_tensor.transpose(2, 1)  # (1, 3, N)
    
    # Initial risk
    with torch.no_grad():
        initial_risk = risk_predictor(original_tensor).item()
    
    print(f"Initial Risk Score: {initial_risk:.4f}")
    
    if initial_risk < 0.5:
        print("⚠️  This artery is already low-risk!")
    
    # Build Laplacian neighbors for smoothing
    k_neighbors = 8
    tree = cKDTree(original_points)
    _, neighbor_indices = tree.query(original_points, k=k_neighbors + 1)
    neighbor_indices = neighbor_indices[:, 1:]  # Remove self
    neighbor_indices_tensor = torch.tensor(neighbor_indices, dtype=torch.long, device=device)
    
    # Setup optimization
    points_delta = torch.zeros_like(original_tensor, requires_grad=True)
    optimizer = optim.Adam([points_delta], lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['num_steps'], eta_min=1e-5)
    
    # Optimization loop with dynamic weight schedule
    print("\nOptimizing...")
    print("-" * 60)
    
    lambda_smooth = config.get('lambda_smooth', 0.1)
    lambda_inward = config.get('lambda_inward', 0.3)
    
    history = {'risk': [], 'movement': []}
    best_risk = initial_risk
    best_delta = points_delta.clone().detach()
    
    for step in range(1, config['num_steps'] + 1):
        optimizer.zero_grad()
        
        # Dynamic lambda schedule (matches engine.py)
        if step <= 150:
            lambda_dist = 0.0   # Phase 1: free deformation
        elif step <= 300:
            lambda_dist = 0.5   # Phase 2: gentle constraint
        else:
            lambda_dist = 2.0   # Phase 3: tighten up
        
        current_points = original_tensor + points_delta
        
        # Loss 1: Risk score (primary objective)
        risk_score = risk_predictor(current_points)
        loss_risk = risk_score.squeeze()
        
        # Loss 2: Distance regularization (dynamic weight)
        movement_sq = torch.mean(points_delta ** 2)
        loss_distance = lambda_dist * movement_sq
        
        # Loss 3: Laplacian smoothing (prevents crumpled geometry)
        delta_transposed = points_delta.squeeze(0).transpose(0, 1)  # (N, 3)
        neighbor_deltas = delta_transposed[neighbor_indices_tensor]  # (N, k, 3)
        neighbor_avg = torch.mean(neighbor_deltas, dim=1)            # (N, 3)
        laplacian = delta_transposed - neighbor_avg                  # (N, 3)
        loss_smooth = lambda_smooth * torch.mean(laplacian ** 2)
        
        # Loss 4: Inward bias (promotes aneurysm deflation)
        centroid = torch.mean(original_tensor, dim=2, keepdim=True)
        radial_vec = original_tensor - centroid
        radial_norm = torch.norm(radial_vec, dim=1, keepdim=True) + 1e-8
        radial_unit = radial_vec / radial_norm
        radial_movement = torch.sum(points_delta * radial_unit, dim=1)
        loss_inward = lambda_inward * torch.mean(torch.relu(radial_movement) ** 2)
        
        total_loss = loss_risk + loss_distance + loss_smooth + loss_inward
        total_loss.backward()
        
        max_grad_norm = 5.0 if step <= 150 else 2.0
        torch.nn.utils.clip_grad_norm_([points_delta], max_norm=max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        current_risk = risk_score.item()
        current_movement = torch.sqrt(movement_sq).item()
        
        history['risk'].append(current_risk)
        history['movement'].append(current_movement)
        
        if current_risk < best_risk:
            best_risk = current_risk
            best_delta = points_delta.clone().detach()
            status = "✓ NEW BEST"
        else:
            status = ""
        
        if step % 20 == 0 or step == 1:
            print(f"Step {step:4d} | Risk: {current_risk:.4f} | Move: {current_movement:.4f} | λ_d: {lambda_dist} | {status}")
        
        if current_risk < config['target_risk']:
            print(f"\n🎉 Target risk {config['target_risk']} reached!")
            break
    
    print("-" * 60)
    
    # Generate final healed artery
    with torch.no_grad():
        healed_tensor = original_tensor + best_delta
        healed_points = healed_tensor.squeeze(0).transpose(0, 1).cpu().numpy()
        final_risk = risk_predictor(healed_tensor).item()
    
    # Save outputs
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_points_as_obj(original_points, str(output_dir / "original.obj"))
    save_points_as_obj(healed_points, str(output_dir / "healed.obj"))
    
    # Movement analysis
    delta_np = best_delta.squeeze(0).transpose(0, 1).cpu().numpy()
    point_movements = np.linalg.norm(delta_np, axis=1)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['risk'], 'r-')
    axes[0].axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
    axes[0].axhline(y=config['target_risk'], color='g', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Risk Score')
    axes[0].set_title('Risk During Optimization')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(point_movements, bins=50, color='purple', alpha=0.7)
    axes[1].set_xlabel('Point Movement')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Movement Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "optimization.png", dpi=150)
    plt.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Initial Risk:  {initial_risk:.4f}")
    print(f"Final Risk:    {final_risk:.4f}")
    print(f"Reduction:     {initial_risk - final_risk:.4f} ({100*(initial_risk-final_risk)/initial_risk:.1f}%)")
    print(f"Mean Movement: {np.mean(point_movements):.4f}")
    print(f"\nOutput: {output_dir}")
    
    if final_risk < 0.5:
        print("\n✅ SUCCESS: Artery transformed to LOW RISK!")
    else:
        print("\n⚠️  Risk reduced but still above threshold.")
    
    return healed_points, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate counterfactual healed artery")
    parser.add_argument("--input", "-i", required=True, help="Input mesh file (.obj)")
    parser.add_argument("--output", "-o", default="outputs/counterfactual", help="Output directory")
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps")
    parser.add_argument("--target", type=float, default=0.3, help="Target risk score")
    args = parser.parse_args()
    
    config = {
        'input_file': args.input,
        'output_dir': args.output,
        'num_points': 2048,
        'latent_dim': 128,
        'learning_rate': 0.015,
        'num_steps': args.steps,
        'lambda_smooth': 0.1,
        'lambda_inward': 0.3,
        'target_risk': args.target,
    }
    
    generate_counterfactual(config)
