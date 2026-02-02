"""
Prepare Labels - Generate Combined Labels CSV

Creates a unified labels file from all data sources.

Usage:
    python -m training.scripts.prepare_labels
"""

import os
import sys
import glob
from pathlib import Path
import pandas as pd
import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def compute_curvature_score(file_path: str) -> float:
    """
    Compute a curvature-based risk score for a mesh.
    Higher curvature variance = higher risk (aneurysm-like).
    
    Note: This is a simplified proxy. In practice, use the base scores.
    """
    try:
        mesh = trimesh.load(file_path, force='mesh')
        
        # Simple geometric complexity as proxy
        num_vertices = len(mesh.vertices) if hasattr(mesh, 'vertices') else 0  # type: ignore
        num_faces = len(mesh.faces) if hasattr(mesh, 'faces') else 0  # type: ignore
        
        # Higher vertex/face ratio suggests more complex geometry
        complexity = (num_vertices / max(num_faces, 1)) if num_faces > 0 else 1.0
        score = np.clip(complexity / 5.0, 0, 1)
        return float(score)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return 0.5  # Default middle score


def prepare_labels():
    """Generate combined labels CSV from all data sources."""
    
    print("=" * 60)
    print("PREPARING LABELS")
    print("=" * 60)
    
    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    
    # Data sources
    sources = [
        {
            'name': 'complete',
            'folder': PROJECT_ROOT / "IntrA" / "complete",
            'base_score': 0.3,  # Complete arteries are mostly healthy
        },
        {
            'name': 'vessel',
            'folder': PROJECT_ROOT / "IntrA" / "generated" / "vessel" / "obj",
            'base_score': 0.2,  # Generated vessels are healthy
        },
        {
            'name': 'aneurysm',
            'folder': PROJECT_ROOT / "IntrA" / "generated" / "aneurysm" / "obj",
            'base_score': 0.8,  # Aneurysms are high risk
        },
    ]
    
    for source in sources:
        folder = source['folder']
        if not folder.exists():
            print(f"Skipping {source['name']}: folder not found")
            continue
        
        files = list(folder.glob("*.obj"))
        print(f"\n{source['name']}: Found {len(files)} files")
        
        for file_path in files:
            # Use base score with small random variation
            score = source['base_score'] + np.random.uniform(-0.1, 0.1)
            score = np.clip(score, 0, 1)
            
            records.append({
                'filename': file_path.name,
                'data_folder': str(folder),
                'source': source['name'],
                'curvature_score': round(score, 4)
            })
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Save
    output_path = data_dir / "combined_labels.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(df)}")
    print(f"\nBy source:")
    print(df.groupby('source').agg({
        'curvature_score': ['count', 'mean', 'std']
    }))
    print(f"\nSaved to: {output_path}")
    
    # Create balanced version
    low_risk = df[df['curvature_score'] < 0.4]
    high_risk = df[df['curvature_score'] >= 0.4]
    
    # Oversample minority class
    if len(low_risk) < len(high_risk):
        factor = len(high_risk) // len(low_risk)
        low_risk_oversampled = pd.concat([low_risk] * factor)
        balanced_df = pd.concat([low_risk_oversampled, high_risk])
    else:
        factor = len(low_risk) // len(high_risk)
        high_risk_oversampled = pd.concat([high_risk] * factor)
        balanced_df = pd.concat([low_risk, high_risk_oversampled])
    
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    balanced_path = data_dir / "balanced_labels.csv"
    balanced_df.to_csv(balanced_path, index=False)
    print(f"Balanced version: {balanced_path} ({len(balanced_df)} samples)")
    
    return df


if __name__ == "__main__":
    prepare_labels()
