"""
Prepare Labels - Ground Truth Binary Labels from IntrA Dataset

Assigns risk labels based on the IntrA dataset's folder structure:
  - generated/vessel/   → 0.0  (healthy)
  - generated/aneurysm/ → 1.0  (diseased)
  - complete/           → 1.0  (all contain aneurysm regions)

This replaces the old geometry-only scoring which produced overlapping
distributions (vessel mean ~0.53, aneurysm mean ~0.59) that gave the
risk predictor no signal to learn from.

Usage:
    python -m training.scripts.prepare_labels
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def prepare_labels():
    """Generate ground-truth binary labels CSV from IntrA folder structure."""

    print("=" * 60)
    print("PREPARING GROUND-TRUTH BINARY LABELS")
    print("=" * 60)

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Each source gets a hard binary label based on folder identity
    sources = [
        {
            'name': 'vessel',
            'folder': PROJECT_ROOT / "IntrA" / "generated" / "vessel" / "obj",
            'label': 0.0,
        },
        {
            'name': 'aneurysm',
            'folder': PROJECT_ROOT / "IntrA" / "generated" / "aneurysm" / "obj",
            'label': 1.0,
        },
        {
            'name': 'complete',
            'folder': PROJECT_ROOT / "IntrA" / "complete",
            'label': 1.0,  # All complete arteries contain aneurysm regions
        },
    ]

    records = []
    errors = 0

    for source in sources:
        folder = source['folder']
        if not folder.exists():
            print(f"Skipping {source['name']}: folder not found at {folder}")
            continue

        files = sorted(folder.glob("*.obj"))
        print(f"\n{source['name']}: {len(files)} files → label = {source['label']}")

        for i, file_path in enumerate(files):
            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(files)}]")

            record = {
                'filename': file_path.name,
                'data_folder': str(folder),
                'source': source['name'],
                'risk_score': source['label'],
            }
            records.append(record)

    if not records:
        print("ERROR: No files found in any source folder!")
        return None

    df = pd.DataFrame(records)

    # Save
    output_path = data_dir / "combined_labels.csv"
    df.to_csv(output_path, index=False)

    # ---- Validation ----
    print(f"\n{'='*60}")
    print("LABEL DISTRIBUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total samples: {len(df)}, Errors: {errors}")

    for src in ['vessel', 'aneurysm', 'complete']:
        subset = df[df['source'] == src]
        if len(subset) > 0:
            print(f"  {src:>10s}: {len(subset):5d} samples, "
                  f"mean={subset['risk_score'].mean():.3f}, "
                  f"range=[{subset['risk_score'].min():.1f}, {subset['risk_score'].max():.1f}]")

    vessel_scores = df[df['source'] == 'vessel']['risk_score']
    aneurysm_scores = df[df['source'].isin(['aneurysm', 'complete'])]['risk_score']

    vessel_mean = vessel_scores.mean() if len(vessel_scores) > 0 else float('nan')
    aneurysm_mean = aneurysm_scores.mean() if len(aneurysm_scores) > 0 else float('nan')
    gap = aneurysm_mean - vessel_mean

    print(f"\n  Vessel mean label:   {vessel_mean:.3f}  (target: < 0.15)")
    print(f"  Aneurysm mean label: {aneurysm_mean:.3f}  (target: > 0.85)")
    print(f"  Gap:                 {gap:.3f}  (target: > 0.70)")

    # Class balance
    n_healthy = len(df[df['risk_score'] < 0.5])
    n_diseased = len(df[df['risk_score'] >= 0.5])
    print(f"\n  Class balance: {n_healthy} healthy ({100*n_healthy/len(df):.1f}%) "
          f"/ {n_diseased} diseased ({100*n_diseased/len(df):.1f}%)")
    print(f"  Imbalance ratio: 1:{n_healthy/n_diseased:.1f}" if n_diseased > 0 else "")

    # Validation checks
    checks_passed = True
    if vessel_mean > 0.15:
        print("\n  WARNING: Vessel mean > 0.15!")
        checks_passed = False
    if aneurysm_mean < 0.85:
        print("\n  WARNING: Aneurysm mean < 0.85!")
        checks_passed = False
    if gap < 0.70:
        print("\n  WARNING: Gap < 0.70!")
        checks_passed = False

    if checks_passed:
        print("\n  ALL VALIDATION CHECKS PASSED")

    print(f"\nSaved to: {output_path}")

    return df


if __name__ == "__main__":
    prepare_labels()
