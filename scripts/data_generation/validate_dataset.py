"""
Dataset Validation Script

Validates the generated dataset and creates sample visualizations.
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.synthetic.visualizer import PatternVisualizer


def load_random_samples(metadata_file: Path, n_samples: int = 9) -> List[Dict]:
    """Load random samples from dataset."""
    with open(metadata_file, 'r') as f:
        all_metadata = json.load(f)

    # Random sample
    indices = np.random.choice(len(all_metadata), size=min(n_samples, len(all_metadata)), replace=False)
    samples = [all_metadata[i] for i in indices]

    return samples


def validate_dataset(metadata_file: Path, output_dir: Path):
    """
    Validate dataset and generate reports.

    Args:
        metadata_file: Path to dataset metadata JSON
        output_dir: Output directory for validation results
    """
    print("="*70)
    print("Dataset Validation")
    print("="*70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    with open(metadata_file, 'r') as f:
        all_metadata = json.load(f)

    # Load summary
    summary_file = metadata_file.parent / metadata_file.name.replace('metadata', 'summary')
    with open(summary_file, 'r') as f:
        summary = json.load(f)

    print(f"\nDataset Summary:")
    print(f"  Total patterns: {summary['total_samples']}")
    print(f"  Gratings: {summary['pattern_counts']['gratings']}")
    print(f"  Contacts: {summary['pattern_counts']['contacts']}")
    print(f"  Isolated: {summary['pattern_counts']['isolated']}")
    print(f"  Generation time: {summary['generation_time_seconds']:.1f}s")
    print(f"  Generation speed: {summary['patterns_per_second']:.2f} patterns/sec")

    # Validate files exist
    print(f"\nValidating files...")
    missing_files = 0
    for metadata in all_metadata[:10]:  # Check first 10
        filepath = Path(metadata['filepath'])
        if not filepath.exists():
            print(f"  Missing: {filepath}")
            missing_files += 1

    if missing_files == 0:
        print(f"  ✓ All checked files exist")
    else:
        print(f"  ⚠ {missing_files} missing files detected!")

    # Check pattern statistics
    print(f"\nPattern Statistics:")

    # Sample patterns
    samples = load_random_samples(metadata_file, n_samples=10)

    for sample in samples[:3]:
        pattern = np.load(sample['filepath'])
        print(f"  {sample['pattern_type']}: shape={pattern.shape}, "
              f"range=[{pattern.min():.3f}, {pattern.max():.3f}], "
              f"mean={pattern.mean():.3f}")

    # Visualize random samples
    print(f"\nGenerating visualizations...")

    visualizer = PatternVisualizer()

    # Load 9 random samples
    samples = load_random_samples(metadata_file, n_samples=9)
    patterns = []
    metadatas = []

    for sample in samples:
        pattern = np.load(sample['filepath'])
        patterns.append(pattern)
        metadatas.append(sample)

    # Save visualization
    viz_path = output_dir / "dataset_samples.png"
    visualizer.visualize_multiple_patterns(
        patterns,
        metadatas,
        save_path=viz_path,
        show=False
    )
    print(f"  ✓ Sample visualization saved: {viz_path}")

    # Create detailed view of one pattern
    sample = samples[0]
    pattern = np.load(sample['filepath'])

    detail_path = output_dir / f"detailed_sample_{sample['pattern_type']}.png"
    visualizer.visualize_statistics(
        pattern,
        sample,
        save_path=detail_path,
        show=False
    )
    print(f"  ✓ Detailed visualization saved: {detail_path}")

    # Distribution analysis
    print(f"\nPattern Distribution Analysis:")
    pattern_types = [m['pattern_type'] for m in all_metadata]
    from collections import Counter
    counts = Counter(pattern_types)

    for ptype, count in counts.items():
        percentage = count / len(all_metadata) * 100
        print(f"  {ptype}: {count} ({percentage:.1f}%)")

    print("\n" + "="*70)
    print("✓ Validation complete!")
    print("="*70)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate synthetic dataset")
    parser.add_argument(
        '--metadata',
        type=Path,
        default=project_root / 'data' / 'raw' / 'synthetic' / 'metadata' / 'dataset_metadata_1000.json',
        help='Path to dataset metadata file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=project_root / 'results' / 'dataset_validation',
        help='Output directory for validation results'
    )

    args = parser.parse_args()

    validate_dataset(args.metadata, args.output)


if __name__ == '__main__':
    main()
