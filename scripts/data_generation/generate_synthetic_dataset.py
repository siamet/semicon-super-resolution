"""
Synthetic Dataset Generation Script

Generates a large dataset of synthetic semiconductor patterns
with configurable variety and saves them with metadata.

Usage:
    python scripts/data_generation/generate_synthetic_dataset.py --config config/data_config.yaml --num_samples 1000
"""

import sys
from pathlib import Path
import argparse
import yaml
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.synthetic.pattern_generator import (
    PatternConfig,
    GratingGenerator,
    ContactHoleGenerator,
    IsolatedFeatureGenerator
)


class DatasetGenerator:
    """Generate synthetic dataset with diverse patterns."""

    def __init__(self, config_path: Path):
        """
        Initialize dataset generator.

        Args:
            config_path: Path to data configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Extract configuration
        self.synthetic_config = self.config['synthetic']
        self.output_dir = Path(self.config['paths']['raw_data'])
        self.metadata_dir = Path(self.config['paths']['metadata'])

        # Create pattern config
        self.pattern_config = PatternConfig(
            image_size=self.synthetic_config['image_size'],
            pixel_size_nm=self.synthetic_config['pixel_size_nm']
        )

        # Initialize generators
        self.grating_gen = GratingGenerator(self.pattern_config)
        self.contact_gen = ContactHoleGenerator(self.pattern_config)
        self.isolated_gen = IsolatedFeatureGenerator(self.pattern_config)

        # Initialize random number generator
        self.rng = np.random.default_rng()

    def create_output_directories(self):
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for each pattern type
        for pattern_type in ['gratings', 'contacts', 'logic_cells', 'isolated']:
            (self.output_dir / pattern_type).mkdir(exist_ok=True)

        print(f"✓ Output directories created at: {self.output_dir}")

    def generate_grating(self, idx: int) -> Dict[str, Any]:
        """Generate a random grating pattern."""
        cfg = self.synthetic_config['patterns']['gratings']
        ler_cfg = self.synthetic_config['ler']

        # Random parameters
        pitch = self.rng.uniform(*cfg['pitch_range'])
        duty_cycle = self.rng.uniform(*cfg['duty_cycle_range'])
        orientation = self.rng.choice(cfg['orientations'])

        # Generate pattern
        pattern, metadata = self.grating_gen.generate(
            pitch_nm=pitch,
            duty_cycle=duty_cycle,
            orientation_deg=orientation,
            add_ler=True,
            ler_sigma_nm=ler_cfg['sigma_nm'],
            ler_correlation_nm=ler_cfg['correlation_length_nm']
        )

        # Save pattern
        filename = f"grating_{idx:06d}.npy"
        filepath = self.output_dir / "gratings" / filename
        np.save(filepath, pattern.astype(np.float32))

        # Add file info to metadata
        metadata['filename'] = filename
        metadata['filepath'] = str(filepath)
        metadata['index'] = idx

        return metadata

    def generate_contact_holes(self, idx: int) -> Dict[str, Any]:
        """Generate a random contact hole pattern."""
        cfg = self.synthetic_config['patterns']['contacts']
        ler_cfg = self.synthetic_config['ler']

        # Random parameters
        diameter = self.rng.uniform(*cfg['diameter_range'])
        shape = self.rng.choice(cfg['shapes'])
        array_type = self.rng.choice(cfg['array_types'])
        pitch = diameter * self.rng.uniform(2.0, 3.0)  # 2-3x diameter

        # Generate pattern
        pattern, metadata = self.contact_gen.generate(
            diameter_nm=diameter,
            pitch_nm=pitch,
            shape=shape,
            array_type=array_type,
            add_ler=True,
            ler_sigma_nm=ler_cfg['sigma_nm'],
            ler_correlation_nm=ler_cfg['correlation_length_nm']
        )

        # Save pattern
        filename = f"contacts_{idx:06d}.npy"
        filepath = self.output_dir / "contacts" / filename
        np.save(filepath, pattern.astype(np.float32))

        # Add file info to metadata
        metadata['filename'] = filename
        metadata['filepath'] = str(filepath)
        metadata['index'] = idx

        return metadata

    def generate_isolated_feature(self, idx: int) -> Dict[str, Any]:
        """Generate a random isolated feature pattern."""
        ler_cfg = self.synthetic_config['ler']

        # Random parameters
        feature_type = self.rng.choice(['line', 'space', 'post'])
        width = self.rng.uniform(30, 100)  # nm
        orientation = self.rng.uniform(0, 180)

        # Generate pattern
        pattern, metadata = self.isolated_gen.generate(
            feature_type=feature_type,
            width_nm=width,
            orientation_deg=orientation,
            add_ler=True,
            ler_sigma_nm=ler_cfg['sigma_nm'],
            ler_correlation_nm=ler_cfg['correlation_length_nm']
        )

        # Save pattern
        filename = f"isolated_{idx:06d}.npy"
        filepath = self.output_dir / "isolated" / filename
        np.save(filepath, pattern.astype(np.float32))

        # Add file info to metadata
        metadata['filename'] = filename
        metadata['filepath'] = str(filepath)
        metadata['index'] = idx

        return metadata

    def generate_dataset(self, num_samples: int):
        """
        Generate full dataset with diverse patterns.

        Args:
            num_samples: Total number of patterns to generate
        """
        print(f"\nGenerating {num_samples} synthetic patterns...")
        print("="*70)

        self.create_output_directories()

        all_metadata = []
        pattern_counts = {
            'gratings': 0,
            'contacts': 0,
            'isolated': 0
        }

        start_time = time.time()

        # Generate patterns with progress bar
        with tqdm(total=num_samples, desc="Generating patterns") as pbar:
            for idx in range(num_samples):
                # Distribute pattern types roughly equally
                # 40% gratings, 40% contacts, 20% isolated
                rand = self.rng.random()

                if rand < 0.4:
                    metadata = self.generate_grating(idx)
                    pattern_counts['gratings'] += 1
                elif rand < 0.8:
                    metadata = self.generate_contact_holes(idx)
                    pattern_counts['contacts'] += 1
                else:
                    metadata = self.generate_isolated_feature(idx)
                    pattern_counts['isolated'] += 1

                all_metadata.append(metadata)
                pbar.update(1)

        elapsed_time = time.time() - start_time

        # Save metadata (convert numpy types to Python types for JSON serialization)
        metadata_file = self.metadata_dir / f"dataset_metadata_{num_samples}.json"

        # Convert numpy types to Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        all_metadata_converted = convert_numpy(all_metadata)

        with open(metadata_file, 'w') as f:
            json.dump(all_metadata_converted, f, indent=2)

        # Save summary
        summary = {
            'total_samples': num_samples,
            'pattern_counts': pattern_counts,
            'generation_time_seconds': elapsed_time,
            'patterns_per_second': num_samples / elapsed_time,
            'config': self.synthetic_config,
            'output_directory': str(self.output_dir),
            'metadata_file': str(metadata_file)
        }

        summary_file = self.metadata_dir / f"dataset_summary_{num_samples}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\n" + "="*70)
        print("✓ Dataset generation complete!")
        print("="*70)
        print(f"Total patterns: {num_samples}")
        print(f"  - Gratings: {pattern_counts['gratings']} ({pattern_counts['gratings']/num_samples*100:.1f}%)")
        print(f"  - Contacts: {pattern_counts['contacts']} ({pattern_counts['contacts']/num_samples*100:.1f}%)")
        print(f"  - Isolated: {pattern_counts['isolated']} ({pattern_counts['isolated']/num_samples*100:.1f}%)")
        print(f"\nGeneration time: {elapsed_time:.2f}s ({num_samples/elapsed_time:.1f} patterns/sec)")
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Metadata saved: {metadata_file}")
        print(f"Summary saved: {summary_file}")
        print("="*70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic semiconductor pattern dataset"
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=project_root / 'config' / 'data_config.yaml',
        help='Path to data configuration file'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=1000,
        help='Number of patterns to generate'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")

    # Generate dataset
    generator = DatasetGenerator(args.config)
    generator.generate_dataset(args.num_samples)


if __name__ == '__main__':
    main()
