"""
Test Script for Pattern Generator

Generates sample patterns and validates them through visualization.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from src.data.synthetic.pattern_generator import (
    PatternConfig,
    GratingGenerator,
    ContactHoleGenerator,
    IsolatedFeatureGenerator,
    create_pattern_generator
)
from src.data.synthetic.visualizer import PatternVisualizer, quick_visualize


def test_grating_generator():
    """Test grating pattern generation."""
    print("Testing Grating Generator...")

    config = PatternConfig(image_size=512, pixel_size_nm=2.0)
    generator = GratingGenerator(config)

    # Test different parameters
    patterns = []
    metadatas = []

    # Vary pitch
    for pitch in [50, 100, 150]:
        pattern, metadata = generator.generate(
            pitch_nm=pitch,
            duty_cycle=0.5,
            orientation_deg=0,
            add_ler=True
        )
        patterns.append(pattern)
        metadatas.append(metadata)
        print(f"  Generated grating: pitch={pitch}nm")

    # Vary orientation
    pattern, metadata = generator.generate(
        pitch_nm=100,
        duty_cycle=0.5,
        orientation_deg=45,
        add_ler=True
    )
    patterns.append(pattern)
    metadatas.append(metadata)
    print(f"  Generated grating: 45° orientation")

    return patterns, metadatas


def test_contact_hole_generator():
    """Test contact hole pattern generation."""
    print("\nTesting Contact Hole Generator...")

    config = PatternConfig(image_size=512, pixel_size_nm=2.0)
    generator = ContactHoleGenerator(config)

    patterns = []
    metadatas = []

    # Circular regular array
    pattern, metadata = generator.generate(
        diameter_nm=50,
        pitch_nm=120,
        shape='circular',
        array_type='regular',
        add_ler=True
    )
    patterns.append(pattern)
    metadatas.append(metadata)
    print(f"  Generated circular contacts (regular)")

    # Square staggered array
    pattern, metadata = generator.generate(
        diameter_nm=60,
        pitch_nm=150,
        shape='square',
        array_type='staggered',
        add_ler=True
    )
    patterns.append(pattern)
    metadatas.append(metadata)
    print(f"  Generated square contacts (staggered)")

    return patterns, metadatas


def test_isolated_feature_generator():
    """Test isolated feature pattern generation."""
    print("\nTesting Isolated Feature Generator...")

    config = PatternConfig(image_size=512, pixel_size_nm=2.0)
    generator = IsolatedFeatureGenerator(config)

    patterns = []
    metadatas = []

    # Line
    pattern, metadata = generator.generate(
        feature_type='line',
        width_nm=50,
        orientation_deg=0,
        add_ler=True
    )
    patterns.append(pattern)
    metadatas.append(metadata)
    print(f"  Generated isolated line")

    # Space
    pattern, metadata = generator.generate(
        feature_type='space',
        width_nm=60,
        orientation_deg=90,
        add_ler=True
    )
    patterns.append(pattern)
    metadatas.append(metadata)
    print(f"  Generated isolated space")

    # Post
    pattern, metadata = generator.generate(
        feature_type='post',
        width_nm=80,
        add_ler=True
    )
    patterns.append(pattern)
    metadatas.append(metadata)
    print(f"  Generated isolated post")

    return patterns, metadatas


def test_pattern_statistics():
    """Test pattern with detailed statistics."""
    print("\nGenerating detailed statistics for sample pattern...")

    config = PatternConfig(image_size=512, pixel_size_nm=2.0)
    generator = GratingGenerator(config)

    pattern, metadata = generator.generate(
        pitch_nm=80,
        duty_cycle=0.5,
        orientation_deg=0,
        add_ler=True,
        ler_sigma_nm=2.0,
        ler_correlation_nm=20.0
    )

    visualizer = PatternVisualizer()
    visualizer.visualize_statistics(pattern, metadata)

    print(f"  Pattern shape: {pattern.shape}")
    print(f"  Value range: [{pattern.min():.3f}, {pattern.max():.3f}]")
    print(f"  Mean: {pattern.mean():.3f}")
    print(f"  Std: {pattern.std():.3f}")


def main():
    """Main test function."""
    print("="*70)
    print("Pattern Generator Validation Test")
    print("="*70)

    visualizer = PatternVisualizer()

    # Test gratings
    grating_patterns, grating_metadatas = test_grating_generator()
    print(f"\n  Visualizing {len(grating_patterns)} grating patterns...")
    visualizer.visualize_multiple_patterns(
        grating_patterns,
        grating_metadatas,
        save_path=project_root / "results" / "test_gratings.png",
        show=False
    )

    # Test contact holes
    contact_patterns, contact_metadatas = test_contact_hole_generator()
    print(f"  Visualizing {len(contact_patterns)} contact patterns...")
    visualizer.visualize_multiple_patterns(
        contact_patterns,
        contact_metadatas,
        save_path=project_root / "results" / "test_contacts.png",
        show=False
    )

    # Test isolated features
    isolated_patterns, isolated_metadatas = test_isolated_feature_generator()
    print(f"  Visualizing {len(isolated_patterns)} isolated features...")
    visualizer.visualize_multiple_patterns(
        isolated_patterns,
        isolated_metadatas,
        save_path=project_root / "results" / "test_isolated.png",
        show=False
    )

    # Detailed statistics
    test_pattern_statistics()

    # Test one pattern with profile
    print("\n  Generating pattern with line profiles...")
    config = PatternConfig(image_size=512, pixel_size_nm=2.0)
    generator = GratingGenerator(config)
    pattern, metadata = generator.generate(pitch_nm=100, duty_cycle=0.5, add_ler=True)
    visualizer.visualize_pattern_with_profile(
        pattern,
        metadata,
        save_path=project_root / "results" / "test_profile.png",
        show=False
    )

    print("\n" + "="*70)
    print("✓ All tests completed successfully!")
    print(f"  Results saved to: {project_root / 'results'}")
    print("="*70)


if __name__ == '__main__':
    # Create results directory if it doesn't exist
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    main()
