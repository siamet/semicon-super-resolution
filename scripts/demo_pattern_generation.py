#!/usr/bin/env python3
"""
Demo Script: Synthetic Pattern Generation

This script demonstrates how to use the pattern generation module to create
various semiconductor patterns including gratings, contact holes, and isolated features.

Usage:
    python scripts/demo_pattern_generation.py [--save-path OUTPUT_DIR]
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic.pattern_generator import (
    PatternConfig,
    GratingGenerator,
    ContactHoleGenerator,
    IsolatedFeatureGenerator,
    create_pattern_generator
)
from src.data.synthetic.visualizer import PatternVisualizer, quick_visualize


def demo_basic_patterns(save_path: Path = None):
    """Demonstrate basic pattern generation for each type."""
    print("\n" + "="*60)
    print("DEMO 1: Basic Pattern Generation")
    print("="*60)

    config = PatternConfig(image_size=256, pixel_size_nm=2.0)
    visualizer = PatternVisualizer(figsize=(15, 5))

    patterns = []
    metadatas = []

    # Grating
    print("\n1. Generating vertical grating (pitch=80nm, duty_cycle=0.5)...")
    grating_gen = GratingGenerator(config)
    pattern_grating, meta_grating = grating_gen.generate(
        pitch_nm=80.0,
        duty_cycle=0.5,
        orientation_deg=0.0,
        add_ler=False
    )
    patterns.append(pattern_grating)
    metadatas.append(meta_grating)
    print(f"   ✓ Generated {pattern_grating.shape} pattern")

    # Contact holes
    print("\n2. Generating circular contact hole array (diameter=50nm, pitch=100nm)...")
    contact_gen = ContactHoleGenerator(config)
    pattern_contacts, meta_contacts = contact_gen.generate(
        diameter_nm=50.0,
        pitch_nm=100.0,
        shape='circular',
        array_type='regular',
        add_ler=False
    )
    patterns.append(pattern_contacts)
    metadatas.append(meta_contacts)
    print(f"   ✓ Generated {pattern_contacts.shape} pattern")

    # Isolated line
    print("\n3. Generating isolated line (width=40nm, length=300nm)...")
    isolated_gen = IsolatedFeatureGenerator(config)
    pattern_line, meta_line = isolated_gen.generate(
        feature_type='line',
        width_nm=40.0,
        length_nm=300.0,
        orientation_deg=0.0,
        add_ler=False
    )
    patterns.append(pattern_line)
    metadatas.append(meta_line)
    print(f"   ✓ Generated {pattern_line.shape} pattern")

    # Visualize all
    print("\n4. Visualizing patterns...")
    save_file = save_path / "demo1_basic_patterns.png" if save_path else None
    visualizer.visualize_multiple_patterns(
        patterns, metadatas,
        save_path=save_file,
        show=save_path is None
    )
    if save_file:
        print(f"   ✓ Saved to {save_file}")


def demo_grating_variations(save_path: Path = None):
    """Demonstrate grating variations (pitch, duty cycle, orientation)."""
    print("\n" + "="*60)
    print("DEMO 2: Grating Variations")
    print("="*60)

    config = PatternConfig(image_size=256, pixel_size_nm=2.0)
    generator = GratingGenerator(config)
    visualizer = PatternVisualizer(figsize=(16, 12))

    patterns = []
    metadatas = []

    variations = [
        # (pitch, duty_cycle, orientation, description)
        (60.0, 0.5, 0.0, "Vertical, 60nm pitch"),
        (60.0, 0.3, 0.0, "Vertical, 30% duty cycle"),
        (60.0, 0.7, 0.0, "Vertical, 70% duty cycle"),
        (60.0, 0.5, 45.0, "45° rotation"),
        (40.0, 0.5, 0.0, "Dense: 40nm pitch"),
        (120.0, 0.5, 0.0, "Sparse: 120nm pitch"),
    ]

    print("\nGenerating grating variations:")
    for pitch, dc, angle, desc in variations:
        print(f"   - {desc}")
        pattern, metadata = generator.generate(
            pitch_nm=pitch,
            duty_cycle=dc,
            orientation_deg=angle,
            add_ler=False
        )
        patterns.append(pattern)
        metadatas.append(metadata)

    save_file = save_path / "demo2_grating_variations.png" if save_path else None
    visualizer.visualize_multiple_patterns(
        patterns, metadatas,
        save_path=save_file,
        show=save_path is None
    )
    if save_file:
        print(f"   ✓ Saved to {save_file}")


def demo_line_edge_roughness(save_path: Path = None):
    """Demonstrate line edge roughness effects."""
    print("\n" + "="*60)
    print("DEMO 3: Line Edge Roughness (LER)")
    print("="*60)

    config = PatternConfig(image_size=256, pixel_size_nm=2.0)
    generator = GratingGenerator(config)
    visualizer = PatternVisualizer()

    print("\nComparing patterns with and without LER:")

    # Without LER
    print("   - Generating ideal pattern (no LER)...")
    pattern_ideal, meta_ideal = generator.generate(
        pitch_nm=80.0,
        duty_cycle=0.5,
        add_ler=False
    )

    # With LER
    print("   - Generating pattern with LER (sigma=2nm, correlation=30nm)...")
    pattern_ler, meta_ler = generator.generate(
        pitch_nm=80.0,
        duty_cycle=0.5,
        add_ler=True,
        ler_sigma_nm=2.0,
        ler_correlation_nm=30.0
    )

    # Visualize with profiles
    save_file1 = save_path / "demo3_ler_comparison_ideal.png" if save_path else None
    visualizer.visualize_pattern_with_profile(
        pattern_ideal, meta_ideal,
        title="Ideal Grating (No LER)",
        save_path=save_file1,
        show=save_path is None
    )

    save_file2 = save_path / "demo3_ler_comparison_realistic.png" if save_path else None
    visualizer.visualize_pattern_with_profile(
        pattern_ler, meta_ler,
        title="Realistic Grating (With LER)",
        save_path=save_file2,
        show=save_path is None
    )

    if save_file1:
        print(f"   ✓ Saved to {save_file1}")
        print(f"   ✓ Saved to {save_file2}")


def demo_lwr_vs_ler(save_path: Path = None):
    """Demonstrate LWR vs LER comparison."""
    print("\n" + "="*60)
    print("DEMO 3B: Line Width Roughness (LWR) vs Line Edge Roughness (LER)")
    print("="*60)

    config = PatternConfig(image_size=256, pixel_size_nm=2.0)
    generator = GratingGenerator(config)
    visualizer = PatternVisualizer(figsize=(15, 5))

    print("\nComparing three roughness models:")
    print("   1. Ideal pattern (no roughness)")
    print("   2. LER (independent edge noise)")
    print("   3. LWR (correlated edge noise - more realistic)")

    # Ideal pattern
    print("   - Generating ideal pattern...")
    pattern_ideal, meta_ideal = generator.generate(
        pitch_nm=80.0,
        duty_cycle=0.5,
        add_ler=False,
        add_lwr=False
    )

    # Pattern with LER (independent edges)
    print("   - Generating pattern with LER (independent edges)...")
    pattern_ler, meta_ler = generator.generate(
        pitch_nm=80.0,
        duty_cycle=0.5,
        add_ler=True,
        ler_sigma_nm=2.0,
        ler_correlation_nm=30.0,
        add_lwr=False
    )

    # Pattern with LWR (correlated edges)
    print("   - Generating pattern with LWR (correlated edges, rho=0.5)...")
    pattern_lwr, meta_lwr = generator.generate(
        pitch_nm=80.0,
        duty_cycle=0.5,
        add_ler=False,
        add_lwr=True,
        lwr_sigma_nm=1.5,
        lwr_correlation_nm=30.0,
        lwr_edge_correlation=0.5
    )

    # Create side-by-side comparison
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Ideal
    ax = axes[0]
    ax.imshow(pattern_ideal, cmap='gray', origin='upper')
    ax.set_title("Ideal (No Roughness)", fontsize=12, fontweight='bold')
    ax.axis('off')

    # LER
    ax = axes[1]
    ax.imshow(pattern_ler, cmap='gray', origin='upper')
    ax.set_title("LER (Independent Edges)\nσ=2.0nm", fontsize=12, fontweight='bold')
    ax.axis('off')

    # LWR
    ax = axes[2]
    ax.imshow(pattern_lwr, cmap='gray', origin='upper')
    ax.set_title("LWR (Correlated Edges)\nσ=1.5nm, ρ=0.5", fontsize=12, fontweight='bold')
    ax.axis('off')

    plt.tight_layout()

    if save_path:
        save_file = save_path / "demo3b_lwr_vs_ler.png"
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved to {save_file}")
        plt.close()
    else:
        plt.show()

    print("\n   Key Differences:")
    print("   - LER: Each edge varies independently (unrealistic)")
    print("   - LWR: Edges are partially correlated (realistic lithography)")
    print("   - LWR captures width variations from shared resist/developer effects")


def demo_contact_hole_arrays(save_path: Path = None):
    """Demonstrate contact hole array variations."""
    print("\n" + "="*60)
    print("DEMO 4: Contact Hole Arrays")
    print("="*60)

    config = PatternConfig(image_size=256, pixel_size_nm=2.0)
    generator = ContactHoleGenerator(config)
    visualizer = PatternVisualizer(figsize=(12, 8))

    patterns = []
    metadatas = []

    variations = [
        # (diameter, shape, array_type, description)
        (40.0, 'circular', 'regular', "Circular, regular array"),
        (40.0, 'circular', 'staggered', "Circular, staggered array"),
        (40.0, 'square', 'regular', "Square, regular array"),
        (40.0, 'square', 'staggered', "Square, staggered array"),
    ]

    print("\nGenerating contact hole variations:")
    for diameter, shape, array_type, desc in variations:
        print(f"   - {desc}")
        pattern, metadata = generator.generate(
            diameter_nm=diameter,
            pitch_nm=80.0,
            shape=shape,
            array_type=array_type,
            add_ler=False
        )
        patterns.append(pattern)
        metadatas.append(metadata)

    save_file = save_path / "demo4_contact_arrays.png" if save_path else None
    visualizer.visualize_multiple_patterns(
        patterns, metadatas,
        save_path=save_file,
        show=save_path is None
    )
    if save_file:
        print(f"   ✓ Saved to {save_file}")


def demo_corner_rounding(save_path: Path = None):
    """Demonstrate corner rounding for lithographic realism."""
    print("\n" + "="*60)
    print("DEMO 5: Corner Rounding (Optical Proximity Effects)")
    print("="*60)

    config = PatternConfig(image_size=256, pixel_size_nm=2.0)
    generator = ContactHoleGenerator(config)
    visualizer = PatternVisualizer(figsize=(15, 5))

    print("\nGenerating square contacts with different corner rounding:")

    # Sharp corners
    print("   - Sharp corners (no rounding)...")
    pattern_sharp, meta_sharp = generator.generate(
        diameter_nm=60.0,
        pitch_nm=120.0,
        shape='square',
        add_ler=False
    )

    # Slight rounding
    print("   - Slight rounding (5nm blur)...")
    pattern_rounded = generator.add_corner_rounding(pattern_sharp, blur_sigma_nm=5.0)

    # Heavy rounding
    print("   - Heavy rounding (10nm blur)...")
    pattern_heavy_rounded = generator.add_corner_rounding(pattern_sharp, blur_sigma_nm=10.0)

    patterns = [pattern_sharp, pattern_rounded, pattern_heavy_rounded]
    titles = ["Sharp Corners", "Slight Rounding (5nm)", "Heavy Rounding (10nm)"]

    save_file = save_path / "demo5_corner_rounding.png" if save_path else None
    visualizer.visualize_multiple_patterns(
        patterns,
        titles=titles,
        save_path=save_file,
        show=save_path is None
    )
    if save_file:
        print(f"   ✓ Saved to {save_file}")


def demo_statistics_analysis(save_path: Path = None):
    """Demonstrate pattern statistics visualization."""
    print("\n" + "="*60)
    print("DEMO 6: Pattern Statistics Analysis")
    print("="*60)

    config = PatternConfig(image_size=256, pixel_size_nm=2.0)
    generator = GratingGenerator(config)
    visualizer = PatternVisualizer()

    print("\nGenerating grating with LER and analyzing statistics...")
    pattern, metadata = generator.generate(
        pitch_nm=80.0,
        duty_cycle=0.5,
        orientation_deg=45.0,
        add_ler=True,
        ler_sigma_nm=2.0
    )

    save_file = save_path / "demo6_statistics.png" if save_path else None
    visualizer.visualize_statistics(
        pattern, metadata,
        save_path=save_file,
        show=save_path is None
    )
    if save_file:
        print(f"   ✓ Saved to {save_file}")

    # Print some statistics
    print(f"\nPattern Statistics:")
    print(f"   - Shape: {pattern.shape}")
    print(f"   - Min/Max: {pattern.min():.3f} / {pattern.max():.3f}")
    print(f"   - Mean: {pattern.mean():.3f}")
    print(f"   - Std Dev: {pattern.std():.3f}")


def demo_factory_pattern(save_path: Path = None):
    """Demonstrate factory function usage."""
    print("\n" + "="*60)
    print("DEMO 7: Using Factory Pattern")
    print("="*60)

    config = PatternConfig(image_size=256, pixel_size_nm=2.0)
    visualizer = PatternVisualizer(figsize=(15, 5))

    patterns = []
    metadatas = []

    print("\nUsing create_pattern_generator() factory:")

    # Create generators using factory
    pattern_types = ['grating', 'contacts', 'isolated']
    params = [
        {'pitch_nm': 60.0, 'duty_cycle': 0.5, 'add_ler': False},
        {'diameter_nm': 40.0, 'shape': 'circular', 'add_ler': False},
        {'feature_type': 'line', 'width_nm': 30.0, 'length_nm': 200.0, 'add_ler': False}
    ]

    for ptype, param in zip(pattern_types, params):
        print(f"   - Creating {ptype} generator...")
        generator = create_pattern_generator(ptype, config)
        pattern, metadata = generator.generate(**param)
        patterns.append(pattern)
        metadatas.append(metadata)

    save_file = save_path / "demo7_factory_pattern.png" if save_path else None
    visualizer.visualize_multiple_patterns(
        patterns, metadatas,
        save_path=save_file,
        show=save_path is None
    )
    if save_file:
        print(f"   ✓ Saved to {save_file}")


def main():
    """Run all demos."""
    parser = argparse.ArgumentParser(
        description="Demo script for synthetic pattern generation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--save-path',
        type=str,
        default=None,
        help='Directory to save output images (if not specified, displays interactively)'
    )
    parser.add_argument(
        '--demo',
        type=str,
        choices=['all', '1', '2', '3', '3b', '4', '5', '6', '7'],
        default='all',
        help='Which demo to run (default: all)'
    )

    args = parser.parse_args()

    # Setup save path
    save_path = None
    if args.save_path:
        save_path = Path(args.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput will be saved to: {save_path.absolute()}")
    else:
        print("\nRunning in interactive mode (displaying plots)")

    print("\n" + "="*60)
    print("SEMICONDUCTOR PATTERN GENERATION DEMO")
    print("="*60)
    print(f"Running demo: {args.demo}")

    # Run selected demos
    demos = {
        '1': demo_basic_patterns,
        '2': demo_grating_variations,
        '3': demo_line_edge_roughness,
        '3b': demo_lwr_vs_ler,
        '4': demo_contact_hole_arrays,
        '5': demo_corner_rounding,
        '6': demo_statistics_analysis,
        '7': demo_factory_pattern,
    }

    if args.demo == 'all':
        for demo_func in demos.values():
            demo_func(save_path)
    else:
        demos[args.demo](save_path)

    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    if save_path:
        print(f"\nAll outputs saved to: {save_path.absolute()}")
    print("\nFor more information, see:")
    print("  - src/data/synthetic/pattern_generator.py")
    print("  - src/data/synthetic/visualizer.py")
    print("  - tests/data/synthetic/test_pattern_generator.py")
    print()


if __name__ == '__main__':
    main()
