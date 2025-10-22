# Pattern Generator Quick Start Guide

A practical guide to using the synthetic semiconductor pattern generation module.

---

## Installation & Setup

```bash
# Ensure you're in the project root
cd /path/to/semicon-super-resolution

# Activate environment
conda activate semicon-sr
# OR
source venv/bin/activate

# Run tests to verify installation
pytest tests/data/synthetic/test_pattern_generator.py
```

---

## Basic Usage

### 1. Generate a Simple Grating

```python
from src.data.synthetic.pattern_generator import PatternConfig, GratingGenerator

# Configure pattern generation
config = PatternConfig(image_size=512, pixel_size_nm=2.0)

# Create generator
generator = GratingGenerator(config)

# Generate pattern
pattern, metadata = generator.generate(
    pitch_nm=80.0,       # 80nm period
    duty_cycle=0.5,      # 50% lines, 50% spaces
    orientation_deg=0.0, # Vertical lines
    add_ler=False        # No line edge roughness
)

print(f"Pattern shape: {pattern.shape}")
print(f"Value range: [{pattern.min():.2f}, {pattern.max():.2f}]")
print(f"Metadata: {metadata}")
```

### 2. Generate Contact Holes

```python
from src.data.synthetic.pattern_generator import ContactHoleGenerator

generator = ContactHoleGenerator(config)

pattern, metadata = generator.generate(
    diameter_nm=50.0,
    pitch_nm=100.0,
    shape='circular',      # or 'square'
    array_type='regular',  # or 'staggered'
    add_ler=False
)
```

### 3. Generate Isolated Feature

```python
from src.data.synthetic.pattern_generator import IsolatedFeatureGenerator

generator = IsolatedFeatureGenerator(config)

pattern, metadata = generator.generate(
    feature_type='line',  # or 'space' or 'post'
    width_nm=40.0,
    length_nm=300.0,
    orientation_deg=45.0,
    add_ler=False
)
```

---

## Advanced Features

### Add Line Width Roughness (LWR) - **Recommended for Realistic Patterns**

```python
# Generate with realistic line width roughness (correlated edges)
pattern, metadata = generator.generate(
    pitch_nm=80.0,
    duty_cycle=0.5,
    add_lwr=True,
    lwr_sigma_nm=1.5,              # 1σ = 1.5nm (typically < LER)
    lwr_correlation_nm=30.0,       # 30nm correlation length
    lwr_edge_correlation=0.5       # 0.5 = partially correlated edges
)
```

**Key Points:**
- **LWR models correlated roughness** on both edges (realistic for lithography)
- `lwr_edge_correlation`: 0 = independent edges, 1 = identical edges, 0.3-0.7 = realistic
- LWR captures shared effects: resist properties, developer uniformity
- **Use LWR by default** for production datasets

### Add Line Edge Roughness (LER) - **For Testing Only**

```python
# Generate with independent edge roughness (less realistic)
pattern, metadata = generator.generate(
    pitch_nm=80.0,
    duty_cycle=0.5,
    add_ler=True,
    ler_sigma_nm=2.0,        # 1σ = 2nm (3σ = 6nm)
    ler_correlation_nm=30.0,  # 30nm correlation length
    add_lwr=False            # Disable LWR when using LER
)
```

**Note:** LER and LWR are mutually exclusive. LWR is preferred for realistic patterns.

### Add Corner Rounding

```python
# Generate sharp pattern
pattern_sharp, _ = generator.generate(pitch_nm=80.0, add_ler=False)

# Add corner rounding (simulates optical proximity effects)
pattern_rounded = generator.add_corner_rounding(pattern_sharp, blur_sigma_nm=5.0)
```

### Use Factory Pattern

```python
from src.data.synthetic.pattern_generator import create_pattern_generator

# Create generator by type
generator = create_pattern_generator('grating', config)
pattern, metadata = generator.generate(pitch_nm=60.0)

# Switch types easily
for pattern_type in ['grating', 'contacts', 'isolated']:
    gen = create_pattern_generator(pattern_type, config)
    # ... generate patterns
```

---

## Visualization

### Simple Visualization

```python
from src.data.synthetic.visualizer import quick_visualize

# Quick display
quick_visualize(pattern, metadata, mode='simple')

# With line profiles
quick_visualize(pattern, metadata, mode='profile')

# With statistics (histogram, FFT)
quick_visualize(pattern, metadata, mode='stats')
```

### Advanced Visualization

```python
from src.data.synthetic.visualizer import PatternVisualizer

visualizer = PatternVisualizer()

# Single pattern with physical scale (nm instead of pixels)
visualizer.visualize_pattern(
    pattern, 
    metadata,
    save_path='output/grating.png',
    show=True,
    show_physical_scale=True  # Show nm instead of pixels
)

# Multiple patterns in grid
patterns = [pattern1, pattern2, pattern3]
metadatas = [meta1, meta2, meta3]
visualizer.visualize_multiple_patterns(patterns, metadatas)
```

---

## Common Patterns & Parameters

### Typical Semiconductor Patterns

```python
# Dense metal lines (10nm node)
grating_10nm = generator.generate(pitch_nm=20.0, duty_cycle=0.5)

# Contact via array (7nm node)
contacts_7nm = generator.generate(diameter_nm=14.0, pitch_nm=28.0)

# Critical dimension (CD) measurement target
cd_target = generator.generate(width_nm=30.0, length_nm=200.0)
```

### Realistic Parameters

| Parameter | Typical Range | Notes |
|-----------|---------------|-------|
| **Pitch** | 20-500 nm | Technology node dependent |
| **Duty Cycle** | 0.3-0.7 | 0.5 = equal lines/spaces |
| **LWR σ** | 1-3 nm | 1σ value, typically < LER (RECOMMENDED) |
| **LWR Correlation** | 20-50 nm | Spatial correlation length |
| **LWR Edge Correlation** | 0.3-0.7 | 0 = independent, 1 = identical edges |
| **LER σ** | 1-5 nm | 1σ value (for testing only) |
| **LER Correlation** | 20-50 nm | Spatial correlation length |
| **Corner Rounding** | 3-10 nm | Depends on lithography |

---

## Complete Workflow Example

```python
import numpy as np
from pathlib import Path
from src.data.synthetic.pattern_generator import (
    PatternConfig, GratingGenerator
)
from src.data.synthetic.visualizer import PatternVisualizer

# 1. Configure
config = PatternConfig(image_size=512, pixel_size_nm=2.0)
generator = GratingGenerator(config)
visualizer = PatternVisualizer()

# 2. Generate multiple patterns
patterns = []
metadatas = []

for pitch in [40, 60, 80, 100]:
    pattern, metadata = generator.generate(
        pitch_nm=float(pitch),
        duty_cycle=0.5,
        add_lwr=True,
        lwr_sigma_nm=1.5,
        lwr_edge_correlation=0.5
    )
    patterns.append(pattern)
    metadatas.append(metadata)

# 3. Visualize
output_dir = Path('results/patterns')
output_dir.mkdir(parents=True, exist_ok=True)

visualizer.visualize_multiple_patterns(
    patterns,
    metadatas,
    save_path=output_dir / 'pitch_sweep.png',
    show=False
)

# 4. Save patterns as numpy arrays
for i, (pattern, meta) in enumerate(zip(patterns, metadatas)):
    np.save(output_dir / f'pattern_{i}.npy', pattern)
    
print(f"Generated {len(patterns)} patterns")
print(f"Saved to {output_dir}")
```

---

## Running the Demo Script

```bash
# Run all demos interactively (displays plots)
python scripts/demo_pattern_generation.py

# Run specific demo
python scripts/demo_pattern_generation.py --demo 1   # Basic patterns
python scripts/demo_pattern_generation.py --demo 3   # LER comparison
python scripts/demo_pattern_generation.py --demo 3b  # LWR vs LER (NEW!)

# Save all outputs to directory
python scripts/demo_pattern_generation.py --save-path results/demo_output

# Get help
python scripts/demo_pattern_generation.py --help
```

---

## Parameter Validation

The module validates inputs and provides helpful error messages:

```python
# This will raise ValueError
generator.generate(pitch_nm=-10.0)  # Error: pitch must be positive

# This will raise ValueError  
generator.generate(duty_cycle=1.5)  # Error: duty_cycle must be in [0, 1]

# This will warn (but still generate)
generator.generate(pitch_nm=2.0)  # Warning: pitch very small, use finer pixels
```

---

## Tips & Best Practices

### Resolution Selection
```python
# For fine features (<30nm), use smaller pixels
config_fine = PatternConfig(image_size=512, pixel_size_nm=1.0)

# For larger patterns, coarser pixels save memory
config_coarse = PatternConfig(image_size=256, pixel_size_nm=4.0)
```

### Reproducibility
```python
# Set random seed for reproducible LER
import numpy as np
np.random.seed(42)

generator.rng = np.random.default_rng(42)
pattern, _ = generator.generate(pitch_nm=80.0, add_ler=True)
# Same pattern will be generated with same seed
```

### Performance
```python
# Disable LER for faster generation during prototyping
pattern, _ = generator.generate(pitch_nm=80.0, add_ler=False)  # Fast

# Enable LER only for final dataset generation
pattern, _ = generator.generate(pitch_nm=80.0, add_ler=True)   # Slower
```

---

## Troubleshooting

### Import Errors
```python
# If you get "ModuleNotFoundError: No module named 'src'"
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Memory Issues
```python
# Reduce image size for memory-constrained environments
config = PatternConfig(image_size=256, pixel_size_nm=2.0)  # 256×256 instead of 512×512
```

### Visualization Not Showing
```python
# Ensure matplotlib backend is correct
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg' depending on your system
```

---

## API Reference

See docstrings in the code for complete API documentation:
- `src/data/synthetic/pattern_generator.py` - Pattern generation
- `src/data/synthetic/visualizer.py` - Visualization tools
- `tests/data/synthetic/test_pattern_generator.py` - Usage examples

---

## Further Reading

- **System Architecture**: `docs/architecture/ARCHITECTURE.md`
- **Research Proposal**: `RESEARCH_PROPOSAL.md`
- **Project Roadmap**: `ROADMAP.md`

---

**Last Updated**: October 15, 2025
