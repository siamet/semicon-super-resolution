# Physics-Informed Deep Learning Super-Resolution for Semiconductor Inspection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Master's Thesis Research Project**
> Bridging the resolution gap between optical and electron microscopy in semiconductor manufacturing through physics-informed deep learning.

---

## 🎯 Project Overview

This research investigates physics-informed deep learning super-resolution techniques for enhancing automated optical inspection (AOI) systems in semiconductor manufacturing. As the industry transitions to sub-5nm process nodes, conventional optical inspection faces fundamental diffraction limits preventing detection of critical 3-7nm defects.

### Research Objectives

- **Primary Goal**: Develop physics-informed SR methods that enhance optical inspection images toward electron microscopy quality while maintaining metrological accuracy
- **Target Performance**:
  - >3dB PSNR improvement over traditional deconvolution
  - <5% critical dimension measurement error
  - >95% defect detection recall
  - Successful sim-to-real transfer (>85% performance retention)

### Key Innovations

1. **Physics-Informed Architecture**: Integration of optical system models (PSF/OTF) into neural network architectures and loss functions
2. **Semiconductor-Specific Benchmarking**: First comprehensive benchmark for SR evaluation on semiconductor patterns
3. **Hallucination Mitigation**: Multi-modal detection strategies to ensure metrological fidelity
4. **Uncertainty Quantification**: Confidence estimation for automated decision-making

---

## 📊 Current Project Status

**Phase**: Phase 1 - Foundation Development (29% Complete)
**Timeline**: 13-18 month master's thesis project
**Current Progress**: Month 1, Week 2 Complete
**Last Updated**: 2025-10-15

### ✅ Completed

#### Week 1: Environment Setup (2025-10-08)
- [x] Comprehensive research proposal (74KB detailed analysis)
- [x] System architecture design
- [x] Technology stack selection
- [x] Development workflow definition
- [x] Success metrics establishment
- [x] Complete project structure (80+ directories)
- [x] Environment setup (requirements.txt, environment.yml)
- [x] Git configuration (.gitignore, .gitattributes, LICENSE)
- [x] Configuration files (5 comprehensive YAML files)

#### Week 2: Pattern Generation (2025-10-15)
- [x] **Pattern generator module** (`src/data/synthetic/pattern_generator.py`)
  - [x] 3 pattern types: gratings, contact holes, isolated features
  - [x] Line edge roughness (LER) modeling
  - [x] Corner rounding for lithographic realism
  - [x] Comprehensive input validation
  - [x] Factory pattern implementation
- [x] **Visualizer module** (`src/data/synthetic/visualizer.py`)
  - [x] 4 visualization modes (simple, profile, stats, multi-pattern)
  - [x] Physical scale display (nanometers)
- [x] **Unit tests**: 39 tests, 100% passing
- [x] **Demo script**: 7 example scenarios
- [x] **Documentation**: Progress report + Quick Start Guide

### 🔄 In Progress (Week 3)
- [ ] PSF/OTF modeling (Airy disk, Hopkins formulation)
- [ ] Image degradation pipeline (convolution + noise)
- [ ] Batch dataset generation

### ⏳ Upcoming (Week 4)
- [ ] Baseline methods (Richardson-Lucy, Wiener filtering)
- [ ] Evaluation metrics framework
- [ ] First 5,000 synthetic image pairs generated

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Input: Low-Res Optical Image              │
│                   + Optical System Parameters               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Physics-Informed SR Models                      │
│  ┌──────────┬─────────┬──────────┬─────────┬──────────┐   │
│  │  U-Net   │  RCAN   │ ESRGAN   │ SwinIR  │   HAT    │   │
│  └──────────┴─────────┴──────────┴─────────┴──────────┘   │
│                                                              │
│  Physics Components:                                         │
│  • PSF/OTF Models                                           │
│  • Optical Parameter Conditioning                            │
│  • Physics-Informed Loss Functions                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Output: Super-Resolved Image                    │
│              + Uncertainty Map                               │
│              + Hallucination Detection                       │
└─────────────────────────────────────────────────────────────┘
```

### Core Components

- **Data Pipeline**: Synthetic pattern generation + real microscope data
- **Physics Modeling**: PSF/OTF simulation for optical degradation
- **Model Zoo**: 6 baseline SR architectures
- **Training**: Physics-informed loss functions with optical constraints
- **Evaluation**: Standard + semiconductor-specific metrics
- **Analysis**: Uncertainty quantification, hallucination detection

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended: RTX 3060+ for development)
- 16GB+ system RAM
- ~50GB storage for development

### Installation

```bash
# Clone the repository
git clone https://github.com/siamet/semicon-super-resolution.git
cd semicon-super-resolution

# Option 1: Conda environment (recommended)
conda env create -f environment.yml
conda activate semicon-sr

# Option 2: Pip in virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import src; print('Package structure ready!')"
```

### Run Pattern Generation Demo

```bash
# Run all demos (interactive mode)
python scripts/demo_pattern_generation.py

# Run specific demo
python scripts/demo_pattern_generation.py --demo 3  # LER comparison

# Save outputs to directory
python scripts/demo_pattern_generation.py --save-path results/demo_output

# Run unit tests
pytest tests/data/synthetic/test_pattern_generator.py -v
```

### Generate Custom Patterns

```python
from src.data.synthetic.pattern_generator import PatternConfig, GratingGenerator
from src.data.synthetic.visualizer import PatternVisualizer

# Configure and generate
config = PatternConfig(image_size=512, pixel_size_nm=2.0)
generator = GratingGenerator(config)
pattern, metadata = generator.generate(
    pitch_nm=80.0,
    duty_cycle=0.5,
    add_ler=True,
    ler_sigma_nm=2.0
)

# Visualize
visualizer = PatternVisualizer()
visualizer.visualize_pattern(pattern, metadata, show_physical_scale=True)
```

See [Pattern Generator Quick Start Guide](docs/PATTERN_GENERATOR_QUICKSTART.md) for detailed usage examples.

### Train Models

```bash
# Train baseline traditional methods
python scripts/training/train_traditional.py \
    --method richardson_lucy \
    --data_dir data/processed/train

# Train deep learning models
python scripts/training/train_deep_models.py \
    --model swinir \
    --config config/training_config.yaml \
    --checkpoint_dir models/checkpoints

# Train with physics-informed loss
python scripts/training/train_deep_models.py \
    --model swinir \
    --config config/training_config.yaml \
    --physics_informed \
    --wavelength 248 \
    --NA 0.95
```

### Evaluate Performance

```bash
# Run comprehensive benchmark
python scripts/evaluation/run_benchmark.py \
    --checkpoint models/final/swinir_best.pth \
    --test_data data/processed/test \
    --output_dir results/benchmarks

# Compare multiple methods
python scripts/evaluation/compare_methods.py \
    --methods unet rcan esrgan swinir \
    --test_data data/processed/test
```

---

## 📁 Project Structure

```
semicon-super-resolution/
├── README.md                          # This file
├── RESEARCH_PROPOSAL.md              # Comprehensive research proposal
├── CLAUDE.md                         # AI development context
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda environment
│
├── config/                          # Configuration files
│   ├── data_config.yaml            # Data generation parameters
│   ├── model_config.yaml           # Model hyperparameters
│   └── training_config.yaml        # Training configurations
│
├── src/                            # Source code (to be implemented)
│   ├── data/                       # Data pipeline
│   ├── models/                     # SR model implementations
│   ├── physics/                    # PSF/OTF modeling
│   ├── training/                   # Training infrastructure
│   ├── evaluation/                 # Metrics and benchmarking
│   └── analysis/                   # Analysis tools
│
├── scripts/                        # Executable scripts
│   ├── data_generation/           # Data generation
│   ├── training/                  # Training scripts
│   ├── evaluation/                # Evaluation scripts
│   └── analysis/                  # Analysis scripts
│
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_analysis.ipynb
│   └── ...
│
├── docs/                          # Documentation
│   ├── architecture/              # Architecture design
│   ├── installation.md           # Setup instructions
│   └── api_reference/            # API documentation
│
├── data/                          # Data directory (gitignored)
│   ├── raw/                       # Raw data
│   └── processed/                 # Processed datasets
│
├── models/                        # Trained models (gitignored)
│   ├── checkpoints/              # Training checkpoints
│   └── final/                    # Final models
│
├── results/                       # Experimental results
│   ├── benchmarks/               # Benchmark results
│   ├── figures/                  # Generated figures
│   └── logs/                     # Training logs
│
└── tests/                         # Unit tests
    ├── test_data/
    ├── test_models/
    └── test_evaluation/
```

---

## 🔬 Research Methodology

### Phase 1: Foundation (Months 1-2)
- Synthetic data pipeline development
- Baseline method implementation
- Evaluation framework establishment

### Phase 2: Model Development (Months 3-6)
- Core CNN models (U-Net, RCAN, ESRGAN)
- Advanced transformers (SwinIR, HAT)
- Physics-informed modifications

### Phase 3: Training & Validation (Months 7-10)
- Hyperparameter optimization
- Synthetic benchmarking
- Real-world validation on microscope data

### Phase 4: Analysis & Synthesis (Months 11-13)
- Failure mode analysis
- Uncertainty quantification
- Thesis writing and publication

---

## 📊 Evaluation Metrics

### Standard Image Quality
- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)
- **LPIPS** (Learned Perceptual Image Patch Similarity)

### Semiconductor-Specific
- **CD Error**: Critical dimension measurement accuracy
- **EPE**: Edge placement error
- **Defect Detection**: Precision, recall, F1 score
- **Hallucination Rate**: False feature generation

---

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch 2.1.0, torchvision
- **Vision Transformers**: timm (PyTorch Image Models)
- **Scientific Computing**: numpy, scipy, scikit-image
- **Image Processing**: OpenCV, tifffile
- **Optical Simulation**: Custom PSF/OTF models
- **Experiment Tracking**: Weights & Biases, TensorBoard
- **Deployment**: ONNX Runtime, TensorRT

---

## 📚 Key References

1. Liang et al., "SwinIR: Image Restoration Using Swin Transformer" (ICCV 2021)
2. Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution" (ICCV 2021)
3. Chen et al., "HAT: Hybrid Attention Transformer for Image Restoration" (2023)
4. Ren et al., "Physics-informed Deep Super-resolution for Spatiotemporal Data" (JCP 2023)
5. Kim et al., "Electron Microscopy-based Automatic Defect Inspection" (arXiv 2024)

See [RESEARCH_PROPOSAL.md](RESEARCH_PROPOSAL.md) for comprehensive literature review.

---

## 🎓 Academic Context

**Type**: Master's Thesis Research  
**Duration**: 13-18 months
**Field**: Computer Vision × Semiconductor Manufacturing    
**Institution**: National Taiwan University     
**Advisor**: Dr. Liang Chia Chen

### Expected Outcomes
- Master's thesis document
- 2-4 peer-reviewed publications
- Open-source software package
- Pre-trained model weights
- Comprehensive benchmarking dataset

---

## 📈 Success Criteria

**Technical Performance**:
- ✓ >3dB PSNR improvement over deconvolution
- ✓ <5% CD measurement error
- ✓ >95% defect detection recall
- ✓ >10 fps inference on single GPU

**Scientific Contribution**:
- ✓ First semiconductor-specific SR benchmark
- ✓ Physics-informed methodologies
- ✓ Hallucination mitigation strategies
- ✓ Uncertainty quantification framework

**Practical Impact**:
- ✓ Successful sim-to-real transfer
- ✓ Production-ready deployment pipeline
- ✓ Industry collaboration/validation
- ✓ Open-source community adoption

---

## 🤝 Contributing

This is a research project for a master's thesis. While direct code contributions are not being accepted at this time, feedback, suggestions, and discussions are welcome:

- **Issues**: Report bugs or suggest features via GitHub Issues
- **Discussions**: Join technical discussions in GitHub Discussions
- **Citations**: If you use this work, please cite appropriately (citation format TBD)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Thesis advisor and research group
- Semiconductor industry partners for data and validation
- Open-source community for foundational models (SwinIR, ESRGAN, etc.)
- Funding sources (to be acknowledged)

---

## 📝 Development Notes

For comprehensive research background and methodology, see [RESEARCH_PROPOSAL.md](RESEARCH_PROPOSAL.md).

For system architecture and design patterns, see [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md).

For detailed development context and AI collaboration guidelines, see [CLAUDE.md](CLAUDE.md).

---

**🚧 Note**: This project is in the early planning phase. The codebase is currently being implemented based on the completed architecture design. Check back for updates as development progresses!
