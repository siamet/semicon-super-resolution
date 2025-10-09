# CLAUDE.md - AI Development Context

This file provides comprehensive guidance to Claude Code when working with this repository. It defines coding standards, development practices, and project-specific context that should persist across all sessions.

## 🎯 Quick Session Start
- Use `/status` to check current progress and priorities
- Use `/prime` for comprehensive project analysis and context updates
- Current development focus: **[Set by /prime command]**
- Active development phase: **[Set by /prime command]**

---

## 🧱 Core Development Philosophy

### KISS (Keep It Simple, Stupid)
Simplicity should be a key goal in design. Choose straightforward solutions over complex ones whenever possible. Simple solutions are easier to understand, maintain, and debug.

### YAGNI (You Aren't Gonna Need It)
Avoid building functionality on speculation. Implement features only when they are needed, not when you anticipate they might be useful in the future.

### Design Principles
- **Dependency Inversion**: High-level modules should not depend on low-level modules. Both should depend on abstractions.
- **Open/Closed Principle**: Software entities should be open for extension but closed for modification.
- **Single Responsibility**: Each function, class, and module should have one clear purpose.
- **Fail Fast**: Check for potential errors early and raise exceptions immediately when issues occur.

---

## 🏗️ Code Structure & Standards

### File and Function Limits
- **Files should be under 500 lines**. If approaching this limit, refactor by splitting into modules/components
- **Functions should be under 50 lines** with a single, clear responsibility
- **Classes/Components should be under 100 lines** and represent a single concept or entity
- **Line length should be max 100-120 characters** (following project linting rules)
- **Use project-specific environment** (virtual env, node_modules, etc.) for all commands

### Code Organization
- **Organize code into clearly separated modules/components**, grouped by feature or responsibility
- **Follow consistent directory structure** as defined in project architecture
- **Maintain clear separation of concerns** between layers (UI, business logic, data)

---

## 🛠️ Technology Stack & Environment
**[Last updated: 2025-10-08 via /prime]**

### Primary Technologies
- **Language**: Python 3.10+
- **Deep Learning**: PyTorch 2.1.0 with CUDA 11.8/12.1
- **Computer Vision**: OpenCV, scikit-image, scipy
- **Vision Transformers**: timm (PyTorch Image Models)
- **Optical Simulation**: Custom PSF/OTF models, FDTD-based (Lumerical)
- **Process Simulation**: TCAD (Synopsys Sentaurus) for synthetic data
- **Testing**: pytest, unittest
- **Experiment Tracking**: Weights & Biases, TensorBoard, MLflow

### Development Environment
**Environment Management**: conda/venv for Python dependencies
**Package Manager**: pip with requirements.txt
**GPU Acceleration**: CUDA-enabled GPU required for training (4×RTX 4090 or 2×A6000 recommended)

### Essential Commands
```bash
# Environment setup
conda env create -f environment.yml
# OR
pip install -r requirements.txt

# Data generation
python scripts/data_generation/generate_synthetic_dataset.py

# Training models
python scripts/training/train_deep_models.py --model swinir --config config/training_config.yaml

# Evaluation
python scripts/evaluation/run_benchmark.py --checkpoint models/final/swinir_best.pth

# Testing
pytest tests/

# Jupyter notebooks for exploration
jupyter notebook notebooks/
```

---

## 📋 Style & Conventions

### Code Style
- **Follow language-specific best practices** (PEP 8 for Python, ESLint for JS/TS, etc.)
- **Use type annotations** where supported (TypeScript, Python type hints, etc.)
- **Prefer explicit over implicit** - make intentions clear in any language
- **Use descriptive names** that explain purpose and context

### Naming Conventions
**[Adapt based on language/framework conventions]**
- **Variables/Functions**: [e.g., camelCase (JS/TS), snake_case (Python), PascalCase (C#)]
- **Classes/Components**: [e.g., PascalCase (most languages), kebab-case (Vue components)]
- **Constants**: [e.g., UPPER_SNAKE_CASE, ALL_CAPS]
- **Files**: [e.g., kebab-case.js, snake_case.py, PascalCase.cs]
- **Directories**: [e.g., kebab-case, snake_case, consistent with project]

### Documentation Standards
**[Language-specific documentation format]**
```
// JavaScript/TypeScript JSDoc
/**
 * Brief description of function purpose
 * @param {string} param1 - Description of parameter
 * @param {number} param2 - Description of parameter  
 * @returns {boolean} Description of return value
 * @throws {Error} When validation fails
 */

# Python docstrings
"""Brief description of function purpose.

Args:
    param1: Description of parameter
    param2: Description of parameter
    
Returns:
    Description of return value
    
Raises:
    ValueError: When validation fails
"""
```

---

## 🧪 Testing Strategy

### Test-Driven Development (TDD)
1. **Write tests first** - Define expected behavior before implementation
2. **Run tests and confirm they fail** - Ensure tests are actually testing something
3. **Write minimal code** to make tests pass
4. **Refactor** while keeping tests green

### Test Organization
- **Unit tests**: Test individual functions and classes in isolation
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete user workflows
- **Test file naming**: `test_[module_name].py`

---

## 🔧 Project-Specific Context
**[Last updated: 2025-10-08 via /prime]**

### Project Type
**Research Project**: Master's thesis on physics-informed deep learning super-resolution for semiconductor inspection

### Current Architecture

This is a **research-focused machine learning project** with the following architecture:

```
Core Components:
├── Data Pipeline: Synthetic pattern generation + Real data acquisition
├── Physics Modeling: PSF/OTF simulation for optical degradation
├── Model Zoo: 6+ SR architectures (U-Net, RCAN, ESRGAN, SwinIR, Real-ESRGAN, HAT)
├── Physics-Informed Training: Custom loss functions with optical constraints
├── Evaluation Framework: Standard + semiconductor-specific metrics
└── Deployment: ONNX/TensorRT export for production inference
```

**Key Innovation**: Bridging physics-based methods (deconvolution) with deep learning through physics-informed loss functions and optical system parameter conditioning.

### Key Files and Directories

**Documentation**:
- `RESEARCH_PROPOSAL.md` - Comprehensive research proposal (74KB, highly detailed)
- `docs/architecture/ARCHITECTURE.md` - System architecture and design patterns
- `CLAUDE.md` - This file, AI development context
- `ROADMAP.md` - Detailed 13-month roadmap

**Current State**:
- ✅ Research proposal complete
- ✅ Architecture design complete
- ✅ **Project structure initialized** (80+ directories, all packages ready)
- ✅ **Development environment configured** (requirements.txt, environment.yml, setup.py)
- ✅ **Git properly configured** (.gitignore, .gitattributes, LICENSE)
- ✅ **Comprehensive configs created** (5 YAML files for all workflows)
- 🔄 Implementation starting: Week 2 - Pattern generation
- ⏳ Data generation pipeline: In progress
- ⏳ Model implementations: To be implemented Month 3-6

**Project Structure** (✅ Complete):
```
src/
├── data/          # Synthetic + real data handling ✅ initialized
│   ├── synthetic/ # Pattern generation, PSF models, degradation
│   └── real/      # Real data loading and preprocessing
├── models/        # SR model implementations ✅ initialized
│   ├── base/      # Abstract base classes
│   ├── traditional/  # Deconvolution, Wiener filter
│   ├── deep_learning/  # U-Net, RCAN, ESRGAN, SwinIR, HAT
│   └── physics_informed/  # Physics-aware components
├── training/      # Training infrastructure ✅ initialized
├── evaluation/    # Metrics and benchmarking ✅ initialized
├── analysis/      # Uncertainty, hallucination detection ✅ initialized
└── utils/         # General utilities ✅ initialized
```

### Research Context

**Research Question**: Can physics-informed deep learning super-resolution bridge the resolution gap between optical inspection (fast, low-res) and SEM inspection (slow, high-res) for semiconductor manufacturing?

**Target Outcome**:
- >3dB PSNR improvement over traditional deconvolution
- <5% critical dimension measurement error
- >95% defect detection recall
- Successful sim-to-real transfer (>85% performance retention)

**Timeline**: 13-18 months (master's thesis)
- Months 1-2: Data pipeline + baselines
- Months 3-6: Model development
- Months 7-10: Training + validation
- Months 11-13: Analysis + thesis writing

### Key Technical Challenges

1. **Hallucination Detection**: Preventing false features in metrological applications
2. **Sim-to-Real Transfer**: Training on synthetic data, deploying on real microscope images
3. **Physics Consistency**: Ensuring SR outputs respect optical physics constraints
4. **Validation**: Limited ground truth (SEM imaging is expensive/destructive)
5. **Domain-Specific Metrics**: CD accuracy, edge placement error, defect detection

### External Dependencies

**Critical**:
- PyTorch ecosystem (torch, torchvision, timm)
- Scientific computing (numpy, scipy, scikit-image)
- Image processing (opencv-python, tifffile for 16-bit images)
- Optical physics (custom PSF/OTF models)

**Optional but Recommended**:
- TCAD simulation tools (Synopsys Sentaurus) - synthetic data
- Optical simulation (Lumerical FDTD) - rigorous PSF modeling
- Experiment tracking (Weights & Biases)
- Production deployment (ONNX Runtime, TensorRT)

---

## 🚨 Error Handling & Logging

### Exception Best Practices
- **Use language-specific exception types** rather than generic errors
- **Provide meaningful error messages** that help with debugging
- **Log errors with context** including relevant data for diagnosis
- **Fail fast** - validate inputs early and handle errors gracefully

### Logging Strategy
**[Adapt to language/framework logging system]**
```
// JavaScript/Node.js
console.debug('Detailed information for diagnosis');
console.info('General information about execution');
console.warn('Something unexpected happened');
console.error('A serious problem occurred');

// Python
import logging
logger = logging.getLogger(__name__)
logger.debug("Detailed information")
logger.info("General information")
logger.warning("Something unexpected")
logger.error("Serious problem")

// Java
logger.debug("Detailed information");
logger.info("General information");  
logger.warn("Something unexpected");
logger.error("Serious problem");
```

---

## 🔄 Development Workflow

### Git Workflow
- **Feature branches** for all new development
- **Descriptive commit messages** following conventional format
- **Small, focused commits** that represent single logical changes
- **Pull request reviews** before merging to main

### Branch Strategy
```bash
main              # Production-ready code
develop           # Integration branch for features
feature/[name]    # Individual feature development
hotfix/[name]     # Critical production fixes
```

### Commit Message Format
```
type(scope): brief description

Detailed explanation if needed

- List any breaking changes
- Reference related issues (#123)

Types: feat, fix, docs, style, refactor, test, chore
```

---

## 🛡️ Security & Performance

### Security Guidelines
- **Never commit secrets** - use environment variables or secure vaults
- **Validate all inputs** - sanitize and validate user data appropriately
- **Use parameterized queries** - prevent injection attacks
- **Implement proper authentication** and authorization patterns
- **Follow OWASP guidelines** for web applications
- **Keep dependencies updated** - regularly update packages/libraries

### Performance Considerations
- **Database optimization**: Use indexes, avoid N+1 queries, optimize query patterns
- **Caching strategies**: Implement appropriate caching at multiple levels
- **Resource management**: Properly close connections, manage memory usage
- **Profiling**: Measure performance before optimizing - don't guess
- **Bundle/Build optimization**: Minimize bundle size, optimize assets

---

## 📚 Development Commands & Tools

### Essential Development Commands
```bash
# [Auto-updated by /prime command based on project analysis]
# Environment setup: [e.g., npm install, pip install -r requirements.txt, go mod tidy]
# Development server: [e.g., npm run dev, python manage.py runserver, go run main.go]
# Testing: [e.g., npm test, pytest, mvn test, go test ./...]
# Build: [e.g., npm run build, python setup.py build, mvn package]
# Linting: [e.g., npm run lint, flake8, golangci-lint run]
```

### Debugging Tools
- **Language debugger**: [e.g., Node.js inspector, pdb/ipdb, gdb, IDE debuggers]
- **Logging**: Use structured logging appropriate for the stack
- **Testing**: Verbose test output for debugging test failures  
- **Profiling**: [e.g., Chrome DevTools, cProfile, pprof, perf]
- **Network debugging**: [e.g., browser DevTools, curl, Postman, wire protocol tools]

---

## 🎯 Current Development Context
**[Last updated: 2025-10-08 via /prime]**

### Active Phase
**Phase**: Phase 1, Month 1, Week 2 - Foundation Development
**Focus**: Synthetic pattern generation and PSF/OTF modeling
**Progress**: Week 1 complete (14% of Phase 1), Week 2 in progress

### Current Project State

**Completed**:
✅ Comprehensive research proposal (RESEARCH_PROPOSAL.md)   
✅ System architecture design (docs/architecture/ARCHITECTURE.md)   
✅ Technology stack selection   
✅ Development workflow defined 
✅ Success metrics established  
✅ Project structure created  
✅ Environment configured   
✅ Git configured   
✅ **Config files created** (5 comprehensive YAML files)
✅ **Week 1 complete** (2025-10-08)

**In Progress (Week 2)**:
🔄 Pattern generation implementation
🔄 PSF/OTF modeling
🔄 Degradation pipeline

**Not Started**:
⏳ Baseline methods (Week 3-4)
⏳ Evaluation metrics (Week 3-4)
⏳ Model implementations (Month 3-6)
⏳ Training infrastructure (Month 3-8)
⏳ Tests (Ongoing with each component)

### Recent Decisions
**[Latest 3-5 technical decisions]**

- **2025-10-08**: Week 1 environment setup completed
  - Created complete project structure with 80+ directories
  - All Python packages properly initialized with __init__.py
  - Comprehensive .gitignore for ML projects (data, models, results ignored)
  - Binary file handling in .gitattributes (TIFF, models, archives)
  - 70+ dependencies specified in requirements.txt

- **2025-10-08**: Configuration architecture finalized
  - 5 YAML config files created (data, model, training, evaluation, paths)
  - Centralized path management for reproducibility
  - Support for 6 SR models + traditional baselines
  - Physics-informed loss components fully specified

- **2025-10-07**: Comprehensive architecture design completed
  - Decided on modular architecture with physics-informed components
  - Selected 6 baseline models: U-Net, RCAN, ESRGAN, SwinIR, Real-ESRGAN, HAT
  - Defined physics-informed loss function incorporating PSF consistency

- **2025-10-07**: Technology stack finalized
  - PyTorch 2.1.0 as primary framework (over TensorFlow)
  - CUDA 12.1 for GPU acceleration (RTX 3060 12GB confirmed)
  - Weights & Biases for experiment tracking
  - ONNX/TensorRT for production deployment

### Next Priorities

**Immediate (Month 1-2: Foundation)**:
1. **Create project structure** - Initialize `src/` directory with all subdirectories
2. **Implement data pipeline** - Synthetic pattern generation (gratings, contacts, logic cells)
3. **PSF/OTF modeling** - Physics simulation for realistic image degradation
4. **Baseline methods** - Richardson-Lucy deconvolution, Wiener filtering
5. **Evaluation metrics** - PSNR, SSIM, CD error, edge placement error

**Next Phase (Month 3-4: Core Models)**:
1. Implement U-Net with physics embedding
2. Implement RCAN with channel attention
3. Implement ESRGAN for GAN-based SR
4. Initial training on synthetic data
5. Benchmark against baselines

**Critical Path Items**:
- Data generation pipeline (blocks everything)
- Physics modeling (blocks physics-informed training)
- Evaluation framework (blocks model comparison)

### Known Issues & Blockers

**Current Blockers**:
- ⚠️ **GPU for heavy training** - RTX 3060 12GB confirmed, may need cloud for 4x models in parallel
- ⚠️ **TCAD simulation access** - Synopsys Sentaurus for high-fidelity synthetic data (Decision: Start with analytical, add TCAD later)
- ⚠️ **Real data acquisition** - Need cleanroom access for validation data (timeline: Month 9-10)
- ⚠️ **W&B account setup** - Need to initialize experiment tracking

**Design Decisions Made**:
- ✅ **Data strategy**: Start with analytical patterns (fast), add TCAD simulation later for realism
- ✅ **Model development**: Sequential implementation (validate each before moving on)
- ✅ **Data format**: TIFF 16-bit for semiconductor images (specified in config)
- ✅ **Pre-training**: Will try both pre-trained and from-scratch approaches

**Resolved Dependencies**:
- ✅ requirements.txt created (70+ packages)
- ✅ environment.yml created (Conda with CUDA 12.1)
- ✅ setup.py created (editable installation)
- ✅ Git configuration complete
- ✅ All config files created

---

## ⚠️ Important Development Notes

### Critical Guidelines
- **NEVER ASSUME OR GUESS** - When in doubt, ask for clarification
- **Always verify file paths and imports** before use
- **Use project environment** (venv, node_modules, etc.) for all commands
- **Test your code** - No feature is complete without tests
- **Update documentation** when making architectural changes
- **Follow the planning workflow** - use `/plan-feature` before coding

### Session Management
- **Use `/clear`** when switching to completely different features
- **Use `/update-planning`** to save progress and decisions
- **Keep CLAUDE.md current** - update when patterns or practices change

### Quality Checklist
Before completing any feature:
- [ ] Code follows style guidelines
- [ ] Tests are written and passing
- [ ] Documentation is updated
- [ ] Error handling is implemented
- [ ] Performance impact is considered
- [ ] Security implications are reviewed

---

## 🔍 Search & Discovery Commands

When analyzing code or debugging:
1. **Use `tree` command** to understand project structure
2. **Search for patterns** using `grep` or `rg` (ripgrep)
3. **Check git history** for context on changes
4. **Review test files** to understand expected behavior
5. **Examine config files** for environment setup

---

*This document is automatically updated by `/prime` command and should be maintained as the project evolves. Last updated: [Auto-generated timestamp]*