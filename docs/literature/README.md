# Research Literature & References

This directory contains research papers, literature reviews, and references organized by topic.

## 📁 Directory Structure

- **`reviews/`** - Literature review documents and summaries
- **`super_resolution/`** - Super-resolution papers (ESRGAN, SwinIR, HAT, etc.)
- **`semiconductor/`** - Semiconductor inspection and metrology papers
- **`physics_informed/`** - Physics-informed machine learning papers
- **`hallucination/`** - Hallucination detection and uncertainty quantification
- **`datasets/`** - Dataset papers and documentation

## 📝 File Naming Convention

Use descriptive filenames with year and first author:

```
YYYY_FirstAuthor_ShortTitle.pdf
2024_Wang_SwinIR_ImageRestorationTransformer.pdf
2023_Liang_HAT_HybridAttentionTransformer.pdf
```

## 🔖 Key Papers by Topic

### Super-Resolution Architectures

**Transformer-Based:**
- [ ] 2024 - Wang et al. - SwinIR: Image Restoration Using Swin Transformer
- [ ] 2023 - Liang et al. - HAT: Hybrid Attention Transformer for Image Restoration
- [ ] 2023 - Chen et al. - Activating More Pixels in Image Super-Resolution

**GAN-Based:**
- [ ] 2021 - Wang et al. - Real-ESRGAN: Training Real-World Blind SR with Pure Synthetic Data
- [ ] 2018 - Wang et al. - ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks
- [ ] 2017 - Ledig et al. - SRGAN: Photo-Realistic Single Image SR Using GANs

**CNN-Based:**
- [ ] 2018 - Zhang et al. - RCAN: Image Super-Resolution Using Very Deep Residual Channel Attention Networks
- [ ] 2017 - Zhang et al. - Learning Deep CNN Denoiser Prior for Image Restoration
- [ ] 2015 - Dong et al. - SRCNN: Image Super-Resolution Using Deep Convolutional Networks

### Physics-Informed Deep Learning

**General Physics-Informed ML:**
- [ ] 2024 - NeurIPS Workshop - Physics-Informed Machine Learning
- [ ] 2023 - Karniadakis et al. - Physics-Informed Neural Networks (Review)
- [ ] 2019 - Raissi et al. - Physics-Informed Neural Networks: A Deep Learning Framework

**Optical/Imaging Applications:**
- [ ] 2023 - Physics-aware SR for microscopy
- [ ] 2022 - PSF-aware image reconstruction
- [ ] 2021 - Deconvolution with learned priors

### Semiconductor Inspection & Metrology

**Optical Inspection:**
- [ ] KLA Corporation - Advanced optical inspection systems (whitepapers)
- [ ] Applied Materials - Optical metrology for advanced nodes
- [ ] ASML - Lithography and inspection challenges at 3nm/5nm nodes

**SEM & High-Resolution Imaging:**
- [ ] CD-SEM metrology papers
- [ ] Overlay metrology at advanced nodes
- [ ] Defect detection and classification

### Hallucination Detection & Uncertainty

**Hallucination in SR:**
- [ ] 2024 - Recent work on hallucination in super-resolution
- [ ] 2023 - Quantifying uncertainty in image reconstruction
- [ ] 2022 - Perceptual quality vs. reconstruction accuracy

**Uncertainty Quantification:**
- [ ] Bayesian deep learning for image reconstruction
- [ ] Ensemble methods for uncertainty estimation
- [ ] Calibration of neural network predictions

### Simulation-to-Real Transfer

**Domain Adaptation:**
- [ ] Sim-to-real transfer learning
- [ ] Domain randomization techniques
- [ ] Self-supervised learning for adaptation

**Synthetic Data Generation:**
- [ ] Procedural generation for training data
- [ ] Physics-based rendering for synthetic datasets
- [ ] Validation of synthetic vs. real data

## 📚 Citation Management

For BibTeX entries and formal citations, see:
- **`thesis/bibliography/`** - Master BibTeX file for thesis
- **`RESEARCH_PROPOSAL.md`** - Initial references

## 🔗 Useful Resources

### Conferences & Journals
- **CVPR** - Computer Vision and Pattern Recognition
- **ICCV** - International Conference on Computer Vision
- **NeurIPS** - Neural Information Processing Systems
- **IEEE TIP** - Transactions on Image Processing
- **SPIE** - Advanced Lithography, Metrology conferences

### Industry Resources
- **SEMI** - Semiconductor Equipment and Materials International
- **IEEE IRDS** - International Roadmap for Devices and Systems
- **ITRS** - International Technology Roadmap for Semiconductors

### Code & Model Repositories
- **Papers With Code** - https://paperswithcode.com/task/super-resolution
- **Hugging Face** - Pre-trained SR models
- **GitHub** - Official implementations

## 📖 Literature Review Workflow

1. **Search** - Use Google Scholar, arXiv, IEEE Xplore, SPIE Digital Library
2. **Organize** - Save PDFs to appropriate subdirectory with naming convention
3. **Read** - Take notes in `reviews/TOPIC_review_notes.md`
4. **Cite** - Add BibTeX entry to `thesis/bibliography/references.bib`
5. **Analyze** - Create Jupyter notebook in `notebooks/literature_analysis/` for quantitative comparisons

## ✅ Reading Progress Tracker

Track your progress on key papers:

- [ ] Read foundational SR papers (SRCNN, SRGAN, ESRGAN)
- [ ] Read recent transformer papers (SwinIR, HAT)
- [ ] Read physics-informed ML papers
- [ ] Read semiconductor metrology papers
- [ ] Survey hallucination detection literature

---

**Last Updated:** 2025-10-22
