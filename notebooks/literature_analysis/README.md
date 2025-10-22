# Literature Analysis Notebooks

This directory contains Jupyter notebooks for **quantitative analysis** of research papers.

## 📊 What are these notebooks for?

Instead of just reading papers and taking notes, these notebooks let you:

1. **Compare papers quantitatively** - Create tables and plots comparing PSNR, SSIM, inference time, model size
2. **Visualize trends** - Plot performance evolution over time, accuracy vs computational cost
3. **Make data-driven decisions** - Score and rank models based on your criteria
4. **Track implementation progress** - Checklist of papers to read and models to implement

## 📚 Available Notebooks

### 01_SR_architecture_comparison.ipynb
**Purpose:** Compare super-resolution architectures from literature

**What it does:**
- ✅ Creates comparison tables (PSNR, SSIM, params, FLOPs, inference time)
- ✅ Plots performance evolution (PSNR over years)
- ✅ Plots accuracy vs model complexity
- ✅ Ranks models by efficiency score
- ✅ Provides implementation recommendations

**Example output:**
```
📊 Models Ranked by Efficiency (PSNR / Inference Time):
============================================================
SRCNN           | Score: 203.20 | PSNR: 30.48 dB | Time: 15 ms
ESRGAN          | Score: 39.62  | PSNR: 30.90 dB | Time: 78 ms
RCAN            | Score: 34.35  | PSNR: 32.63 dB | Time: 95 ms

✅ Top 3 Models for Implementation:
1. SRCNN (2015)
2. ESRGAN (2018)
3. RCAN (2018)
```

### 02_hallucination_metrics_comparison.ipynb (To be created)
**Purpose:** Compare hallucination detection methods

**Will include:**
- Hallucination metrics from different papers
- False positive/negative rates
- Computational overhead
- Integration with SR models

### 03_physics_informed_methods.ipynb (To be created)
**Purpose:** Compare physics-informed approaches

**Will include:**
- Different physics constraint formulations
- Loss function designs
- Performance improvements vs vanilla models
- Computational cost analysis

## 🎯 Why Use Notebooks Instead of Just Reading?

### Traditional Approach (Just Reading):
```
✅ Read SRCNN paper → Take notes
✅ Read SRGAN paper → Take notes
✅ Read ESRGAN paper → Take notes
❓ Which one is best for my project?
❓ How do they compare quantitatively?
❓ What's the trade-off between speed and accuracy?
```

### Notebook Approach (Analytical):
```
✅ Read SRCNN paper → Extract data → Add to comparison table
✅ Read SRGAN paper → Extract data → Update plots
✅ Read ESRGAN paper → Extract data → Re-rank models
✅ See visualizations: PSNR trend, speed vs accuracy plot
✅ Run scoring algorithm: "RCAN is best for your use case"
✅ Make informed decision based on data
```

## 🚀 How to Use

1. **Read papers** from `docs/literature/[topic]/`
2. **Extract key metrics** (PSNR, SSIM, inference time, etc.)
3. **Open notebook** in Jupyter
4. **Add data** to the comparison tables
5. **Run cells** to update plots and rankings
6. **Make decisions** based on quantitative analysis


## 🚀 Run the Literature Analysis Notebook (3 Easy Options)

### **Option 1: Interactive in Browser (RECOMMENDED)**

```bash

# Start Jupyter Notebook inside the project root directory
jupyter notebook notebooks/literature_analysis/01_SR_architecture_comparison.ipynb
```

**This will:**
1. Open your web browser
2. Show the notebook
3. Click **"Cell → Run All"** to execute everything
4. See tables, plots, and rankings

---

### **Option 2: JupyterLab (More Features)**

```bash

# Start JupyterLab inside the project root directory
jupyter lab notebooks/literature_analysis/01_SR_architecture_comparison.ipynb
```

**Better for:**
- Multiple notebooks
- File browsing
- Split-screen editing
- Terminal access

---

### **Option 3: VSCode (If you have Python extension)**

1. Open VSCode
2. Navigate to: `notebooks/literature_analysis/01_SR_architecture_comparison.ipynb`
3. VSCode will show the notebook
4. Click "Run All" button at the top

---

## 📊 What You'll See

When you run the notebook, it will:

1. **Create a comparison table** of 7 SR models:
   ```
   Model      | PSNR  | Inference Time | Parameters | Efficiency
   ================================================================
   SRCNN      | 30.48 | 15 ms          | 0.06M      | 203.20
   SRGAN      | 29.40 | 45 ms          | 1.55M      | 65.33
   ESRGAN     | 30.90 | 78 ms          | 16.7M      | 39.62
   ...
   ```

2. **Generate 2 plots**:
   - PSNR evolution over time (2015-2024)
   - Accuracy vs Model Size (bubble size = inference time)

3. **Rank models by efficiency**:
   ```
   Top 3 Models for Implementation:
   1. SRCNN (2015) - CNN
   2. ESRGAN (2018) - GAN
   3. RCAN (2018) - CNN+Attention
   ```

4. **Save plot** to: `results/literature_analysis/SR_architecture_comparison.png`

---
## 📈 Example Use Cases

### Use Case 1: Selecting Baseline Models
**Question:** Which SR models should I implement as baselines?

**Notebook:** `01_SR_architecture_comparison.ipynb`

**Answer:**
- Plot PSNR vs inference time → See which models are fast AND accurate
- Calculate efficiency score → Rank models
- Result: "Implement RCAN, SwinIR, and HAT as baselines"

### Use Case 2: Hallucination Risk Assessment
**Question:** Which SR models have lowest hallucination risk for metrology?

**Notebook:** `02_hallucination_metrics_comparison.ipynb`

**Answer:**
- Compare false positive rates from papers
- Analyze hallucination detection methods
- Result: "Avoid ESRGAN for CD measurement, use SwinIR instead"

### Use Case 3: Physics Loss Function Design
**Question:** What physics constraints do other papers use?

**Notebook:** `03_physics_informed_methods.ipynb`

**Answer:**
- Compare different loss function formulations
- Analyze performance improvements
- Result: "PSF consistency loss + edge sharpness term works best"

## 🆚 Notebooks vs Markdown Notess

| Aspect | Markdown Notes | Jupyter Notebooks |
|--------|---------------|-------------------|
| **Purpose** | Qualitative insights | Quantitative analysis |
| **Content** | Text, bullet points | Code, plots, tables |
| **Use Case** | "What did the paper say?" | "How does it compare?" |
| **Output** | Static text | Interactive visualizations |
| **Example** | "SwinIR achieves 32.92 dB PSNR" | "SwinIR is 2.29 dB better than SRCNN, see plot" |

**Both are useful!** Use markdown for reading notes, notebooks for analysis.

## 📊 Example Workflow

```
Week 1: Read 5 SR papers
├── Take notes in: docs/literature/reviews/SR_architectures.md
├── Extract metrics for: notebooks/literature_analysis/01_SR_architecture_comparison.ipynb
└── Run notebook → Generate comparison plots

Week 2: Read 3 hallucination papers
├── Take notes in: docs/literature/reviews/hallucination_methods.md
├── Extract metrics for: notebooks/literature_analysis/02_hallucination_metrics.ipynb
└── Run notebook → Identify lowest-risk models

Week 3: Implementation decision
├── Review notebook rankings
├── Choose top 3 models based on data
└── Start implementation in src/models/
```

## 🛠️ Tips

1. **Update notebooks as you read** - Add new papers incrementally
2. **Version control notebooks** - Git tracks your analysis evolution
3. **Share with advisor** - Notebooks are great for showing your literature review
4. **Export plots** - Save figures to `results/literature_analysis/` for presentations

## 📖 Next Steps

- [ ] Run `01_SR_architecture_comparison.ipynb` example
- [ ] Add your own paper data
- [ ] Create `02_hallucination_metrics_comparison.ipynb`
- [ ] Create `03_physics_informed_methods.ipynb`

---

**Location in Project:**
- Notebooks: `notebooks/literature_analysis/`
- Papers: `docs/literature/[topic]/`
- Notes: `docs/literature/reviews/`
- BibTeX: `thesis/bibliography/`
