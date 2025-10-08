# Prime Context for Claude Code

## Project Analysis and Onboarding

### Step 1: Initial Project Discovery
Use the `tree` command to get a comprehensive understanding of the project structure:
```bash
tree -I 'node_modules|.git|dist|build|coverage' -L 3
```

If tree is not available, use:
```bash
find . -type f -name "*.md" -o -name "*.json" -o -name "*.js" -o -name "*.ts" -o -name "*.py" | head -20
```

### Step 2: Core Documentation Analysis
Read these files in order of priority:

#### Essential Context Files
1. **CLAUDE.md** - If it exists, read first for project-specific Claude context
2. **CLAUDE.local.md** - Local development context (if exists)
3. **README.md** - Primary project documentation
4. **docs/README.md** - Additional documentation overview

#### Planning and Requirements (if new project)
5. **RESEARCH_PROPOSAL.md** or **requirements.md** - Product Requirements Document
6. **docs/planning/** - Development planning documents (if exists)
7. **ARCHITECTURE.md** or **docs/architecture/** - System design

#### Configuration Files
8. **package.json** / **pyproject.toml** / **Cargo.toml** - Dependencies and scripts
9. **requirements.txt** / **environment.yml** / **Pipfile** - Python dependencies
10. **thesis.tex** / **main.tex** - LaTeX thesis document
11. **tsconfig.json** / **jsconfig.json** - TypeScript/JavaScript configuration
12. **docker-compose.yml** / **Dockerfile** - Containerization setup
13. **.env.example** - Environment variables template

### Step 3: Source Code Analysis
Dynamically analyze the actual project structure. Look for these common patterns and identify what type of project this is:

#### Identify Project Type
Based on the directory structure and files found, determine:
- **Frontend-only** (React, Vue, Angular, etc.)
- **Backend API** (Express, FastAPI, Django, etc.) 
- **Full-stack application** (Next.js, Nuxt, etc.)
- **Monorepo** (multiple apps/packages)
- **Library/Package** (npm package, Python package, etc.)
- **Research Project** (academic research, thesis, data analysis)
- **Data Science/ML Project** (Jupyter notebooks, datasets, models)
- **Documentation Project** (technical writing, academic papers)

#### Analyze Main Source Directories
Examine the primary content locations based on project type:

**For Software Projects** (typically `src/`, `app/`, `lib/`, or root-level files):
1. **Find the entry point** - main.js, index.ts, app.py, etc.
2. **Identify key directories** and their apparent purposes
3. **Look for configuration files** that indicate framework choices
4. **Examine file naming patterns** and organization strategy
5. **Note any unusual or project-specific structure**

**For Research/Academic Projects** (typically organized around research workflow):
1. **Literature review** - papers/, references/, bibliography/
2. **Data collection** - data/, datasets/, raw_data/, processed_data/
3. **Analysis/Code** - scripts/, analysis/, notebooks/, experiments/
4. **Writing** - thesis/, chapters/, papers/, reports/
5. **Results** - results/, figures/, plots/, outputs/
6. **Documentation** - docs/, methodology/, protocols/

**For Data Science/ML Projects**:
1. **Notebooks** - Jupyter notebooks for exploration and analysis
2. **Data pipelines** - ETL scripts, preprocessing
3. **Models** - Training scripts, model definitions, saved models
4. **Experiments** - Experiment tracking, hyperparameter tuning
5. **Deployment** - Model serving, inference code

#### Key Files to Examine
**Software Projects:**
- Main application entry points
- Route/page definitions  
- Component/module organization
- Database schemas or models (if present)
- API endpoints or handlers (if present)
- Type definitions (if TypeScript)
- Test file organization

**Research/Academic Projects:**
- Thesis document (thesis.tex, thesis.md, main.tex)
- Research proposal or methodology documents
- Data analysis scripts (R, Python, MATLAB)
- Jupyter notebooks with analysis
- Bibliography/references files (.bib)
- Experiment protocols or procedures
- Results summary documents

**Data Science/ML Projects:**
- Main analysis notebooks
- Data preprocessing scripts
- Model training and evaluation code
- Configuration files for experiments
- Dataset documentation
- Model performance metrics

### Step 4: Testing and Build Configuration
- **tests/** or **__tests__/** - Test files
- **jest.config.js** / **vitest.config.ts** - Testing configuration
- **webpack.config.js** / **vite.config.ts** - Build configuration
- **eslint.config.js** / **.eslintrc** - Linting rules
- **prettier.config.js** - Code formatting rules

### Step 5: Analysis and Auto-Documentation
After reading the above files, automatically update project documentation and provide comprehensive analysis:

#### Auto-Update CLAUDE.md
Update the CLAUDE.md file with current project context:

1. **Technology Stack Section**: Update with discovered technologies, frameworks, and tools
2. **Essential Commands Section**: Add the actual development commands found (npm run dev, pytest, etc.)
3. **Project-Specific Context Section**: Fill in with:
   - Current architecture overview
   - Key files and directories
   - Database schema (if applicable)
   - API endpoints (if applicable)
   - External dependencies
4. **Current Development Context**: Set initial phase and priorities based on analysis

#### Auto-Update README.md
Update or create README.md with comprehensive project documentation:

1. **Project Overview**: Clear description of what the project does
2. **Installation Instructions**: Step-by-step setup process
3. **Usage Guide**: How to run and use the project
4. **Development Setup**: Environment setup and development workflow
5. **Architecture Overview**: High-level system design
6. **API Documentation**: Endpoint documentation (if applicable)
7. **Contributing Guidelines**: How others can contribute
8. **Deployment Instructions**: How to deploy the project

#### Comprehensive Analysis Report
After updating documentation, provide analysis covering:

### Special Instructions for Research/Academic Projects

If this appears to be a research or academic project:

1. **Research Context** - Understand the research question, hypothesis, or thesis topic
2. **Methodology** - What research methods are being used?
3. **Data Sources** - What data is being collected or analyzed?
4. **Academic Timeline** - What are the key milestones (proposal defense, thesis submission, etc.)?
5. **Collaboration** - Are there supervisors, committee members, or co-authors?
6. **Publication Goals** - Are there target conferences, journals, or thesis requirements?

Ask clarifying questions if:
- The research question or objectives aren't clear
- Methodology needs refinement or validation
- Data collection or analysis approaches need guidance
- Writing and documentation structure needs organization
- Timeline and milestone planning needs development

### Special Instructions for New Projects

If this appears to be a new project (minimal files, mostly documentation):

1. **Read PRD thoroughly** - Understand business requirements and user stories
2. **Identify MVP scope** - What needs to be built first?
3. **Technology decisions** - Are tech stack choices already made?
4. **Architecture planning** - Is there a proposed system design?
5. **Development phases** - Are there defined milestones or phases?

Ask clarifying questions if:
- The PRD is unclear about specific features
- Technology choices haven't been finalized
- Architecture decisions need validation
- Development priorities aren't clear

### Context Completion Checklist

Confirm I understand:
- [ ] Project's business purpose and target users
- [ ] Complete technology stack and dependencies
- [ ] Current development phase and priorities
- [ ] Code organization and key files
- [ ] Development workflow and commands
- [ ] Testing strategy and build process
- [ ] Any project-specific conventions or decisions
- [ ] Immediate next steps or current focus area

### Ready to Code
Once this analysis is complete, I'll have the full context needed to:
- Write code that follows project conventions
- Understand how new features fit into existing architecture  
- Make appropriate technology choices consistent with the stack
- Follow established patterns and practices
- Focus on the right priorities for the current development phase

---