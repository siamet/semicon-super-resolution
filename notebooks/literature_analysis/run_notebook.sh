#!/bin/bash
# Quick script to run the literature analysis notebook

echo "🚀 Running Super-Resolution Architecture Comparison Notebook"
echo "============================================================"
echo ""

# Check if results directory exists
mkdir -p results/literature_analysis

# Execute the notebook
echo "📊 Executing notebook..."
jupyter nbconvert --to notebook --execute \
    notebooks/literature_analysis/01_SR_architecture_comparison.ipynb \
    --output 01_SR_architecture_comparison_executed.ipynb \
    --ExecutePreprocessor.timeout=600

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Notebook executed successfully!"
    echo ""
    echo "📁 Output files:"
    echo "   - Executed notebook: notebooks/literature_analysis/01_SR_architecture_comparison_executed.ipynb"
    echo "   - Comparison plot: results/literature_analysis/SR_architecture_comparison.png"
    echo ""
    echo "🔍 To view results:"
    echo "   jupyter notebook notebooks/literature_analysis/01_SR_architecture_comparison_executed.ipynb"
    echo ""
else
    echo ""
    echo "❌ Notebook execution failed"
    echo "Try running interactively: jupyter notebook notebooks/literature_analysis/01_SR_architecture_comparison.ipynb"
fi
