"""
Physics-Informed Deep Learning Super-Resolution for Semiconductor Inspection

Setup configuration for package installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements from requirements.txt
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() for line in f
            if line.strip() and not line.startswith('#') and not line.startswith('-')
        ]
else:
    requirements = []

setup(
    name="semicon-super-resolution",
    version="0.1.0",
    author="siamet",
    author_email="siamet@protonmail.com",
    description="Physics-informed deep learning super-resolution for semiconductor inspection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/siamet/semicon-super-resolution",
    project_urls={
        "Bug Tracker": "https://github.com/siamet/semicon-super-resolution/issues",
        "Documentation": "https://github.com/siamet/semicon-super-resolution/tree/main/docs",
        "Source Code": "https://github.com/siamet/semicon-super-resolution",
    },
    packages=find_packages(include=['src', 'src.*']),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "isort>=5.12.0",
            "pylint>=2.17.0",
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-xdist>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
            "sphinx-autodoc-typehints>=1.24.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipython>=8.15.0",
            "ipywidgets>=8.1.0",
            "notebook>=7.0.0",
        ],
        "deployment": [
            "onnx>=1.15.0",
            "onnxruntime-gpu>=1.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "semicon-sr-train=scripts.training.train_deep_models:main",
            "semicon-sr-eval=scripts.evaluation.run_benchmark:main",
            "semicon-sr-generate=scripts.data_generation.generate_synthetic_dataset:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json"],
    },
    zip_safe=False,
    keywords=[
        "super-resolution",
        "semiconductor",
        "deep learning",
        "physics-informed",
        "computer vision",
        "image processing",
        "optical inspection",
        "metrology",
    ],
)
