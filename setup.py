#!/usr/bin/env python3
"""Setup script for CNSBM package."""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Version management
def get_version():
    """Get version from cnsbm/__init__.py"""
    version_file = os.path.join("cnsbm", "__init__.py")
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"

setup(
    name="cnsbm",
    version=get_version(),
    author="Kevin Lam",
    author_email="kevin.lam@example.com",
    description="Categorical Block Modelling For Primary and Residual Copy Number Variation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/cnsbm2025",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/cnsbm2025/issues",
        "Documentation": "https://github.com/your-username/cnsbm2025#readme",
        "Paper": "https://arxiv.org/abs/2506.22963",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.9",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "jupyter",
        ],
        "gpu": ["jax[cuda]>=0.3.0"],
    },
    keywords=[
        "machine learning",
        "stochastic block model", 
        "copy number variation",
        "bioinformatics",
        "clustering",
        "variational inference",
    ],
    include_package_data=True,
    zip_safe=False,
)
