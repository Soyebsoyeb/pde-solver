"""Setup script for PDE solver package."""

from setuptools import setup, find_packages

setup(
    name="pde-solver",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "torch>=2.0.0",
        "sympy>=1.12",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "omegaconf>=2.3.0",
        "typer>=0.9.0",
        "wandb>=0.15.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.10",
)

