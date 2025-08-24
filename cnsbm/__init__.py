"""
CNSBM: Categorical Block Modelling For Primary and Residual Copy Number Variation

A Python package for stochastic block modeling of categorical matrices with support
for missing values and copy number variation analysis.
"""

__version__ = "1.0.1"
__author__ = "Kevin Lam"
__email__ = "kevin.lam@stat.ubc.ca"


from .cnsbm import CNSBM
from .cnsbm_trainer import CNSBMTrainer

# Runtime check for JAX GPU support
import jax
import warnings
if hasattr(jax, 'devices'):
    gpu_devices = [d for d in jax.devices() if d.platform == 'gpu']
    if not gpu_devices:
        warnings.warn(
            "JAX is running on CPU. For GPU support, manually install the correct JAX CUDA wheel. "
            "See the README for instructions: https://github.com/google/jax#pip-installation-gpu-cuda"
        )

__all__ = [
    "CNSBM",
    "CNSBMTrainer",
]
