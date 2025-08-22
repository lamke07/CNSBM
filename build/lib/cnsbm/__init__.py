"""
CNSBM: Categorical Block Modelling For Primary and Residual Copy Number Variation

A Python package for stochastic block modeling of categorical matrices with support
for missing values and copy number variation analysis.
"""

__version__ = "1.0.0"
__author__ = "Kevin Lam"
__email__ = "kevin.lam@stat.ubc.ca"

from .cnsbm import CNSBM
from .cnsbm_trainer import CNSBMTrainer

__all__ = [
    "CNSBM",
    "CNSBMTrainer",
]
