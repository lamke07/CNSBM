# CN-SBM: Categorical Block Modelling For Primary and Residual Copy Number Variation
This repository contains the implementation of the model described in the paper: **CN-SBM: Categorical Block Modelling For Primary and Residual Copy Number Variation** ([arXiv:2506.22963](https://arxiv.org/abs/2506.22963)), to appear in MLCB 2025.

# Note on NumPy Version
If you plan to use pickle files in an environment with NumPy < 2, you should install numpy < 2 for compatibility. Otherwise, it is fine to use NumPy 2 or newer.

# Installation

## Option 1: Install from PyPI (Recommended)

```bash
pip install cnsbm
```

## Option 2: Development installation

```bash
git clone https://github.com/lamke07/CNSBM.git
cd CNSBM
pip install -e .
```

### GPU Support (Optional)

For GPU acceleration with JAX, you must manually install the correct JAX GPU wheel for your CUDA version. The pip extra `cnsbm[gpu]` is not supported.

**To install JAX with CUDA support:**

1. Find your CUDA version (e.g., CUDA 12).
2. Run the following command (replace `cuda12_pip` with your CUDA version):

```bash
pip install --upgrade "jax[cuda12_pip]>=0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

See the [JAX GPU installation guide](https://github.com/google/jax#pip-installation-gpu-cuda) or [here](https://docs.jax.dev/en/latest/installation.html) for details.

# Simple usage

```python
import os
import jax.numpy as jnp
from cnsbm import CNSBM

cwd = os.getcwd()

# C is a categorical matrix (integer-encoded categories starting from 0),
# missing values are encoded as -1. The number of categories will be inferred by C.max().
# For an example of how to construct and use C, see cn_vi-simple.ipynb in this repository
C = jnp.asarray(C)
K, L = 15, 10

# Initialize Jax model
sbm_test = CNSBM(C, K, L, rand_init='spectral_bi', fill_na=2)
# Run batch variational inference
_ = sbm_test.batch_vi(75, batch_print=1, fitted=False, tol=1e-6)

# plot reordered output and get summary information
sbm_test.plt_blocks(plt_init=True)
sbm_test.summary()
_ = sbm_test.ICL(verbose=True, slow=True)

# Save model outputs and export cluster labels / probabilities
os.makedirs(os.path.join(cwd, 'output'), exist_ok=True)
sbm_test.export_outputs_csv(os.path.join(cwd, 'output'), model_name='test_sbm')
sbm_test.save_jax_model(os.path.join(cwd, 'output', f'test_sbm.pickle'))
```