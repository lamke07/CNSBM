import jax
import jax.numpy as jnp
from jax.scipy.special import digamma
from functools import partial

"""Dual Data Versions of SBM"""

"""VI Update Functions"""

# full data versions
@jax.jit
def update_phi_g_full(phi_h, phi_h_D, gamma_kl_di, gamma_kl_sum_di, delta_kl_di, delta_kl_sum_di, gamma_g_di, c_row, d_row):
    """Update q(g_i | phi^g)."""
    sum1 = jnp.einsum("ml, klm -> k", phi_h, gamma_kl_di[:, :, c_row]-gamma_kl_sum_di, precision='highest')
    sum2 = jnp.einsum("ml, klm -> k", phi_h_D, delta_kl_di[:, :, d_row]-delta_kl_sum_di, precision='highest')
    out = sum1 + sum2 - gamma_g_di

    return out

# shared phi_h across both columns
@jax.jit
def update_phi_h_full(phi_g, gamma_kl_di, gamma_kl_sum_di, delta_kl_di, delta_kl_sum_di, gamma_h_di, c_col, d_col):
    """q(h_j | phi^h)."""
    sum1 = jnp.einsum("nk, kln -> l", phi_g, gamma_kl_di[:, :, c_col]-gamma_kl_sum_di, precision='highest')
    sum2 = jnp.einsum("nk, kln -> l", phi_g, delta_kl_di[:, :, d_col]-delta_kl_sum_di, precision='highest')
    out = sum1 + sum2 - gamma_h_di

    return out

vmap_update_phi_g_full = jax.vmap(update_phi_g_full, in_axes=(None, None, None, None, None, None, None, 0, 0))
vmap_update_phi_h_full = jax.vmap(update_phi_h_full, in_axes=(None, None, None, None, None, None, None, 1, 1))
jax.jit(vmap_update_phi_g_full)
jax.jit(vmap_update_phi_h_full)

# Missing data versions
@jax.jit
def update_phi_g_mis(
    phi_h, phi_h_D,
    gamma_kl_di, gamma_kl_sum_di, delta_kl_di, delta_kl_sum_di, gamma_g_di, 
    c_row, c_row_mask, d_row, d_row_mask,
):
    """Update q(g_i | phi^g)."""
    sum1 = jnp.einsum("ml, klm -> km", phi_h, gamma_kl_di[:, :, c_row]-gamma_kl_sum_di, precision='highest')
    sum1 = jnp.einsum("m, km -> k", c_row_mask, sum1, precision='highest')
    sum2 = jnp.einsum("ml, klm -> km", phi_h_D, delta_kl_di[:, :, d_row]-delta_kl_sum_di, precision='highest')
    sum2 = jnp.einsum("m, km -> k", d_row_mask, sum2, precision='highest')
    out = sum1 + sum2 - gamma_g_di

    return out

@jax.jit
def update_phi_h_mis(
    phi_g,
    gamma_kl_di, gamma_kl_sum_di, delta_kl_di, delta_kl_sum_di, gamma_h_di, 
    c_col, c_col_mask, d_col, d_col_mask,
):
    """Update q(g_i | phi^g)."""
    sum1 = jnp.einsum("nk, kln -> ln", phi_g, gamma_kl_di[:, :, c_col]-gamma_kl_sum_di, precision='highest')
    sum1 = jnp.einsum("n,ln -> l", c_col_mask, sum1, precision='highest')
    sum2 = jnp.einsum("nk, kln -> ln", phi_g, delta_kl_di[:, :, d_col]-delta_kl_sum_di, precision='highest')
    sum2 = jnp.einsum("n,ln -> l", d_col_mask, sum2, precision='highest')
    out = sum1 + sum2 - gamma_h_di

    return out

vmap_update_phi_g_mis = jax.vmap(update_phi_g_mis, in_axes=(None, None, None, None, None, None, None, 0, 0, 0, 0))
vmap_update_phi_h_mis = jax.vmap(update_phi_h_mis, in_axes=(None, None, None, None, None, None, None, 1, 1, 1, 1))
jax.jit(vmap_update_phi_g_mis)
jax.jit(vmap_update_phi_h_mis)

def update_phi_g(
    phi_h, phi_h_D,
    gamma_kl_di, gamma_kl_sum_di, delta_kl_di, delta_kl_sum_di, gamma_g_di, 
    C, C_mask, D, D_mask,
    missing = False
):
    if not missing:
        return vmap_update_phi_g_full(phi_h, phi_h_D, gamma_kl_di, gamma_kl_sum_di, delta_kl_di, delta_kl_sum_di, gamma_g_di, C, D)
    else:
        assert C_mask is not None and D_mask is not None
        return vmap_update_phi_g_mis(phi_h, phi_h_D, gamma_kl_di, gamma_kl_sum_di, delta_kl_di, delta_kl_sum_di, gamma_g_di, C, C_mask, D, D_mask)

def update_phi_h(
    phi_g,
    gamma_kl_di, gamma_kl_sum_di, delta_kl_di, delta_kl_sum_di, gamma_h_di, 
    C, C_mask, D, D_mask,
    missing = False
):
    if not missing:
        return vmap_update_phi_h_mis(phi_g, gamma_kl_di, gamma_kl_sum_di, delta_kl_di, delta_kl_sum_di, gamma_h_di, C, D)
    else:
        assert C_mask is not None and D_mask is not None
        return vmap_update_phi_h_mis(phi_g, gamma_kl_di, gamma_kl_sum_di, delta_kl_di, delta_kl_sum_di, gamma_h_di, C, C_mask, D, D_mask)

