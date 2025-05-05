from .utils_mnsbm import (
    normalize_log_probs, 
    update_gamma_g, update_gamma_h,
    KLD_dirichlet, KLD_gpi, post_dir, log_post_dir,
    plt_blocks,
    cluster_proportions, sc_init, sc_bi_init,
    generate_phi_g, generate_phi_h, ICL_penalty, jax_array_to_csv
)
from . import utils_dual
import os, time, gc, pickle
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax.scipy.special import digamma


"""SBM Model"""
def robbins_monro_schedule(t, eta_0=0.01, tau=5000, kappa=0.75):
    return eta_0 / (1 + t / tau) ** kappa

class MNSBMDual:
    def __init__(self, C, D, K, L, alphas=None, phis=None, gammas=None,
                 init_clusters=None, rand_init='random', target_cats=None, target_concentration=None, 
                 concentration=0.9, warm_start=False):
        # Matrix Size
        self.C = C
        self.D = D
        assert (C != -1).all() and (D != -1).all()
        assert self.C.shape == self.D.shape

        self.missing = False
        self.vmap_update_phi_g = utils_dual.vmap_update_phi_g
        self.vmap_update_phi_h = utils_dual.vmap_update_phi_h
        self.update_gamma_kl = utils_dual.update_gamma_kl
        self.update_delta_kl = utils_dual.update_delta_kl
        self.loglik_q = utils_dual.loglik_q
        self.loglik_q_D = utils_dual.loglik_q_D
        self.sbm_log_lik = utils_dual.sbm_log_lik
        self.sbm_log_lik_D = utils_dual.sbm_log_lik_D

        self.N, self.M = self.C.shape
        self.num_cat = int(C.max() + 1)
        self.num_cat_D = int(D.max() + 1)

        # Cluster dimensions
        self.K, self.L = K, L

        # SBM object
        self.rand_init = rand_init
        self.fitted = False

        # Training history (store iteration metrics)
        self.training_history = []
        self.ICL_fitted = {}

        print("Initializing variational distributions...")

        # If using spectral clustering initialization
        if self.rand_init == 'spectral':
            self.init_g, self.init_h, self.init_proportions = sc_init(self.C, self.K, self.L, self.num_cat, None)
            self.init_proportions_D = cluster_proportions(self.D, self.K, self.L, self.num_cat_D, self.init_g, self.init_h)
        elif self.rand_init == 'spectral_bi':
            self.init_g, self.init_h, self.init_proportions = sc_bi_init(self.C, self.K, self.L, self.num_cat, None)
            self.init_proportions_D = cluster_proportions(self.D, self.K, self.L, self.num_cat_D, self.init_g, self.init_h)
        elif self.rand_init == 'init':
            self.init_g, self.init_h = init_clusters['g'], init_clusters['h']
            if gammas is None and not warm_start:
                # when initializing a new model with initial values based on cluster proportions
                self.init_proportions = cluster_proportions(self.C, self.K, self.L, self.num_cat, self.init_g, self.init_h)
                self.init_proportions_D = cluster_proportions(self.D, self.K, self.L, self.num_cat_D, self.init_g, self.init_h)
            elif gammas is not None and warm_start:
                # when using existing gammas as a warm start
                self.init_proportions = gammas['gamma_kl'] - 1
                self.init_proportions = gammas['delta_kl'] - 1
            else:
                # when using warm start based on cluster labels (self.init_proportions will be skipped by using update_gammas_warm)
                self.init_proportions = np.random.dirichlet(alpha=np.ones(self.num_cat), size=(self.K, self.L))  # Shape (K, L, num_cat)
                self.init_proportions_D = np.random.dirichlet(alpha=np.ones(self.num_cat_D), size=(self.K, self.L))  # Shape (K, L, num_cat)

        # Initialize priors
        if alphas is None:
            # uniform prior over the categories
            self.alpha_g, self.alpha_h, self.alpha_pi = 1,1,1
        else:
            self.alphas = alphas
            self.alpha_g, self.alpha_h, self.alpha_pi = alphas['alpha_g'], alphas['alpha_h'], alphas['alpha_pi']

        if phis is None:
            if self.rand_init == 'uniform':
                # uniform over the cluster assignments
                self.phi_g = jnp.full((self.N, self.K), 1.0 / self.K)  # q(g_i)
                self.phi_h = jnp.full((self.M, self.L), 1.0 / self.L)  # q(h_j)
            elif self.rand_init in ['random', 'random_target']:
                self.phi_g = jnp.asarray(np.random.dirichlet(alpha=np.ones(self.K), size=self.N))  # Shape (N, K)
                self.phi_h = jnp.asarray(np.random.dirichlet(alpha=np.ones(self.L), size=self.M))  # Shape (M, L)
            elif self.rand_init in ['spectral', 'spectral_bi', 'init']:
                self.phi_g = generate_phi_g(self.N, self.K, self.init_g, concentration=concentration)
                self.phi_h = generate_phi_h(self.M, self.L, self.init_h, concentration=concentration)
        else:
            self.phis = phis
            self.phi_g, self.phi_h, = phis['phi_g'], phis['phi_h']

        if gammas is None:
            if self.rand_init == 'uniform':
                # uniform over the cluster probabilities and block distributions
                self.gamma_g = jnp.ones(self.K)    # q(pi^g)
                self.gamma_h = jnp.ones(self.L)    # q(pi^h)
                self.gamma_kl = jnp.ones((self.K, self.L, self.num_cat))  # q(pi^(k,l))
                self.delta_kl = jnp.ones((self.K, self.L, self.num_cat_D))  # q(pi^(k,l))
            elif self.rand_init == 'random':
                # 2. Initialize gamma_g and gamma_h as positive values
                self.gamma_g = np.random.gamma(shape=2.0, scale=1.0, size=self.K)  # Shape (K,)
                self.gamma_h = np.random.gamma(shape=2.0, scale=1.0, size=self.L)  # Shape (L,)
                self.gamma_kl = np.random.dirichlet(alpha=np.ones(self.num_cat), size=(self.K, self.L))  # Shape (K, L, num_cat)
                self.delta_kl = np.random.dirichlet(alpha=np.ones(self.num_cat_D), size=(self.K, self.L))  # Shape (K, L, num_cat)
            elif self.rand_init == 'random_target':
                assert target_cats is not None and target_concentration is not None
                # 2. Initialize gamma_g and gamma_h as positive values
                self.gamma_g = np.random.gamma(shape=2.0, scale=1.0, size=self.K)  # Shape (K,)
                self.gamma_h = np.random.gamma(shape=2.0, scale=1.0, size=self.L)  # Shape (L,)
                
                alpha_kl = np.ones(self.num_cat)  # Base concentration
                alpha_kl[target_cats] += target_concentration  # Add weight to target categories
                self.gamma_kl = np.random.dirichlet(alpha=alpha_kl, size=(self.K, self.L))  # Shape (K, L, num_cat)
                self.delta_kl = np.random.dirichlet(alpha=alpha_kl, size=(self.K, self.L))  # Shape (K, L, num_cat)
            elif self.rand_init in ['spectral', 'spectral_bi', 'init']:
                self.gamma_g = np.random.gamma(shape=2.0, scale=1.0, size=self.K)  # Shape (K,)
                self.gamma_h = np.random.gamma(shape=2.0, scale=1.0, size=self.L)  # Shape (L,)
                self.gamma_kl = 1 + self.init_proportions
                self.delta_kl = 1 + self.init_proportions_D
        else:
            self.gammas = gammas
            self.gamma_g, self.gamma_h, self.gamma_kl, self.delta_kl = (gammas[key] for key in ['gamma_g', 'gamma_h', 'gamma_kl', 'delta_kl'])

    def batch_vi(self, num_iters, tol=1e-6, batch_print=50, fitted=False):
        self.C_1h = jax.nn.one_hot(self.C, num_classes=self.num_cat)
        self.D_1h = jax.nn.one_hot(self.D, num_classes=self.num_cat_D)
        
        if fitted:
            phi_g, phi_h, gamma_g, gamma_h, gamma_kl, delta_kl = (self.fitted_params[keys] for keys in ('phi_g', 'phi_h', 'gamma_g', 'gamma_h', 'gamma_kl', 'delta_kl'))
        else:
            phi_g, phi_h = self.phi_g, self.phi_h
            gamma_g, gamma_h, gamma_kl, delta_kl = self.gamma_g, self.gamma_h, self.gamma_kl, self.delta_kl

        print("Running batch variational inference...")

        # Run VI loop
        elbo_prev = -jnp.inf
        for i in range(num_iters):
            start_time = time.time()
            # Local updates
            gamma_g_di, gamma_h_di = digamma(gamma_g), digamma(gamma_h)
            gamma_kl_sum_di = digamma(gamma_kl.sum(axis=2)[:, :, jnp.newaxis])
            gamma_kl_di = digamma(gamma_kl)
            delta_kl_sum_di = digamma(delta_kl.sum(axis=2)[:, :, jnp.newaxis])
            delta_kl_di = digamma(delta_kl)

            phi_g = normalize_log_probs(self.vmap_update_phi_g(phi_h, gamma_kl_di, gamma_kl_sum_di, delta_kl_di, delta_kl_sum_di, gamma_g_di, self.C, self.D))
            phi_h = normalize_log_probs(self.vmap_update_phi_h(phi_g, gamma_kl_di, gamma_kl_sum_di, delta_kl_di, delta_kl_sum_di, gamma_h_di, self.C, self.D))

            # Global updates
            gamma_g = update_gamma_g(self.alpha_g, phi_g, update_factor=1)
            gamma_h = update_gamma_h(self.alpha_h, phi_h, update_factor=1)
            
            gamma_kl = self.update_gamma_kl(self.alpha_pi, phi_g, phi_h, self.C_1h, self.K, self.L, self.num_cat, update_factor=1)
            delta_kl = self.update_delta_kl(self.alpha_pi, phi_g, phi_h, self.D_1h, self.K, self.L, self.num_cat_D, update_factor=1)

            elbo, ll, ll_D, KL_g, KL_h, KL_kl, KL_kl_D = self.elbo(phi_g, phi_h, gamma_g, gamma_h, gamma_kl, delta_kl, fitted=False, verbose=False)
            end_time = time.time()

            # Save iteration results
            self.training_history.append({
                "iteration": i, "ELBO": elbo,
                "LogLik": ll, "LogLikD": ll_D, "KL-g": KL_g, "KL-h": KL_h, "KL-kl": KL_kl, "KL-kl_D": KL_kl_D
            })

            if jnp.abs(elbo - elbo_prev) < tol:
                print(f'Iteration {i}, ELBO: {elbo:,.3f}, Loglik: {ll:,.3f}, Loglik_D: {ll_D:,.3f}, KL-g: {KL_g:,.3f}, KL-h: {KL_h:,.3f}, KL-kl: {KL_kl:,.3f}, KL_kl_D: {KL_kl_D:,.3f}')
                print(f"ELBO converged at iteration {i}.")
                break
            else:
                elbo_prev = elbo

            if i % batch_print == 0:
                print(f'Iteration {i}, ELBO: {elbo:,.3f}, Loglik: {ll:,.3f}, Loglik_D: {ll_D:,.3f}, KL-g: {KL_g:,.3f}, KL-h: {KL_h:,.3f}, KL-kl: {KL_kl:,.3f}, KL_kl_D: {KL_kl_D:,.3f}')
                print(f"Time elapsed: {end_time - start_time:,.2f} seconds")

        self.fitted = True
        self.fitted_params = {'phi_g': phi_g, 'phi_h': phi_h, 'gamma_g': gamma_g, 'gamma_h': gamma_h, 'gamma_kl': gamma_kl, 'delta_kl': delta_kl}
        self.set_posteriors()

        return phi_g, phi_h, gamma_g, gamma_h, gamma_kl, delta_kl

    def elbo(self, phi_g=None, phi_h=None, gamma_g=None, gamma_h=None, gamma_kl=None, delta_kl=None, fitted=False, verbose=False):
        if fitted:
            assert self.fitted
            phi_g, phi_h, gamma_g, gamma_h, gamma_kl, delta_kl = (self.fitted_params[keys] for keys in ('phi_g', 'phi_h', 'gamma_g', 'gamma_h', 'gamma_kl', 'delta_kl'))
        else:
            assert phi_g is not None and phi_h is not None
            assert gamma_g is not None and gamma_h is not None
            assert gamma_kl is not None and delta_kl is not None

        if not self.missing:
            ll = self.loglik_q(self.C, phi_g, phi_h, gamma_kl)
            ll_D = self.loglik_q_D(self.D, phi_g, phi_h, delta_kl)
        else:
            ll = self.loglik_q(self.C, self.C_mask, phi_g, phi_h, gamma_kl)
        KL_g = KLD_gpi(phi_g, gamma_g, self.alpha_g)
        KL_h = KLD_gpi(phi_h, gamma_h, self.alpha_h)
        KL_kl = KLD_dirichlet(gamma_kl, self.alpha_pi, axis=2).sum()
        KL_kl_D = KLD_dirichlet(delta_kl, self.alpha_pi, axis=2).sum()

        elbo = ll + ll_D - KL_g - KL_h - KL_kl - KL_kl_D

        if verbose:
            print(f'ELBO: {elbo:,.3f}, Loglik: {ll:,.3f}, Loglik_D: {ll_D:,.3f}, KL-g: {KL_g:,.3f}, KL-h: {KL_h:,.3f}, KL-kl: {KL_kl:,.3f}, KL-kl: {KL_kl_D:,.3f}')

        return elbo, ll, ll_D, KL_g, KL_h, KL_kl, KL_kl_D

    def loglik_fitted(self, slow=False):
        phi_g, phi_h, gamma_kl, delta_kl = (self.fitted_params[keys] for keys in ('phi_g', 'phi_h', 'gamma_kl', 'delta_kl'))
        log_dir_mean, _ = log_post_dir(gamma_kl)
        log_dir_mean_D, _ = log_post_dir(delta_kl)

        post_g = jnp.argmax(phi_g, axis=1)
        post_h = jnp.argmax(phi_h, axis=1)

        if slow:
            return self.sbm_log_lik_slow(self.C, log_dir_mean, post_g, post_h, self.num_cat), self.sbm_log_lik_slow(self.D, log_dir_mean_D, post_g, post_h, self.num_cat_D)
        else:
            return self.sbm_log_lik(self.C, log_dir_mean, post_g, post_h), self.sbm_log_lik(self.D, log_dir_mean_D, post_g, post_h)

        
    def ICL(self, slow=False, verbose=False):
        assert self.fitted

        g_labels, h_labels = (self.posterior_dist[keys] for keys in ('g_labels', 'h_labels'))
        K_unique = len(jnp.unique(g_labels))
        L_unique = len(jnp.unique(h_labels))

        ICL_pen = ICL_penalty(self.N, self.M, K_unique, L_unique, self.num_cat)
        ll, ll_D = self.loglik_fitted(slow=slow)

        ICL = ll + ll_D - ICL_pen

        if verbose:
            print(f'ICL: {ICL:,.3f}, Loglik: {ll:,.3f}, Loglik_D: {ll_D:,.3f}, ICL-penalty: {ICL_pen:,.3f}, K-eff: {K_unique}, L-eff: {L_unique}')

        self.ICL_fitted = {'ICL': ICL, 'Loglik': ll, 'Loglik_D': ll_D, 'ICL_pen': ICL_pen, 'K-eff': K_unique, 'L-eff': L_unique}

        return ICL

    def set_posteriors(self):
        # Obtain fitted parameters and MAP
        assert self.fitted
        
        phi_g, phi_h, gamma_kl, delta_kl = (self.fitted_params[keys] for keys in ('phi_g', 'phi_h', 'gamma_kl', 'delta_kl'))
        cluster_argmax = np.array(jnp.argmax(gamma_kl, axis=2))
        dir_mean, dir_variance = post_dir(gamma_kl)

        cluster_argmax_D = np.array(jnp.argmax(delta_kl, axis=2))
        dir_mean_D, dir_variance_D = post_dir(gamma_kl)

        self.posterior_dist = {
            'g_labels': jnp.argmax(phi_g, 1), 'h_labels': jnp.argmax(phi_h, 1),
            'cluster_argmax': cluster_argmax, 'dir_mean': dir_mean, 'dir_variance': dir_variance,
            'cluster_argmax_D': cluster_argmax_D, 'dir_mean_D': dir_mean_D, 'dir_variance_D': dir_variance_D
            }

        return 1
    
    def summary(self):
        """Print summary of output"""
        assert self.fitted

        _ = self.elbo(fitted=True, verbose=True)

        clusters_g = np.unique(self.posterior_dist['g_labels'])
        clusters_h = np.unique(self.posterior_dist['h_labels'])
        print(f"{len(clusters_g)} row clusters:", clusters_g)
        print(f"{len(clusters_h)} col clusters:", clusters_h)

        for threshold in [0.5, 0.75, 0.9, 0.95, 1]:
            print(f"Rows with <{threshold} probability: {int((self.fitted_params['phi_g'].max(1) < threshold).sum())}")
            print(f"Cols with <{threshold} probability: {int((self.fitted_params['phi_h'].max(1) < threshold).sum())}")

    def plt_blocks(self, plt_init=False, print_labels=False):
        assert self.fitted

        if plt_init:
            g_labels = self.phi_g.argmax(1)
            h_labels = self.phi_h.argmax(1)
            cluster_means = np.array(jnp.argmax(self.gamma_kl, axis=2))
            plt_blocks(self.C, g_labels, h_labels, cluster_means, title=' (init ' + self.rand_init + ')', print_labels=False)

        # Get posterior labels
        g_labels = self.fitted_params['phi_g'].argmax(1)
        h_labels = self.fitted_params['phi_h'].argmax(1)

        # Plot for C
        gamma_mean = self.fitted_params['gamma_kl'] / self.fitted_params['gamma_kl'].sum(axis=2, keepdims=True)
        cluster_means = np.array(jnp.argmax(gamma_mean, axis=2))

        plt_blocks(self.C, g_labels, h_labels, cluster_means, title=' (VI)', print_labels=print_labels)
        
        # Plot for D
        gamma_mean = self.fitted_params['delta_kl'] / self.fitted_params['delta_kl'].sum(axis=2, keepdims=True)
        cluster_means = np.array(jnp.argmax(gamma_mean, axis=2))

        plt_blocks(self.D, g_labels, h_labels, cluster_means, title=' (VI)', print_labels=print_labels)

    def to_cpu(self):
        """Move all JAX arrays in the class to the CPU."""
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, jax.Array):  # Check if it's a JAX array
                setattr(self, attr_name, jax.device_put(attr_value, device=jax.devices("cpu")[0]))

        gc.collect()
        return None
    
    def fitted_model_params(self):
        assert self.fitted
        params = {
            'prior': {'alpha_g': self.alpha_g, 'alpha_h': self.alpha_h, 'alpha_pi': self.alpha_pi},
            'vi_init': {'phi_g': self.phi_g, 'phi_h': self.phi_h, 'gamma_g': self.gamma_g, 'gamma_h': self.gamma_h, 'gamma_kl': self.gamma_kl, 'delta_kl': self.delta_kl},
            'vi_posterior': self.fitted_params,
            'training_history': self.training_history,
            'ICL_fitted': self.ICL_fitted
        }

        return params
    
    def export_outputs_csv(self, folderpath, model_name=None):
        foldersave = os.path.join(folderpath, model_name)
        os.makedirs(foldersave, exist_ok=True)

        post_mean = jax_array_to_csv(self.posterior_dist['dir_mean'], os.path.join(foldersave, 'sbm_fitted_gamma_kl.csv'))
        cluster_argmax = pd.DataFrame(self.posterior_dist['cluster_argmax']).to_csv(os.path.join(foldersave, 'sbm_fitted_cluster_argmax_D.csv'), index=False)

        post_mean = jax_array_to_csv(self.posterior_dist['dir_mean_D'], os.path.join(foldersave, 'sbm_fitted_delta_kl.csv'))
        cluster_argmax = pd.DataFrame(self.posterior_dist['cluster_argmax_D']).to_csv(os.path.join(foldersave, 'sbm_fitted_cluster_argmax_D.csv'), index=False)

        post_g = pd.DataFrame(self.fitted_params['phi_g']).to_csv(os.path.join(foldersave, 'sbm_fitted_phi_g.csv'), index=False)
        post_h = pd.DataFrame(self.fitted_params['phi_h']).to_csv(os.path.join(foldersave, 'sbm_fitted_phi_h.csv'), index=False)

        g_labels = pd.DataFrame(self.posterior_dist['g_labels']).to_csv(os.path.join(foldersave, 'sbm_fitted_g_labels.csv'), index=False)
        h_labels = pd.DataFrame(self.posterior_dist['h_labels']).to_csv(os.path.join(foldersave, 'sbm_fitted_h_labels.csv'), index=False)
        
    
    def save_jax_model(self, filepath):
        assert self.fitted
        params = {
            'prior': {'alpha_g': self.alpha_g, 'alpha_h': self.alpha_h, 'alpha_pi': self.alpha_pi},
            'vi_init': {'phi_g': self.phi_g, 'phi_h': self.phi_h, 'gamma_g': self.gamma_g, 'gamma_h': self.gamma_h, 'gamma_kl': self.gamma_kl, 'delta_kl': self.delta_kl},
            'vi_posterior': self.fitted_params,
            'training_history': self.training_history,
            'posterior_dist': self.posterior_dist,
            'ICL_fitted': self.ICL_fitted
        }
        
        # Save as pickle
        print("Saving model...")
        with open(filepath, "wb") as f:
            pickle.dump(params, f)
    
    def load_jax_model(self, filepath):
        print("Loading saved model...")
        with open(filepath, "rb") as f:
            params= pickle.load(f)
        
        self.alpha_g, self.alpha_h, self.alpha_pi = (params['prior'][key] for key in ['alpha_g', 'alpha_h', 'alpha_pi'])
        self.phi_g, self.phi_h, self.gamma_g, self.gamma_h, self.gamma_kl, self.delta_kl = (params['vi_init'][key] for key in ['phi_g', 'phi_h', 'gamma_g', 'gamma_h', 'gamma_kl', 'delta_kl'])
        self.fitted_params = params['vi_posterior']
        self.training_history = params['training_history']
        self.ICL_fitted = params['ICL_fitted']

        assert self.phi_g.shape[1] == self.K and self.phi_h.shape[1] == self.L
        assert self.gamma_g.shape[0] == self.K and self.gamma_h.shape[0] == self.L
        assert self.gamma_kl.shape == (self.K, self.L, self.num_cat)

        self.fitted = True
        self.set_posteriors()

    def load_jax_model_dict(self, params):
        print("Loading saved model...")
        
        self.alpha_g, self.alpha_h, self.alpha_pi = (params['prior'][key] for key in ['alpha_g', 'alpha_h', 'alpha_pi'])
        self.phi_g, self.phi_h, self.gamma_g, self.gamma_h, self.gamma_kl, self.delta_kl = (params['vi_init'][key] for key in ['phi_g', 'phi_h', 'gamma_g', 'gamma_h', 'gamma_kl', 'delta_kl'])
        self.fitted_params = params['vi_posterior']
        self.training_history = params['training_history']
        self.ICL_fitted = params['ICL_fitted']

        assert self.phi_g.shape[1] == self.K and self.phi_h.shape[1] == self.L
        assert self.gamma_g.shape[0] == self.K and self.gamma_h.shape[0] == self.L
        assert self.gamma_kl.shape == (self.K, self.L, self.num_cat)

        self.fitted = True
        self.set_posteriors()



