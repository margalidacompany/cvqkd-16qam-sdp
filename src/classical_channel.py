"""
classical_channel.py

Computes the classical channel P(Z|X) induced by heterodyne detection and
rectangular discretization (including post-selection).
"""

import numpy as np
from scipy.special import erf
from region_operators import build_16qam_regions_with_postselection


def gaussian_cdf_interval(a, b, mu, sigma):
    """
    Compute P(a <= X <= b) for X ~ N(mu, sigma^2).
    Inputs:
        - a (float): lower bound (can be -np.inf)
        - b (float): upper bound (can be np.inf)
        - mu (float): mean
        - sigma (float): standard deviation
    Outputs:
        - p (float): probability mass in [a, b]
    """
    sqrt2 = np.sqrt(2)
    
    if a == -np.inf:
        term_a = -1.0
    else:
        term_a = erf((a - mu) / (sqrt2 * sigma))
    
    if b == np.inf:
        term_b = 1.0
    else:
        term_b = erf((b - mu) / (sqrt2 * sigma))
    
    return 0.5 * (term_b - term_a)


def build_classical_channel(alpha, alpha0_B, Delta, eta_t, eta_d=1.0, xi=0.01, nu_el=0.0):
    """
    Build the classical channel matrix P(Z=z | X=k) for 16-QAM with post-selection.
    For a rectangular decision region A_z = [x1,x2] x [y1,y2], we use:
        P(Z=z|X=k) = P(x1 <= Re(y) <= x2) * P(y1 <= Im(y) <= y2)
    Inputs:
        - alpha (np.ndarray, complex): Alice's constellation points (length 16).
        - alpha0_B (float): constellation spacing parameter at Bob.
        - Delta (float): post-selection parameter.
        - eta_t (float): channel transmittance.
        - eta_d (float): detector efficiency.
        - xi (float): excess noise.
        - nu_el (float): electronic noise.
    Outputs:
        - P_Z_given_X (np.ndarray): shape (16, 17), last column is discard (⊥).
    """
    dim_A = len(alpha)  
    dim_Z = 17  
    
    regions = build_16qam_regions_with_postselection(alpha0_B, Delta)
    
    beta_factor = np.sqrt(eta_d * eta_t)
    
    sigma_total_sq = 1.0 + 0.5 * eta_d * eta_t * xi + nu_el
    sigma = np.sqrt(sigma_total_sq / 2.0)
    
    P_Z_given_X = np.zeros((dim_A, dim_Z))
    
    for k in range(dim_A):
        mu_k = beta_factor * alpha[k]
        mu_x = np.real(mu_k)
        mu_y = np.imag(mu_k)
        
        p_sum = 0.0
        for z in range(16):
            region = regions[z]
            p_x = gaussian_cdf_interval(region.x_low, region.x_up, mu_x, sigma)
            p_y = gaussian_cdf_interval(region.y_low, region.y_up, mu_y, sigma)
            P_Z_given_X[k, z] = p_x * p_y
            p_sum += P_Z_given_X[k, z]
        
        P_Z_given_X[k, 16] = 1.0 - p_sum
    
    return P_Z_given_X


def compute_pass_probability(P_Z_given_X, px):
    """
    Compute the post-selection pass probability.
    p_pass = sum_k px[k] * (1 - P(discard|X=k))
    Inputs:
        - P_Z_given_X (np.ndarray): classical channel.
        - px (np.ndarray): Alice's distribution.
    Outputs:
        - p_pass (float): probability of a conclusive outcome.
    """
    p_discard_given_X = P_Z_given_X[:, -1]
    p_pass = np.sum(px * (1.0 - p_discard_given_X))
    return p_pass

