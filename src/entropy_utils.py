"""
entropy_utils.py
Entropy-related utilities and correction terms used in the key-rate calculation.
"""

import numpy as np
import math


def binary_entropy(p, tol=1e-15):
    """
    Compute binary (Bernoulli) entropy h(p) = -p log2 p - (1-p) log2(1-p).
    Inputs:
        - p (float): probability.
        - tol (float): numerical tolerance to avoid log(0).
    Outputs:
        - b_entropy (float): binary entropy in bits.
    """
    p = float(p)
    if p < tol or p > 1 - tol:
        return 0.0
    b_entropy = -p*np.log2(p) - (1-p)*np.log2(1-p)
    return b_entropy



def delta_w(w, Z_dim):
    """
    Compute the finite-size energy correction term Delta(w) (Eq. (6) in the paper).
    Delta(w) = sqrt(w) log2|Z| + (1+sqrt(w)) h( sqrt(w)/(1+sqrt(w)) )
    Inputs:
        - w (float): bound weight.
        - Z_dim (int): size of the key alphabet |Z|.
    Outputs:
        - dw (float): correction term in bits
    """
    if w <= 0:
        return 0.0
    aux = np.sqrt(w)
    term1 = aux * np.log2(Z_dim)
    term2 = (1.0 + aux) * (binary_entropy(aux / (1.0 + aux)))
    dw = term1 + term2
    return dw



def delta_smooth(rank_rhoA, n_key, eps_bar):
    """
    Compute the smoothing correction term delta(eps_bar) (between Eq. (5) and Eq. (6)).
    Inputs:
        - rank_rhoA (int): upper bound on rank(rho_A)
        - n_key (int): number of key samples
        - eps_bar (float): security parameter for smoothing
    Outputs:
        - ds (float): correction term in bits
    """
    if n_key <= 0:
        raise ValueError("must be > 0")
    ds = (2.0 * np.log2(rank_rhoA + 3.0)) * np.sqrt(np.log2(2.0 / eps_bar) / float(n_key))
    return ds



def shannon_entropy(p, tol=1e-15):
    """
    Computes the Shannon entropy H(p).
    Inputs: 
        - p (array of floats): probability vector.
        - tol (float): numerical tolerance to drop zero entries.
    Outputs:
        - H(float): Shannon entropy in bits.
    """
    p = np.asarray(p, dtype=float)
   
    if p.ndim != 1:
        raise ValueError("[ERROR] shannon_entropy: p must be  1D vector")
    
    p = p[p > tol]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))



def conditional_entropy(P_Z_given_X, p_X, tol=1e-15):
    """
    Compute conditional entropy H(Z|X).
    Inputs:
        - P_Z_given_X (2D array of float): matrix P[z|x_k].
        - p_X (1d array, float): vector p_X[k].
        - tol (float): tolerance for normalization and zeros.
    Outputs:
        - H (float): conditional entropy H(Z|X) in bits.
    """
    P = np.asarray(P_Z_given_X, dtype=float)
    p_X = np.asarray(p_X, dtype=float)

    if P.ndim != 2:
        raise ValueError("[ERROR] conditional_entropy: P_Z_given_X must be 2D")
    K, M = P.shape #K pos values of X and M is pos values of Z
    if p_X.shape != (K,):
        raise ValueError("[ERROR] conditional_entropy: p_X must have K elements")

    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums <= tol] = 1.0
    P = P / row_sums

    H = 0.0
    for k in range(K):
        pk = p_X[k]
        if pk <= tol:
            continue
        H_k = shannon_entropy(P[k], tol) 
        H += pk * H_k

    return float(H)



def delta_leak_EC(P_Z_given_X, p_X, n_key, beta_rec, eps_EC, p_pass=1.0, tol=1e-15):
    """
    Compute error-correction leakage delta_leak^EC (Eq. (15) in the paper).
    Inputs:
        - P_Z_given_X (2D array of float): conditional matrix P[z|x]
        - p_X (1D array of float): distribution p_X[x]
        - n_key (int): number of key samples used for error correction
        - beta_rec (float): reconciliation efficiency
        - eps_EC (float): error-correction failure probability
        - p_pass (float): sifting/post-selection success probability
        - tol (float): numerical tolerance
    Outputs:
        - delta_leak (float): leakage term in bits
        - HZ (float): H(Z) in bits
        - HZ_given_X (float): H(Z|X) in bits
    """
    P = np.asarray(P_Z_given_X, dtype=float)
    p_X = np.asarray(p_X, dtype=float)

    if P.ndim != 2:
        raise ValueError(" [ERROR]delta_leak_EC: P_Z_given_X must be 2D")
    K, M = P.shape
    if p_X.shape != (K,):
        raise ValueError("[ERROR] delta_leak_EC: p_X must have K elements")
    if n_key <= 0:
        raise ValueError("[ERROR] delta_leak_EC: n_key must be > 0")
    if not (0.0 < beta_rec <= 1.0):
        raise ValueError("[ERROR] delta_leak_EC: beta_rec must be in (0,1]")
    if not (0.0 < eps_EC < 1.0):
        raise ValueError("[ERROR] delta_leak_EC: eps_EC must be in (0,1)")
    if not (0.0 <= p_pass <= 1.0):
        raise ValueError("[ERROR] delta_leak_EC: p_pass must be in [0,1]")

    row_sums = P.sum(axis=1, keepdims=True)
    row_sums[row_sums <= tol] = 1.0
    P = P / row_sums

    P_Z = p_X @ P  # prob marginal
    HZ = shannon_entropy(P_Z, tol)
    HZ_given_X = conditional_entropy(P, p_X, tol)

    # 1st term
    term_inside = n_key * ((1.0 - beta_rec) * HZ + beta_rec * HZ_given_X)
    # 2nd term
    term_log = math.log2(2.0 / eps_EC)

    delta_leak = p_pass * (term_inside + term_log)

    return float(delta_leak), float(HZ), float(HZ_given_X)



