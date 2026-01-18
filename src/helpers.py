"""
helpers.py

Helper functions used on Alice's side (source-replacement state and
probability-shaped 16-QAM constellation generation).
"""


import numpy as np
import qutip as qt

def build_tau_A(alpha, px):
    """
    Build the source-replacement state tau_A on Alice's label space.
    Input:
        - alpha (np.ndarray, complex): constellation points alpha_k (length dim_A).
        - px (np.ndarray, float): probability vector p_k (length dim_A).
    Output:
        - tau_A (qutip.Qobj): density matrix tau_A with dims = [[dim_A], [dim_A]].

    """
    dim_A = len(alpha)
    tau = np.zeros((dim_A, dim_A), dtype=complex)

    for k in range(dim_A):
        for l in range(dim_A): 
            overlap = np.exp(  
                -0.5 * (abs(alpha[k])**2 + abs(alpha[l])**2)
                + np.conj(alpha[l]) * alpha[k]
            )
            tau[k, l] = np.sqrt(px[k] * px[l]) * overlap

    return qt.Qobj(tau, dims=[[dim_A], [dim_A]])



def compute_alpha_probs_VA(alpha0, levels, nu_shape):
    """
    Build the 16-QAM constellation, its Maxwell-Boltzmann shaped probabilities, and the modulation variance V_A (Eq. (2) in the paper).
    Inputs:
        - alpha0 (float): global scaling factor for the constellation amplitudes.
        - levels (array of floats or int): amplitude levels (in this code [-3, -1, 1, 3]).
        - nu_shape (float): Maxwell–Boltzmann shaping parameter nu.
    Outputs:
        - alpha (np.ndarray, complex): array of constellation points.
        - px (np.ndarray, float): normalized Maxwell–Boltzmann probabilities p_k.
        - VA    (float): modulation variance V_A.
    """

    # 16-QAM constellation scaled by alpha0
    alpha = np.array([alpha0 * (x + 1j * y) for x in levels for y in levels])
    abs_alpha_sq = np.abs(alpha) ** 2

    p_unnorm = np.exp(-nu_shape * abs_alpha_sq)
    px = p_unnorm / np.sum(p_unnorm)

    VA = np.sum(px * abs_alpha_sq)
    return alpha, px, VA

