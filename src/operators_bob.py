"""
operators_bob.py: 
This module defines Bob's single-mode truncated Fock space and the displaced photon-number operators n_beta and 
n_beta^2 used as energy observables in the semidefinite program (SDP) of the security analysis.
"""


import qutip as qt
import numpy as np
from dataclasses import dataclass


@dataclass
class FockSpace:
    dim_B: int
    a: qt.Qobj
    adag: qt.Qobj
    n: qt.Qobj


def fock_ops (Nc):
    """
    Construct Fock operators truncated for Bob's mode.
    Inputs: 
        -Nc (int): photon-number cutoff (Hilbert-space dimension is Nc + 1)
    Outputs: 
        - FockSpace: container with dim_B, a, adag, n.
    """
    if Nc < 0:
        raise ValueError("Nc must be >= 0")
    dim_B = Nc + 1
    
    return FockSpace(
        dim_B=dim_B,
        a=qt.destroy(dim_B),
        adag=qt.create(dim_B),
        n=qt.num(dim_B),
    )


@dataclass
class DisplacedEnergy:
    D_beta: qt.Qobj
    n_beta: qt.Qobj
    n_beta_sq: qt.Qobj
    n_beta_norm: float
    n_beta_sq_norm: float


def displaced_energy_ops(beta, fock):
    """
    Build displaced energy operators for Bob for a given displacement beta.
    Inputs: 
        - beta (complex): displacement amplitude for Bob's mode.
        - fock (FockSpace): truncated Fock space returned by fock_ops(Nc).
    Outputs: 
        - DisplacedEnergy: container with D_beta, n_beta, n_beta_sq, n_beta_norm, n_beta_sq_norm
    """
    if not isinstance(fock, FockSpace):
        raise TypeError("fock must be an instance of FockSpace, returned by fock_ops(Nc).")

    D_beta = qt.displace(fock.dim_B, beta)

    n_beta = D_beta * fock.n * D_beta.dag()

    n_beta_sq = n_beta * n_beta

    n_beta_norm = float(np.max(np.real(n_beta.eigenenergies()))) #max --> worst case
    n_beta_sq_norm = float(np.max(np.real(n_beta_sq.eigenenergies()))) #np.real, keeps only the real part, not imaginary
    return DisplacedEnergy(D_beta, n_beta, n_beta_sq, n_beta_norm, n_beta_sq_norm)


