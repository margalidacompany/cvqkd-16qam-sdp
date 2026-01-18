"""
region_operators.py
Computes the region operators R_z for a 16-QAM CV-QKD protocol with heterodyne detection. 
These operators are used to construct the key map.
"""

import numpy as np
from scipy import integrate
from math import factorial
from dataclasses import dataclass


@dataclass
class RegionLimits:  # Defines the rectangular limits of a decision region in phase space.
    x_low: float
    x_up: float
    y_low: float
    y_up: float



def region_matrix_element_integrand(x, y, m, n):
    """
    Compute the integrand for [R_z]_{mn}, returning both real and imaginary parts.
    Inputs:
        - x (float): real coordinate
        - y (float): imaginary coordinate
        - m (int): row index
        - n (int): column index
    Output:
        - Tuple (real_part, imag_part)
    """
    r_sq = x*x + y*y
    zeta = x + 1j*y
    zeta_conj = x - 1j*y
    
    product = (zeta**m) * (zeta_conj**n) # (x+iy)^m * (x-iy)^n = ζ^m * (ζ*)^n
    value = np.exp(-r_sq) * product / np.sqrt(factorial(m) * factorial(n)) / np.pi
    return np.real(value), np.imag(value)



def compute_region_matrix_element(m, n, region, epsabs= 1e-10, epsrel = 1e-10):
    """
    Compute one matrix element [R_z]_{mn} by numerical integration over the region.
    Inputs:
        - m (int): row index
        - n (int): column index
        - region (RegionLimits): integration limits in phase space
        - epsabs (float): absolute error tolerance for integration
        - epsrel (float): relative error tolerance for integration
    Outputs:
        - complex: value of [R_z]_{mn}
    """
    # Approximate infinite bounds by truncating to [-15, 15] for numerical stability.
    x_low = max(region.x_low, -15.0)
    x_up = min(region.x_up, 15.0)
    y_low = max(region.y_low, -15.0)
    y_up = min(region.y_up, 15.0)

    real_part, _ = integrate.dblquad(
        lambda y, x: region_matrix_element_integrand(x, y, m, n)[0],
        x_low, x_up,  # x limits
        lambda x: y_low, lambda x: y_up,  # y limits (constant)
        epsabs=epsabs, epsrel=epsrel #absolute error and relative error allowed (def in the funct) 
    )
    
    imag_part, _ = integrate.dblquad(
        lambda y, x: region_matrix_element_integrand(x, y, m, n)[1],
        x_low, x_up,
        lambda x: y_low, lambda x: y_up,
        epsabs=epsabs, epsrel=epsrel
    )
    
    return real_part + 1j * imag_part


def build_region_operator(Nc, region, epsabs=1e-10, epsrel=1e-10):
    """
    Build the region operator R_z as a (Nc+1) x (Nc+1) matrix.
    Inputs:
        - Nc (int): photon-number cutoff (Hilbert-space dimension is Nc + 1)
        - region (RegionLimits): decision region limits in phase space
        - epsabs (float): absolute error tolerance for integration
        - epsrel (float): relative error tolerance for integration
    Outputs:
        - np.ndarray: complex matrix of shape (Nc+1, Nc+1)
    """
    dim = Nc + 1
    R = np.zeros((dim, dim), dtype=complex)
    
    for m in range(dim):
        for n in range(m, dim): 
            R[m, n] = compute_region_matrix_element(m, n, region, epsabs, epsrel)
            if m != n:
                R[n, m] = np.conj(R[m, n])  # complete the rest with the conjugate by hermitian symmetry
    
    return R


def build_16qam_regions_no_postselection(alpha0_B):
    """
    Build the 16 decision regions for 16-QAM WITHOUT post-selection.
    Inputs:
        - alpha0_B (float): Bob's constellation scaling parameter
    Outputs:
        - list[RegionLimits]: list of 16 RegionLimits objects
    """
    g = 2 * alpha0_B 
    
    x_boundaries = [-np.inf, -g, 0.0, g, np.inf] 
    y_boundaries = [-np.inf, -g, 0.0, g, np.inf]
    
    regions = []
    
    for row in range(4):  # y direction
        for col in range(4):  # x direction
            region = RegionLimits(
                x_low=x_boundaries[col],
                x_up=x_boundaries[col + 1],
                y_low=y_boundaries[row],
                y_up=y_boundaries[row + 1]
            )
            regions.append(region)
    
    return regions


def build_16qam_regions_with_postselection(alpha0_B, Delta):
    """
    Build the 16 decision regions for 16-QAM WITH post-selection.
    Inputs:
        - alpha0_B (float): Bob's constellation scaling parameter
        - Delta (float): post-selection gap around the decision boundaries
    Outputs:
        - list[RegionLimits]: list of 16 RegionLimits objects
    """
    g = 2 * alpha0_B

    x_lows = [-np.inf, -g, Delta, g]
    x_ups  = [-g, -Delta, g, np.inf]

    y_lows = [-np.inf,-g, Delta, g]
    y_ups  = [-g, -Delta, g, np.inf]

    regions = []
    for row in range(4):
        for col in range(4):
            regions.append(
                RegionLimits(
                    x_low=x_lows[col],
                    x_up=x_ups[col],
                    y_low=y_lows[row],
                    y_up=y_ups[row]
                )
            )

    return regions





def build_all_region_operators(Nc, alpha0_B, with_postselection=False, Delta=0.0):
    """
    Build all region operators R_z for the 16 decision regions.
    Inputs:
        - Nc (int): photon-number cutoff (Hilbert-space dimension is Nc + 1)
        - alpha0_B (float): Bob's constellation scaling parameter
        - with_postselection (bool): whether to use post-selection regions
        - Delta (float): post-selection parameter (only used if with_postselection=True)
    Outputs:
        - list[np.ndarray]: list of 16 complex matrices R_z, each of shape (Nc+1, Nc+1)
    """
    if with_postselection:
        regions= build_16qam_regions_with_postselection(alpha0_B, Delta)
    else:
        regions = build_16qam_regions_no_postselection(alpha0_B)
    
    R_operators = []
    
    for z, region in enumerate(regions):
        R_z = build_region_operator(Nc, region)
        R_operators.append(R_z)
        
    
    return R_operators