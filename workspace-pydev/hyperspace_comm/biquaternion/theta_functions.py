"""
Biquaternionized Jacobi Theta Functions.

Implements theta functions with biquaternion arguments according to
the Unified Biquaternion Theory (UBT) framework.

Author: David Jaros
Site: www.octonion-multiverse.com
"""

import numpy as np
import cmath
import math
from .biquaternion import Biquaternion


def jacobi_theta3_complex(z, tau, max_terms=50):
    """
    Compute the Jacobi theta function ϑ₃(z, τ) for complex arguments.
    
    ϑ₃(z, τ) = Σ_{n=-∞}^{∞} exp(πin²τ + 2πinz)
    
    Args:
        z: complex position parameter
        tau: complex modular parameter (Im(tau) > 0)
        max_terms: maximum number of terms in the sum (uses ±max_terms)
    
    Returns:
        Complex value of theta function
    """
    if tau.imag <= 0:
        raise ValueError("tau must have positive imaginary part")
    
    result = 0
    for n in range(-max_terms, max_terms + 1):
        exponent = cmath.pi * 1j * n**2 * tau + 2 * cmath.pi * 1j * n * z
        result += cmath.exp(exponent)
    
    return result


def biquaternion_theta(z_bq, tau_bq, max_terms=30):
    """
    Compute biquaternionized Jacobi theta function Θ₃(Z_BQ, T_BQ).
    
    Θ₃(Z_BQ, T_BQ) = Σ_{n=-∞}^{∞} exp(πin²T_BQ + 2πinZ_BQ)
    
    where both Z_BQ and T_BQ are biquaternions, and the exponential
    is computed using biquaternion exponential.
    
    Args:
        z_bq: Biquaternion position parameter
        tau_bq: Biquaternion modular parameter
        max_terms: maximum number of terms (±max_terms range)
    
    Returns:
        Biquaternion value of theta function
    """
    # Check tau_bq has appropriate structure (imaginary part of scalar component > 0)
    if tau_bq.w.imag <= 0:
        raise ValueError("tau_bq scalar component must have positive imaginary part")
    
    result = Biquaternion(0, 0, 0, 0)
    
    for n in range(-max_terms, max_terms + 1):
        # Compute exponent: πin²T_BQ + 2πinZ_BQ
        n_squared_term = tau_bq * (cmath.pi * 1j * n**2)
        n_linear_term = z_bq * (2 * cmath.pi * 1j * n)
        
        exponent = n_squared_term + n_linear_term
        
        # Compute exp(exponent) using biquaternion exponential
        term = exponent.exp()
        result = result + term
    
    return result


def theta_expansion_coefficients(wave_freq, base_tau, num_modes=10):
    """
    Compute coefficients for theta function expansion of a hyperspace wave.
    
    Decomposes a hyperspace wave into a sum of theta function modes:
    Ψ(t) ≈ Σ_n c_n ϑ₃(ft, τ_n)
    
    Args:
        wave_freq: frequency of the hyperspace wave
        base_tau: base modular parameter (complex)
        num_modes: number of theta function modes to use
    
    Returns:
        List of (coefficient, tau_n) tuples
    """
    # For balanced hyperspace waves: tau_n = base_tau + i/sqrt(2) + corrections
    coeffs = []
    
    balanced_tau_correction = 1j / math.sqrt(2)
    
    for n in range(num_modes):
        # Each mode has a slightly different tau parameter
        # This creates the frequency spectrum of the hyperspace wave
        delta_tau = (n - num_modes/2) * 0.1 * 1j  # Small imaginary corrections
        tau_n = base_tau + balanced_tau_correction + delta_tau
        
        # Coefficient weights decrease with mode number
        # Using a Gaussian envelope for smooth spectrum
        weight = math.exp(-(n - num_modes/2)**2 / (num_modes/4)**2)
        
        # Phase factor for proper reconstruction
        phase = cmath.exp(2j * cmath.pi * n / num_modes)
        
        coeff = weight * phase
        
        coeffs.append((coeff, tau_n))
    
    return coeffs


def reconstruct_wave_from_theta(t_array, freq, base_tau, num_modes=10, max_terms=30):
    """
    Reconstruct a hyperspace wave from theta function expansion.
    
    Args:
        t_array: array of time values
        freq: wave frequency
        base_tau: base modular parameter
        num_modes: number of theta modes
        max_terms: terms in each theta function sum
    
    Returns:
        Array of complex wave values at each time point
    """
    coeffs = theta_expansion_coefficients(freq, base_tau, num_modes)
    
    wave = np.zeros(len(t_array), dtype=complex)
    
    for t_idx, t in enumerate(t_array):
        z = freq * t  # Position parameter
        
        for coeff, tau_n in coeffs:
            theta_val = jacobi_theta3_complex(z, tau_n, max_terms)
            wave[t_idx] += coeff * theta_val
    
    # Normalize
    wave = wave / num_modes
    
    return wave


def biquaternion_theta_modular_transform(z_bq, tau_bq):
    """
    Apply modular transformation to biquaternion theta function.
    
    Implements: Θ₃(Z_BQ/T_BQ, -1/T_BQ) = √Det(T_BQ) Θ₃(Z_BQ, T_BQ)
    
    This is a key symmetry in the UBT framework.
    
    Args:
        z_bq: Biquaternion position
        tau_bq: Biquaternion modular parameter
    
    Returns:
        Tuple (z_transformed, tau_transformed, prefactor)
    """
    # Compute -1/T_BQ
    tau_inv = tau_bq.inverse()
    tau_transformed = tau_inv * (-1)
    
    # Compute Z_BQ/T_BQ
    z_transformed = z_bq / tau_bq
    
    # Compute determinant (for biquaternion, this is related to norm)
    # Prefactor: sqrt(Det(T_BQ)) ≈ sqrt(T_BQ.norm())
    # For proper definition, we use the scalar part as dominant
    det_approx = tau_bq.norm()
    prefactor = cmath.sqrt(det_approx)
    
    return z_transformed, tau_transformed, prefactor


def theta_quantization_condition(k_vec, L, tau_bq):
    """
    Compute quantization condition for hyperspace momentum from theta functions.
    
    The periodicity of theta functions imposes quantization:
    Θ₃(Z + N, T) = exp(πiTr(N²T + 2NZ)) Θ₃(Z, T)
    
    For periodicity, we need: πTr(N²T) = 2πm (integer m)
    This gives quantization of allowed momenta.
    
    Args:
        k_vec: momentum vector (3-tuple)
        L: size of periodic box (3-tuple)
        tau_bq: biquaternion modular parameter
    
    Returns:
        Boolean indicating if the momentum satisfies quantization condition
    """
    # Momentum must satisfy: k_i = 2πn_i/L_i for integers n_i
    tolerance = 1e-6
    
    for ki, Li in zip(k_vec, L):
        ni_approx = ki * Li / (2 * math.pi)
        ni_int = round(ni_approx.real)
        
        if abs(ni_approx - ni_int) > tolerance:
            return False
    
    return True


def theta_zeros_spectrum(tau_bq, search_range=10):
    """
    Find the zeros of the biquaternion theta function in a given range.
    
    These zeros represent stable wave solutions and forbidden energies.
    
    Args:
        tau_bq: biquaternion modular parameter
        search_range: size of search region
    
    Returns:
        List of approximate zero locations (as Biquaternion objects)
    """
    zeros = []
    grid_size = 20
    
    # Search in a grid
    for wx in np.linspace(-search_range, search_range, grid_size):
        for wy in np.linspace(-search_range, search_range, grid_size):
            z_bq = Biquaternion(wx + 1j*wy, 0, 0, 0)
            
            try:
                theta_val = biquaternion_theta(z_bq, tau_bq, max_terms=20)
                
                # Check if norm is very small (near zero)
                if theta_val.norm() < 0.1:
                    zeros.append(z_bq)
            except:
                pass
    
    return zeros
