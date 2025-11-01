"""
Hyperspace wave representation in biquaternion formalism.

Implements hyperspace waves as biquaternion-valued fields
according to the Unified Biquaternion Theory (UBT).

Author: David Jaros
Site: www.octonion-multiverse.com
"""

import numpy as np
import cmath
import math
from .biquaternion import Biquaternion, bq_exp


class HyperspaceWaveBQ:
    """
    Represents a hyperspace wave in biquaternion form.
    
    The wave is described by four components forming a biquaternion field:
    Ψ_BQ = Ψ₀ + Ψ₁i + Ψ₂j + Ψ₃k
    """
    
    def __init__(self, freq=2e6, fsample=25e6, s1=-1/math.sqrt(2), s2=1, 
                 amplitudes=None):
        """
        Initialize a biquaternion hyperspace wave.
        
        Args:
            freq: frequency of wave (Hz)
            fsample: sampling frequency (Hz)
            s1: damping coefficient (should be -1/√2 for balanced waves)
            s2: frequency modulation (±1)
            amplitudes: tuple of 4 complex amplitudes (A₀, A₁, A₂, A₃) for the
                       biquaternion components. If None, uses default balanced form.
        """
        self.freq = freq
        self.fsample = fsample
        self.s1 = s1
        self.s2 = s2
        
        if amplitudes is None:
            # Default: balanced wave in scalar component only
            self.amplitudes = Biquaternion(1+0j, 0, 0, 0)
        else:
            if len(amplitudes) == 4:
                self.amplitudes = Biquaternion(*amplitudes)
            else:
                raise ValueError("amplitudes must be a tuple of 4 complex numbers")
    
    def generate(self, N=1024):
        """
        Generate N samples of the biquaternion hyperspace wave.
        
        Returns:
            List of N Biquaternion objects representing the wave samples.
        """
        index_k = np.arange(N)
        
        # Complex argument (matching original implementation):
        # arg = k * 2π * (s1 + i*s2) * (freq/fsample)
        arg_factor = 2 * cmath.pi * (self.s1 + 1j * self.s2) * (self.freq / self.fsample)
        
        # Generate wave samples
        samples = []
        for k in index_k:
            # Phase: exp(k * arg_factor)
            # This matches the original: exp(k * 2π * (s1+s2*i) * (freq/fsample))
            phase = cmath.exp(k * arg_factor)
            
            # Apply to each component of the amplitude biquaternion
            wave_bq = Biquaternion(
                self.amplitudes.w * phase,
                self.amplitudes.x * phase,
                self.amplitudes.y * phase,
                self.amplitudes.z * phase
            )
            samples.append(wave_bq)
        
        return samples
    
    def generate_components(self, N=1024):
        """
        Generate N samples and return as separate component arrays.
        
        Returns:
            Tuple of 4 numpy arrays (w_arr, x_arr, y_arr, z_arr) containing
            the complex values of each biquaternion component.
        """
        samples = self.generate(N)
        
        w_arr = np.array([s.w for s in samples])
        x_arr = np.array([s.x for s in samples])
        y_arr = np.array([s.y for s in samples])
        z_arr = np.array([s.z for s in samples])
        
        return (w_arr, x_arr, y_arr, z_arr)
    
    def generate_real_parts(self, N=1024):
        """
        Generate N samples and return real parts only (for compatibility).
        
        Returns:
            Tuple of 4 numpy arrays containing the real parts of each component.
        """
        w_arr, x_arr, y_arr, z_arr = self.generate_components(N)
        
        return (np.real(w_arr), np.real(x_arr), np.real(y_arr), np.real(z_arr))
    
    def energy_density(self, N=1024):
        """
        Calculate energy density of the wave over N samples.
        
        Returns:
            Numpy array of energy densities: |Ψ_BQ|² at each sample point.
        """
        samples = self.generate(N)
        energy = np.array([s.norm_squared() for s in samples])
        return energy
    
    def total_energy(self, N=1024):
        """
        Calculate total energy of the wave over N samples.
        
        Returns:
            Total energy (sum of energy densities).
        """
        return np.sum(self.energy_density(N))


def generate_balanced_bq_wave(freq=2e6, fsample=25e6, N=1024):
    """
    Generate a balanced hyperspace wave in biquaternion form.
    
    Convenience function for creating standard balanced waves.
    
    Args:
        freq: frequency in Hz
        fsample: sampling frequency in Hz
        N: number of samples
    
    Returns:
        Tuple (samples, wave) where:
        - samples: list of N Biquaternion objects
        - wave: HyperspaceWaveBQ object
    """
    wave = HyperspaceWaveBQ(freq=freq, fsample=fsample, 
                            s1=-1/math.sqrt(2), s2=1)
    samples = wave.generate(N)
    return samples, wave


def generate_polarized_bq_wave(freq=2e6, fsample=25e6, N=1024, 
                                polarization='circular'):
    """
    Generate a polarized hyperspace wave in biquaternion form.
    
    Args:
        freq: frequency in Hz
        fsample: sampling frequency in Hz
        N: number of samples
        polarization: 'circular', 'linear_x', 'linear_y', 'linear_z', or tuple of amplitudes
    
    Returns:
        Tuple (samples, wave)
    """
    if polarization == 'circular':
        # Circular polarization: equal x and y components with 90° phase
        amplitudes = (0, 1, 1j, 0)
    elif polarization == 'linear_x':
        amplitudes = (0, 1, 0, 0)
    elif polarization == 'linear_y':
        amplitudes = (0, 0, 1, 0)
    elif polarization == 'linear_z':
        amplitudes = (0, 0, 0, 1)
    elif isinstance(polarization, tuple):
        amplitudes = polarization
    else:
        raise ValueError(f"Unknown polarization: {polarization}")
    
    wave = HyperspaceWaveBQ(freq=freq, fsample=fsample,
                            s1=-1/math.sqrt(2), s2=1,
                            amplitudes=amplitudes)
    samples = wave.generate(N)
    return samples, wave


def bq_wave_superposition(waves, coefficients=None):
    """
    Create a superposition of multiple biquaternion waves.
    
    Args:
        waves: list of (samples, HyperspaceWaveBQ) tuples
        coefficients: list of complex coefficients for each wave
                     If None, uses equal weighting
    
    Returns:
        List of Biquaternion samples representing the superposition
    """
    if coefficients is None:
        coefficients = [1.0] * len(waves)
    
    if len(waves) == 0:
        raise ValueError("Need at least one wave for superposition")
    
    N = len(waves[0][0])
    result = []
    
    for i in range(N):
        sum_bq = Biquaternion(0, 0, 0, 0)
        for (samples, wave), coeff in zip(waves, coefficients):
            sum_bq = sum_bq + samples[i] * coeff
        result.append(sum_bq)
    
    return result
