"""
Demonstration of Hyperspace Waves in Biquaternion UBT Framework

This script demonstrates the connection between hyperspace waves and
the Unified Biquaternion Theory (UBT), including:
1. Biquaternion representation of hyperspace waves
2. Theta function expansions
3. Curved space generalizations

Author: David Jaros
Site: www.octonion-multiverse.com
UBT: https://github.com/DavJ/unified-biquaternion-theory
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from biquaternion.biquaternion import Biquaternion
from biquaternion.wave_bq import (HyperspaceWaveBQ, generate_balanced_bq_wave,
                                   generate_polarized_bq_wave, bq_wave_superposition)
from biquaternion.theta_functions import (jacobi_theta3_complex, biquaternion_theta,
                                           reconstruct_wave_from_theta, 
                                           theta_expansion_coefficients)
from generator.generate import generate_hyperspace_wave

# Create output directory for plots
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), 'hyperspace_waves_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def demo_biquaternion_basics():
    """Demonstrate basic biquaternion operations."""
    print("\n" + "="*70)
    print("DEMO 1: Biquaternion Basics")
    print("="*70)
    
    # Create biquaternions
    q1 = Biquaternion(1+1j, 2, 3j, 4)
    q2 = Biquaternion(0, 1, 0, 0)  # Unit i
    
    print(f"\nq1 = {q1}")
    print(f"q2 = {q2}")
    
    # Operations
    print(f"\nq1 + q2 = {q1 + q2}")
    print(f"q1 * q2 = {q1 * q2}")
    print(f"q1 conjugate = {q1.conjugate()}")
    print(f"|q1| = {q1.norm():.4f}")
    
    # Quaternion unit relations
    qi = Biquaternion(0, 1, 0, 0)
    qj = Biquaternion(0, 0, 1, 0)
    qk = Biquaternion(0, 0, 0, 1)
    
    print("\nQuaternion unit relations:")
    print(f"i² = {qi * qi}")
    print(f"j² = {qj * qj}")
    print(f"k² = {qk * qk}")
    print(f"ij = {qi * qj}")
    print(f"jk = {qj * qk}")
    print(f"ki = {qk * qi}")


def demo_hyperspace_wave_bq():
    """Demonstrate biquaternion hyperspace wave generation."""
    print("\n" + "="*70)
    print("DEMO 2: Biquaternion Hyperspace Waves")
    print("="*70)
    
    freq = 2e6  # 2 MHz
    fsample = 25e6  # 25 MHz sampling
    N = 1024
    
    # Generate balanced wave
    print("\nGenerating balanced hyperspace wave...")
    samples, wave = generate_balanced_bq_wave(freq=freq, fsample=fsample, N=N)
    
    print(f"  Frequency: {freq/1e6:.1f} MHz")
    print(f"  Samples: {N}")
    print(f"  Total energy: {wave.total_energy(N):.6f}")
    
    # Extract components
    w_arr, x_arr, y_arr, z_arr = wave.generate_components(N)
    
    print(f"\nFirst 5 samples (scalar component w):")
    for i in range(5):
        print(f"  Sample {i}: {samples[i].w:.6f}")
    
    # Compare with original implementation
    original = generate_hyperspace_wave(freq=freq, fsample=fsample, N=N)
    diff = np.max(np.abs(np.real(w_arr) - original))
    print(f"\nCompatibility with original: max diff = {diff:.2e}")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Biquaternion Hyperspace Wave Components')
    
    t = np.arange(N) / fsample * 1e6  # time in microseconds
    
    axes[0, 0].plot(t, np.real(w_arr))
    axes[0, 0].set_title('Scalar Component (w) - Real')
    axes[0, 0].set_xlabel('Time (μs)')
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(t, np.imag(w_arr))
    axes[0, 1].set_title('Scalar Component (w) - Imaginary')
    axes[0, 1].set_xlabel('Time (μs)')
    axes[0, 1].grid(True)
    
    energy = wave.energy_density(N)
    axes[1, 0].plot(t, energy)
    axes[1, 0].set_title('Energy Density |Ψ_BQ|²')
    axes[1, 0].set_xlabel('Time (μs)')
    axes[1, 0].set_ylabel('Energy')
    axes[1, 0].grid(True)
    
    # Phase plot
    phase = np.angle(w_arr)
    axes[1, 1].plot(t, phase)
    axes[1, 1].set_title('Phase Angle')
    axes[1, 1].set_xlabel('Time (μs)')
    axes[1, 1].set_ylabel('Phase (rad)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hyperspace_wave_bq_components.png'), dpi=150)
    print("\nPlot saved to: " + OUTPUT_DIR + "/hyperspace_wave_bq_components.png")


def demo_polarized_waves():
    """Demonstrate polarized biquaternion waves."""
    print("\n" + "="*70)
    print("DEMO 3: Polarized Hyperspace Waves")
    print("="*70)
    
    N = 512
    freq = 2e6
    fsample = 25e6
    
    polarizations = {
        'Circular': 'circular',
        'Linear X': 'linear_x',
        'Linear Y': 'linear_y',
        'Linear Z': 'linear_z'
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Polarized Hyperspace Waves - Energy Components')
    
    t = np.arange(N) / fsample * 1e6
    
    for idx, (name, pol) in enumerate(polarizations.items()):
        samples, wave = generate_polarized_bq_wave(
            freq=freq, fsample=fsample, N=N, polarization=pol
        )
        
        w_arr, x_arr, y_arr, z_arr = wave.generate_components(N)
        
        ax = axes.flat[idx]
        ax.plot(t, np.abs(w_arr)**2, label='|w|²', alpha=0.7)
        ax.plot(t, np.abs(x_arr)**2, label='|x|²', alpha=0.7)
        ax.plot(t, np.abs(y_arr)**2, label='|y|²', alpha=0.7)
        ax.plot(t, np.abs(z_arr)**2, label='|z|²', alpha=0.7)
        ax.set_title(f'{name} Polarization')
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Component Energy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        print(f"\n{name} Polarization:")
        print(f"  Total energy: {wave.total_energy(N):.6f}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'hyperspace_wave_polarizations.png'), dpi=150)
    print("\nPlot saved to: " + OUTPUT_DIR + "/hyperspace_wave_polarizations.png")


def demo_theta_functions():
    """Demonstrate theta function connection."""
    print("\n" + "="*70)
    print("DEMO 4: Jacobi Theta Functions and Hyperspace Waves")
    print("="*70)
    
    # Complex theta function
    print("\nComplex Jacobi Theta Function ϑ₃(z, τ):")
    z = 0.5 + 0.5j
    tau = 1j
    theta_val = jacobi_theta3_complex(z, tau, max_terms=50)
    print(f"  ϑ₃({z}, {tau}) = {theta_val:.6f}")
    
    # Biquaternion theta function
    print("\nBiquaternion Theta Function Θ₃(Z_BQ, T_BQ):")
    z_bq = Biquaternion(0.5+0.5j, 0.1, 0, 0)
    tau_bq = Biquaternion(1j, 0.1j, 0, 0)
    theta_bq = biquaternion_theta(z_bq, tau_bq, max_terms=20)
    print(f"  Θ₃(Z_BQ, T_BQ) = {theta_bq}")
    print(f"  |Θ₃| = {theta_bq.norm():.6f}")
    
    # Theta function expansion coefficients
    print("\nTheta Function Expansion for Hyperspace Wave:")
    freq = 2e6
    base_tau = 1j
    coeffs = theta_expansion_coefficients(freq, base_tau, num_modes=10)
    print(f"  Number of modes: {len(coeffs)}")
    print(f"  First 3 coefficients:")
    for i, (coeff, tau_n) in enumerate(coeffs[:3]):
        print(f"    Mode {i}: c = {coeff:.4f}, τ = {tau_n:.4f}")
    
    # Wave reconstruction
    print("\nReconstructing wave from theta functions...")
    N = 256
    fsample = 25e6
    t_array = np.arange(N) / fsample
    
    wave_theta = reconstruct_wave_from_theta(t_array, freq, base_tau, 
                                             num_modes=10, max_terms=20)
    wave_original = generate_hyperspace_wave(freq=freq, fsample=fsample, N=N)
    
    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Theta Function vs Original Hyperspace Wave')
    
    t_us = t_array * 1e6
    
    axes[0].plot(t_us, np.real(wave_original), label='Original Wave', alpha=0.7)
    axes[0].plot(t_us, np.real(wave_theta), label='Theta Expansion', alpha=0.7, linestyle='--')
    axes[0].set_title('Real Part Comparison')
    axes[0].set_xlabel('Time (μs)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t_us, np.abs(wave_original), label='|Original|', alpha=0.7)
    axes[1].plot(t_us, np.abs(wave_theta), label='|Theta|', alpha=0.7, linestyle='--')
    axes[1].set_title('Magnitude Comparison')
    axes[1].set_xlabel('Time (μs)')
    axes[1].set_ylabel('|Amplitude|')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'theta_function_comparison.png'), dpi=150)
    print("Plot saved to: " + OUTPUT_DIR + "/theta_function_comparison.png")


def demo_wave_superposition():
    """Demonstrate superposition of biquaternion waves."""
    print("\n" + "="*70)
    print("DEMO 5: Wave Superposition in Biquaternion Space")
    print("="*70)
    
    N = 512
    fsample = 25e6
    
    # Create multiple waves at different frequencies
    freqs = [1e6, 2e6, 3e6]  # 1, 2, 3 MHz
    waves = []
    
    for freq in freqs:
        samples, wave = generate_balanced_bq_wave(freq=freq, fsample=fsample, N=N)
        waves.append((samples, wave))
    
    # Superpose with equal weights
    superposed = bq_wave_superposition(waves)
    
    print(f"\nCreated superposition of {len(freqs)} waves:")
    for freq in freqs:
        print(f"  - {freq/1e6:.1f} MHz")
    
    # Extract scalar component
    w_superposed = np.array([s.w for s in superposed])
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Superposition of Multiple Hyperspace Waves')
    
    t = np.arange(N) / fsample * 1e6
    
    axes[0].plot(t, np.real(w_superposed))
    axes[0].set_title('Real Part of Superposed Wave')
    axes[0].set_xlabel('Time (μs)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True)
    
    # Compute power spectrum
    fft_result = np.fft.fft(w_superposed)
    freqs_fft = np.fft.fftfreq(N, 1/fsample)
    power = np.abs(fft_result)**2
    
    # Only positive frequencies
    positive_freq_idx = freqs_fft > 0
    
    axes[1].semilogy(freqs_fft[positive_freq_idx]/1e6, power[positive_freq_idx])
    axes[1].set_title('Power Spectrum')
    axes[1].set_xlabel('Frequency (MHz)')
    axes[1].set_ylabel('Power')
    axes[1].grid(True)
    axes[1].set_xlim([0, 5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'wave_superposition.png'), dpi=150)
    print("Plot saved to: " + OUTPUT_DIR + "/wave_superposition.png")


def demo_curved_space_concepts():
    """Demonstrate curved space concepts (theoretical)."""
    print("\n" + "="*70)
    print("DEMO 6: Curved Space Generalization (Theoretical)")
    print("="*70)
    
    print("\nIn the Unified Biquaternion Theory framework:")
    print("\n1. Flat Space Wave Equation:")
    print("   □_BQ Ψ_BQ = (∇^μ∇_μ - m²)Ψ_BQ = 0")
    
    print("\n2. Curved Space Metric:")
    print("   g_μν = g_μν^(R) + i·g_μν^(I)")
    print("   where g^(I)_μν couples ordinary space to hyperspace")
    
    print("\n3. Dispersion Relation for Balanced Waves:")
    print("   With s₁ = -1/√2, s₂ = 1:")
    print("   ω²/2 = k⃗² + m²")
    print("   This allows superluminal group velocities!")
    
    print("\n4. Complex Wave Vector:")
    print("   K^μ = k^μ + i·κ^μ")
    print("   - Real part (k^μ): propagation in ordinary space")
    print("   - Imaginary part (κ^μ): propagation in hyperspace")
    
    # Demonstrate modified dispersion
    k_values = np.linspace(0, 10, 100)
    m = 1.0
    
    # Standard dispersion: ω² = k² + m²
    omega_standard = np.sqrt(k_values**2 + m**2)
    
    # Balanced hyperspace dispersion: ω²/2 = k² + m²
    omega_balanced = np.sqrt(2*(k_values**2 + m**2))
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, omega_standard, label='Standard: ω² = k² + m²', linewidth=2)
    plt.plot(k_values, omega_balanced, label='Balanced Hyperspace: ω²/2 = k² + m²', 
             linewidth=2, linestyle='--')
    
    # Add light line for reference
    plt.plot(k_values, k_values, 'k:', alpha=0.5, label='Light line (ω = k)')
    
    plt.xlabel('Wave Vector k', fontsize=12)
    plt.ylabel('Frequency ω', fontsize=12)
    plt.title('Dispersion Relations: Standard vs Hyperspace Waves', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'dispersion_relations.png'), dpi=150)
    print("\nDispersion relation plot saved to: " + OUTPUT_DIR + "/dispersion_relations.png")
    
    print("\n5. Physical Implications:")
    print("   - Enhanced barrier penetration (tunneling)")
    print("   - Faster-than-light group velocities in hyperspace component")
    print("   - Potential for retrocausal effects")
    print("   - Quantized hyperspace momenta from theta function periodicity")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("HYPERSPACE WAVES - BIQUATERNION UBT FRAMEWORK DEMONSTRATION")
    print("="*70)
    print("\nAuthor: David Jaros")
    print("Site: www.octonion-multiverse.com")
    print("UBT: https://github.com/DavJ/unified-biquaternion-theory")
    
    try:
        demo_biquaternion_basics()
        demo_hyperspace_wave_bq()
        demo_polarized_waves()
        demo_theta_functions()
        demo_wave_superposition()
        demo_curved_space_concepts()
        
        print("\n" + "="*70)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("="*70)
        print(f"\nGenerated plots in: {OUTPUT_DIR}")
        print("  - hyperspace_wave_bq_components.png")
        print("  - hyperspace_wave_polarizations.png")
        print("  - theta_function_comparison.png")
        print("  - wave_superposition.png")
        print("  - dispersion_relations.png")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
