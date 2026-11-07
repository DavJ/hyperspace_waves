"""Reproduce a single return in the dimensionless compact tidal-target model."""
import json
import math

import numpy as np
from scipy.integrate import quad_vec

from verify_source_matching import compact_layer_coefficients


def tidal_target(frequency, target_frequency, distance, angle):
    """Reflect and transmit unit incidence in mode +1 for the example target."""
    # Radius one, normalized box along z and a real even profile on the circle.
    modes = np.arange(-2, 3)
    profile = np.array([.10, .25, 1., .25, .10])
    half_width, coupling = .4, .12
    length = 2 * half_width
    assert coupling**2 * (profile @ profile) / length < 1
    open_modes = np.abs(modes) < frequency
    wave_numbers = np.sqrt((frequency**2 - modes**2).astype(complex))
    assert frequency > 1 and np.all(wave_numbers != 0)

    arg = 1j * wave_numbers * length
    averaged_green = 1j / wave_numbers * (np.expm1(arg) - arg) / arg**2
    # Check the analytic box average independently by integrating the Green kernel.
    numerical, _ = quad_vec(lambda x: (1 - x) * np.exp(arg * x), 0., 1.)
    np.testing.assert_allclose(averaged_green, 1j * numerical / wave_numbers, atol=1e-12)
    self_energy = np.dot(profile**2, averaged_green)
    strength = coupling**2 * frequency**4
    gap = target_frequency**2 - frequency**2
    scattering = strength / (gap - strength * self_energy)

    modes, profile, wave_numbers = (x[open_modes] for x in (modes, profile, wave_numbers))
    wave_numbers = wave_numbers.real
    overlap = profile * np.sinc(half_width * wave_numbers / math.pi)
    incident = int(np.flatnonzero(modes == 1)[0])
    np.testing.assert_allclose(self_energy.imag, np.sum(overlap**2 / (2 * wave_numbers)), atol=1e-12)
    np.testing.assert_allclose(scattering.imag, abs(scattering)**2 * self_energy.imag, atol=1e-12)

    phase = np.exp(1j * (1 - modes) * angle)
    amplitude = 1j * scattering * overlap * overlap[incident] * phase / (2 * wave_numbers)
    reflection = amplitude * np.exp(1j * (wave_numbers[incident] + wave_numbers) * distance)
    transmission = amplitude * np.exp(1j * (wave_numbers[incident] - wave_numbers) * distance)
    transmission[incident] += 1
    flux = np.dot(wave_numbers, abs(reflection)**2 + abs(transmission)**2)
    np.testing.assert_allclose(flux, wave_numbers[incident], atol=1e-12, rtol=0)
    return modes, wave_numbers, overlap, reflection


def calibrated_echo(frequency, target_frequency, distance, angle):
    modes, wave_numbers, overlap, reflection = tidal_target(
        frequency, target_frequency, distance, angle)
    first, second = (int(np.flatnonzero(modes == n)[0]) for n in (1, 2))
    _, station1 = compact_layer_coefficients(1., frequency - 1, 1.)
    _, station2 = compact_layer_coefficients(2., frequency - 2, 1.)
    # Unit pump amplitudes, M5^3 = 1, common weak layer contrast.
    transmit, receive1 = .02 * station1
    receive2 = .02 * station2[1]
    echoes = transmit * reflection[[first, second]] * np.array([receive1, receive2])
    k1, k2 = wave_numbers[[first, second]]
    calibration = receive2 * k1 * overlap[second] / (receive1 * k2 * overlap[first])
    return k2 - k1, echoes, (echoes[1] / echoes[0]) / calibration


def verify_compact_echo():
    distance, angle = 7.4, 1.2
    # Include a frequency with closed +/-2 channels in the optical-theorem check.
    tidal_target(1.55, 2.45, distance, angle)
    results = []
    for target_frequency in (2.45, 4.):
        lower, upper = (calibrated_echo(nu, target_frequency, distance, angle)
                        for nu in (2.3, 2.8))
        delta = upper[0] - lower[0]
        unambiguous_interval = 2 * math.pi / delta
        recovered_distance = (np.angle(upper[2] / lower[2]) % (2 * math.pi)) / delta
        recovered_angle = (lower[0] * recovered_distance - np.angle(lower[2])) % (2 * math.pi)
        np.testing.assert_allclose([recovered_distance, recovered_angle], [distance, angle], atol=1e-11)
        results.append({
            "target_resonance": target_frequency,
            "echo_magnitudes_at_lower_frequency": abs(lower[1]).tolist(),
            "recovered_distance": float(recovered_distance),
            "recovered_compact_angle": float(recovered_angle),
        })
    # Translation leaves the same-mode echo fixed and changes the mixed phase.
    _, before, _ = calibrated_echo(2.3, 2.45, distance, angle)
    _, after, _ = calibrated_echo(2.3, 2.45, distance, angle + .6)
    np.testing.assert_allclose(after / before, [1., np.exp(-.6j)], atol=1e-12)
    return {
        "status": "conditional_single_return_at_isolated_ports",
        "metric_frequencies": [2.3, 2.8],
        "unambiguous_distance_interval": [0., float(unambiguous_interval)],
        "examples": results,
    }


if __name__ == "__main__":
    print(json.dumps(verify_compact_echo(), indent=2))
