"""Check repeated station–target returns in a dimensionless passive port model."""
import json
import math

import numpy as np

from verify_compact_echo import recover_coordinates, tidal_target_factors


def feedback_echo(frequency, target_frequency, distance, angle):
    modes, wave_numbers, overlap, scattering = tidal_target_factors(frequency, target_frequency)
    first, second = (int(np.flatnonzero(modes == n)[0]) for n in (1, 2))
    factors = overlap / np.sqrt(2 * wave_numbers)
    travel = np.exp(1j * wave_numbers * distance)
    angular = np.exp(-1j * modes * angle)
    pattern = factors * travel * angular
    sense = factors * travel * angular.conj()
    strength = 1j * scattering
    reflection = strength * np.outer(pattern, sense)
    identity = np.eye(len(modes))
    transmission = identity + strength * np.outer(factors * travel.conj() * angular, sense)
    target_flux = reflection.conj().T @ reflection + transmission.conj().T @ transmission
    np.testing.assert_allclose(target_flux, identity, atol=1e-12)

    # Unit flux in every channel. The station is unitary and mixes modes internally.
    # Its return has norm one because mode zero has no external readout.
    indices = np.arange(len(modes))
    mixer = np.exp(2j * math.pi * np.outer(indices, indices) / len(modes)) / np.sqrt(len(modes))
    reflectivity = np.where(modes == 0, 1., .87)
    readout = np.sqrt(1 - reflectivity**2)
    station_return = mixer * reflectivity
    station_launch = mixer * readout
    station = np.block([[-np.diag(reflectivity), np.diag(readout)],
                        [station_launch, station_return]])
    np.testing.assert_allclose(station.conj().T @ station, np.eye(2 * len(modes)), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(station_return, ord=2), 1., atol=1e-12)

    incident = identity[:, first]
    launch = station_launch @ incident
    baseline = -reflectivity * incident
    first_return = reflection @ launch
    returned = np.linalg.solve(identity - reflection @ station_return, first_return)
    loop = strength * (sense @ station_return @ pattern)
    readout_fraction = np.linalg.norm(readout * pattern)**2 / np.linalg.norm(pattern)**2
    assert abs(loop) <= np.sqrt(1 - readout_fraction) < 1
    np.testing.assert_allclose(returned, first_return / (1 - loop), atol=1e-12)

    external = baseline + readout * returned
    through = transmission @ (launch + station_return @ returned)
    np.testing.assert_allclose(np.linalg.norm(external)**2 + np.linalg.norm(through)**2, 1., atol=1e-12)

    echo = external - baseline
    calibration = readout[second] * factors[second] / (readout[first] * factors[first])
    delta = wave_numbers[second] - wave_numbers[first]
    ratio = echo[second] / echo[first] / calibration
    np.testing.assert_allclose(ratio, np.exp(1j * (delta * distance - angle)), atol=1e-12)
    # The direct path is coherent and biases the same ratio on the raw output.
    raw_ratio = external[second] / external[first] / calibration
    assert not np.isclose(raw_ratio, ratio)
    return delta, ratio, loop


def verify_compact_echo_feedback():
    distance, angle = 7.4, 1.2
    results = []
    for target_frequency in (2.27, 2.30):
        lower, upper = (feedback_echo(nu, target_frequency, distance, angle) for nu in (2.3, 2.8))
        recovered_distance, recovered_angle, _ = recover_coordinates(lower[0], lower[1], upper[0], upper[1])
        np.testing.assert_allclose([recovered_distance, recovered_angle], [distance, angle], atol=1e-11)
        results.append({
            "target_resonance": target_frequency,
            "loop_magnitudes": [abs(lower[2]), abs(upper[2])],
            "amplitude_factors": [1 / abs(1 - lower[2]), 1 / abs(1 - upper[2])],
            "recovered_distance": recovered_distance,
            "recovered_compact_angle": recovered_angle,
        })
    return {"status": "conditional_repeated_returns_in_open_ports", "examples": results}


if __name__ == "__main__":
    print(json.dumps(verify_compact_echo_feedback(), indent=2))
