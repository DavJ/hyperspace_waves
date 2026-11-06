import math

import numpy as np
import pytest
from scipy.integrate import quad

from compact_radar import (circle_modal_echo, cross_port_times, locate_circle_modal_target,
    locate_cross_port_target, torus_born_echo_2d)


@pytest.mark.parametrize("periods, target", [
    ((12., 8.), (10., 5.)),
    ((12., 8.), (6., 4.)),       # both seams
    ((12., 8.), (1.3, 0.7)),     # central branches with floating-point cancellation
    ((12., 8., 5.), (9., 2., 4.4)),
])
def test_cross_ports_recover_target_modulo_all_periods(periods, target):
    times = cross_port_times(periods, 1.7, target)
    recovered = locate_cross_port_target(periods, 1.7, times)
    np.testing.assert_allclose(recovered, target, rtol=0, atol=1e-12)


def test_inconsistent_arrivals_are_rejected():
    times = cross_port_times((12., 8.), 1., (10., 5.))
    times[2] += .3
    with pytest.raises(ValueError, match="inconsistent"):
        locate_cross_port_target((12., 8.), 1., times)


def test_bistatic_born_response_matches_convolution_and_first_front():
    periods, speed, stiffness = (40., 40.), 1.4, .2
    source, target, receiver = (0., 0.), (2., 0.), (2., 3.)
    a, b = 2 / speed, 3 / speed
    time = 5.
    # The endpoint substitution removes both square-root singularities.
    excess = time - a - b
    integral = quad(lambda theta: 2 / math.sqrt(
        (2 * a + excess * math.sin(theta)**2) *
        (2 * b + excess * math.cos(theta)**2)), 0, math.pi / 2)[0]
    expected = -stiffness * integral / (2 * math.pi * speed**2)**2
    result = torus_born_echo_2d(periods, speed, source, target, receiver, time, stiffness)
    assert result == pytest.approx(expected, rel=1e-13)
    assert torus_born_echo_2d(periods, speed, source, target, receiver, a + b, stiffness) == 0.


def test_antipodal_paths_add_on_both_legs():
    port, target, time = (0., 0.), (2., 0.), 4.1
    one_path = torus_born_echo_2d((40., 40.), 1., port, target, port, time, .25)
    antipodal = torus_born_echo_2d((4., 40.), 1., port, target, port, time, .25)
    assert antipodal == pytest.approx(4 * one_path)


def test_circle_modal_echo_matches_direct_spectral_solve():
    length, speed, mass, damping = 11., 1.3, .6, .04
    frequency, target, stiffness = math.hypot(mass, 2 * math.pi * speed / length), 2.8, .15
    _, echo = circle_modal_echo(length, speed, mass, damping, frequency, target, stiffness)

    # Solve a larger standing-wave system with the actual point stiffness.
    cutoff = 80
    harmonics = np.arange(1, cutoff + 1)
    angles = 2 * math.pi * target * harmonics / length
    profiles = np.concatenate(([1 / math.sqrt(length)], math.sqrt(2 / length) *
                               np.column_stack((np.cos(angles), np.sin(angles))).ravel()))
    omega_squared = mass**2 + (2 * math.pi * speed / length)**2 * np.repeat(
        np.arange(cutoff + 1), [1] + [2] * cutoff)**2
    inverse_free = omega_squared - frequency**2 - 2j * damping * frequency
    operator = np.diag(inverse_free) + stiffness * np.outer(profiles, profiles)
    response = np.linalg.solve(operator, np.eye(len(profiles))[:, 1:5])[1:5]
    difference = response - np.diag(1 / inverse_free[1:5])
    np.testing.assert_allclose(echo, difference, rtol=.002, atol=1e-12)


@pytest.mark.parametrize("harmonic", [1, 3])
def test_adjacent_mode_echo_resolves_antipodal_targets(harmonic):
    length, speed, mass, damping = 12., 1., 1., .015
    frequency = math.hypot(mass, 2 * math.pi * speed * harmonic / length)
    targets = (3.3, 9.3)
    calibration, echo = circle_modal_echo(
        length, speed, mass, damping, frequency, targets[0], .01, harmonic=harmonic)
    _, antipodal = circle_modal_echo(
        length, speed, mass, damping, frequency, targets[1], .01, harmonic=harmonic)
    np.testing.assert_allclose(antipodal[:2, :2], echo[:2, :2], atol=1e-12)
    np.testing.assert_allclose(antipodal[2:, 2:], echo[2:, 2:], atol=1e-12)
    np.testing.assert_allclose(antipodal[2:, :2], -echo[2:, :2], atol=1e-12)
    recovered = [locate_circle_modal_target(length, calibration, sample)
                 for sample in (echo, antipodal)]
    np.testing.assert_allclose(recovered, targets, rtol=0, atol=1e-12)


def test_mixed_echo_stays_finite_at_fixed_stiffness_in_low_damping_limit():
    length, speed, mass = 12., 1., 1.
    frequency = math.hypot(mass, 2 * math.pi * speed / length)
    calibration, echo = circle_modal_echo(length, speed, mass, 1e-7, frequency, 3.3, .25)
    assert np.linalg.norm(echo[2:, :2]) == pytest.approx(abs(calibration[1]), rel=1e-6)
