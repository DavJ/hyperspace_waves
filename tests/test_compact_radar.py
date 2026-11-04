import math

import numpy as np
import pytest
from scipy.integrate import quad

from compact_radar import cross_port_times, locate_cross_port_target, torus_born_echo_2d


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
