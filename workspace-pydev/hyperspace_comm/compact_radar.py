"""Travel-time localization and a massless Born echo on an assumed real torus."""
from itertools import product
import math

import numpy as np
from scipy.special import ellipkm1

from compact_locality import _positive


def _periods(circumferences):
    periods = np.asarray(circumferences, dtype=float)
    if periods.ndim != 1 or not periods.size or not np.all(np.isfinite(periods) & (periods > 0)):
        raise ValueError("circumferences must be a nonempty vector of finite positive lengths")
    return periods


def _point(coordinates, periods):
    point = np.asarray(coordinates, dtype=float)
    if point.shape != periods.shape or not np.all(np.isfinite(point)):
        raise ValueError("a point needs one finite coordinate per circumference")
    return np.remainder(point, periods)


def cross_ports(circumferences):
    """Return the origin followed by the positive and negative quarter ports on each axis."""
    periods = _periods(circumferences)
    quarter = np.diag(periods / 4)
    ports = np.zeros((2 * len(periods) + 1, len(periods)))
    ports[1::2] = quarter
    ports[2::2] = np.remainder(-quarter, periods)
    return ports


def cross_port_times(circumferences, speed, target):
    """Monostatic first-return times in cross_ports order on an isotropic torus."""
    periods = _periods(circumferences)
    speed = _positive(speed, "speed")
    separation = np.abs(cross_ports(periods) - _point(target, periods))
    shortest = np.minimum(separation, periods - separation)
    return 2 * (np.hypot.reduce(shortest, axis=1) / speed)


def locate_cross_port_target(circumferences, speed, times):
    """Recover a point modulo the periods from consistent, noiseless first returns."""
    periods = _periods(circumferences)
    speed = _positive(speed, "speed")
    times = np.asarray(times, dtype=float)
    if times.shape != (2 * len(periods) + 1,) or not np.all(np.isfinite(times) & (times >= 0)):
        raise ValueError("one finite nonnegative arrival per cross port is required")

    # Use a common length scale before squaring the ranges.
    scale = max(periods)
    normalized_ranges = (speed / scale) * (times / 2)
    if not np.all(np.isfinite(normalized_ranges)) or np.any(normalized_ranges > math.sqrt(len(periods))):
        raise ValueError("arrivals exceed the torus range")
    squared = normalized_ranges**2
    quarter = periods / scale / 4
    plus = (squared[1::2] - squared[0]) / quarter**2
    minus = (squared[2::2] - squared[0]) / quarter**2
    difference = minus - plus
    total = minus + plus
    sign = np.where(difference < 0, -1., 1.)
    roundoff = 64 * np.finfo(float).eps / quarter**2
    central_branch = total >= 2 - roundoff
    relative_coordinate = np.where(central_branch, difference / 4, sign * (10 - total) / 8)
    position = np.remainder(relative_coordinate * (periods / 4), periods)
    if not np.allclose(cross_port_times(periods, speed, position), times,
                       rtol=1e-10, atol=1e-12 * (scale / speed)):
        raise ValueError("arrivals are inconsistent with one target in the cross-port model")
    return position


def _image_travel_times(displacement, periods, speed, time):
    reach = speed * time
    if not math.isfinite(reach):
        raise ValueError("wave reach must be finite")
    bounds = [
        range(math.ceil((-reach - delta) / period), math.floor((reach - delta) / period) + 1)
        for delta, period in zip(displacement, periods)
    ]
    for winding in product(*bounds):
        travel_time = math.hypot(*(displacement + periods * winding)) / speed
        if travel_time < time:
            yield travel_time


def torus_born_echo_2d(circumferences, speed, source, target, receiver, time, stiffness):
    """Scattered impulse response to first order in a massless point defect.

    Coordinates are real and periodic. Source and receiver must each be
    separated from the target. The coincident free-port baseline and repeated
    point scattering are outside this first-order continuum calculation.
    At each arrival the value uses the left limit.
    """
    periods = _periods(circumferences)
    if periods.shape != (2,):
        raise ValueError("the echo kernel requires two circumferences")
    speed = _positive(speed, "speed")
    stiffness = _positive(stiffness, "stiffness", allow_zero=True)
    time = float(time)
    if not math.isfinite(time):
        raise ValueError("time must be finite")
    target = _point(target, periods)
    outbound = _point(source, periods) - target
    inbound = _point(receiver, periods) - target
    if not np.any(outbound) or not np.any(inbound):
        raise ValueError("point target must be separated from both ports")
    if time <= 0 or stiffness == 0:
        return 0.

    out_times = list(_image_travel_times(outbound, periods, speed, time))
    in_times = list(_image_travel_times(inbound, periods, speed, time))
    terms = []
    for a, b in product(out_times, in_times):
        excess = time - (a + b)
        if excess > 0:
            left, right = excess + 2 * a, excess + 2 * b
            complementary_parameter = (2 * a / left) * (2 * b / right)
            terms.append(ellipkm1(complementary_parameter) / math.sqrt(left) / math.sqrt(right))
    return -stiffness * math.fsum(terms) / (2 * math.pi**2 * speed**4)
