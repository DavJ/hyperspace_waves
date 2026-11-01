"""Pair pumping and collisionless readout in a declared scalar-QED comparator.

Natural units, one canonical complex massive mode, external homogeneous pulse.
No UBT spectrum, fiber-localized transmitter or multiverse channel is selected.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.integrate import solve_ivp


def _pulse_parameters(mass, duration, field, charge, p_parallel, p_perp, potential_offset):
    values = (mass, duration, field, charge, p_parallel, p_perp, potential_offset)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("pulse parameters must be finite real numbers")
    if mass <= 0 or duration <= 0 or p_perp < 0:
        raise ValueError("mass and duration must be positive; p_perp must be nonnegative")
    # A_z = offset - E*T*tanh(t/T); p is canonical, p-q*A is kinetic.
    m, transverse = mass*duration, p_perp*duration
    central_p = (p_parallel-charge*potential_offset)*duration
    impulse = charge*field*duration*duration
    incoming = math.hypot(m, transverse, central_p-impulse)
    outgoing = math.hypot(m, transverse, central_p+impulse)
    if not all(math.isfinite(x) for x in (m, transverse, central_p, impulse, incoming, outgoing)):
        raise ValueError("dimensionless pulse parameters exceed finite arithmetic")
    if incoming == 0 or outgoing == 0:
        raise ValueError("asymptotic frequencies underflowed")
    return m, transverse, central_p, impulse, incoming, outgoing


def _log_sinh(x):
    """Log(sinh(x)) for x >= 0, including the zero limit."""
    return x+math.log(-math.expm1(-2*x))-math.log(2) if x else -math.inf


def _log_cosh(x):
    x = abs(x)
    return x+math.log1p(math.exp(-2*x))-math.log(2)


def sauter_pair_number(mass, duration, field, charge, p_parallel=0., p_perp=0., *, potential_offset=0.):
    """Mean pairs in one scalar momentum pair, not a density or a probability.

    E_z=field*sech^2(t/duration). ``charge`` means the physical coupling e*q.
    This is the scalar (plus-sign) Sauter result, Kim/Lee/Yoon 0807.2696 Eq.68.
    The imaginary-lambda branch is evaluated without subtracting nearly equal
    cosines. Large hyperbolic factors are evaluated in logarithms.
    """
    _, _, p, impulse, wi, wo = _pulse_parameters(
        mass, duration, field, charge, p_parallel, p_perp, potential_offset)
    k = abs(impulse)
    if k == 0:
        return 0.
    # Rationalize wo-wi: avoids cancellation in weak fields.
    denominator = wi/2+wo/2
    difference = 2*(p/denominator)*impulse
    b = math.pi*abs(difference)
    if k < .5:
        # cosh(b)+cos(2*pi*sqrt(1/4-k^2))
        # = 2*sinh(b/2)^2 + 2*sin(delta/2)^2.
        delta = 4*math.pi*k*k/(1+math.sqrt(1-4*k*k))
        sine = math.sin(delta/2)
        sine_log = math.log(sine) if sine else -math.inf
        log_numerator = math.log(2)+float(np.logaddexp(2*_log_sinh(b/2), 2*sine_log))
    else:
        lam = k*math.sqrt((1-.5/k)*(1+.5/k))
        log_numerator = float(np.logaddexp(_log_cosh(2*math.pi*lam), _log_cosh(b)))
    log_result = log_numerator-math.log(2)-_log_sinh(math.pi*wi)-_log_sinh(math.pi*wo)
    if not math.isfinite(log_result) and log_result != -math.inf:
        raise ValueError("pulse result exceeds finite arithmetic")
    try:
        return math.exp(log_result)
    except OverflowError as exc:
        raise ValueError("pair occupation exceeds finite arithmetic") from exc


@dataclass(frozen=True)
class BogoliubovResult:
    alpha: complex
    beta: complex
    wronskian_defect: float
    window: float

    @property
    def mean_pairs(self):
        return abs(self.beta)**2


def sauter_bogoliubov(mass, duration, field, charge, p_parallel=0., p_perp=0., *,
                      potential_offset=0., window=16., rtol=2e-11, atol=2e-13):
    """Independent finite-window KG integration; never renormalizes its answer.

    ``window`` is the half-window in pulse durations. Tail truncation and ODE
    error must be checked against window/tolerance convergence, not hidden by
    enforcing |alpha|^2-|beta|^2=1 after integration.
    """
    m, pt, p, impulse, wi, wo = _pulse_parameters(
        mass, duration, field, charge, p_parallel, p_perp, potential_offset)
    if not all(math.isfinite(x) and x > 0 for x in (window, rtol, atol)):
        raise ValueError("window and ODE tolerances must be finite and positive")

    def equation(x, state):
        omega2 = m*m+pt*pt+(p+impulse*np.tanh(x))**2
        return np.array([state[1], -omega2*state[0]])

    u = 1/math.sqrt(2*wi)
    solution = solve_ivp(equation, (-window, window), [complex(u), -1j*wi*u],
                         method="DOP853", rtol=rtol, atol=atol)
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError("Klein-Gordon integration failed")
    final_u, final_du = solution.y[:, -1]
    alpha = math.sqrt(wo/2)*final_u+1j*final_du/math.sqrt(2*wo)
    beta = math.sqrt(wo/2)*final_u-1j*final_du/math.sqrt(2*wo)
    return BogoliubovResult(complex(alpha), complex(beta), float(abs(alpha)**2-abs(beta)**2-1), window)


def _population(mass, momenta, occupations, weights):
    momenta = np.asarray(momenta, dtype=float)
    occupations, weights = np.asarray(occupations, dtype=float), np.asarray(weights, dtype=float)
    if not math.isfinite(mass) or mass <= 0:
        raise ValueError("mass must be finite and positive")
    if momenta.ndim != 2 or momenta.shape[1] != 3 or occupations.shape != (len(momenta),) or weights.shape != occupations.shape:
        raise ValueError("supply N kinetic three-momenta, occupations and phase-space weights")
    if not all(np.isfinite(x).all() for x in (momenta, occupations, weights)) or np.any(occupations < 0) or np.any(weights < 0):
        raise ValueError("finite momenta and finite nonnegative populations/weights required")
    energies = np.hypot(np.hypot(momenta[:, 0], momenta[:, 1]), np.hypot(momenta[:, 2], mass))
    if not np.isfinite(energies).all():
        raise ValueError("energies exceed finite arithmetic")
    return momenta, occupations, weights, energies


def collisionless_pair_drude_tensor(mass, charge, kinetic_momenta, occupations, weights):
    """State-induced response delta j = -D delta A in a soft homogeneous probe.

    For each row the particle at k and antiparticle at -k have the SAME pair
    occupation N; the factor two counts both. Weights approximate d^3p/(2*pi)^3
    (mass dimension three), NOT fitted transmitter/detector gains. Use after
    dephasing/averaging anomalous pair coherence, with fixed initial distribution,
    negligible collisions/backreaction and a physical time-dependent probe.
    This is a medium Drude weight, not a vacuum photon mass or a DC gauge test.
    """
    if not math.isfinite(charge):
        raise ValueError("charge must be finite")
    k, n, w, energy = _population(mass, kinetic_momenta, occupations, weights)
    velocities = k/energy[:, None]
    hessian = (np.eye(3)[None, :, :]-velocities[:, :, None]*velocities[:, None, :])/energy[:, None, None]
    with np.errstate(over="ignore", invalid="ignore"):
        tensor = 2*charge**2*np.einsum("a,a,aij->ij", w, n, hessian)
    if not np.isfinite(tensor).all():
        raise ValueError("Drude weight exceeds finite arithmetic")
    return tensor


def pair_energy_density(mass, kinetic_momenta, occupations, weights):
    """Produced out-particle energy density, vacuum-subtracted; factor two/pair."""
    _, n, w, energy = _population(mass, kinetic_momenta, occupations, weights)
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(2*np.sum(w*n*energy))
    if not math.isfinite(result):
        raise ValueError("energy density exceeds finite arithmetic")
    return result


def pair_count_pmf(count, mean_pairs):
    """Geometric pair count for one bosonic squeezed-vacuum mode pair."""
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
        raise ValueError("pair count must be a nonnegative integer")
    if not math.isfinite(mean_pairs) or mean_pairs < 0:
        raise ValueError("mean pair number must be finite and nonnegative")
    if mean_pairs == 0:
        return float(count == 0)
    log_ratio = math.log(mean_pairs)-math.log1p(mean_pairs) if mean_pairs < 1 else -math.log1p(1/mean_pairs)
    return math.exp(-math.log1p(mean_pairs)+count*log_ratio)


def single_arm_click_probability(mean_pairs, efficiency=1.):
    """At least one count in ONE arm, no dark counts, independent efficiency."""
    if not math.isfinite(mean_pairs) or mean_pairs < 0 or not math.isfinite(efficiency) or not 0 <= efficiency <= 1:
        raise ValueError("nonnegative finite occupation and efficiency in [0,1] required")
    detected_mean = efficiency*mean_pairs
    return detected_mean/(1+detected_mean)
