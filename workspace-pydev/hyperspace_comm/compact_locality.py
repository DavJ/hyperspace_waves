"""Localization and causal response on an explicitly assumed real circle.

This extends the reduced-mode comparator, not the canonical UBT action. The
real coordinate y, its gradient term and all apparatus overlaps are conditional;
none is identified with imaginary time psi by these calculations.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import j0


def _positive(value, name, *, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def _cutoff(cutoff):
    if isinstance(cutoff, (bool, np.bool_)) or not isinstance(cutoff, (int, np.integer)) or cutoff < 0:
        raise ValueError("cutoff must be a nonnegative integer")
    return int(cutoff)


def _circle(circumference, separation):
    length = _positive(circumference, "circumference")
    separation = np.asarray(separation, dtype=float)
    if not np.isfinite(separation).all():
        raise ValueError("separations must be finite")
    d = np.remainder(separation, length)
    return length, np.minimum(d, length-d)


def _subgap(mass, frequency):
    mass = _positive(mass, "mass")
    frequency = float(frequency)
    if not math.isfinite(frequency) or abs(frequency) >= mass:
        raise ValueError("a real frequency strictly below the positive gap is required")
    gap = (mass-abs(frequency))*(mass+abs(frequency))
    if not math.isfinite(gap) or gap <= 0:
        raise ValueError("squared gap exceeds finite arithmetic")
    return gap


def interval_concentration_matrix(cutoff, width, circumference, *, center=0.):
    """Exact interval Gram matrix for phi_n=exp(2*pi*i*n*y/L)/sqrt(L).

    For a unit L2 amplitude in modes -N,...,N, a^dagger C a is its fraction in
    the interval. Largest eigenvalue is the achievable maximum, not the weaker
    trace bound min(1,(2*N+1)*width/L). It is NOT a population gain or capacity.
    """
    cutoff = _cutoff(cutoff)
    length = _positive(circumference, "circumference")
    width = _positive(width, "width", allow_zero=True)
    if width > length or not math.isfinite(center):
        raise ValueError("width must not exceed the circumference; center must be finite")
    count = 2*cutoff+1
    if width == length:
        return np.eye(count, dtype=complex)
    n = np.arange(-cutoff, cutoff+1, dtype=float)
    difference = n[None, :]-n[:, None]
    return width/length*np.sinc(difference*width/length)*np.exp(2j*np.pi*difference*((center % length)/length))


def maximum_interval_concentration(cutoff, width, circumference):
    return float(np.linalg.eigvalsh(interval_concentration_matrix(cutoff, width, circumference))[-1])


def interval_population_fraction(density_matrix, cutoff, width, circumference, *, center=0.):
    """One-body fraction for a supplied positive semidefinite unit-trace matrix.

    A diagonal mixture is uniform in y irrespective of populated Fourier labels.
    Coherent superpositions may localize. No physical UBT state is selected.
    """
    c = interval_concentration_matrix(cutoff, width, circumference, center=center)
    rho = np.asarray(density_matrix, dtype=complex)
    if rho.shape != c.shape or not np.isfinite(rho).all() or not np.allclose(rho, rho.conj().T, atol=1e-13, rtol=1e-13):
        raise ValueError("density matrix must be finite, Hermitian and match the mode space")
    if abs(np.trace(rho)-1) > 1e-12 or np.linalg.eigvalsh(rho).min() < -1e-13:
        raise ValueError("density matrix must be positive semidefinite with unit trace")
    return float(np.trace(rho@c).real)


def periodic_subgap_green(mass, frequency, speed, circumference, separation):
    """Exact infinite-mode inverse of mu^2-omega^2-v^2*d_y^2, v>0.

    Stable image form avoids cosh/sinh overflow. There is no damping, new
    composite pole or UBT-derived compact speed. A point-source kernel is not
    a finite-energy localized state, hardware response or psi distance.
    """
    gap = _subgap(mass, frequency)
    speed = _positive(speed, "speed")
    length, d = _circle(circumference, separation)
    kappa = math.sqrt(gap)/speed
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        result = (np.exp(-kappa*d)+np.exp(-kappa*(length-d)))/(
            2*speed*math.sqrt(gap)*(-np.expm1(-kappa*length)))
    if not np.isfinite(result).all():
        raise ValueError("Green kernel exceeds finite arithmetic")
    return result


def periodic_subgap_contrast(mass, frequency, speed, circumference, separation):
    """G(omega,0)-G(omega,d), evaluated without subtracting a large zero mode."""
    gap = _subgap(mass, frequency)
    speed = _positive(speed, "speed")
    length, d = _circle(circumference, separation)
    kappa = math.sqrt(gap)/speed
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        result = (-np.expm1(-kappa*d))*(-np.expm1(-kappa*(length-d)))/(
            2*speed*math.sqrt(gap)*(-np.expm1(-kappa*length)))
    if not np.isfinite(result).all():
        raise ValueError("Green contrast exceeds finite arithmetic")
    return result


def resonant_contrast(speed, circumference, separation):
    """Finite limit of G(0)-G(d) as omega approaches mu from below, not G itself."""
    speed = _positive(speed, "speed")
    length, d = _circle(circumference, separation)
    return d*(1-d/length)/(2*speed**2)


def box_transfer_from_modes(mass, frequency, speed, circumference, width_a, width_b, separation, *, cutoff):
    """Finite Galerkin transfer for two L1-normalized top-hat source/readout maps.

    Each width is in (0,L]. Truncation is explicit: at speed=0 a finite mode
    sum can falsely couple disjoint intervals; use contact_box_transfer for
    that exact local limit. For speed>0 the absolute omitted-tail bound is
    L/(2*pi^2*v^2*N), N>0 (also valid for point-profile sinc limits).
    """
    gap = _subgap(mass, frequency)
    speed = _positive(speed, "speed", allow_zero=True)
    length, d = _circle(circumference, separation)
    if np.ndim(d):
        raise ValueError("one separation is required for a modal transfer")
    widths = [_positive(w, "width") for w in (width_a, width_b)]
    if any(w > length for w in widths):
        raise ValueError("profile widths must not exceed the circumference")
    cutoff = _cutoff(cutoff)
    if length in widths or cutoff == 0:
        return 1/(length*gap)
    n = np.arange(1, cutoff+1, dtype=float)
    harmonics = np.sinc(n*widths[0]/length)*np.sinc(n*widths[1]/length)
    return float((1/gap+2*np.sum(harmonics*np.cos(2*np.pi*n*float(d)/length)/(
        gap+(2*np.pi*speed*n/length)**2)))/length)


def contact_box_transfer(mass, frequency, circumference, width_a, width_b, separation):
    """Exact v=0 limit: overlap/(width_a*width_b*(mu^2-omega^2)).

    This assumes local independent copies and no other nonlocal/mixing term.
    Disjoint supports give zero to all times, not just below a gap.
    """
    gap = _subgap(mass, frequency)
    length, d = _circle(circumference, separation)
    if np.ndim(d):
        raise ValueError("one separation is required for a contact transfer")
    wa, wb = (_positive(w, "width") for w in (width_a, width_b))
    if max(wa, wb) > length:
        raise ValueError("profile widths must not exceed the circumference")
    overlap = sum(max(0., min(wa/2, float(d)+shift*length+wb/2)-max(-wa/2, float(d)+shift*length-wb/2))
                  for shift in (-1, 0, 1))
    return overlap/(wa*wb*gap)


def periodic_retarded_green(mass, speed, circumference, time, separation):
    """Exact finite image sum at finite t for d_tt-v^2*d_yy+mu^2.

    A 1+1-dimensional reduced comparator. Theta(0)=1/2 at positive-time image
    fronts; the isolated value at t=0 is set to zero. No spectral truncation.
    """
    mass = _positive(mass, "mass", allow_zero=True)
    speed = _positive(speed, "speed")
    length, d = _circle(circumference, separation)
    time = float(time)
    if np.ndim(d) or not math.isfinite(time):
        raise ValueError("a finite scalar time and separation are required")
    if time <= 0 or time < float(d)/speed:
        return 0.
    reach = speed*time
    if not math.isfinite(reach):
        raise ValueError("image reach exceeds finite arithmetic")
    first = math.ceil((-reach-float(d))/length)
    last = math.floor((reach-float(d))/length)
    total = 0.
    for winding in range(first, last+1):
        flight = abs(float(d)+winding*length)/speed
        if flight <= time:
            proper_time = math.sqrt(max(0., (time-flight)*(time+flight)))
            total += (.5 if time == flight else 1.)*j0(mass*proper_time)
    return float(total/(2*speed))


def truncated_retarded_green(mass, speed, circumference, time, separation, *, cutoff):
    """Finite Fourier sum, deliberately not advertised as a local causal kernel.

    A projected point source has nonlocal Dirichlet tails. Apparent pre-front
    transfer here is a truncation/source-smearing artifact, not faster signalling.
    """
    mass = _positive(mass, "mass", allow_zero=True)
    speed = _positive(speed, "speed", allow_zero=True)
    length, d = _circle(circumference, separation)
    cutoff = _cutoff(cutoff)
    if np.ndim(d) or not math.isfinite(time):
        raise ValueError("a finite scalar time and separation are required")
    if time <= 0:
        return 0.
    n = np.arange(cutoff+1, dtype=float)
    omega = np.hypot(mass, 2*np.pi*speed*n/length)
    weights = np.where(n == 0, 1., 2.)
    return float(np.sum(weights*np.cos(2*np.pi*n*float(d)/length)*time*np.sinc(omega*time/np.pi))/length)
