"""Conditional neutral-composite kernels, hbar=c=1 in ordinary 3+1 dimensions.

The matrix Wick identity does not select a UBT carrier or its physical modes.
The scalar kernels assume a canonical positive-energy complex physical mode
in its Gaussian vacuum. r is ordinary spatial separation, never psi-distance.
Mass-source matching and the photon tail use the same minimal scalar-QED
comparator as source_matching.py, with no independent neutral field or portal.
"""
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import k1

from source_matching import complex_scalar_threshold_coefficients


def _finite(value, name, *, nonnegative=False, positive=False):
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f'{name} is outside its finite physical domain')
    return value


def gaussian_bilinear_connected(bilinear, cross_covariance):
    """<H(x)H(0)>_c=2 Tr[B G(x) B G(x)^T], H=z^T B z, <z>=0.

    Real Euclidean Gaussian coordinates and real symmetric B, of any finite
    dimension. G is the position-space cross covariance BEFORE any time
    integration; multiplying static propagators would give the wrong bubble.
    Indefinite B is allowed. Positivity of the physical Hilbert space and the
    derivation of G from the constrained UBT action are separate requirements.
    """
    b, g = np.asarray(bilinear), np.asarray(cross_covariance)
    if np.iscomplexobj(b) or np.iscomplexobj(g):
        raise ValueError('This Euclidean real-coordinate identity requires real matrices')
    b, g = b.astype(float), g.astype(float)
    if (b.ndim != 2 or b.shape[0] != b.shape[1] or b.size == 0
            or g.shape != b.shape or not np.isfinite(b).all() or not np.isfinite(g).all()
            or not np.allclose(b, b.T, rtol=1e-12, atol=1e-13)):
        raise ValueError('Finite equal-sized square matrices and symmetric B are required')
    return float(2 * np.trace(b @ g @ b @ g.T))


def pair_spectral_density(s, mass_a, mass_b):
    """One normalized scalar-pair bubble, rho=sqrt(lambda)/(16*pi^2*s).

    Identical-real-field Wick multiplicities belong in the bilinear weights;
    O=phi^dagger phi for one complex scalar has exactly this normalization.
    """
    s = _finite(s, 's')
    a, b = _finite(mass_a, 'mass_a', positive=True), _finite(mass_b, 'mass_b', positive=True)
    if s <= (a + b)**2:
        return 0.0
    return math.sqrt((1 - (a + b)**2/s) * (1 - (a - b)**2/s)) / (16 * math.pi**2)


def subtracted_density_susceptibility(z, mass):
    """Pi(z)-Pi(0), real z<=4*mass^2, from the Feynman-parameter logarithm.

    Fixed subtraction removes the local UV constant; it is not a free pole
    residue or an arbitrary source gain. The cut above threshold is not
    evaluated by this real-boundary routine. At threshold the bubble is finite.
    """
    mass, z = _finite(mass, 'mass', positive=True), _finite(z, 'z')
    if z > 4 * mass**2:
        raise ValueError('This real routine stops at the pair threshold')
    if z == 4 * mass**2:
        return 1 / (8 * math.pi**2)
    integral = quad(lambda x: -math.log1p(-z*x*(1-x)/mass**2), 0, 1,
                    epsabs=1e-12, epsrel=1e-11)[0]
    return integral / (16 * math.pi**2)


def density_static_kernel(mass, radius):
    """Separated-point static free density response; contact terms excluded.

    mass*K1(2*mass*r)/(16*pi^3*r^2). The massless limit is supported only
    for this free kernel, not for the massive low-energy EM matching below.
    """
    mass, radius = _finite(mass, 'mass', nonnegative=True), _finite(radius, 'radius', positive=True)
    if mass == 0:
        return 1 / (32 * math.pi**3 * radius**3)
    return float(mass * k1(2*mass*radius) / (16 * math.pi**3 * radius**2))


def density_evanescent_kernel(mass, omega, radius):
    """Free density response at r>0 and |omega|<=2m; no delta-function pole.

    Spectral Yukawa superposition, evaluated with a dimensionless integration
    variable and an extracted exponential. It is a forced, below-threshold
    response. No imaginary-time metric sign or physical psi mass is assumed.
    """
    mass, radius = _finite(mass, 'mass', positive=True), _finite(radius, 'radius', positive=True)
    omega = _finite(omega, 'omega')
    if abs(omega) > 2*mass:
        raise ValueError('Above-threshold propagation needs the oscillatory retarded kernel')
    ar = 2 * mass * radius
    kr = math.sqrt(max(0., 4*mass**2 - omega**2)) * radius

    def integrand(u):
        if u == 0:
            return 0.
        root = math.hypot(u, kr)
        exponent = u*u/(root+kr)  # stable sqrt(u^2+kr^2)-kr
        return u*u/math.hypot(u, ar) * math.exp(-exponent)

    value = quad(integrand, 0, np.inf, epsabs=1e-12, epsrel=1e-11)[0]
    return value * math.exp(-kr) / (32 * math.pi**3 * radius**3)


def mass_source_em_coefficient(mass, e, charge=1.):
    """<O>_{F^2}=c_O F^2, O=delta Gamma/delta J, M^2=m^2-J.

    c_O=-e^2*q^2/(192*pi^2*m^2), under the already specified heavy-scalar,
    weak-field, soft-momentum matching assumptions, with no bare J*F^2 term.
    J is an external mass-squared source, NOT a new dynamical field. This
    local mixed derivative must not be multiplied by an invented h propagator.
    It is invalid as a near-2m pump coefficient without full momentum matching.
    """
    mass = _finite(mass, 'mass', positive=True)
    charge = _finite(charge, 'charge')
    return float(complex_scalar_threshold_coefficients(
        [[mass**2]], [[[-1.]]], [[charge]], e)[0])


def photon_pair_static_asymptotic(mass, e, charge, radius):
    """Leading scalar-QED two-photon tail: 15*c_O^2/(pi^3*r^7).

    Requires m*r>>1, weak coupling, canonical vacuum Maxwell modes and the
    mass-source operator definition above. This ordinary photon-pair continuum
    is order e^4, with no one-photon/composite pole. It is not the full
    loop-corrected kernel, nor automatically a nonzero-psi-harmonic channel.
    """
    radius = _finite(radius, 'radius', positive=True)
    c = mass_source_em_coefficient(mass, e, charge)
    return 15*c*c / (math.pi**3 * radius**7)
