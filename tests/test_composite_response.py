"""Independent Gaussian, spectral, Euclidean and matching controls."""
import math
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import k1

from composite_response import (gaussian_bilinear_connected, pair_spectral_density,
    subtracted_density_susceptibility, density_static_kernel, density_evanescent_kernel,
    mass_source_em_coefficient, photon_pair_static_asymptotic)
from source_matching import complex_scalar_gauge_kinetic_shift


def test_matrix_wick_identity_by_explicit_four_index_contractions():
    rng = np.random.default_rng(918)
    for n in (1, 3, 8):
        b = rng.normal(size=(n, n)); b = (b+b.T)/2
        g = rng.normal(size=(n, n))
        direct = sum(b[i,j]*b[k,l]*(g[i,k]*g[j,l]+g[i,l]*g[j,k])
                     for i in range(n) for j in range(n) for k in range(n) for l in range(n))
        assert gaussian_bilinear_connected(b, g) == pytest.approx(direct, abs=3e-11)
    assert gaussian_bilinear_connected(np.eye(2)/2, np.eye(2)*3) == 9


def test_spectral_threshold_and_unequal_mass_symmetry():
    assert pair_spectral_density(8, 1, 2) == 0
    assert pair_spectral_density(9, 1, 2) == 0
    assert pair_spectral_density(13, 1, 2) == pytest.approx(pair_spectral_density(13, 2, 1))
    assert pair_spectral_density(8, 1, 1) == pytest.approx(math.sqrt(.5)/(16*math.pi**2))


def test_nonzero_background_keeps_its_existing_linear_projection():
    b, g = np.diag([1.,-2.,.5]), np.diag([.2,.4,.7])
    background = np.array([1.,2.,3.])
    ell = 2*b@background
    pair = gaussian_bilinear_connected(b,g)
    full = gaussian_bilinear_connected(b,g,linear_form=ell)
    assert full-pair == pytest.approx(4*background@b@g@b@background)
    assert full > pair


@pytest.mark.parametrize('z', [-8., -.1, 0., .3, 2., 3.9, 4.])
def test_subtracted_bubble_against_independent_spectral_dispersion(z):
    ratio = z/4
    spectral = quad(lambda t: 2*ratio*t*t/(1-ratio+ratio*t*t), 0, 1)[0]/(16*math.pi**2) if z else 0.
    assert subtracted_density_susceptibility(z, 1) == pytest.approx(spectral, rel=3e-10, abs=1e-13)


@pytest.mark.parametrize('mass,radius', [(1., .3), (1., 1.), (.4, 3.), (2., 2.)])
def test_static_kernel_bessel_spectral_and_euclidean_time_integral(mass, radius):
    def euclidean_time(t):
        x = math.hypot(t, radius)
        prop = mass*k1(mass*x)/(4*math.pi**2*x)
        return prop*prop
    independent = 2*quad(euclidean_time, 0, np.inf, epsabs=1e-14, epsrel=1e-10)[0]
    target = density_static_kernel(mass, radius)
    assert target == pytest.approx(independent, rel=2e-8)
    assert density_evanescent_kernel(mass, 0, radius) == pytest.approx(target, rel=2e-10)


def test_massless_free_limit_and_threshold_power_are_distinct():
    r = 2.
    assert density_static_kernel(0, r) == pytest.approx(1/(32*math.pi**3*r**3))
    assert density_static_kernel(1e-7, r) == pytest.approx(density_static_kernel(0, r), rel=1e-11)
    # Exactly at the positive-mass pair threshold the asymptote is r^-4,
    # not the nonuniform kappa->0 limit of the fixed-kappa r^-5/2 expansion.
    radius = 150.
    expected = 1/(32*math.pi**3*radius**4)
    assert density_evanescent_kernel(1, 2, radius) == pytest.approx(expected, rel=2e-4)
    assert subtracted_density_susceptibility(4, 1) == pytest.approx(1/(8*math.pi**2))


def test_near_threshold_response_is_finite_and_frequency_symmetric():
    below = density_evanescent_kernel(1, 1.9, 3)
    assert below > density_static_kernel(1, 3)
    assert density_evanescent_kernel(1, -1.9, 3) == below
    assert density_evanescent_kernel(1, 2, 3) > below


def test_uniform_density_source_still_couples_to_nonzero_constituent_harmonics():
    # A source Fourier index is not the constituent's Fourier index.
    count, n = 64, 3
    psi = 2*np.pi*np.arange(count)/count
    phi = np.exp(1j*n*psi)/np.sqrt(2*np.pi)
    density = abs(phi)**2
    assert abs(np.fft.fft(phi)[n]) > 1
    assert 2*np.pi*np.mean(density) == pytest.approx(1)
    np.testing.assert_allclose(np.fft.fft(density)[1:],0,atol=2e-14)
    # A charge-neutral pair can retain a nonzero correlation under phase
    # averaging even though either single nonzero-index mean vanishes.
    np.testing.assert_allclose(np.mean(np.exp(1j*n*psi)),0,atol=2e-15)
    assert np.mean(np.exp(1j*n*psi)*np.exp(-1j*n*psi)) == pytest.approx(1)


def test_mass_source_matching_is_same_action_mixed_derivative():
    mass, e, q, step = 2., .3, -2., 1e-5
    coeff = mass_source_em_coefficient(mass, e, q)
    assert coeff == pytest.approx(-e*e*q*q/(192*math.pi**2*mass**2))
    dz = (complex_scalar_gauge_kinetic_shift([[mass**2-step]], [[mass**2]], [[q]], e)
          - complex_scalar_gauge_kinetic_shift([[mass**2+step]], [[mass**2]], [[q]], e))/(2*step)
    assert coeff == pytest.approx(-dz/4, rel=1e-8)


def test_mass_source_sign_against_independent_magnetic_proper_time_kernel():
    mass, e, field = 1., .3, 1e-3
    def proper_time(t):
        x=e*field*t
        # Stable expansion of x/sinh(x)-1 at the integrable lower endpoint.
        correction=-x*x/6+7*x**4/360 if x<1e-3 else (2*x*math.exp(-x)/(1-math.exp(-2*x))-1)
        return math.exp(-mass*mass*t)*correction/(16*math.pi**2*t*t)
    direct=quad(proper_time,0,np.inf,epsabs=1e-18)[0]
    expected=mass_source_em_coefficient(mass,e)*(2*field*field)
    assert direct == pytest.approx(expected,rel=3e-7)


def test_photon_tail_is_gapless_power_law_and_can_dominate_the_free_tail():
    first = photon_pair_static_asymptotic(1., .3, 1., 20.)
    second = photon_pair_static_asymptotic(1., .3, 1., 40.)
    assert first/second == pytest.approx(128)
    assert first > density_static_kernel(1., 20.)
    assert photon_pair_static_asymptotic(1., 0., 1., 20.) == 0
    assert photon_pair_static_asymptotic(1., .3, 0., 20.) == 0


@pytest.mark.parametrize('call', [
    lambda: density_static_kernel(-1, 1), lambda: density_static_kernel(1, 0),
    lambda: density_evanescent_kernel(1, 2.01, 1),
    lambda: subtracted_density_susceptibility(4.01, 1),
    lambda: mass_source_em_coefficient(0, .3),
    lambda: pair_spectral_density(np.nan, 1, 1),
    lambda: gaussian_bilinear_connected([[1,2],[0,1]], np.eye(2)),
    lambda: gaussian_bilinear_connected([[1]], [[1]], linear_form=[1,2]),
    lambda: gaussian_bilinear_connected([[1]], [[1]], linear_form=[1j]),
])
def test_invalid_domains_are_not_silently_reinterpreted(call):
    with pytest.raises(ValueError): call()
