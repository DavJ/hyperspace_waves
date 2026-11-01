"""Independent mode evolution, weak-pump and energy-Hessian controls."""
import math

import numpy as np
import pytest

from occupation_channel import (sauter_pair_number, sauter_bogoliubov,
    collisionless_pair_drude_tensor, pair_energy_density, pair_count_pmf,
    single_arm_click_probability)


@pytest.mark.parametrize("mass,tau,field,charge,pz,pt", [
    (1., 1., .2, 1., .7, .2),  # imaginary scalar lambda
    (1., 1., .5, 1., 0., 0.),  # lambda=0, no singularity
    (1., 1., .8, 1., .4, .3),  # real scalar lambda
    (.6, 1.4, 1.2, -.8, -.9, .4),
    (.2, .3, 3., .7, 0., .1),
])
def test_exact_scalar_spectrum_against_independent_kg_evolution(mass,tau,field,charge,pz,pt):
    exact = sauter_pair_number(mass,tau,field,charge,pz,pt)
    ode = sauter_bogoliubov(mass,tau,field,charge,pz,pt)
    assert ode.mean_pairs == pytest.approx(exact, rel=5e-7, abs=2e-12)
    assert abs(ode.wronskian_defect) < 2e-9


def test_finite_window_and_tolerance_convergence_without_renormalizing():
    args = (.7, 1.2, .6, .9, .3, .2)
    exact = sauter_pair_number(*args)
    coarse = sauter_bogoliubov(*args,window=10,rtol=2e-9,atol=2e-11)
    fine = sauter_bogoliubov(*args,window=18,rtol=2e-12,atol=2e-14)
    assert abs(fine.mean_pairs-exact) < 1e-10
    assert abs(fine.wronskian_defect) < abs(coarse.wronskian_defect)


def test_constant_A_neutral_modes_and_zero_pump_have_zero_pair_creation():
    for field, charge in ((0.,1.), (3.,0.), (0.,0.)):
        assert sauter_pair_number(1.,1.,field,charge,.7,.2,potential_offset=2.) == 0
    integrated = sauter_bogoliubov(1.,1.,0.,.8,.4,.2,potential_offset=2.)
    assert integrated.mean_pairs < 1e-23


def test_canonical_momentum_relabeling_removes_constant_gauge_offset():
    args = (.8,1.3,.6,-.7)
    p, offset = .4, 2.5
    expected = sauter_pair_number(*args,p,.2)
    shifted = sauter_pair_number(*args,p+args[3]*offset,.2,potential_offset=offset)
    assert shifted == pytest.approx(expected,rel=2e-14)
    integrated = sauter_bogoliubov(*args,p+args[3]*offset,.2,potential_offset=offset)
    assert integrated.mean_pairs == pytest.approx(expected,rel=2e-8)


def test_charge_field_and_central_momentum_reversal_symmetries():
    target = sauter_pair_number(.9,1.2,.8,.7,.5,.2)
    assert sauter_pair_number(.9,1.2,.8,-.7,.5,.2) == pytest.approx(target)
    assert sauter_pair_number(.9,1.2,-.8,.7,.5,.2) == pytest.approx(target)
    assert sauter_pair_number(.9,1.2,.8,.7,-.5,.2) == pytest.approx(target)


def test_weak_field_matches_first_order_adiabatic_fourier_integral():
    # beta_1 ~ g*pz*E/(2*omega^2) int sech^2(t/T) exp(-2i*omega*t) dt.
    mass, tau, pz, pt, charge = .8,.9,.6,.2,.7
    omega = math.sqrt(mass**2+pz**2+pt**2)
    field = 1e-5
    expected = (math.pi*charge*field*tau**2*pz/omega/math.sinh(math.pi*tau*omega))**2
    assert sauter_pair_number(mass,tau,field,charge,pz,pt) == pytest.approx(expected,rel=2e-8)


def test_zero_parallel_momentum_starts_at_fourth_power_in_weak_field():
    mass,tau,charge,field = .8,.9,.7,1e-5
    expected = math.pi**2*(charge*field*tau**2)**4/math.sinh(math.pi*tau*mass)**2
    actual = sauter_pair_number(mass,tau,field,charge)
    assert actual == pytest.approx(expected,rel=2e-8)
    assert sauter_pair_number(mass,tau,2*field,charge)/actual == pytest.approx(16,rel=2e-8)


def test_logarithmic_evaluation_survives_large_hyperbolic_arguments():
    # Direct cosh/sinh evaluation overflows; physical occupation is finite.
    actual = sauter_pair_number(1.,100.,1.,1.,0.,0.)
    assert 0 < actual < 1
    # Adiabatic long-pulse central mode approaches the scalar Schwinger exponent.
    assert actual == pytest.approx(math.exp(-math.pi),rel=2e-4)
    assert sauter_pair_number(1.,1000.,.001,1.) == 0  # genuine float underflow


def test_drude_tensor_is_the_energy_hessian_of_both_opposite_charges():
    mass,charge = 1.1,.4
    momenta = np.array([[.2,.3,-.7],[-.5,.1,.9],[.4,-.2,.6]])
    n,w = np.array([.2,.8,1.7]),np.array([.03,.04,.02])
    tensor = collisionless_pair_drude_tensor(mass,charge,momenta,n,w)
    def energy(a):
        particle = np.sqrt(mass**2+np.sum((momenta-charge*a)**2,axis=1))
        antiparticle = np.sqrt(mass**2+np.sum((-momenta+charge*a)**2,axis=1))
        return np.dot(w*n,particle+antiparticle)
    h = 3e-4
    basis = np.eye(3)*h
    direct = np.array([[(energy(a+b)-energy(a-b)-energy(-a+b)+energy(-a-b))/(4*h*h)
                        for b in basis] for a in basis])
    np.testing.assert_allclose(tensor,direct,rtol=3e-6,atol=3e-10)
    assert pair_energy_density(mass,momenta,n,w) == pytest.approx(energy(np.zeros(3)))


def test_neutral_pair_drude_weight_adds_and_is_positive_and_rotation_covariant():
    n,w,mass,charge = np.array([2.]),np.array([.3]),1.5,.4
    expected = 2*charge**2*n[0]*w[0]/mass*np.eye(3)
    np.testing.assert_allclose(collisionless_pair_drude_tensor(mass,charge,[[0,0,0]],n,w),expected)
    k = np.array([[.2,.4,.7],[-.8,.5,.3]])
    n,w = np.array([.7,.3]),np.array([.1,.2])
    a = collisionless_pair_drude_tensor(mass,charge,k,n,w)
    rotation,_ = np.linalg.qr(np.array([[1.,2.,3.],[-2.,.2,.5],[.4,-1.,2.]]))
    b = collisionless_pair_drude_tensor(mass,-charge,k@rotation.T,n,w)
    np.testing.assert_allclose(b,rotation@a@rotation.T,rtol=1e-13,atol=1e-16)
    assert np.linalg.eigvalsh(a).min() > 0
    assert np.count_nonzero(collisionless_pair_drude_tensor(mass,0,k,n,w)) == 0
    assert np.count_nonzero(collisionless_pair_drude_tensor(mass,charge,k,n*0,w)) == 0


@pytest.mark.parametrize("mean", [0.,.03,.7,3.])
def test_pair_count_statistics_and_one_arm_detection_by_explicit_sum(mean):
    counts = np.arange(800)
    probabilities = np.array([pair_count_pmf(int(k),mean) for k in counts])
    assert probabilities.sum() == pytest.approx(1,abs=2e-14)
    assert counts@probabilities == pytest.approx(mean,abs=2e-14)
    variance = (counts-mean)**2@probabilities
    assert variance == pytest.approx(mean*(1+mean),abs=2e-13)
    for efficiency in (0.,.2,.8,1.):
        direct = np.sum(probabilities*(1-(1-efficiency)**counts))
        assert single_arm_click_probability(mean,efficiency) == pytest.approx(direct,abs=3e-15)
    # N is a mean, not P(any pair); a single arm of N=3 clicks with P=3/4.
    if mean == 3:
        assert single_arm_click_probability(mean) == .75


@pytest.mark.parametrize("args", [(-1,1,1,1),(1,0,1,1),(1,1,float('nan'),1),(1,1,1,float('inf'))])
def test_invalid_pulse_domains_fail(args):
    with pytest.raises(ValueError): sauter_pair_number(*args)
    with pytest.raises(ValueError): sauter_bogoliubov(*args)


def test_readout_and_statistics_reject_unphysical_inputs():
    with pytest.raises(ValueError): collisionless_pair_drude_tensor(1,1,[[0,0,0]],[-1],[1])
    with pytest.raises(ValueError): collisionless_pair_drude_tensor(1,1,[[0,0]],[1],[1])
    with pytest.raises(ValueError): pair_energy_density(0,[[0,0,0]],[1],[1])
    with pytest.raises(ValueError): pair_count_pmf(.5,1)
    with pytest.raises(ValueError): pair_count_pmf(True,1)
    with pytest.raises(ValueError): pair_count_pmf(0,-1)
    with pytest.raises(ValueError): single_arm_click_probability(1,1.1)
    with pytest.raises(ValueError): sauter_pair_number(1,1,1,1,p_perp=-1)
    with pytest.raises(ValueError): sauter_bogoliubov(1,1,1,1,window=0)
