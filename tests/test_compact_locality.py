"""Independent interval integrals, Green identities and causal/Laplace controls."""
import math

import numpy as np
import pytest
from scipy.integrate import quad

from compact_locality import (interval_concentration_matrix,
    maximum_interval_concentration, interval_population_fraction,
    periodic_subgap_green, periodic_subgap_contrast, resonant_contrast,
    box_transfer_from_modes, contact_box_transfer, periodic_retarded_green,
    truncated_retarded_green)


def test_interval_gram_matrix_against_continuous_complex_wavefunctions():
    cutoff,width,length,center = 3,.7,4.,3.9
    c = interval_concentration_matrix(cutoff,width,length,center=center)
    rng = np.random.default_rng(138)
    a = rng.normal(size=7)+1j*rng.normal(size=7)
    a /= np.linalg.norm(a)
    n = np.arange(-cutoff,cutoff+1)
    def intensity(y):
        return abs(np.sum(a*np.exp(2j*np.pi*n*y/length))/math.sqrt(length))**2
    direct = quad(intensity,center-width/2,center+width/2,epsabs=1e-13)[0]
    assert np.vdot(a,c@a).real == pytest.approx(direct,abs=2e-14)
    assert interval_population_fraction(np.outer(a,a.conj()),cutoff,width,length,center=center) == pytest.approx(direct)


@pytest.mark.parametrize('cutoff,width', [(0,.4),(2,.4),(5,.1),(8,.8)])
def test_concentration_eigenvalue_is_achievable_and_bounded_by_trace(cutoff,width):
    c = interval_concentration_matrix(cutoff,width,1.)
    eig,vec = np.linalg.eigh(c)
    best = vec[:,-1]
    assert eig.min() >= -1e-13 and eig.max() <= 1+1e-13
    assert eig[-1] <= min(1,(2*cutoff+1)*width)+1e-13
    assert interval_population_fraction(np.outer(best,best.conj()),cutoff,width,1.) == pytest.approx(eig[-1])
    assert maximum_interval_concentration(cutoff,width,1.) == pytest.approx(eig[-1])


def test_concentration_translation_invariance_nesting_and_boundary_widths():
    eig = np.linalg.eigvalsh(interval_concentration_matrix(5,.3,2.))
    shifted = np.linalg.eigvalsh(interval_concentration_matrix(5,.3,2.,center=1.9))
    np.testing.assert_allclose(eig,shifted,atol=4e-16)
    assert np.count_nonzero(interval_concentration_matrix(3,0,1.)) == 0
    np.testing.assert_array_equal(interval_concentration_matrix(3,1,1.),np.eye(7))
    concentrations = [maximum_interval_concentration(n,.1,1.) for n in range(9)]
    assert np.all(np.diff(concentrations) > 0)


def test_diagonal_populations_stay_uniform_but_coherence_can_localize():
    populations = np.array([1.,2.,4.,3.,9.,.1,6.])
    rho = np.diag(populations/populations.sum())
    for center in (0,.2,.95):
        assert interval_population_fraction(rho,3,.1,1.,center=center) == pytest.approx(.1)
    c = interval_concentration_matrix(3,.1,1.)
    _,v = np.linalg.eigh(c)
    coherent = np.outer(v[:,-1],v[:,-1].conj())
    assert interval_population_fraction(coherent,3,.1,1.) > .5
    assert interval_population_fraction(np.diag(np.diag(coherent)),3,.1,1.) == pytest.approx(.1)


@pytest.mark.parametrize('mass,omega,speed,length,d', [
    (1.,0.,.8,3.,.7),(1.,.8,.4,4.,1.9),(.5,.1,1.2,2.,.3),
])
def test_periodic_green_against_images_and_fourier_tail_bound(mass,omega,speed,length,d):
    gap=mass**2-omega**2
    kappa=math.sqrt(gap)/speed
    images=sum(math.exp(-kappa*abs(d+j*length))/(2*speed*math.sqrt(gap)) for j in range(-100,101))
    exact=periodic_subgap_green(mass,omega,speed,length,d)
    assert exact == pytest.approx(images,rel=2e-14)
    cutoff=6000
    n=np.arange(1,cutoff+1)
    fourier=(1/gap+2*np.sum(np.cos(2*np.pi*n*d/length)/(gap+(2*np.pi*speed*n/length)**2)))/length
    bound=length/(2*np.pi**2*speed**2*cutoff)
    assert abs(fourier-exact) <= bound
    assert periodic_subgap_green(mass,-omega,speed,length,d+3*length) == pytest.approx(exact)
    assert periodic_subgap_green(mass,omega,speed,length,-d) == pytest.approx(exact)


def test_periodic_green_solves_homogeneous_ode_jump_and_zero_mode_normalization():
    mass,omega,speed,length,d = .9,.3,.7,3.2,.8
    gap=mass**2-omega**2
    g=lambda y: float(periodic_subgap_green(mass,omega,speed,length,y))
    h=2e-4
    second=(g(d+h)-2*g(d)+g(d-h))/h**2
    assert speed**2*second == pytest.approx(gap*g(d),rel=2e-7)
    h=1e-6
    jump=2*(g(h)-g(0))/h
    assert jump == pytest.approx(-1/speed**2,rel=2e-6)
    integrated=quad(g,0,length,points=[length/2],epsabs=1e-12)[0]
    assert integrated == pytest.approx(1/gap,rel=2e-13)


def test_small_speed_at_disjoint_points_is_exponentially_small_not_a_zero_mode():
    assert periodic_subgap_green(1.,0.,.001,3.,1.) == 0  # underflow far from source
    assert periodic_subgap_green(1.,0.,1.,3.,1.) > 0


def test_resonance_amplifies_uniform_component_and_keeps_differential_response_finite():
    mass,speed,length,d = 1.,.7,3.,.8
    omega=np.nextafter(mass,0.)
    origin=periodic_subgap_green(mass,omega,speed,length,0.)
    remote=periodic_subgap_green(mass,omega,speed,length,d)
    assert remote/origin == pytest.approx(1.,abs=4e-15)
    contrast=periodic_subgap_contrast(mass,omega,speed,length,d)
    assert contrast == pytest.approx(resonant_contrast(speed,length,d),rel=2e-14)
    # At ordinary detuning the stable expression agrees with direct subtraction.
    direct=periodic_subgap_green(1.,.4,speed,length,0)-periodic_subgap_green(1.,.4,speed,length,d)
    assert periodic_subgap_contrast(1.,.4,speed,length,d) == pytest.approx(direct,rel=2e-14)


def test_finite_profile_transfer_against_double_real_space_integral():
    mass,omega,speed,length,wa,wb,d = .8,.1,.7,4.,.3,.4,1.2
    def receive(y):
        return quad(lambda x: periodic_subgap_green(mass,omega,speed,length,y-x),-wa/2,wa/2,epsabs=1e-12)[0]/wa
    direct=quad(receive,d-wb/2,d+wb/2,epsabs=1e-12)[0]/wb
    modal=box_transfer_from_modes(mass,omega,speed,length,wa,wb,d,cutoff=2000)
    assert modal == pytest.approx(direct,rel=2e-10)


def test_uniform_profile_eliminates_position_addressing_at_either_end():
    mass,omega,speed,length = 1.,.3,.8,4.
    expected=1/(length*(mass**2-omega**2))
    for d in (0.,.5,1.5,3.9):
        for wa,wb in ((length,.2),(.3,length),(length,length)):
            assert box_transfer_from_modes(mass,omega,speed,length,wa,wb,d,cutoff=15) == pytest.approx(expected,rel=2e-15)
            assert contact_box_transfer(mass,omega,length,wa,wb,d) == pytest.approx(expected)


def test_no_gradient_local_copies_have_exactly_zero_disjoint_transfer():
    assert contact_box_transfer(1.,0.,4.,.4,.6,1.) == 0
    assert contact_box_transfer(1.,0.,4.,.4,.6,.1) == pytest.approx(.4/(.4*.6))
    assert contact_box_transfer(1.,0.,4.,.4,.6,3.9) == pytest.approx(.4/(.4*.6))
    # Finite Fourier projection smears the source across the circle.
    spurious=box_transfer_from_modes(1.,0.,0.,4.,.4,.6,1.,cutoff=3)
    assert abs(spurious) > .01


def test_retarded_image_front_and_wrapped_massless_arrivals():
    g=lambda t: periodic_retarded_green(0.,2.,4.,t,1.)
    assert g(-1.) == 0 and g(0.) == 0 and g(.49) == 0
    assert g(.5) == .125  # half value on the first front
    assert g(.6) == .25
    assert g(1.6) == .5
    assert g(2.6) == .75
    assert periodic_retarded_green(1.,.8,4.,.7,.9) == 0
    assert periodic_retarded_green(1.,.8,4.,1.2,.9) != 0


@pytest.mark.parametrize('mass,sigma,speed,length,d', [(1.,1.3,.8,3.,.7),(0.,1.5,.7,2.,.4)])
def test_laplace_transform_of_retarded_images_matches_periodic_resolvent(mass,sigma,speed,length,d):
    end=32/sigma
    fronts=sorted({0.,end}|{abs(d+j*length)/speed for j in range(-40,41) if 0 < abs(d+j*length)/speed < end})
    integral=sum(quad(lambda t: math.exp(-sigma*t)*periodic_retarded_green(mass,speed,length,t,d),a,b,
                      epsabs=2e-13,epsrel=1e-11)[0] for a,b in zip(fronts,fronts[1:]))
    exact=periodic_subgap_green(math.hypot(mass,sigma),0.,speed,length,d)
    tail=math.exp(-sigma*end)*((end/length+1/(2*speed))/sigma+1/(length*sigma**2))
    assert abs(integral-exact) < tail+1e-11


def test_truncated_point_source_can_fake_a_prefront_signal():
    mass,speed,length,time,d = .8,1.,4.,.3,1.
    assert periodic_retarded_green(mass,speed,length,time,d) == 0
    finite=truncated_retarded_green(mass,speed,length,time,d,cutoff=3)
    assert abs(finite) > .01
    assert truncated_retarded_green(mass,speed,length,-1.,d,cutoff=3) == 0


@pytest.mark.parametrize('kwargs', [dict(cutoff=-1,width=.1,circumference=1.),
    dict(cutoff=True,width=.1,circumference=1.),dict(cutoff=1,width=1.1,circumference=1.),
    dict(cutoff=1,width=-.1,circumference=1.)])
def test_invalid_concentration_domains_fail(kwargs):
    with pytest.raises(ValueError): interval_concentration_matrix(**kwargs)


def test_invalid_response_and_state_domains_fail():
    with pytest.raises(ValueError): periodic_subgap_green(1.,1.,1.,4.,.5)
    with pytest.raises(ValueError): periodic_subgap_green(1.,0.,0.,4.,.5)
    with pytest.raises(ValueError): periodic_subgap_green(1.,0.,1.,4.,float('nan'))
    with pytest.raises(ValueError): periodic_retarded_green(1.,1.,4.,float('inf'),.5)
    with pytest.raises(ValueError): contact_box_transfer(1.,0.,4.,0.,.3,.5)
    with pytest.raises(ValueError): interval_population_fraction(np.diag([1.,1.,-1.]),1,.2,1.)
    with pytest.raises(ValueError): interval_population_fraction(np.eye(3),1,.2,1.)
