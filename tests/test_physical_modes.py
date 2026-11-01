"""Independent controls, not experimental verification of hyperspace waves."""
import numpy as np
import pytest
from scipy.linalg import expm
from biquaternion.biquaternion import Biquaternion
from biquaternion.theta_functions import (biquaternion_theta, jacobi_theta3_complex,
    biquaternion_theta_modular_transform, theta_quantization_condition)
from causal_modes import QuadraticModes, mass_shell_residual, damped_poles, damped_template, matched_filter
from transform.transform import mfft
from applications import FTLCommunicationLink


@pytest.mark.parametrize('seed', range(8))
def test_complex_algebra_against_faithful_matrices(seed):
    rng = np.random.default_rng(seed)
    a, b = [Biquaternion(*(rng.normal(size=4)+1j*rng.normal(size=4))) for _ in range(2)]
    np.testing.assert_allclose((a*b).to_matrix(), a.to_matrix()@b.to_matrix(), atol=5e-15)
    np.testing.assert_allclose(a.inverse().to_matrix(), np.linalg.inv(a.to_matrix()), atol=5e-15)
    np.testing.assert_allclose(a.exp().to_matrix(), expm(a.to_matrix()), rtol=2e-13, atol=2e-13)


def test_zero_divisor_and_nilpotent_are_not_positive_norm_cases():
    with pytest.raises(ValueError, match='zero divisor'):
        Biquaternion(1, 1j).inverse()
    q = Biquaternion(0, 1, 1j, 0)
    assert q.norm_squared() == 2
    np.testing.assert_array_equal((q*q).to_array(), 0)
    np.testing.assert_allclose(q.exp().to_matrix(), np.eye(2)+q.to_matrix())


def test_biquaternion_theta_against_independent_matrix_exponentials():
    z, tau = Biquaternion(.13+.1j, .1j, .04, .08), Biquaternion(1j, .1j, 0, 0)
    expected = sum((expm(1j*np.pi*n*n*tau.to_matrix()+2j*np.pi*n*z.to_matrix())
                    for n in range(-9,10)), start=np.zeros((2,2),dtype=complex))
    np.testing.assert_allclose(biquaternion_theta(z,tau,9).to_matrix(), expected, atol=2e-14)


def test_theta_modular_identity_and_convergence_gate():
    z, tau = Biquaternion(.13+.07j), Biquaternion(.2+.9j)
    zp, tp, prefactor = biquaternion_theta_modular_transform(z,tau)
    assert jacobi_theta3_complex(zp.w,tp.w) == pytest.approx(prefactor*jacobi_theta3_complex(z.w,tau.w), rel=1e-12)
    with pytest.raises(NotImplementedError):
        biquaternion_theta_modular_transform(Biquaternion(.1,.2),tau)
    with pytest.raises(ValueError, match='positive definite'):
        biquaternion_theta(z,Biquaternion(1j,2))
    assert not theta_quantization_condition([2*np.pi+1j,0,0],[1,1,1],tau)


def test_old_sqrt2_dispersion_drops_a_nonzero_imaginary_equation():
    residual = mass_shell_residual([1,1/np.sqrt(2),0,0],[1/np.sqrt(2),0,0,0])
    assert residual.real == pytest.approx(0,abs=1e-15)
    assert residual.imag == pytest.approx(np.sqrt(2))
    assert mass_shell_residual([2,np.sqrt(3),0,0],[0,0,0,0],1) == pytest.approx(0)


def test_coupled_physical_modes_and_group_slopes_have_causal_bound():
    # Noncommuting C,M: independent finite differences test HF slopes.
    k = np.array([[2.,.2],[.2,1.]])
    c = .25*k + np.array([[.1,.02],[.02,.08]])
    m = np.array([[2.,.3],[.3,1.]])
    modes = QuadraticModes(k,c,m)
    assert modes.is_causal(1)
    for wave_number in (.1,1.,10.):
        h = wave_number*1e-5
        expected = (modes.frequencies(wave_number+h)-modes.frequencies(wave_number-h))/(2*h)
        np.testing.assert_allclose(modes.group_velocities(wave_number),expected,rtol=2e-8)
        assert np.max(modes.group_velocities(wave_number)) <= np.max(modes.front_speeds())
    assert not QuadraticModes([[1]],[[2]],[[1]]).is_causal(1)


@pytest.mark.parametrize('k,c,m', [([[0]],[[1]],[[1]]),([[-1]],[[1]],[[1]]),
    ([[1]],[[-1]],[[1]]),([[1]],[[1]],[[-1]])])
def test_invalid_raw_hessians_are_rejected(k,c,m):
    with pytest.raises(ValueError):
        QuadraticModes(k,c,m)


def test_passive_damping_roots_and_ordinary_ringdown():
    for damping in (0,.4,3.):
        for pole in damped_poles(1,speed=1,gap=.5,damping=damping):
            assert -pole*pole-2j*damping*pole+1.25 == pytest.approx(0,abs=2e-15)
            assert pole.imag <= 1e-15
    t=np.arange(40)/100
    np.testing.assert_allclose(damped_template(t,2,3), np.exp(-3*t+4j*np.pi*t))


def test_one_template_white_noise_null_is_calibrated_and_scale_invariant():
    rng=np.random.default_rng(107)
    template=damped_template(np.arange(32)/100,3,2)
    signal=template+(.1+.2j)
    a=matched_filter(signal,template,.5)
    b=matched_filter(signal,template*100j,.5)
    assert a['normalized_power'] == pytest.approx(b['normalized_power'])
    noise=(rng.normal(size=(6000,32))+1j*rng.normal(size=(6000,32)))/np.sqrt(2)
    unit=template/np.linalg.norm(template)
    powers=abs(noise@unit.conj())**2
    assert np.mean(powers) == pytest.approx(1,abs=.04)
    assert np.mean(powers>-np.log(.05)) == pytest.approx(.05,abs=.012)
    assert a['single_template_noise_tail'] == pytest.approx(np.exp(-a['normalized_power']))


@pytest.mark.parametrize('size',[1,2,7,16])
@pytest.mark.parametrize('p',[0,-.02,.04])
def test_spiral_transform_matches_defining_finite_sum(size,p):
    signal=np.arange(size)+1j*np.linspace(0,1,size)
    expected=np.array([sum(signal[n]*np.exp(-2*np.pi*(p+1j)*k*n/size)
                           for n in range(size)) for k in range(size)])
    np.testing.assert_allclose(mfft(signal,p),expected,atol=3e-12,rtol=2e-12)


def test_unsafe_transform_and_implicit_ftl_predictions_are_rejected():
    with pytest.raises(ValueError,match='dynamic range'):
        mfft(np.zeros(1024),-1)
    link=FTLCommunicationLink()
    with pytest.raises(ValueError,match='explicit assumed'):
        link.estimate_transmission_time(100)
    with pytest.raises(NotImplementedError):
        link.calculate_group_velocity(1,1)
    assumed=FTLCommunicationLink(propagation_speed_m_s=299792458.)
    assert assumed.estimate_transmission_time(1e6)['speedup_factor'] == 1


def test_source_to_detector_response_and_passivity_controls():
    mode=QuadraticModes([[2]],[[.5]],[[3]])
    omega,k,gamma=1.1,.3,.2
    value=mode.response(k,omega,[1],[1],[[gamma]])
    assert value==pytest.approx(1/(3+.5*k*k-2*omega*omega-2j*omega*gamma))
    assert value.imag>0  # positive absorbed power for positive omega
    assert mode.response(k,omega,[0],[1],[[gamma]])==0
    assert mode.response(k,omega,[1],[0],[[gamma]])==0
    with pytest.raises(ValueError,match='positive semidefinite'):
        mode.response(k,omega,[1],[1],[[-1]])
