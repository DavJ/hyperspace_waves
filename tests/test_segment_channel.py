"""Conditional channel checks against analytic inverses, integrals and ODEs."""
import numpy as np
import pytest
from scipy.integrate import quad,solve_ivp
from segment_channel import (TwoSectorChannel,retarded_oscillator_green,
                              compact_operator_matrix,schur_reduction)


def test_transfer_and_local_self_energy_match_exact_two_by_two_inverse():
    c=TwoSectorChannel(1,1.3,.2,.1,.15)
    w=.8
    a,b=c.operator(w).diagonal()
    assert c.transfer(w,source_gain=2,detector_gain=3)==pytest.approx(6*c.eta/(a*b-c.eta**2))
    assert c.local_response(w,source_gain=1,detector_gain=1)==pytest.approx(1/(a-c.eta**2/b))
    assert c.transfer(w,source_gain=0,detector_gain=1)==0
    assert c.transfer(w,source_gain=1,detector_gain=0)==0


def test_decoupled_sectors_have_zero_transfer_for_arbitrary_nonzero_local_drives():
    c=TwoSectorChannel(1,1.3,0,.1,.15)
    for w in (0,.1,1,1.3,5):
        assert c.transfer(w,source_gain=2+1j,detector_gain=.3)==0
    # A positive link adds diagonal energy too; a naked off-diagonal term in
    # two massless modes would have an unphysical negative squared frequency.
    assert np.linalg.eigvalsh(TwoSectorChannel(0,0,.2).stiffness)[0]>=-1e-15


def test_symmetric_retarded_impulse_matches_independent_time_domain_equations():
    c=TwoSectorChannel(1.2,1.2,.3,.2,.2)
    times=np.linspace(0,20,501)
    def ode(t,y):
        return np.r_[y[2:],-c.stiffness@y[:2]-2*c.damping@y[2:]]
    out=solve_ivp(ode,(0,20),[0,0,1,0],t_eval=times,rtol=2e-11,atol=2e-12)
    np.testing.assert_allclose(c.symmetric_retarded_impulse(times),out.y[1],atol=1e-10)
    assert c.symmetric_retarded_impulse(np.array([-1,-.1,0])).tolist()==[0,0,0]


@pytest.mark.parametrize('w0,gamma',[(2,.2),(1,1),(.3,2),(0,1),(0,0)])
def test_single_channel_green_function_covers_all_damping_regimes(w0,gamma):
    times=np.linspace(0,5,101)
    out=solve_ivp(lambda t,y:[y[1],-2*gamma*y[1]-w0*w0*y[0]],(0,5),[0,1],
                  t_eval=times,rtol=1e-11,atol=1e-12)
    np.testing.assert_allclose(retarded_oscillator_green(times,w0,gamma),out.y[0],atol=3e-11)
    assert np.isfinite(retarded_oscillator_green([1e6],w0,gamma)).all()


def test_impulse_fourier_transform_agrees_with_frequency_response():
    c=TwoSectorChannel(1.2,1.2,.3,.2,.2)
    w=.9
    real=quad(lambda t:float(c.symmetric_retarded_impulse(t)),0,np.inf,
              weight='cos',wvar=w,epsabs=1e-11)[0]
    imag=quad(lambda t:float(c.symmetric_retarded_impulse(t)),0,np.inf,
              weight='sin',wvar=w,epsabs=1e-11)[0]
    assert real+1j*imag==pytest.approx(c.transfer(w,source_gain=1,detector_gain=1),abs=1e-10)


def test_matrix_fourier_selection_matches_independent_continuous_integrals():
    c0=np.array([[2,.1],[.1,1.]])
    c1=np.array([[.1,.03j],[.02,.05]])
    u0=np.array([[1,.2],[.2,2.]])
    u1=np.array([[.1j,.02],[.03j,.04]])
    C={0:c0,1:c1,-1:c1.conj().T}
    U={0:u0,1:u1,-1:u1.conj().T}
    indices=(-1,0,1,2)
    R=1.7
    matrix=compact_operator_matrix(indices,radius=R,gradient_coefficients=C,potential_coefficients=U)
    np.testing.assert_allclose(matrix,matrix.conj().T,atol=1e-15)
    for im,m in enumerate(indices):
        for jn,n in enumerate(indices):
            for a in range(2):
                for b in range(2):
                    def integrand(theta):
                        cy=sum(v[a,b]*np.exp(1j*r*theta) for r,v in C.items())
                        uy=sum(v[a,b]*np.exp(1j*r*theta) for r,v in U.items())
                        return np.exp(1j*(n-m)*theta)*(m*n/R**2*cy+uy)/(2*np.pi)
                    value=quad(lambda t:integrand(t).real,0,2*np.pi,epsabs=1e-12)[0]+1j*quad(lambda t:integrand(t).imag,0,2*np.pi,epsabs=1e-12)[0]
                    assert matrix[im*2+a,jn*2+b]==pytest.approx(value,abs=2e-14)


def test_translation_invariance_and_first_harmonic_selection_rules():
    modes=(137,138,139)
    identity={0:np.eye(1)}
    uniform=compact_operator_matrix(modes,radius=1,gradient_coefficients=identity,potential_coefficients={})
    np.testing.assert_array_equal(uniform,np.diag(np.array(modes)**2))
    perturbed=compact_operator_matrix(modes,radius=1,gradient_coefficients=identity,
                                     potential_coefficients={1:np.array([[.2]]),-1:np.array([[.2]])})
    assert perturbed[0,2]==0
    assert perturbed[0,1]==perturbed[1,2]==.2
    operator=perturbed-(100+1j)**2*np.eye(3)
    reduced=schur_reduction(operator,[0,2])
    assert reduced[0,1]==pytest.approx(-.2**2/operator[1,1])
    np.testing.assert_allclose(np.linalg.inv(reduced),np.linalg.inv(operator)[np.ix_([0,2],[0,2])])


@pytest.mark.parametrize('kwargs',[{'eta':-.1},{'damping_a':-.1},{'omega_a':float('inf')}])
def test_nonpassive_or_nonfinite_channel_parameters_are_rejected(kwargs):
    args={'omega_a':1,'omega_b':1,'eta':.1}|kwargs
    with pytest.raises(ValueError):
        TwoSectorChannel(**args)


def test_invalid_fourier_reality_and_singular_intermediate_are_not_hidden():
    with pytest.raises(ValueError,match='Fourier coefficients'):
        compact_operator_matrix([0,1],radius=1,gradient_coefficients={},potential_coefficients={1:[[1]]})
    with pytest.raises(ValueError,match='radius'):
        compact_operator_matrix([0,1],radius=0,gradient_coefficients={0:[[1]]},potential_coefficients={})
    with pytest.raises(np.linalg.LinAlgError):
        schur_reduction(np.diag([1,0,1]),[0,2])
