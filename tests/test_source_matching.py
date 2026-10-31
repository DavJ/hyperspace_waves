"""Independent controls for a SPECIFIED scalar-loop comparator, not UBT hardware."""
import numpy as np
import pytest
from scipy.integrate import quad
from source_matching import (complex_scalar_photon_mass_squared,
    complex_scalar_threshold_coefficients,complex_scalar_gauge_kinetic_shift,
    NeutralPortalExample)


def example():
    return NeutralPortalExample(v=1,quartic=.3,locking_mass_squared=.6,
        heavy_bare_mass_squared=100,y_a_squared=.02,y_b_squared=.01,e=.3,charge=1)


def test_real_constant_is_charged_under_left_phase_but_kernel_background_is_neutral():
    assert complex_scalar_photon_mass_squared([2],[[1]],.3)==pytest.approx(.72)
    assert complex_scalar_photon_mass_squared([2,3,0],np.diag([0,0,1]),.3)==0
    assert complex_scalar_photon_mass_squared([2,3,1j],np.diag([0,0,1]),.3)==pytest.approx(.18)


def test_scalar_threshold_matches_independent_proper_time_and_feynman_parameter_integrals():
    mass2,kappa,e,q=3.7,.8,.4,-2
    c=complex_scalar_threshold_coefficients([[mass2]],[[[kappa]]],[[q]],e)[0]
    proper=quad(lambda s:np.exp(-mass2*s),0,np.inf,epsabs=1e-13)[0]
    shape=quad(lambda x:(1-2*x)**2,0,1,epsabs=1e-13)[0]
    assert c==pytest.approx(e*e*q*q*kappa/(192*np.pi**2)*proper,rel=1e-11)
    assert c==pytest.approx(e*e*q*q*kappa/(64*np.pi**2*mass2)*shape,rel=1e-13)


def test_matrix_threshold_is_charged_basis_invariant_and_matches_log_derivative():
    rng=np.random.default_rng(64)
    u,_=np.linalg.qr(rng.normal(size=(3,3))+1j*rng.normal(size=(3,3)))
    mass=np.diag([2.,3.,5.])
    # Equal-charge block permits a mass derivative that does NOT commute with mass.
    derivative=np.array([[.3,.2j,0],[-.2j,.4,0],[0,0,.7]])
    charge=np.diag([1.,1.,-2.])
    coefficients=complex_scalar_threshold_coefficients(mass,[derivative],charge,.3)
    rotated=complex_scalar_threshold_coefficients(u@mass@u.conj().T,[u@derivative@u.conj().T],u@charge@u.conj().T,.3)
    np.testing.assert_allclose(rotated,coefficients,rtol=2e-13)
    h=1e-4
    dz=(complex_scalar_gauge_kinetic_shift(mass+h*derivative,mass,charge,.3)
        -complex_scalar_gauge_kinetic_shift(mass-h*derivative,mass,charge,.3))/(2*h)
    assert coefficients[0]==pytest.approx(-dz/4,rel=3e-8)


def test_neutral_or_mass_independent_heavy_modes_have_no_scalar_portal():
    assert complex_scalar_threshold_coefficients([[3]],[[[1]]],[[0]],.3)[0]==0
    assert complex_scalar_threshold_coefficients([[3]],[[[0]]],[[1]],.3)[0]==0
    opposite=complex_scalar_threshold_coefficients(np.diag([3,3]),[np.diag([1,-1])],np.diag([1,-1]),.3)
    assert opposite[0]==pytest.approx(0,abs=1e-18)
    same=complex_scalar_threshold_coefficients(np.diag([3,3]),[np.eye(2)],np.diag([1,-1]),.3)
    single=complex_scalar_threshold_coefficients([[3]],[[[1]]],[[1]],.3)
    np.testing.assert_allclose(same,2*single)


def test_neutral_potential_hessian_matches_finite_differences_of_one_potential():
    model=example()
    def potential(h):
        return model.quartic/4*(np.dot(h,h)-model.v**2)**2+model.locking_mass_squared/2*(h[0]-h[1])**2
    p=model.background
    step=1e-4
    Hess=np.zeros((2,2))
    for i in range(2):
        ei=np.eye(2)[i]*step
        Hess[i,i]=(potential(p+ei)-2*potential(p)+potential(p-ei))/step**2
    a,b=np.eye(2)*step
    Hess[0,1]=Hess[1,0]=(potential(p+a+b)-potential(p+a-b)-potential(p-a+b)+potential(p-a-b))/(4*step**2)
    np.testing.assert_allclose(Hess,model.neutral_hessian,atol=2e-8)
    np.testing.assert_allclose(np.linalg.eigvalsh(model.neutral_hessian),[2*model.quartic*model.v**2,2*model.locking_mass_squared])
    assert model.derived_eta==-model.neutral_hessian[0,1]


def test_one_potential_supplies_mass_derivatives_source_and_readout_response():
    model=example()
    mass=model.heavy_mass_squared(model.background)
    step=1e-3
    derivatives=[]
    for direction in np.eye(2):
        derivatives.append((model.heavy_mass_squared(model.background+step*direction)
                           -model.heavy_mass_squared(model.background-step*direction))/(2*step))
    expected=model.e**2*model.charge**2*np.array(derivatives)/(192*np.pi**2*mass)
    np.testing.assert_allclose(model.portal_coefficients,expected,rtol=1e-8)
    omega=.4
    damping=np.diag([.05,.08])
    c=model.portal_coefficients
    oracle=-4*c@np.linalg.solve(model.neutral_hessian-omega**2*np.eye(2)-2j*omega*damping,c)
    assert model.delta_z_per_pump_f_squared(omega,damping)==pytest.approx(oracle,rel=1e-12,abs=1e-28)


@pytest.mark.parametrize('mass,derivative,charge',[
    ([[-1]],[[[1]]],[[1]]),
    ([[0]],[[[1]]],[[1]]),
    ([[2,1],[1,2]],[[[1,0],[0,1]]],[[1,0],[0,2]]),
    ([[2,0],[0,2]],[[[0,1],[1,0]]],[[1,0],[0,2]]),
])
def test_invalid_heavy_spectrum_or_charged_derivative_is_rejected(mass,derivative,charge):
    with pytest.raises(ValueError):
        complex_scalar_threshold_coefficients(mass,derivative,charge,.3)


def test_plane_wave_has_zero_source_but_near_field_has_nonzero_quadratic_source():
    from source_matching import electromagnetic_f_squared
    assert electromagnetic_f_squared([2,0,0],[0,2,0])==0
    assert electromagnetic_f_squared([0,0,0],[0,0,2])==8
    phase=np.arange(4096)*2*np.pi/4096
    values=np.array([electromagnetic_f_squared([0,0,0],[0,0,np.cos(t)]) for t in phase])
    assert np.mean(values)==pytest.approx(1)
    assert 2*np.mean(values*np.cos(phase))==pytest.approx(0,abs=1e-15)
    assert 2*np.mean(values*np.cos(2*phase))==pytest.approx(1)
