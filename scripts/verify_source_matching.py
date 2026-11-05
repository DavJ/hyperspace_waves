"""Exact source/mixing identities plus conditional scalar-loop matching checks."""
import argparse
import json
import platform
from pathlib import Path
import sys
import numpy as np
import scipy
import sympy as s
from scipy.integrate import quad
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace-pydev'/'hyperspace_comm'))
from source_matching import (NeutralPortalExample,complex_scalar_threshold_coefficients,
    complex_scalar_gauge_kinetic_shift,complex_scalar_photon_mass_squared)


def verify_geometric_coupling():
    """Compare direct Maxwell variation and the tetrad pullback in flat space."""
    metric = s.diag(-1, 1, 1, 1)
    ex, ey, ez, bx, by, bz = s.symbols("Ex Ey Ez Bx By Bz", real=True)
    field = s.Matrix([
        [0, ex, ey, ez],
        [-ex, 0, bz, -by],
        [-ey, -bz, 0, bx],
        [-ez, by, -bx, 0],
    ])
    raised = metric * field * metric
    invariant = s.trace(field.T * raised)
    stress = raised * metric * raised.T - metric * invariant / 4
    assert s.expand(s.trace(metric * stress)) == 0
    assert stress.subs({ex: 0, ey: 0, ez: 0, bx: 0, by: 0}) == bz**2 * s.diag(1, 1, 1, -1) / 2
    for row in range(4):
        for column in range(row, 4):
            variation = s.zeros(4)
            variation[row, column] = variation[column, row] = 1
            inverse_variation = -metric * variation * metric
            raised_variation = inverse_variation * field * metric + metric * field * inverse_variation
            density_variation = (
                -s.trace(metric * variation) * invariant / 8
                - s.trace(field.T * raised_variation) / 4
            )
            assert s.expand(density_variation - s.trace(stress * variation) / 2) == 0

    identity = s.eye(2)
    basis = [
        s.I * identity,
        -s.I * s.Matrix([[0, 1], [1, 0]]),
        -s.I * s.Matrix([[0, -s.I], [s.I, 0]]),
        -s.I * s.diag(1, -1),
    ]

    def sharp(matrix):
        return s.trace(matrix) * identity - matrix

    def pair(left, right):
        return (sharp(left) * right + sharp(right) * left) / 2

    jet = s.Matrix(4, 4, s.symbols("jet0:16"))
    varied_tetrad = [sum((jet[mu, a] * basis[a] for a in range(4)), s.zeros(2))
                     for mu in range(4)]
    symmetric_gradient = jet * metric + metric * jet.T
    for mu in range(4):
        for nu in range(4):
            variation = pair(varied_tetrad[mu], basis[nu]) + pair(basis[mu], varied_tetrad[nu])
            assert (variation - symmetric_gradient[mu, nu] * identity).applyfunc(s.expand) == s.zeros(2)
    phase = s.symbols("phase")
    assert (pair(phase * basis[0], phase * basis[0]) + phase**2 * identity).applyfunc(s.expand) == s.zeros(2)


def verify_radar_phase_reference():
    """Check the full-matrix marker and a retarded-response translation example."""
    r, b = s.symbols("r b", real=True)
    theta = s.diag(1, -1) + s.I * s.Matrix([[0, r], [r, b]])
    assert s.expand(theta.det()) == r**2 - 1 + s.I * b
    marker = s.expand(s.trace(theta.adjugate() * theta.H))
    assert marker == -2 - 2 * r**2

    # Keep two internal components per mode and solve with the shifted target.
    harmonics = np.repeat(np.arange(-2, 3), 2)
    free = (np.kron(np.eye(5), [[.3, .2], [.2, .7]])
            + np.diag(harmonics**2 - .05j))
    empty_response = np.linalg.inv(free)
    profiles = np.tile([[.3, .2], [.1, .4]], (5, 1))
    target_at_origin = profiles @ profiles.T
    origin_echo = np.linalg.inv(free + target_at_origin) - empty_response
    zero = np.flatnonzero(harmonics == 0)[0]
    one = np.flatnonzero(harmonics == 1)[0]
    uniform_reference = origin_echo[zero, zero]
    phase_reference = origin_echo[one, zero]
    assert abs(uniform_reference) > 1e-4 and abs(phase_reference) > 1e-4

    # Compare full inversions, including all repeated scattering in this matrix.
    samples = []
    for angle in (.4, 2.6, 4.1):
        shift = np.exp(-1j * harmonics * angle)
        phase = np.outer(shift, shift.conj())
        echo = np.linalg.inv(free + phase * target_at_origin) - empty_response
        np.testing.assert_allclose(echo, phase * origin_echo, atol=1e-13, rtol=1e-12)
        recovered = float((-np.angle(echo[one, zero] / phase_reference)) % (2 * np.pi))
        np.testing.assert_allclose(recovered, angle, atol=1e-13, rtol=0)
        samples.append({"target_angle": angle, "recovered_angle": recovered,
                        "uniform_echo_change": abs(echo[zero, zero] - uniform_reference)})
    return {
        "status": "conditional_compact_phase_reference_example",
        "uniform_echo_amplitude": abs(uniform_reference),
        "first_harmonic_echo_amplitude": abs(phase_reference),
        "samples": samples,
    }


def report():
    hA,hB,v,lam,delta,m0,yA,yB=s.symbols('hA hB v lam delta m0 yA yB',positive=True,real=True)
    V=lam*(hA*hA+hB*hB-v*v)**2/4+delta*(hA-hB)**2/2
    background={hA:v/s.sqrt(2),hB:v/s.sqrt(2)}
    assert all(s.simplify(s.diff(V,h).subs(background))==0 for h in (hA,hB))
    Hess=s.hessian(V,(hA,hB)).subs(background).applyfunc(s.simplify)
    expected=s.Matrix([[lam*v*v+delta,lam*v*v-delta],[lam*v*v-delta,lam*v*v+delta]])
    assert Hess==expected
    assert s.simplify(Hess.det()-4*lam*v*v*delta)==0
    mass=m0+yA*hA*hA+yB*hB*hB
    assert s.simplify(s.diff(mass,hA).subs(background)-s.sqrt(2)*v*yA)==0
    assert s.simplify(s.diff(mass,hB).subs(background)-s.sqrt(2)*v*yB)==0
    alpha,phi=s.symbols('alpha phi',real=True)
    assert s.simplify(s.diff(s.exp(s.I*alpha)*phi,alpha).subs(alpha,0)-s.I*phi)==0
    assert s.simplify(s.exp(s.I*alpha)*phi*s.exp(-s.I*alpha)-phi)==0
    A,x,y,e,q=s.symbols('A x y e q',real=True)
    seagull=e*e*q*q*A*A*(x*x+y*y)/2
    assert s.diff(seagull,hA,A,A)==0  # neutral h, no tree h*A^2 vertex
    # One complex scalar: coefficient a4 of the connection curvature is 1/12.
    prefactor=s.Rational(1,12)/(4*s.pi)**2
    assert s.simplify(prefactor-1/(192*s.pi**2))==0
    c,hdot,F2=s.symbols('c hdot F2',real=True)
    Zd=-4*c*hdot
    assert s.simplify(c*hdot*F2+Zd*F2/4)==0  # scalar + EM energy exchange
    z=s.symbols('z',real=True)
    B0,B1=s.symbols('B0 B1',real=True)
    field=2*(B0+B1*s.cos(z))**2
    harmonic1=s.integrate(field*s.cos(z),(z,0,2*s.pi))/s.pi
    harmonic2=s.integrate(field*s.cos(2*z),(z,0,2*s.pi))/s.pi
    assert s.simplify(harmonic1-4*B0*B1)==0
    assert s.simplify(harmonic2-B1**2)==0
    model=NeutralPortalExample(1,.3,.6,100,.02,.01,.3,1)
    mass2=model.heavy_mass_squared(model.background)
    kappa=2*model.background*np.array([model.y_a_squared,model.y_b_squared])
    c=model.portal_coefficients
    proper=quad(lambda t:np.exp(-mass2*t),0,np.inf,epsabs=1e-13)[0]
    from_proper=model.e**2*model.charge**2*kappa*proper/(192*np.pi**2)
    shape=quad(lambda xx:(1-2*xx)**2,0,1,epsabs=1e-13)[0]
    from_parameter=model.e**2*model.charge**2*kappa*shape/(64*np.pi**2*mass2)
    h=1e-3
    massp=model.heavy_mass_squared(model.background+np.array([h,0]))
    massm=model.heavy_mass_squared(model.background-np.array([h,0]))
    dz=(complex_scalar_gauge_kinetic_shift([[massp]],[[mass2]],[[1]],model.e)
        -complex_scalar_gauge_kinetic_shift([[massm]],[[mass2]],[[1]],model.e))/(2*h)
    response=model.delta_z_per_pump_f_squared(.4,np.diag([.05,.08]))
    return {
        'schema':'conditional-source-detector-matching/v1',
        'status':'SCALAR_QED_MATCHING_DERIVED_CONDITIONALLY_UBT_EMBEDDING_OPEN',
        'ubt_snapshot':'9b24d4b41cb486c01f2bfc4002588c85f8e691cd',
        'source_paths':['canonical/ACTION.en.md','canonical/THEORY/canonical/canonical_action.tex',
                        'canonical/interactions/qed.tex','canonical/qed_phi_const/step1_u1_protection.tex',
                        'research_tracks/action_selection/full_theta_reallinear_graded_jet_2026_10.en.md'],
        'primary_background_sources':[
            'https://arxiv.org/abs/hep-th/0306138 (eqs. 4.25, 4.28)',
            'https://arxiv.org/abs/hep-ph/9505225 (low-energy mass-derivative method)'],
        'versions':{'python':platform.python_version(),'numpy':np.__version__,
                    'scipy':scipy.__version__,'sympy':s.__version__},
        'exact_checks':{
            'neutral_hessian':'[[lambda*v^2+delta,lambda*v^2-delta],[lambda*v^2-delta,lambda*v^2+delta]]',
            'eta_in_specified_basis':'delta-lambda*v^2',
            'scalar_loop_coefficient':'c_a=e^2/(192*pi^2)*Tr[Q^2*(M^2)^(-1)*partial_a(M^2)]',
            'photon_mass_for_constant_complex_scalar_v':'2*e^2*norm(Q*v)^2',
            'neutral_minimal_tree_h_A_A_vertex':'0',
            'source':'(Box+H)h=c*F^2',
            'maxwell_readout':'partial_mu f^mu_nu=4*partial_mu[(c.h)*F_probe^mu_nu]',
            'energy_exchange_sum':'c.hdot*F^2+(Zdot/4)*F^2=0',
            'magnetic_pump_omega_coefficient':'4*Bdc*Bac',
            'magnetic_pump_two_omega_coefficient':'Bac^2'},
        'illustrative_units':'natural, dimensionless reference scale; NO UBT/hardware parameter values',
        'illustrative_parameters':{'v':1,'lambda':.3,'delta':.6,'m0_squared':100,'yA_squared':.02,'yB_squared':.01,'e':.3,'charge':1},
        'illustrative_neutral_squared_masses':np.linalg.eigvalsh(model.neutral_hessian).tolist(),
        'illustrative_heavy_mass_squared':mass2,
        'illustrative_derived_eta':model.derived_eta,
        'illustrative_loop_portals':c.tolist(),
        'proper_time_absolute_error':float(np.max(abs(c-from_proper))),
        'feynman_parameter_absolute_error':float(np.max(abs(c-from_parameter))),
        'gauge_kinetic_derivative_relative_error':float(abs(-dz/(4*c[0])-1)),
        'illustrative_delta_Z_over_F_squared':{'real':response.real,'imaginary':response.imag},
        'legacy_neutrality_audit':'Commutation proves adjoint neutrality, not neutrality under the declared LEFT phase; an explicit physical charge representation is required.',
        'lean_status':'LEAN-PENDING: finite CAS identities and numerical integral controls only',
        'not_established':['UBT physical neutral/charged representation and constrained spectrum',
                           'embedding of this effective polynomial in a finalized single-Theta action',
                           'nondegenerate UBT on-shell geometry for the comparator',
                           'actual neighboring multiverse sectors',
                           'quantum-corrected vacuum, full loop propagator and EFT counterterm selection',
                           'finite-momentum/material/lead response or pickup volts',
                           'operating frequency, emitted power, range, hardware or received message']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--radar-reference',action='store_true',
                      help='print the calibrated compact-phase example')
    args=parser.parse_args()
    if args.radar_reference:
        print(json.dumps(verify_radar_phase_reference(), indent=2))
        return
    verify_geometric_coupling()
    result=report()
    path=Path(__file__).resolve().parents[1]/'docs/results/source-matching.json'
    if args.check:
        old=json.loads(path.read_text())
        for key in ('status','ubt_snapshot','source_paths','exact_checks','not_established','legacy_neutrality_audit'):
            assert old[key]==result[key],key
        np.testing.assert_allclose(old['illustrative_loop_portals'],result['illustrative_loop_portals'],rtol=1e-11)
        assert result['proper_time_absolute_error']<1e-16
        assert result['feynman_parameter_absolute_error']<1e-16
        assert result['gauge_kinetic_derivative_relative_error']<1e-5
        verify_radar_phase_reference()
        print('Maxwell variation, compact phase selection and restricted scalar-loop matching checks agree.')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2)+'\n')
        print(path)


if __name__=='__main__':
    main()
