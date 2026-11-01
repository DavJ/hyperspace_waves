"""Exact normalization and independent integral controls; not a UBT spectrum."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import scipy
import sympy as s
from scipy.integrate import quad
from scipy.special import k1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'workspace-pydev'/'hyperspace_comm'))
from composite_response import (subtracted_density_susceptibility, density_static_kernel,
    density_evanescent_kernel, mass_source_em_coefficient, photon_pair_static_asymptotic)


def report():
    x, r, mass, e, q, J = s.symbols('x r mass e q J', positive=True, real=True)
    # Once-subtracted bubble coefficient, independently of Bessel evaluation.
    slope = s.integrate(x*(1-x), (x,0,1))/(16*s.pi**2*mass**2)
    assert s.simplify(slope-1/(96*s.pi**2*mass**2)) == 0
    loop_log = e**2*q**2*s.log((mass**2-J)/mass**2)/(192*s.pi**2)
    source_coeff = s.diff(loop_log,J).subs(J,0)
    assert source_coeff == -e**2*q**2/(192*s.pi**2*mass**2)
    # Derive all Maxwell F-F contractions from D_mn=delta_mn/(4*pi^2*x^2),
    # rather than importing the claimed F^2-F^2 normalization.
    coords=s.symbols('x0:4',real=True)
    prop=1/(4*s.pi**2*sum(z*z for z in coords))
    at={coords[0]:r,**{z:0 for z in coords[1:]}}
    h=s.hessian(prop,coords).subs(at)
    d=lambda a,b:s.Integer(a==b)
    total=0
    for a in range(4):
        for b in range(4):
            for c in range(4):
                for f in range(4):
                    tensor=-d(b,f)*h[a,c]+d(b,c)*h[a,f]+d(a,f)*h[b,c]-d(a,c)*h[b,f]
                    total += 2*tensor**2
    assert s.simplify(total-48/(s.pi**4*r**8)) == 0
    static_maxwell=48/s.pi**4*s.sqrt(s.pi)*s.gamma(s.Rational(7,2))/s.gamma(4)/r**7
    assert s.simplify(static_maxwell-15/(s.pi**3*r**7)) == 0
    # KL density C*s^2 reproduces the same static Maxwell normalization.
    photon_spectral_static=s.Rational(1,4)/s.pi**2 * 2*s.factorial(5)/(4*s.pi*r**7)
    assert s.simplify(photon_spectral_static-static_maxwell) == 0
    # Independent physical two-photon polarizations in the centre-of-mass frame.
    energy=s.symbols('energy',positive=True)
    eta=s.diag(1,-1,-1,-1)
    k_a=s.Matrix([energy,0,0,energy]);k_b=s.Matrix([energy,0,0,-energy])
    polarizations=(s.Matrix([0,1,0,0]),s.Matrix([0,0,1,0]))
    dot=lambda a,b:(a.T*eta*b)[0]
    amplitude2=sum((4*(dot(k_a,k_b)*dot(a,b)-dot(k_a,b)*dot(k_b,a)))**2
                   for a in polarizations for b in polarizations)
    assert s.simplify(amplitude2-8*(4*energy**2)**2) == 0
    # Identical-state factor 1/2, integrated phase space 1/(8*pi), KL 1/(2*pi).
    assert s.simplify(8*s.Rational(1,2)/(8*s.pi)/(2*s.pi)-1/(4*s.pi**2)) == 0
    free_threshold=1/(32*s.pi**3*mass*r**4)
    fixed_kappa=s.symbols('kappa',positive=True)
    fixed_pref=(2*fixed_kappa)**s.Rational(3,2)/(256*s.pi**s.Rational(5,2)*mass)
    assert s.simplify(fixed_pref.subs(fixed_kappa,2*mass)-s.sqrt(mass)/(32*s.pi**s.Rational(5,2))) == 0

    bubble_error=0.
    for z in (-8.,-.1,.3,2.,3.9,4.):
        u=z/4
        dispersion=quad(lambda t:2*u*t*t/(1-u+u*t*t),0,1)[0]/(16*np.pi**2)
        bubble_error=max(bubble_error,abs(subtracted_density_susceptibility(z,1)-dispersion))
    static_error=0.
    for mu,rr in ((1.,.3),(1.,1.),(.4,3.),(2.,2.)):
        def time_integrand(t):
            distance=np.hypot(t,rr)
            return (mu*k1(mu*distance)/(4*np.pi**2*distance))**2
        direct=2*quad(time_integrand,0,np.inf,epsabs=1e-14,epsrel=1e-10)[0]
        exact=density_static_kernel(mu,rr)
        static_error=max(static_error,abs(direct/exact-1),abs(density_evanescent_kernel(mu,0,rr)/exact-1))
    samples=[{'mass':1.,'e':.3,'charge':1.,'radius':rr,
              'free_static':density_static_kernel(1.,rr),
              'photon_pair_static_asymptotic':photon_pair_static_asymptotic(1.,.3,1.,rr)}
             for rr in (10.,20.,40.)]
    assert bubble_error<1e-11 and static_error<1e-7
    assert samples[1]['photon_pair_static_asymptotic']>samples[1]['free_static']
    return {
        'schema':'conditional-composite-response/v1',
        'status':'DERIVED_FOR_DECLARED_GAUSSIAN_AND_SCALAR_QED_COMPARATORS',
        'ubt_context':'https://github.com/UBT-Institute/unified-biquaternion-theory/pull/825',
        'exact':{'wick':'2 Tr(B G B G^T), no selected component count',
                 'bubble_slope_at_zero':str(slope),
                 'static_free_density':'m K1(2mr)/(16 pi^3 r^2)',
                 'positive_mass_pair_threshold_asymptotic':str(free_threshold),
                 'maxwell_euclidean_F2_correlator':str(s.simplify(total)),
                 'mass_source_F2_coefficient':str(source_coeff),
                 'static_maxwell_F2_correlator':str(s.simplify(static_maxwell)),
                 'photon_tail':'15 c_O^2/(pi^3 r^7), leading order e^4'},
        'controls':{'maximum_bubble_dispersion_absolute_error':bubble_error,
                    'maximum_static_euclidean_spectral_relative_error':static_error},
        'illustrative_dimensionless_samples':samples,
        'tools':{'numpy':np.__version__,'scipy':scipy.__version__,'sympy':s.__version__},
        'lean_status':'LEAN-PENDING: lean/lake unavailable; no compiled formal proof',
        'not_established':['a UBT physical-mode projector, masses or positive constrained Hessian',
            'a stable neutral bound-state pole or the full loop-corrected response',
            'a nonzero-psi-harmonic coupling to ordinary zero-harmonic photons',
            'finite-frequency matching near the massive pair threshold',
            'adjacent universes, transmitter power, coil voltage or communication range'],
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    result=report()
    path=ROOT/'docs/results/composite-response.json'
    if args.check:
        old=json.loads(path.read_text())
        for key in ('schema','status','ubt_context','exact','not_established'):
            assert old[key]==result[key],key
        for previous,current in zip(old['illustrative_dimensionless_samples'],result['illustrative_dimensionless_samples']):
            assert previous.keys()==current.keys()
            np.testing.assert_allclose(list(previous.values()),list(current.values()),rtol=1e-10,atol=0)
        assert len(old['illustrative_dimensionless_samples'])==len(result['illustrative_dimensionless_samples'])
        print('Conditional composite, dispersion and ordinary photon-pair tail controls agree.')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2)+'\n')
        print(path)


if __name__=='__main__':main()
