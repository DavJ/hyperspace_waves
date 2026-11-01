#!/usr/bin/env python3
"""Reproduce scalar pair pumping, KG/gamma checks and a state-induced readout."""
from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np
import scipy
from scipy.special import loggamma
import sympy as s

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'workspace-pydev'/'hyperspace_comm'))
from occupation_channel import (sauter_pair_number, sauter_bogoliubov,
    collisionless_pair_drude_tensor, pair_energy_density,
    single_arm_click_probability)


def gamma_pair_number(mass,tau,field,charge,pz,pt):
    """Independent hypergeometric connection formula, not cosh/sinh Eq.68."""
    k = charge*field*tau*tau
    wi = tau*math.hypot(mass,pt,pz-charge*field*tau)
    wo = tau*math.hypot(mass,pt,pz+charge*field*tau)
    a,b = -1j*wi/2,1j*wo/2
    lam = np.lib.scimath.sqrt(k*k-.25)
    aa,bb,c = a+b+.5+1j*lam,a+b+.5-1j*lam,1+2*a
    log_beta = loggamma(c)+loggamma(2*b)-loggamma(aa)-loggamma(bb)
    return wo/wi*math.exp(2*log_beta.real)


def report():
    y,a,b,k = s.symbols('y a b k')
    log_derivative = a/y-b/(1-y)
    omega2 = -4*a*a*(1-y)-4*b*b*y-4*k*k*y*(1-y)
    v_coefficient = y*(1-y)*(s.diff(log_derivative,y)+log_derivative**2)+(1-2*y)*log_derivative+omega2/(4*y*(1-y))
    assert s.simplify(v_coefficient+(a+b)*(a+b+1)+k*k) == 0
    first_derivative = 2*y*(1-y)*log_derivative+1-2*y
    assert s.simplify(first_derivative-(1+2*a-(2+2*a+2*b)*y)) == 0

    # Derive the kinetic readout from the same charged-particle Hamiltonian.
    mom = s.Matrix(s.symbols('p_x p_y p_z',real=True))
    probe = s.Matrix(s.symbols('A_x A_y A_z',real=True))
    mass,charge = s.symbols('mu g',positive=True)
    energy = s.sqrt(mass**2+(mom-charge*probe).dot(mom-charge*probe))
    at_zero = {z:0 for z in probe}
    hessian = s.hessian(energy,probe).subs(at_zero)
    e0 = energy.subs(at_zero)
    assert all(s.simplify(x)==0 for x in hessian-charge**2*(s.eye(3)/e0-mom*mom.T/e0**3))

    # Exact squeezed-pair probability generating function and detector thinning.
    n,z,eta = s.symbols('N z eta',nonnegative=True)
    pgf = 1/(1+n*(1-z))
    mean = s.diff(pgf,z).subs(z,1)
    variance = s.diff(pgf,z,2).subs(z,1)+mean-mean**2
    click = s.factor(1-pgf.subs(z,1-eta))
    assert s.simplify(mean-n)==0 and s.simplify(variance-n*(1+n))==0
    assert s.simplify(click-eta*n/(1+eta*n))==0

    samples = []
    error,wronskian = 0.,0.
    for mass_value,tau,field,charge_value,pz,pt in (
            (1.,1.,.2,1.,.7,.2),(1.,1.,.5,1.,0.,0.),
            (1.,1.,.8,1.,.4,.3),(.6,1.4,1.2,-.8,-.9,.4)):
        args=(mass_value,tau,field,charge_value,pz,pt)
        exact = sauter_pair_number(*args)
        gamma = gamma_pair_number(*args)
        ode = sauter_bogoliubov(*args)
        error = max(error,abs(gamma/exact-1),abs(ode.mean_pairs/exact-1))
        wronskian = max(wronskian,abs(ode.wronskian_defect))
        # One quadrature cell only: this is explicitly NOT an integrated density.
        out_momentum = np.array([[pt,0.,pz+charge_value*field*tau]])
        tensor = collisionless_pair_drude_tensor(mass_value,charge_value,out_momentum,[exact],[1.])
        samples.append({'mass':mass_value,'duration':tau,'field':field,'charge':charge_value,
            'p_parallel':pz,'p_perp':pt,'mean_pairs_per_mode_pair':exact,
            'ideal_single_arm_click_probability':single_arm_click_probability(exact),
            'pair_energy_per_unit_phase_space_weight':pair_energy_density(mass_value,out_momentum,[exact],[1.]),
            'readout_tensor_trace_per_unit_phase_space_weight':float(np.trace(tensor))})
    assert error < 5e-7 and wronskian < 2e-9
    return {'schema':'conditional-occupation-channel/v1',
        'status':'DERIVED_FOR_EXTERNAL_PULSE_SCALAR_QED_AND_DEPHASED_KINETIC_READOUT',
        'exact':{
            'pulse':'A_z=C-E*T*tanh(t/T); E_z=E*sech(t/T)^2',
            'mode_equation':'d^2 u/dt^2 + [mu^2+p_perp^2+(p_z-g*A_z)^2] u=0',
            'hypergeometric':'a=-i*T*omega_in/2; b=i*T*omega_out/2; A,B=a+b+1/2 +/- i*sqrt((g*E*T^2)^2-1/4); c=1+2*a',
            'pair_number':'[cosh(2*pi*lambda)+cosh(pi*T*(omega_out-omega_in))]/[2*sinh(pi*T*omega_in)*sinh(pi*T*omega_out)]',
            'normalization':'|alpha|^2-|beta|^2=1; N=|beta|^2; no a posteriori normalization',
            'pair_statistics':'P(k)=N^k/(1+N)^(k+1); Var(k)=N*(1+N); Cov(N_a,N_b)=N*(1+N)',
            'one_arm_click':str(click),
            'readout':'D_ij=2*g^2*int d^3k/(2*pi)^3 f(k)*(delta_ij/E_k-k_i*k_j/E_k^3); delta j=-D delta A',
            'pump_energy':'epsilon_out=2*int d^3k/(2*pi)^3 E_k*f(k); supplied by external pump',
            'fiber_rule':'particle and antiparticle have opposite psi index; all occupations and pair creation have total index zero'},
        'controls':{'maximum_gamma_or_KG_relative_error':error,'maximum_KG_wronskian_defect':wronskian},
        'illustrative_dimensionless_samples':samples,
        'sample_normalization':'One specified momentum pair or one cell per unit phase-space weight, not a full three-momentum integral or a hardware prediction.',
        'source':'Kim, Lee and Yoon, arXiv:0807.2696, scalar Eqs.53-68; constant-offset/canonical-momentum convention transformed explicitly.',
        'tools':{'numpy':np.__version__,'scipy':scipy.__version__,'sympy':s.__version__},
        'lean_status':'LEAN-PENDING: lean/lake unavailable; no compiled formal proof',
        'not_established':['a UBT constrained positive-energy charged spectrum or masses/charges',
            'a spatially or fiber-localized Maxwell-consistent pump and its source overlap',
            'near-pulse coherent readout, collisions, depletion, full momentum-integrated density or a detector noise budget',
            'adjacent multiverse segments, psi-addressed receivers, faster signal fronts or a communication range']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    result=report()
    path=ROOT/'docs/results/occupation-channel.json'
    if args.check:
        old=json.loads(path.read_text())
        for key in ('schema','status','exact','sample_normalization','source','not_established'):
            assert old[key]==result[key],key
        assert len(old['illustrative_dimensionless_samples'])==len(result['illustrative_dimensionless_samples'])
        for previous,current in zip(old['illustrative_dimensionless_samples'],result['illustrative_dimensionless_samples']):
            assert previous.keys()==current.keys()
            np.testing.assert_allclose(list(previous.values()),list(current.values()),rtol=1e-9,atol=0)
        print('Scalar pulse, independent KG/gamma evolution and conditional occupation readout agree.')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2)+'\n')
        print(path)


if __name__=='__main__':main()
