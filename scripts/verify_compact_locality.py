#!/usr/bin/env python3
"""Check conditional localization, periodic response and exact causal fronts."""
from pathlib import Path
import argparse
import json
import math
import sys

import numpy as np
import scipy
from scipy.integrate import quad
import sympy as s

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'workspace-pydev'/'hyperspace_comm'))
from compact_locality import (interval_concentration_matrix,maximum_interval_concentration,
    periodic_subgap_green,periodic_subgap_contrast,resonant_contrast,
    periodic_retarded_green,truncated_retarded_green,contact_box_transfer)


def radar_echo_example():
    """A dimensionless monostatic impulse experiment before any circle wrap."""
    laplace, line_impedance, shunt_inductance = s.symbols(
        's Z0 Ls', positive=True)
    loaded_impedance = 1 / (1 / line_impedance + 1 / (laplace * shunt_inductance))
    reflection = (loaded_impedance - line_impedance) / (loaded_impedance + line_impedance)
    assert s.simplify(reflection + line_impedance / (2 * laplace * shunt_inductance + line_impedance)) == 0

    speed, circumference, distance, stiffness = 1., 12., 2., .25
    leg_time = distance / speed
    arrival = 2 * leg_time

    def echo(time):
        return math.expm1(-stiffness * max(time - arrival, 0.) / (2 * speed)) / (2 * speed)

    def green(time, separation):
        return periodic_retarded_green(0., speed, circumference, time, separation)

    def target_response(time):
        return green(time, distance) * math.exp(
            -stiffness * max(time - leg_time, 0.) / (2 * speed))

    samples = []
    for time in (3., 4., 5., 6., 8.):
        # The exact target response solves its causal Volterra equation.
        if time > arrival:
            scattering = -stiffness * quad(
                lambda t: green(time - t, distance) * target_response(t),
                leg_time, time - leg_time)[0]
        else:
            scattering = 0.
        assert abs(scattering - echo(time)) < 1e-13
        baseline = green(time, 0.)
        samples.append({"time": time, "target_absent": baseline,
                        "target_present": baseline + scattering, "difference": scattering})
    return {"status": "conditional_real_circle_impulse_example",
            "mass": 0., "speed": speed, "circumference": circumference,
            "shortest_arc": distance, "point_stiffness": stiffness,
            "first_return": arrival, "samples": samples}


def report():
    kappa,length,d,speed=s.symbols('kappa L d v',positive=True)
    green=s.cosh(kappa*(length/2-d))/(2*speed**2*kappa*s.sinh(kappa*length/2))
    assert s.simplify(speed**2*(kappa**2*green-s.diff(green,d,2)))==0
    jump=s.diff(green,d).subs(d,0)-s.diff(green,d).subs(d,length)
    assert s.simplify(jump+1/speed**2)==0
    assert s.simplify(s.integrate(green,(d,0,length))-1/(speed**2*kappa**2))==0
    regular=s.simplify(s.limit(green-1/(length*speed**2*kappa**2),kappa,0))
    bernoulli=length/(2*speed**2)*((d/length)**2-d/length+s.Rational(1,6))
    assert s.simplify(regular-bernoulli)==0
    contrast=s.simplify(regular.subs(d,0)-regular)
    assert s.simplify(contrast-d*(length-d)/(2*length*speed**2))==0
    # Interior light-cone PDE; jump/initial normalization also has a Laplace check.
    z,mu=s.symbols('z mu',positive=True)
    bessel=s.besselj(0,mu*s.sqrt(z))
    assert s.simplify(s.expand_func(4*z*s.diff(bessel,z,2)+4*s.diff(bessel,z)+mu**2*bessel))==0

    concentrations=[{'cutoff':n,'harmonic_count':2*n+1,'interval_fraction':.1,
        'maximum_coherent_fraction':maximum_interval_concentration(n,.1,1.),
        'any_diagonal_population_fraction':.1,
        'trace_upper_bound':min(1.,(2*n+1)*.1)} for n in (0,2,4,6,8)]
    matrix=interval_concentration_matrix(4,.1,1.)
    _,vectors=np.linalg.eigh(matrix)
    a=vectors[:,-1];n=np.arange(-4,5)
    direct=quad(lambda y: abs(np.sum(a*np.exp(2j*np.pi*n*y)))**2,-.05,.05,epsabs=1e-13)[0]
    gram_error=abs(direct-np.vdot(a,matrix@a).real)
    assert gram_error < 1e-13

    mass,sigma,v,ell,dist=1.,1.3,.8,3.,.7
    end=32/sigma
    fronts=sorted({0.,end}|{abs(dist+j*ell)/v for j in range(-40,41) if 0 < abs(dist+j*ell)/v < end})
    integral=sum(quad(lambda t: math.exp(-sigma*t)*periodic_retarded_green(mass,v,ell,t,dist),aa,bb,
                      epsabs=2e-13,epsrel=1e-11)[0] for aa,bb in zip(fronts,fronts[1:]))
    resolvent=float(periodic_subgap_green(math.hypot(mass,sigma),0.,v,ell,dist))
    tail=math.exp(-sigma*end)*((end/ell+1/(2*v))/sigma+1/(ell*sigma*sigma))
    laplace_error=abs(integral-resolvent)
    assert laplace_error < tail+1e-11

    resonance=[]
    for frequency in (0.,.9,.999999):
        local=float(periodic_subgap_green(1.,frequency,.7,3.,0.))
        remote=float(periodic_subgap_green(1.,frequency,.7,3.,.8))
        difference=float(periodic_subgap_contrast(1.,frequency,.7,3.,.8))
        resonance.append({'frequency':frequency,'local_kernel':local,'remote_kernel':remote,
            'remote_over_local':remote/local,'differential_kernel':difference})
    prefront={'mass':.8,'speed':1.,'circumference':4.,'time':.3,'separation':1.,'cutoff':3,
        'exact_retarded_response':periodic_retarded_green(.8,1.,4.,.3,1.),
        'truncated_response':truncated_retarded_green(.8,1.,4.,.3,1.,cutoff=3)}
    assert prefront['exact_retarded_response']==0 and abs(prefront['truncated_response'])>.01
    assert contact_box_transfer(1.,0.,4.,.4,.6,1.)==0
    limiting=float(resonant_contrast(.7,3.,.8))
    np.testing.assert_allclose(resonance[-1]['differential_kernel'],limiting,rtol=4e-6)
    return {'schema':'conditional-compact-locality/v1',
        'status':'DERIVED_FOR_FOURIER_LOCALIZATION_AND_DECLARED_REAL_CIRCLE_COMPARATOR',
        'exact':{
            'coordinate':'y is an assumed independent real circle, L=2*pi*R; not identified with imaginary-time psi',
            'concentration':'C_mn=(w/L)*exp(2*pi*i*(n-m)*center/L)*sinc((n-m)*w/L); |n|,|m|<=N',
            'localization_bound':'fraction<=lambda_max(C)<=min(1,(2*N+1)*w/L); diagonal populations give exactly w/L',
            'conditional_action':'S=int dt dy [(q_t^2-v^2*q_y^2-mu^2*q^2)/2+J*q]',
            'subgap_green':'cosh(kappa*(L/2-d))/(2*v^2*kappa*sinh(kappa*L/2)); kappa=sqrt(mu^2-omega^2)/v',
            'zero_gradient_transfer':'int f_B*f_A dy/(mu^2-omega^2); disjoint profiles give zero for local independent copies',
            'uniform_end_transfer':'1/[L*(mu^2-omega^2)] for L1-normalized profiles; no position addressing',
            'resonant_regular_part':str(regular),
            'resonant_differential_limit':str(contrast),
            'retarded_green':'sum_l theta(t-|d+l*L|/v)*J0(mu*sqrt(t^2-(d+l*L)^2/v^2))/(2*v)',
            'causal_front':'d_geodesic/v for points; at least distance between source/readout supports divided by v',
            'energy':'E=int dy (q_t^2+v^2*q_y^2+mu^2*q^2)/2; dE/dt=int dy J*q_t on periodic boundaries'},
        'controls':{'interval_gram_quadrature_absolute_error':gram_error,
            'retarded_laplace_absolute_error':laplace_error,'retarded_laplace_tail_bound':tail},
        'illustrative_dimensionless_concentrations':concentrations,
        'illustrative_dimensionless_resonance':resonance,
        'prefront_truncation_counterexample':prefront,
        'tools':{'numpy':np.__version__,'scipy':scipy.__version__,'sympy':s.__version__},
        'lean_status':'LEAN-PENDING: lean/lake unavailable; no compiled formal proof',
        'not_established':['a UBT compact gradient term surviving physical constraints or an independent real-circle interpretation of psi',
            'the physical Theta mode count, masses, compact speed or electromagnetic overlaps',
            'apparatus control of nonuniform compact profiles or interharmonic coherence',
            'a microscopic composite pole, an extension of the Sauter occupation model, multiverse communication or a laboratory range']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--radar-example',action='store_true',
                      help='print the dimensionless target-present and target-absent impulse traces')
    args=parser.parse_args()
    echo_example = radar_echo_example()
    if args.radar_example:
        print(json.dumps(echo_example, indent=2))
        return
    result=report()
    path=ROOT/'docs/results/compact-locality.json'
    if args.check:
        old=json.loads(path.read_text())
        for key in ('schema','status','exact','not_established'):
            assert old[key]==result[key],key
        for key in ('illustrative_dimensionless_concentrations','illustrative_dimensionless_resonance'):
            assert len(old[key])==len(result[key])
            for previous,current in zip(old[key],result[key]):
                assert previous.keys()==current.keys()
                np.testing.assert_allclose(list(previous.values()),list(current.values()),rtol=1e-9,atol=1e-14)
        assert old['prefront_truncation_counterexample'].keys()==result['prefront_truncation_counterexample'].keys()
        np.testing.assert_allclose(list(old['prefront_truncation_counterexample'].values()),
            list(result['prefront_truncation_counterexample'].values()),rtol=1e-10,atol=1e-14)
        print('Conditional localization, causal fronts and point-target echo checks agree.')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2)+'\n')
        print(path)


if __name__=='__main__':main()
