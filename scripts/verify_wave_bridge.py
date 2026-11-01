"""Reproducible exact/numerical checks. No experimental or Lean claim."""
import argparse
import json
from pathlib import Path
import sys
import platform
import numpy as np
import scipy
import sympy as sp
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace-pydev'/'hyperspace_comm'))
from causal_modes import QuadraticModes, mass_shell_residual
from biquaternion.biquaternion import Biquaternion
from scipy.linalg import expm


def report():
    omega,k,mu,a=sp.symbols('omega k mu a', real=True)
    square=sp.expand((omega+sp.I*a*omega)**2-k*k-mu*mu)
    assert sp.simplify(sp.im(square)-2*a*omega**2)==0
    t,f,s1,s2=sp.symbols('t f s1 s2', real=True)
    pulse=sp.exp(2*sp.pi*f*(s1+sp.I*s2)*t)
    assert sp.simplify(sp.diff(pulse,t)-2*sp.pi*f*(s1+sp.I*s2)*pulse)==0
    # Damped scalar energy identity: multiplying by q_t gives loss -2 gamma q_t^2.
    qt,qx,qxx,mu2,gamma,v=sp.symbols('qt qx qxx mu2 gamma v',real=True)
    q=sp.symbols('q',real=True)
    qtt=v*v*qxx-mu2*q-2*gamma*qt
    assert sp.expand(qt*qtt+mu2*q*qt-v*v*qt*qxx)==-2*gamma*qt**2
    modes=QuadraticModes([[2,.2],[.2,1]],[[.6,.06],[.06,.35]],[[2,.3],[.3,1]])
    residual=mass_shell_residual([1,1/np.sqrt(2),0,0],[1/np.sqrt(2),0,0,0])
    bq=Biquaternion(.3+.2j,.2+.1j,-.1+.3j,.1)
    matrix_error=float(np.max(abs(bq.exp().to_matrix()-expm(bq.to_matrix()))))
    return {
        'schema':'hyperspace-wave-bridge/v1','status':'CONDITIONAL_EFFECTIVE_MODEL_ONLY',
        'source_snapshots':{
            'hyperspace_waves':'413e34412b4cae2d6481f8da564ef74a4ea1949d',
            'magnetic_field_antihemholz':'9b35e3facf93a763b5dc921f908e706cc7c2cb0e',
            'ubt':'386b6dbed7d35ea63c8fd42fd88d57ac2f9cdaa3'},
        'versions':{'python':platform.python_version(),'numpy':np.__version__,
                    'scipy':scipy.__version__,'sympy':sp.__version__},
        'exact_sympy_checks':{'imaginary_mass_shell':'2*a*omega**2',
                             'generator_exponent':'2*pi*f*(s1+i*s2)*t',
                             'damped_energy_loss':'-2*gamma*q_t**2'},
        'old_balanced_branch_dimensionless_residual':{'real':residual.real,'imaginary':residual.imag},
        'matrix_exponential_absolute_error':matrix_error,
        'illustrative_nondimensional_front_speeds':modes.front_speeds().tolist(),
        'illustrative_cone_bound_1':modes.is_causal(1),
        'lean_status':'LEAN-PENDING: no Lean executable or formal PDE/constraint proof in this repository',
        'not_tested':['final UBT action','physical constraint reduction','physical Theta mode count',
                      'microscopic source/detector coupling','barrier transmission','experimental data',
                      'superluminal or retrocausal information transfer']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='verify committed structure/results within floating tolerance')
    args=parser.parse_args()
    result=report()
    target=Path(__file__).resolve().parents[1]/'docs/results/wave-bridge.json'
    if args.check:
        old=json.loads(target.read_text())
        for key in ('status','source_snapshots','exact_sympy_checks','not_tested'):
            assert old[key]==result[key],key
        np.testing.assert_allclose(result['illustrative_nondimensional_front_speeds'],old['illustrative_nondimensional_front_speeds'],rtol=1e-11)
        assert abs(result['old_balanced_branch_dimensionless_residual']['real'])<1e-14
        assert result['old_balanced_branch_dimensionless_residual']['imaginary']>1
        assert result['matrix_exponential_absolute_error']<1e-12
        print('Exact identities and committed conditional results agree; no physical validation.')
    else:
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(result,indent=2)+'\n')
        print(target)


if __name__=='__main__':
    main()
