"""Exact algebra and independent controls for a conditional sector channel."""
import argparse
import json
from pathlib import Path
import platform
import sys
import numpy as np
import scipy
import sympy as sp
from scipy.integrate import quad
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'workspace-pydev'/'hyperspace_comm'))
from segment_channel import TwoSectorChannel,compact_operator_matrix,schur_reduction


def report():
    A,B,eta=sp.symbols('A B eta')
    inverse=sp.Matrix([[A,-eta],[-eta,B]]).inv()
    assert sp.simplify(inverse[1,0]-eta/(A*B-eta**2))==0
    assert sp.simplify(inverse[0,0]-1/(A-eta**2/B))==0
    qA,qB,vA,vB,wA,wB,gA,gB,J=sp.symbols('qA qB vA vB wA wB gA gB J',real=True)
    energy=(vA*vA+vB*vB+wA*wA*qA*qA+wB*wB*qB*qB+eta*(qA-qB)**2)/2
    accA=J-2*gA*vA-wA*wA*qA-eta*(qA-qB)
    accB=-2*gB*vB-wB*wB*qB-eta*(qB-qA)
    rate=sp.diff(energy,qA)*vA+sp.diff(energy,qB)*vB+sp.diff(energy,vA)*accA+sp.diff(energy,vB)*accB
    assert sp.simplify(rate-J*vA+2*gA*vA*vA+2*gB*vB*vB)==0
    D0,D1,D2,u,v=sp.symbols('D0 D1 D2 u v')
    D=sp.Matrix([[D0,u,0],[u,D1,v],[0,v,D2]])
    reduced=D.extract([0,2],[0,2])-D.extract([0,2],[1])*D.extract([1],[1]).inv()*D.extract([1],[0,2])
    assert sp.simplify(reduced[0,1]+u*v/D1)==0
    channel=TwoSectorChannel(1.2,1.2,.3,.2,.2)
    omega=.9
    expected=channel.transfer(omega,source_gain=1,detector_gain=1)
    real=quad(lambda t:float(channel.symmetric_retarded_impulse(t)),0,np.inf,weight='cos',wvar=omega,epsabs=1e-11)[0]
    imag=quad(lambda t:float(channel.symmetric_retarded_impulse(t)),0,np.inf,weight='sin',wvar=omega,epsabs=1e-11)[0]
    modes=[137,138,139]
    L=compact_operator_matrix(modes,radius=1,gradient_coefficients={0:[[1]]},potential_coefficients={1:[[.2]],-1:[[.2]]})
    operator=L-(100+1j)**2*np.eye(3)
    schur=schur_reduction(operator,[0,2])
    return {
        'schema':'conditional-sector-channel/v1',
        'status':'CONDITIONAL_MATHEMATICAL_COMPARATOR_NOT_MULTIVERSE_COMMUNICATION',
        'ubt_snapshot':'3eb72538fca37c917b98f5925e430700999160aa',
        'source_paths':['canonical/ACTION.en.md','canonical/fields/chronofactor_operator.tex',
                        'research_tracks/complex_time_branch_selection/psi_branch_selection.en.md',
                        'research_tracks/information_first/TORUS_THETA_CANONICAL_ACTION_BRIDGE_GATE.cs.md',
                        'speculative_extensions/bridge_137_139/G137_139_metric_bridge.tex'],
        'versions':{'python':platform.python_version(),'numpy':np.__version__,
                    'scipy':scipy.__version__,'sympy':sp.__version__},
        'exact_sympy_checks':{'cross_response':'eta/(A*B-eta**2)',
                             'local_self_energy':'1/(A-eta**2/B)',
                             'energy_rate':'J*vA-2*gA*vA**2-2*gB*vB**2',
                             'eliminated_two_step_entry':'-u*v/D1'},
        'illustrative_units':'nondimensional frequencies/gains; not hardware or UBT parameters',
        'illustrative_cross_response':{'real':expected.real,'imaginary':expected.imag},
        'impulse_fourier_absolute_error':float(abs(real+1j*imag-expected)),
        'first_harmonic_direct_137_139_entry':float(L[0,2].real),
        'two_step_effective_entry':{'real':schur[0,1].real,'imaginary':schur[0,1].imag},
        'neutral_drive_charge_selection':'If Q_n=q*n and [Q,O]=0, (m-n)*q*<m|O|n>=0; no charged transition without compensation.',
        'lean_status':'LEAN-PENDING: no formal operator/PDE or UBT constraint proof in this repository',
        'not_established':['physical neighboring multiverse segments','prime-label = Fourier-label = charge identification',
                           'physical compact derivative term from finalized UBT action','nonzero physical intersector overlap',
                           'toroid source overlap or detector gain','access to a receiver in another sector',
                           'spatial propagation distance or front speed from lumped model','hardware communication or a received reply']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args()
    result=report()
    path=Path(__file__).resolve().parents[1]/'docs/results/segment-channel.json'
    if args.check:
        old=json.loads(path.read_text())
        for key in ('status','ubt_snapshot','source_paths','exact_sympy_checks','not_established','neutral_drive_charge_selection'):
            assert old[key]==result[key],key
        assert result['impulse_fourier_absolute_error']<1e-9
        assert result['first_harmonic_direct_137_139_entry']==0
        for key in ('illustrative_cross_response','two_step_effective_entry'):
            np.testing.assert_allclose(list(old[key].values()),list(result[key].values()),rtol=1e-10,atol=1e-15)
        print('Conditional sector identities agree; no physical multiverse channel is asserted.')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(result,indent=2)+'\n')
        print(path)


if __name__=='__main__':
    main()
