"""Conditional sector-conversion comparators, not a multiverse transmitter.

Labels A/B and integer Fourier indices acquire a physical meaning only after a
single action supplies physical modes, constraints, source and detector maps.
No identification of a Fourier index with a prime-world label is made here.
"""
from dataclasses import dataclass
import numpy as np


def _nonnegative(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')
    return value


def _gain(value, name):
    value = complex(value)
    if not np.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


@dataclass(frozen=True)
class TwoSectorChannel:
    """Two canonically normalized physical mode amplitudes with a spring link.

    E = (vA^2+vB^2+wA^2*qA^2+wB^2*qB^2+eta*(qA-qB)^2)/2.
    Frequencies/damping use s^-1, eta uses s^-2. These are supplied example
    coefficients, not a UBT prediction. The two q's need not exhaust Theta.
    """
    omega_a: float
    omega_b: float
    eta: float
    damping_a: float = 0.0
    damping_b: float = 0.0

    def __post_init__(self):
        for name in ('omega_a','omega_b','eta','damping_a','damping_b'):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name))

    @property
    def stiffness(self):
        return np.array([[self.omega_a**2+self.eta, -self.eta],
                         [-self.eta, self.omega_b**2+self.eta]])

    @property
    def damping(self):
        return np.diag([self.damping_a,self.damping_b])

    def operator(self, angular_frequency):
        omega=float(angular_frequency)
        if not np.isfinite(omega):
            raise ValueError('angular_frequency must be finite')
        return self.stiffness-omega**2*np.eye(2)-2j*omega*self.damping

    def transfer(self, angular_frequency, *, source_gain, detector_gain):
        """B readout per A drive; supplied gains retain their unknown units.

        A physical prediction requires action-derived gains and segment maps.
        Undamped poles are singular; no arbitrary regulator is inserted.
        """
        s,d=_gain(source_gain,'source_gain'),_gain(detector_gain,'detector_gain')
        solution=np.linalg.solve(self.operator(angular_frequency),[s,0])
        return complex(d*solution[1])

    def local_response(self, angular_frequency, *, source_gain, detector_gain):
        """A readout including a possible A->B->A self-energy, not a reply."""
        s,d=_gain(source_gain,'source_gain'),_gain(detector_gain,'detector_gain')
        solution=np.linalg.solve(self.operator(angular_frequency),[s,0])
        return complex(d*solution[0])

    def symmetric_retarded_impulse(self, times):
        """B response to unit A impulse, for equal gaps and damping only.

        The two normal frequencies are omega_a and sqrt(omega_a^2+2*eta).
        Zero before t=0. This is a lumped oscillator response, not a spatial
        front speed or a signal arriving at an established second universe.
        """
        if self.omega_a!=self.omega_b or self.damping_a!=self.damping_b:
            raise ValueError('analytic impulse requires equal frequencies and damping')
        return .5*(retarded_oscillator_green(times,self.omega_a,self.damping_a)
                   -retarded_oscillator_green(times,np.sqrt(self.omega_a**2+2*self.eta),self.damping_a))


def retarded_oscillator_green(times, omega0, damping):
    """Green function for qtt+2*gamma*qt+omega0^2*q=delta(t).

    Stable formulas cover underdamped, critical and overdamped cases without
    evaluating a growing sinh multiplied by a vanishing exponential.
    """
    t=np.asarray(times,dtype=float)
    if not np.isfinite(t).all():
        raise ValueError('times must be finite')
    w,gamma=_nonnegative(omega0,'omega0'),_nonnegative(damping,'damping')
    s=np.maximum(t,0)
    delta=w*w-gamma*gamma
    if delta>0:
        root=np.sqrt(delta)
        result=np.exp(-gamma*s)*s*np.sinc(root*s/np.pi)
    elif delta==0:
        result=s*np.exp(-gamma*s)
    else:
        root=np.sqrt(-delta)
        # -gamma+root = -w^2/(gamma+root), avoiding cancellation.
        slow=-w*w/(gamma+root)
        result=np.exp(slow*s)*(-np.expm1(-2*root*s))/(2*root)
    return np.where(t>=0,result,0)


def compact_operator_matrix(mode_numbers, *, radius, gradient_coefficients, potential_coefficients):
    """Galerkin matrix of L=-d_y(C(y)d_y)+U(y) on a periodic circle.

    C_r,U_r are matrix Fourier coefficients with convention exp(i*r*y/R).
    L_mn = (m*n/R^2)*C_(m-n)+U_(m-n). Any component dimension is supported.
    Hermitian coefficient pairs are required. This is not a derivation of a
    physical compact coordinate, a UBT kinetic term, or its allowed carrier.
    A finite truncation alone neither proves positivity nor eliminates omitted
    modes. Impose physical charge/gauge selection rules separately.
    """
    indices=tuple(mode_numbers)
    if not indices or any(isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) for n in indices) or len(set(indices))!=len(indices):
        raise ValueError('mode_numbers must be distinct integers')
    radius=float(radius)
    if not np.isfinite(radius) or radius<=0:
        raise ValueError('radius must be positive and finite')
    groups=[]
    dimension=None
    for coefficients in (gradient_coefficients,potential_coefficients):
        group={}
        for r,value in coefficients.items():
            if isinstance(r,(bool,np.bool_)) or not isinstance(r,(int,np.integer)):
                raise ValueError('Fourier labels must be integers')
            matrix=np.array(value,dtype=complex,copy=True)
            if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or matrix.size==0 or not np.isfinite(matrix).all():
                raise ValueError('coefficients must be finite nonempty square matrices')
            if dimension is None:
                dimension=len(matrix)
            if matrix.shape!=(dimension,dimension):
                raise ValueError('coefficient matrices must have the same shape')
            group[int(r)]=matrix
        groups.append(group)
    if dimension is None:
        raise ValueError('at least one Fourier coefficient is required')
    zero=np.zeros((dimension,dimension),dtype=complex)
    for group in groups:
        for r,matrix in group.items():
            if not np.allclose(group.get(-r,zero),matrix.conj().T,rtol=1e-12,atol=1e-14):
                raise ValueError('Fourier coefficients require C_-r=C_r^dagger and U_-r=U_r^dagger')
    c,u=groups
    return np.block([[(m*n/radius**2)*c.get(m-n,zero)+u.get(m-n,zero)
                      for n in indices] for m in indices])


def schur_reduction(operator, retained_indices):
    """Exact D_eff=D_kk-D_ke D_ee^-1 D_ek for invertible eliminated block.

    This is algebraic elimination at the supplied frequency. Near an eliminated
    pole retain the intermediate mode; perturbative detuning expansions fail.
    """
    matrix=np.asarray(operator,dtype=complex)
    if matrix.ndim!=2 or matrix.shape[0]!=matrix.shape[1] or matrix.size==0 or not np.isfinite(matrix).all():
        raise ValueError('operator must be a finite nonempty square matrix')
    keep=tuple(retained_indices)
    if not keep or any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or not 0<=i<len(matrix) for i in keep) or len(set(keep))!=len(keep):
        raise ValueError('retained_indices must be distinct valid indices')
    removed=tuple(i for i in range(len(matrix)) if i not in keep)
    effective=matrix[np.ix_(keep,keep)]
    if not removed:
        return effective.copy()
    return effective-matrix[np.ix_(keep,removed)]@np.linalg.solve(
        matrix[np.ix_(removed,removed)],matrix[np.ix_(removed,keep)])
