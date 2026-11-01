"""Conditional action-to-source/readout matching for neutral physical modes.

Natural units hbar=c=1, canonical photon term -F^2/4. Heavy loop fields are
positive-energy COMPLEX SCALARS, not an assumed UBT spectrum. Neutral h fields
are selected physical mode coordinates, not a scalar replacement of Theta.
See docs/source-detector-matching.md for matching and perturbative boundaries.
"""
from dataclasses import dataclass
import numpy as np
from causal_modes import QuadraticModes


def _hermitian(value, name):
    result=np.array(value,dtype=complex,copy=True)
    if result.ndim!=2 or result.shape[0]!=result.shape[1] or result.size==0 or not np.isfinite(result).all():
        raise ValueError(f'{name} must be a finite nonempty square matrix')
    if not np.allclose(result,result.conj().T,rtol=1e-12,atol=1e-13):
        raise ValueError(f'{name} must be Hermitian')
    return (result+result.conj().T)/2


def _coupling(e):
    e=float(e)
    if not np.isfinite(e) or e<0:
        raise ValueError('e must be finite and nonnegative')
    return e


def _commute(a,b):
    scale=np.linalg.norm(a)*np.linalg.norm(b)
    return np.linalg.norm(a@b-b@a)<=1e-11*max(scale,np.finfo(float).tiny)


def _mass_and_charge(mass_squared,charge_generator):
    mass=_hermitian(mass_squared,'mass_squared')
    charge=_hermitian(charge_generator,'charge_generator')
    if mass.shape!=charge.shape:
        raise ValueError('mass and charge must have equal shape')
    if np.linalg.eigvalsh(mass)[0]<=0:
        raise ValueError('heavy mass_squared must be positive definite')
    if not _commute(mass,charge):
        raise ValueError('neutral-background mass matrix must commute with charge')
    return mass,charge


def complex_scalar_photon_mass_squared(background, charge_generator, e):
    """m_gamma^2=2*e^2*|Q*v|^2 for L=|D phi|^2 and constant v.

    Canonically normalized complex scalars and a positive coefficient metric
    are ASSUMED. The factor two follows from comparing with m_gamma^2*A^2/2.
    A neutral background obeys Q*v=0, not merely [v,phase]=0.
    """
    charge=_hermitian(charge_generator,'charge_generator')
    v=np.asarray(background,dtype=complex)
    if v.shape!=(len(charge),) or not np.isfinite(v).all():
        raise ValueError('background must be a finite vector of the charge dimension')
    charged=charge@v
    return float(2*_coupling(e)**2*np.vdot(charged,charged).real)


def complex_scalar_threshold_coefficients(mass_squared, mass_derivatives, charge_generator, e):
    """c_a=e^2/(192*pi^2) Tr[Q^2 M^-2 d_a(M^2)] in L += c_a*h_a*F^2.

    M^2 is an arbitrary positive Hermitian heavy-scalar mass matrix. Neutral
    derivatives must commute with Q; they need not commute with M^2 within
    equal-charge blocks. Canonical positive kinetic terms, soft external
    momenta, weak fields/coupling and no bare h*F^2 term in this comparator.
    Additional microscopic Wilson terms, species or spins change the total.
    """
    mass,charge=_mass_and_charge(mass_squared,charge_generator)
    raw=np.asarray(mass_derivatives,dtype=complex)
    if raw.ndim!=3 or raw.shape[0]==0 or raw.shape[1:]!=mass.shape:
        raise ValueError('mass_derivatives must have shape (neutral_modes,N,N)')
    result=[]
    prefactor=_coupling(e)**2/(192*np.pi**2)
    for derivative in raw:
        derivative=_hermitian(derivative,'mass_derivative')
        if not _commute(derivative,charge):
            raise ValueError('neutral mass derivative must commute with charge')
        value=np.trace(charge@charge@np.linalg.solve(mass,derivative))
        result.append(prefactor*value.real)
    return np.array(result)


def complex_scalar_gauge_kinetic_shift(mass_squared, reference_mass_squared, charge_generator, e):
    """Delta Z=-e^2/(48*pi^2) Tr Q^2 [log M^2(h)-log M^2(0)].

    L=-Z(h) F^2/4. Subtraction fixes one h-independent photon counterterm;
    differentiating it gives c_a=-Z_,a/4. This is the scalar one-loop local
    low-energy term, not the full finite-momentum effective action.
    """
    mass,charge=_mass_and_charge(mass_squared,charge_generator)
    reference,_=_mass_and_charge(reference_mass_squared,charge_generator)
    if mass.shape!=reference.shape:
        raise ValueError('mass matrices must have equal shape')
    def log_positive(matrix):
        values,vectors=np.linalg.eigh(matrix)
        return (vectors*np.log(values))@vectors.conj().T
    difference=log_positive(mass)-log_positive(reference)
    return float(-_coupling(e)**2/(48*np.pi**2)*np.trace(charge@charge@difference).real)


@dataclass(frozen=True)
class NeutralPortalExample:
    """One explicit RENORMALIZABLE EFFECTIVE COMPARATOR; no new UBT action.

    V0=lambda/4*(hA^2+hB^2-v^2)^2 + delta/2*(hA-hB)^2.
    Mchi^2=m0^2+yA^2*hA^2+yB^2*hB^2, <chi>=0.
    hA,hB are neutral real modes and chi one heavy complex charged mode.
    A positive |chi|^4 term may bound the full potential; it does not change
    the displayed background or one-loop threshold coefficient at chi=0.
    All parameters are supplied; no carrier, mode count or scale is selected.
    """
    v: float
    quartic: float
    locking_mass_squared: float
    heavy_bare_mass_squared: float
    y_a_squared: float
    y_b_squared: float
    e: float
    charge: float

    def __post_init__(self):
        for name in ('v','quartic','locking_mass_squared','heavy_bare_mass_squared'):
            value=float(getattr(self,name))
            if not np.isfinite(value) or value<=0:
                raise ValueError(f'{name} must be positive and finite')
            object.__setattr__(self,name,value)
        for name in ('y_a_squared','y_b_squared','e'):
            value=float(getattr(self,name))
            if not np.isfinite(value) or value<0:
                raise ValueError(f'{name} must be nonnegative and finite')
            object.__setattr__(self,name,value)
        value=float(self.charge)
        if not np.isfinite(value):
            raise ValueError('charge must be finite')
        object.__setattr__(self,'charge',value)

    @property
    def background(self):
        return np.full(2,self.v/np.sqrt(2))

    @property
    def neutral_hessian(self):
        r=self.quartic*self.v**2
        d=self.locking_mass_squared
        return np.array([[r+d,r-d],[r-d,r+d]])

    @property
    def derived_eta(self):
        """Signed -V_,AB in the specified physical hA,hB basis; units mass^2."""
        return self.locking_mass_squared-self.quartic*self.v**2

    def heavy_mass_squared(self, neutral_coordinates):
        h=np.asarray(neutral_coordinates,dtype=float)
        if h.shape!=(2,) or not np.isfinite(h).all():
            raise ValueError('neutral_coordinates must be a finite two-vector')
        return self.heavy_bare_mass_squared+self.y_a_squared*h[0]**2+self.y_b_squared*h[1]**2

    @property
    def portal_coefficients(self):
        derivatives=(2*self.background*np.array([self.y_a_squared,self.y_b_squared])).reshape(2,1,1)
        return complex_scalar_threshold_coefficients([[self.heavy_mass_squared(self.background)]],
                                                     derivatives,[[self.charge]],self.e)

    def delta_z_per_pump_f_squared(self, angular_frequency, damping):
        """-4*c^T[H-omega^2 I-2i omega Gamma]^-1*c; NOT pickup volts.

        Uniform selected modal source/readout profiles are assumed. Real
        experiments require spatial overlaps and the Maxwell detector Green
        function. Tree neutral propagator plus leading scalar-loop vertices;
        quantum vacuum/propagator corrections are beyond this example's order.
        """
        modes=QuadraticModes(np.eye(2),np.zeros((2,2)),self.neutral_hessian)
        c=self.portal_coefficients
        return modes.response(0,angular_frequency,c,-4*c,damping)


def electromagnetic_f_squared(electric, magnetic):
    """F_mn F^mn=2*(B^2-E^2), natural units, real instantaneous fields.

    A single ideal vacuum plane wave has zero invariant. A magnetically
    dominated near field need not. No material/coil dynamics is supplied here.
    """
    e,b=np.asarray(electric,dtype=float),np.asarray(magnetic,dtype=float)
    if e.shape!=(3,) or b.shape!=(3,) or not np.isfinite(e).all() or not np.isfinite(b).all():
        raise ValueError('electric and magnetic fields must be finite real three-vectors')
    return float(2*(np.dot(b,b)-np.dot(e,e)))
