"""Conditional physical-mode tests; NOT a derived UBT action or mode count.

q labels real physical perturbations only after all gauge/Dirac constraints
have been reduced. Constant matrices describe a homogeneous, isotropic,
two-derivative effective sector. SI time/length: C/K has units m^2/s^2,
M/K has units s^-2. See UBT_ANALYSIS.md for the omitted microscopic bridge.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh


def _matrix(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw) and np.any(raw.imag != 0):
        raise ValueError(f"{name} must be real")
    result = np.array(raw.real, dtype=float, copy=True)
    if result.ndim != 2 or result.shape[0] != result.shape[1] or result.size == 0:
        raise ValueError(f"{name} must be a nonempty square matrix")
    if not np.all(np.isfinite(result)) or not np.allclose(result, result.T, rtol=1e-12, atol=0):
        raise ValueError(f"{name} must be finite and symmetric")
    result = (result + result.T) / 2
    result.setflags(write=False)
    return result


def _nonnegative(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


@dataclass(frozen=True)
class QuadraticModes:
    """S2=1/2 int [qdot K qdot - grad(q) C grad(q) - q M q].

    K must be strictly positive definite. C,M must be positive semidefinite.
    Indefinite or constrained raw-Theta Hessians must NOT be silently replaced
    by this class. A singular K requires constraint reduction before use.
    """
    kinetic: np.ndarray
    gradient: np.ndarray
    mass_squared: np.ndarray

    def __post_init__(self):
        for name in ("kinetic", "gradient", "mass_squared"):
            object.__setattr__(self, name, _matrix(getattr(self, name), name))
        if not (self.kinetic.shape == self.gradient.shape == self.mass_squared.shape):
            raise ValueError("matrices must have equal shapes")
        if np.linalg.eigvalsh(self.kinetic)[0] <= 0:
            raise ValueError("kinetic must be positive definite after constraint reduction")
        for name in ("gradient", "mass_squared"):
            # Conservative: do not silently accept a negative squared mass/speed.
            if eigh(getattr(self, name), self.kinetic, eigvals_only=True)[0] < 0:
                raise ValueError(f"{name} must be positive semidefinite")

    def frequencies(self, wave_number):
        """Angular eigenfrequencies for real |k| in rad/m; returns rad/s."""
        k = _nonnegative(wave_number, "wave_number")
        omega2 = eigh(k*k*self.gradient + self.mass_squared, self.kinetic, eigvals_only=True)
        return np.sqrt(np.maximum(omega2, 0))

    def front_speeds(self):
        """Characteristic speeds of this constant, isotropic principal symbol."""
        return np.sqrt(np.maximum(eigh(self.gradient, self.kinetic, eigvals_only=True), 0))

    def group_velocities(self, wave_number):
        """Hellmann-Feynman slopes of positive-frequency eigenbranches.

        At a degeneracy diagonalize the gradient form within the degenerate
        eigenspace. At omega=0 the derivative is undefined here; use the
        principal speeds for a massless k=0 branch. No lossy-media claim.
        """
        k = _nonnegative(wave_number, "wave_number")
        values, vectors = eigh(k*k*self.gradient + self.mass_squared, self.kinetic)
        if np.any(values <= 0):
            raise ValueError("group velocity at zero frequency requires a branch limit")
        slopes = np.empty(len(values))
        start = 0
        while start < len(values):
            stop = start + 1
            while stop < len(values) and abs(values[stop]-values[start]) <= 32*np.finfo(float).eps*max(abs(values[start]), abs(values[stop])):
                stop += 1
            basis = vectors[:, start:stop]
            gradients = np.linalg.eigvalsh(basis.T @ self.gradient @ basis)
            slopes[start:stop] = k*gradients/np.sqrt(values[start:stop])
            start = stop
        return slopes

    def is_causal(self, speed_limit=299792458.0):
        """Check C <= speed_limit^2 K for THIS effective model only."""
        limit = _nonnegative(speed_limit, "speed_limit")
        return bool(np.max(self.front_speeds()) <= limit*(1 + 32*np.finfo(float).eps))

    def response(self, wave_number, angular_frequency, source, detector, damping=None):
        """d^T [M+k^2 C-omega^2 K-2i omega Gamma]^-1 s.

        exp(-i omega t); Gamma is a supplied passive symmetric damping matrix.
        s,d must come from independent source/detector derivations to obtain a
        physical voltage. Unit vectors are illustrative, not inferred gains.
        Undamped real poles are singular; no arbitrary fitted regulator is added.
        """
        k = _nonnegative(wave_number, "wave_number")
        omega = float(angular_frequency)
        if not np.isfinite(omega):
            raise ValueError("angular_frequency must be finite")
        n = len(self.kinetic)
        s, d = np.asarray(source, dtype=complex), np.asarray(detector, dtype=complex)
        if s.shape != (n,) or d.shape != (n,) or not np.isfinite(s).all() or not np.isfinite(d).all():
            raise ValueError("source/detector must be finite vectors of the physical mode dimension")
        gamma = np.zeros_like(self.kinetic) if damping is None else _matrix(damping, "damping")
        if gamma.shape != self.kinetic.shape or np.linalg.eigvalsh(gamma)[0] < 0:
            raise ValueError("damping must have the correct shape and be positive semidefinite")
        operator = self.mass_squared + k*k*self.gradient - omega*omega*self.kinetic - 2j*omega*gamma
        return complex(d @ np.linalg.solve(operator, s))


def mass_shell_residual(k_real, k_imag, mass_inverse_length=0):
    """Complex K.K-mu^2 for K=(omega/c,k), signature (+---).

    All four-vector entries and mu have units m^-1. Both the real AND
    imaginary parts must vanish for a free real-mass Klein-Gordon plane wave.
    This is a test-field signature convention, not a UBT metric definition.
    """
    a, b = np.asarray(k_real, dtype=float), np.asarray(k_imag, dtype=float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("four-vectors must be finite real arrays of length four")
    mu = _nonnegative(mass_inverse_length, "mass_inverse_length")
    z = a + 1j*b
    return complex(z[0]**2 - np.dot(z[1:], z[1:]) - mu**2)


def damped_poles(wave_number, *, speed, gap, damping):
    """omega poles for exp(-i omega t), qtt+2 gamma qt-v^2 Lap q+gap^2 q=0.

    gap,damping in s^-1, speed in m/s, k in rad/m. Passive phenomenological
    damping represents an eliminated environment, not a closed UBT action.
    """
    k, v, mu, gamma = (_nonnegative(x, n) for x, n in
                       ((wave_number, "wave_number"), (speed, "speed"),
                        (gap, "gap"), (damping, "damping")))
    root = np.sqrt(complex(v*v*k*k + mu*mu - gamma*gamma))
    return np.array([-1j*gamma + root, -1j*gamma - root])


def damped_template(times, frequency_hz, decay_rate):
    """Ordinary complex ringdown exp(-gamma*t+2*pi*i*f*t), t>=0."""
    t = np.asarray(times, dtype=float)
    frequency = float(frequency_hz)
    gamma = _nonnegative(decay_rate, "decay_rate")
    if not np.isfinite(t).all() or np.any(t < 0) or not np.isfinite(frequency):
        raise ValueError("times must be finite/nonnegative and frequency finite")
    return np.exp((-gamma + 2j*np.pi*frequency)*t)


def matched_filter(signal, template, noise_variance=1.0):
    """Unit-energy complex matched filter with explicit white-noise null.

    For circular complex Gaussian noise E|n_j|^2=sigma^2, power/sigma^2 is
    exponential with mean one for ONE prespecified template. It is not a
    probability of new physics; scans require trials correction/calibration.
    """
    x, h = np.asarray(signal, dtype=complex), np.asarray(template, dtype=complex)
    variance = float(noise_variance)
    if x.ndim != 1 or x.size == 0 or h.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(h).all():
        raise ValueError("signal/template must be equal nonempty finite 1D arrays")
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("noise_variance must be positive and finite")
    scale = np.max(abs(h))
    if scale == 0:
        raise ValueError("template must be nonzero")
    unit = h/scale
    unit /= np.linalg.norm(unit)
    projection = np.vdot(unit, x)
    statistic = float(abs(projection)**2/variance)
    return {"projection": projection, "normalized_power": statistic,
            "single_template_noise_tail": float(np.exp(-statistic))}
