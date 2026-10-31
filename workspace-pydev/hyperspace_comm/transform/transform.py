"""Finite spiral transform, not a physical hyperspace detector.

mfft(x,p)[k] = sum_n x[n] exp[-2*pi*(p+i)*k*n/N].
For p=0 use NumPy's FFT. For p!=0 this is a finite spiral transform with strongly
unequal noise gains. Large growing kernels are rejected rather than overflow.
Use causal_modes.matched_filter for a normalized, specified noise test.
"""
import numpy as np


def mfft(input, p=0):
    values = np.asarray(input, dtype=complex)
    p = float(p)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all() or not np.isfinite(p):
        raise ValueError("input must be a nonempty finite 1D array and p finite")
    if p == 0:
        return np.fft.fft(values).tolist()
    n = len(values)
    # Bound the growing kernel before constructing a dense reference sum.
    exponent = 2*np.pi*abs(p)*(n-1)**2/n
    if exponent > 300:
        raise ValueError("spiral transform dynamic range is unsafe; use a normalized matched filter")
    if n > 2048:
        raise ValueError("nonzero-p reference transform is limited to 2048 samples")
    indices = np.arange(n)
    kernel = np.exp(-2*np.pi*(p+1j)*np.outer(indices, indices)/n)
    result = kernel @ values
    if not np.isfinite(result).all():
        raise ValueError("spiral transform is numerically ill-conditioned")
    return result.tolist()
