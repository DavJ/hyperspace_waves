"""
Detection methods for hyperspace waves.

Implements various detection algorithms including mFFT-based detection,
correlation detection, and energy signature analysis.

Author: David Jaros
Site: www.octonion-multiverse.com
"""

import numpy as np
import math
from transform.transform import mfft
from biquaternion.wave_bq import HyperspaceWaveBQ


class HyperspaceDetector:
    """
    Multi-method detector for hyperspace waves.
    """
    
    def __init__(self, fsample=25e6):
        """
        Initialize detector.
        
        Args:
            fsample: sampling frequency (Hz)
        """
        self.fsample = fsample
    
    def detect_mfft(self, signal, p=-1, threshold=None):
        """
        Detect hyperspace waves using modified FFT.
        
        Args:
            signal: input signal (list or numpy array)
            p: detection parameter (-1 for balanced divergent waves)
            threshold: detection threshold (auto if None)
        
        Returns:
            dict with detection results
        """
        # Compute mFFT spectrum
        spectrum = mfft(list(signal), p=p)
        spectrum = np.array(spectrum)
        
        # Compute power spectrum
        power = np.abs(spectrum)**2
        
        # Find peaks
        peak_power = np.max(power)
        peak_idx = np.argmax(power)
        
        # Frequency of peak
        N = len(signal)
        freqs = np.fft.fftfreq(N, 1/self.fsample)
        peak_freq = freqs[peak_idx]
        
        # Auto threshold if not provided
        if threshold is None:
            threshold = np.mean(power) + 3*np.std(power)
        
        detected = peak_power > threshold
        
        return {
            'detected': detected,
            'peak_power': peak_power,
            'peak_frequency': abs(peak_freq),
            'spectrum': spectrum,
            'power': power,
            'threshold': threshold,
            'snr_db': 10*np.log10(peak_power / np.mean(power)) if detected else 0
        }
    
    def detect_correlation(self, signal, expected_freq, N=None):
        """
        Detect hyperspace wave using correlation with reference.
        
        Args:
            signal: received signal
            expected_freq: expected wave frequency (Hz)
            N: correlation length (defaults to signal length)
        
        Returns:
            dict with correlation results
        """
        if N is None:
            N = len(signal)
        
        # Generate reference hyperspace wave
        wave_gen = HyperspaceWaveBQ(freq=expected_freq, fsample=self.fsample,
                                    s1=-1/math.sqrt(2), s2=1)
        reference_samples = wave_gen.generate(N=N)
        reference = np.array([s.w for s in reference_samples])
        
        # Ensure signal is correct length
        signal = np.array(signal[:N])
        
        # Complex correlation
        correlation = np.abs(np.sum(signal * np.conj(reference))) / N
        
        # Normalized correlation
        signal_power = np.sqrt(np.sum(np.abs(signal)**2))
        ref_power = np.sqrt(np.sum(np.abs(reference)**2))
        
        if signal_power > 0 and ref_power > 0:
            normalized_corr = correlation / (signal_power * ref_power / N)
        else:
            normalized_corr = 0
        
        # Detection threshold (typical: 0.3 for normalized correlation)
        detected = normalized_corr > 0.3
        
        return {
            'detected': detected,
            'correlation': correlation,
            'normalized_correlation': normalized_corr,
            'expected_frequency': expected_freq
        }
    
    def detect_energy_signature(self, signal, window_size=128):
        """
        Detect by characteristic exponential energy decay.
        
        Args:
            signal: input signal
            window_size: size of sliding window
        
        Returns:
            dict with energy analysis results
        """
        signal = np.array(signal)
        
        # Compute sliding window energy
        energy = []
        for i in range(0, len(signal) - window_size, window_size//4):
            window = signal[i:i+window_size]
            E = np.sum(np.abs(window)**2)
            energy.append(E)
        
        energy = np.array(energy)
        
        if len(energy) < 3:
            return {'detected': False, 'error': 'Signal too short'}
        
        # Fit exponential decay
        t = np.arange(len(energy))
        log_energy = np.log(energy + 1e-10)
        
        # Linear fit in log space
        coeffs = np.polyfit(t, log_energy, 1)
        decay_rate = -coeffs[0]
        fit_quality = 1 - np.var(log_energy - np.polyval(coeffs, t)) / np.var(log_energy)
        
        # Check for negative decay (characteristic of hyperspace waves)
        detected = (decay_rate > 0) and (fit_quality > 0.7)
        
        return {
            'detected': detected,
            'decay_rate': decay_rate,
            'fit_quality': fit_quality,
            'energy_trace': energy
        }
    
    def multi_method_detection(self, signal, expected_freq=None):
        """
        Use multiple detection methods for robust detection.
        
        Args:
            signal: input signal
            expected_freq: expected frequency (if known)
        
        Returns:
            dict with combined results
        """
        results = {}
        
        # Method 1: mFFT
        results['mfft'] = self.detect_mfft(signal, p=-1)
        
        # Method 2: Correlation (if frequency known)
        if expected_freq is not None:
            results['correlation'] = self.detect_correlation(signal, expected_freq)
        
        # Method 3: Energy signature
        results['energy'] = self.detect_energy_signature(signal)
        
        # Combined detection (vote)
        votes = sum([
            results['mfft']['detected'],
            results.get('correlation', {}).get('detected', False),
            results['energy']['detected']
        ])
        
        results['combined_detected'] = votes >= 2
        results['confidence'] = votes / (3 if expected_freq else 2)
        
        return results


class FrequencyScanDetector:
    """
    Scan multiple frequencies to detect hyperspace waves.
    """
    
    def __init__(self, fsample=25e6):
        self.fsample = fsample
        self.detector = HyperspaceDetector(fsample)
    
    def scan_frequencies(self, signal, freq_range=(1e6, 10e6), num_freqs=50):
        """
        Scan frequency range for hyperspace waves.
        
        Args:
            signal: input signal
            freq_range: (f_min, f_max) in Hz
            num_freqs: number of frequencies to test
        
        Returns:
            list of detections at each frequency
        """
        frequencies = np.linspace(freq_range[0], freq_range[1], num_freqs)
        
        detections = []
        for freq in frequencies:
            result = self.detector.detect_correlation(signal, freq)
            result['frequency'] = freq
            detections.append(result)
        
        return detections
    
    def find_peaks(self, detections):
        """
        Find frequency peaks in scan results.
        
        Args:
            detections: list from scan_frequencies
        
        Returns:
            list of detected frequencies
        """
        correlations = [d['normalized_correlation'] for d in detections]
        frequencies = [d['frequency'] for d in detections]
        
        # Find local maxima
        peaks = []
        for i in range(1, len(correlations)-1):
            if (correlations[i] > correlations[i-1] and 
                correlations[i] > correlations[i+1] and
                correlations[i] > 0.3):
                peaks.append({
                    'frequency': frequencies[i],
                    'correlation': correlations[i]
                })
        
        return peaks


def compare_em_vs_hyperspace(em_signal, hs_signal, fsample=25e6):
    """
    Compare EM wave vs hyperspace wave detection.
    
    Useful for barrier penetration experiments.
    
    Args:
        em_signal: signal from EM wave transmission
        hs_signal: signal from hyperspace wave transmission
        fsample: sampling frequency
    
    Returns:
        dict with comparison results
    """
    detector = HyperspaceDetector(fsample)
    
    # Standard FFT (EM detection)
    em_result = detector.detect_mfft(em_signal, p=0)
    
    # mFFT (hyperspace detection)
    hs_result_em = detector.detect_mfft(em_signal, p=-1)
    hs_result_hs = detector.detect_mfft(hs_signal, p=-1)
    
    # Compute penetration ratio
    em_power_baseline = em_result['peak_power']
    hs_power_baseline = hs_result_em['peak_power']
    hs_power_test = hs_result_hs['peak_power']
    
    penetration_ratio = hs_power_test / hs_power_baseline if hs_power_baseline > 0 else 0
    blocking_ratio = em_result['peak_power'] / (em_power_baseline + 1e-10)
    
    return {
        'em_detection': em_result,
        'hyperspace_detection': hs_result_hs,
        'penetration_ratio': penetration_ratio,
        'em_blocking_ratio': blocking_ratio,
        'hyperspace_advantage': penetration_ratio / (blocking_ratio + 1e-10)
    }
