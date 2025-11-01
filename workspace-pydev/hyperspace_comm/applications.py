"""
Applications of hyperspace waves.

Implements practical applications including communication systems,
barrier penetration, and experimental protocols.

Author: David Jaros
Site: www.octonion-multiverse.com
"""

import numpy as np
import math
from biquaternion.wave_bq import HyperspaceWaveBQ
from detection import HyperspaceDetector


class HyperspaceCommunicator:
    """
    Communication system using hyperspace waves.
    
    Features:
    - Binary Phase Shift Keying (BPSK) using s2 parameter
    - Barrier penetration capability
    - Potential FTL propagation
    """
    
    def __init__(self, carrier_freq=2e6, data_rate=1e3, fsample=25e6):
        """
        Initialize communicator.
        
        Args:
            carrier_freq: carrier frequency (Hz)
            data_rate: data rate (bits/second)
            fsample: sampling frequency (Hz)
        """
        self.carrier_freq = carrier_freq
        self.data_rate = data_rate
        self.fsample = fsample
        self.samples_per_bit = int(fsample / data_rate)
    
    def encode_message(self, message_bits):
        """
        Encode binary message using BPSK modulation.
        
        Modulation scheme:
        - Bit 0: s2 = +1
        - Bit 1: s2 = -1
        
        Args:
            message_bits: list of 0s and 1s
        
        Returns:
            numpy array of complex samples
        """
        signal = []
        
        for bit in message_bits:
            # Select s2 based on bit value
            s2 = 1 if bit == 0 else -1
            
            # Generate hyperspace wave for this bit
            wave = HyperspaceWaveBQ(
                freq=self.carrier_freq,
                fsample=self.fsample,
                s1=-1/math.sqrt(2),
                s2=s2
            )
            
            bit_samples = wave.generate(N=self.samples_per_bit)
            signal.extend([s.w for s in bit_samples])
        
        return np.array(signal)
    
    def decode_message(self, received_signal):
        """
        Decode message using correlation detection.
        
        Args:
            received_signal: numpy array of received samples
        
        Returns:
            list of decoded bits
        """
        # Generate reference waves for both bit values
        wave_0 = HyperspaceWaveBQ(
            freq=self.carrier_freq,
            fsample=self.fsample,
            s1=-1/math.sqrt(2),
            s2=1
        ).generate(N=self.samples_per_bit)
        
        wave_1 = HyperspaceWaveBQ(
            freq=self.carrier_freq,
            fsample=self.fsample,
            s1=-1/math.sqrt(2),
            s2=-1
        ).generate(N=self.samples_per_bit)
        
        ref_0 = np.array([s.w for s in wave_0])
        ref_1 = np.array([s.w for s in wave_1])
        
        # Decode each bit period
        num_bits = len(received_signal) // self.samples_per_bit
        bits = []
        
        for i in range(num_bits):
            segment = received_signal[i*self.samples_per_bit:(i+1)*self.samples_per_bit]
            
            # Correlate with both references
            corr_0 = np.abs(np.sum(segment * np.conj(ref_0)))
            corr_1 = np.abs(np.sum(segment * np.conj(ref_1)))
            
            # Choose bit with higher correlation
            bits.append(0 if corr_0 > corr_1 else 1)
        
        return bits
    
    def encode_text(self, text):
        """
        Encode text message to hyperspace wave.
        
        Args:
            text: string message
        
        Returns:
            numpy array of samples
        """
        # Convert text to binary
        binary = ''.join(format(ord(c), '08b') for c in text)
        bits = [int(b) for b in binary]
        
        return self.encode_message(bits)
    
    def decode_text(self, received_signal):
        """
        Decode text from received signal.
        
        Args:
            received_signal: numpy array
        
        Returns:
            decoded text string
        """
        bits = self.decode_message(received_signal)
        
        # Convert bits to bytes
        chars = []
        for i in range(0, len(bits), 8):
            if i+8 <= len(bits):
                byte = bits[i:i+8]
                char_code = sum(bit << (7-j) for j, bit in enumerate(byte))
                if 32 <= char_code <= 126:  # Printable ASCII
                    chars.append(chr(char_code))
        
        return ''.join(chars)


class BarrierPenetrationSystem:
    """
    System for communication through barriers.
    """
    
    def __init__(self, carrier_freq=2e6):
        self.comm = HyperspaceCommunicator(carrier_freq=carrier_freq)
        self.detector = HyperspaceDetector(fsample=self.comm.fsample)
    
    def test_penetration(self, message, barrier_type='faraday_cage'):
        """
        Test barrier penetration capability.
        
        Args:
            message: text message to send
            barrier_type: type of barrier
        
        Returns:
            dict with test results
        """
        # Encode message
        signal = self.comm.encode_text(message)
        
        print(f"Testing penetration through {barrier_type}...")
        print(f"Message: '{message}'")
        print(f"Signal length: {len(signal)} samples")
        
        return {
            'message': message,
            'signal': signal,
            'samples': len(signal),
            'duration_ms': len(signal) / self.comm.fsample * 1000,
            'barrier_type': barrier_type
        }
    
    def analyze_penetration(self, baseline_signal, test_signal):
        """
        Analyze penetration performance.
        
        Args:
            baseline_signal: control signal (no barrier)
            test_signal: signal through barrier
        
        Returns:
            dict with analysis
        """
        # Detect in both signals
        baseline_detect = self.detector.detect_mfft(baseline_signal, p=-1)
        test_detect = self.detector.detect_mfft(test_signal, p=-1)
        
        # Calculate transmission coefficient
        transmission = test_detect['peak_power'] / (baseline_detect['peak_power'] + 1e-10)
        
        # Calculate SNR degradation
        snr_baseline = baseline_detect.get('snr_db', 0)
        snr_test = test_detect.get('snr_db', 0)
        snr_loss = snr_baseline - snr_test
        
        return {
            'transmission_coefficient': transmission,
            'snr_baseline_db': snr_baseline,
            'snr_test_db': snr_test,
            'snr_loss_db': snr_loss,
            'penetrated': test_detect['detected']
        }


class RetrocausalExperiment:
    """
    Experiment to test retrocausal signal propagation.
    """
    
    def __init__(self):
        self.comm = HyperspaceCommunicator(carrier_freq=2e6)
        self.detector = HyperspaceDetector(fsample=self.comm.fsample)
    
    def create_test_signal(self, duration_ms=10):
        """
        Create test signal for retrocausal detection.
        
        Args:
            duration_ms: signal duration in milliseconds
        
        Returns:
            test signal array
        """
        # Create distinctive pattern
        test_bits = [1, 0, 1, 1, 0, 0, 1, 0] * (duration_ms // 8 + 1)
        signal = self.comm.encode_message(test_bits)
        
        return signal
    
    def analyze_timing(self, recorded_buffer, test_signal, transmission_time):
        """
        Analyze buffer for signals before transmission time.
        
        Args:
            recorded_buffer: list of (timestamp, signal_segment) tuples
            test_signal: the transmitted signal
            transmission_time: when signal was transmitted
        
        Returns:
            dict with timing analysis
        """
        detections = []
        
        for timestamp, signal_segment in recorded_buffer:
            if timestamp < transmission_time:
                # Correlate with test signal
                N = min(len(signal_segment), len(test_signal))
                segment = signal_segment[:N]
                reference = test_signal[:N]
                
                correlation = np.abs(np.sum(segment * np.conj(reference))) / N
                
                if correlation > 0.5:  # High threshold for retrocausal claim
                    time_advance = transmission_time - timestamp
                    detections.append({
                        'timestamp': timestamp,
                        'time_advance': time_advance,
                        'correlation': correlation
                    })
        
        return {
            'retrocausal_detected': len(detections) > 0,
            'num_detections': len(detections),
            'detections': detections
        }


class FTLCommunicationLink:
    """
    Faster-than-light communication link using hyperspace waves.
    """
    
    def __init__(self, carrier_freq=2e6):
        self.comm = HyperspaceCommunicator(carrier_freq=carrier_freq)
        self.c = 299792458  # Speed of light (m/s)
    
    def calculate_group_velocity(self, freq, wave_vector):
        """
        Calculate group velocity from dispersion relation.
        
        For balanced waves: ω²/2 = k² + m²
        v_g = dω/dk = k/ω · √2
        
        Args:
            freq: frequency (Hz)
            wave_vector: wave vector magnitude (1/m)
        
        Returns:
            group velocity (m/s)
        """
        omega = 2 * math.pi * freq
        k = wave_vector
        
        # From ω²/2 = k² + m², assume m ≈ 0 for high freq
        # Then ω = √2 · k
        # v_g = dω/dk = √2
        
        if k > 0:
            v_g = omega / k * math.sqrt(2)
        else:
            v_g = self.c
        
        return v_g
    
    def estimate_transmission_time(self, distance_m, freq=None):
        """
        Estimate transmission time for given distance.
        
        Args:
            distance_m: distance in meters
            freq: frequency (Hz), uses carrier if None
        
        Returns:
            dict with timing estimates
        """
        if freq is None:
            freq = self.comm.carrier_freq
        
        # Classical EM propagation time
        time_em = distance_m / self.c
        
        # Hyperspace propagation (assume v_g ≈ √2 · c for high frequencies)
        v_hyperspace = math.sqrt(2) * self.c
        time_hyperspace = distance_m / v_hyperspace
        
        # Time savings
        time_saved = time_em - time_hyperspace
        
        return {
            'distance_m': distance_m,
            'distance_km': distance_m / 1000,
            'time_em_s': time_em,
            'time_hyperspace_s': time_hyperspace,
            'time_saved_s': time_saved,
            'speedup_factor': time_em / time_hyperspace,
            'group_velocity': v_hyperspace
        }


def demonstrate_applications():
    """
    Demonstrate various hyperspace wave applications.
    """
    print("="*70)
    print("HYPERSPACE WAVE APPLICATIONS DEMONSTRATION")
    print("="*70)
    
    # 1. Basic Communication
    print("\n1. BASIC COMMUNICATION")
    print("-" * 70)
    comm = HyperspaceCommunicator(carrier_freq=2e6, data_rate=1e3)
    
    message = "HELLO HYPERSPACE"
    print(f"Message: '{message}'")
    
    encoded = comm.encode_text(message)
    print(f"Encoded to {len(encoded)} samples")
    print(f"Duration: {len(encoded)/comm.fsample*1000:.2f} ms")
    
    # Simulate transmission (in real system, this would go through hardware)
    received = encoded  # Perfect channel for demo
    
    decoded = comm.decode_text(received)
    print(f"Decoded: '{decoded}'")
    print(f"Success: {decoded == message}")
    
    # 2. Barrier Penetration
    print("\n2. BARRIER PENETRATION")
    print("-" * 70)
    barrier = BarrierPenetrationSystem()
    test = barrier.test_penetration("TEST", barrier_type="Faraday cage")
    print(f"Signal duration: {test['duration_ms']:.2f} ms")
    print("Ready for transmission through barrier")
    
    # 3. FTL Communication
    print("\n3. FASTER-THAN-LIGHT COMMUNICATION")
    print("-" * 70)
    ftl = FTLCommunicationLink()
    
    distances = [1e6, 1e9, 1e12]  # 1000 km, 1 million km, 1 billion km
    for dist in distances:
        timing = ftl.estimate_transmission_time(dist)
        print(f"\nDistance: {timing['distance_km']:.0f} km")
        print(f"  EM time: {timing['time_em_s']:.6f} s")
        print(f"  Hyperspace time: {timing['time_hyperspace_s']:.6f} s")
        print(f"  Time saved: {timing['time_saved_s']:.6f} s")
        print(f"  Speedup: {timing['speedup_factor']:.2f}x")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70)


if __name__ == '__main__':
    demonstrate_applications()
