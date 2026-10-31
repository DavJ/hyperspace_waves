# Hyperspace Waves — UBT research and verifiable wave models

**Author:** David Jaroš · [UBT](https://github.com/UBT-Institute/unified-biquaternion-theory)

This repository studies the original complex-frequency wave proposal alongside
biquaternion algebra, signal processing and conditional effective dynamics.
**A new physical wave, faster-than-light communication, retrocausality and
universal barrier penetration have not been established.** The 2025 claims of
completion are superseded by the [current derivation and gap ledger](UBT_ANALYSIS.md).

The original samples are ordinary damped sinusoids:

```math
u(t)=\exp[2\pi f(s_1+i s_2)t],\qquad t\geq0.
```

Any negative `s1` gives a decaying half-line template. The historical value
`-1/sqrt(2)` is not selected by finite energy. Complex frequency alone supplies
neither a spatial dispersion law nor a new propagation channel.

## What is now implemented

- Correct complex biquaternion inverse and exponential, including nonzero zero
  divisors and nilpotents; independently checked with faithful 2×2 matrices.
- Both real and imaginary complex mass-shell equations. The previously claimed
  `sqrt(2)*c` derivation fails the imaginary equation.
- An arbitrary-dimensional physical-mode comparator: positive kinetic energy,
  generalized eigenfrequencies, front speeds, group slopes and source-to-detector
  response. Its coefficients and constraint reduction are **not derived from UBT**.
- Passive damped poles and a unit-energy matched filter with a stated single
  template white-noise null. Detection of a template is not detection of a new field.
- A defined finite spiral transform: `p=0` agrees with FFT, unsafe exponential
  dynamic ranges are rejected. Raw unequal-gain transform peaks are not evidence
  of enhanced penetration.
- A tested central-complex theta modular identity. General noncommuting modular
  identities and the old heuristic theta reconstruction remain unproved.

The current UBT reference is pinned to
[`386b6db`](https://github.com/UBT-Institute/unified-biquaternion-theory/commit/386b6dbed7d35ea63c8fd42fd88d57ac2f9cdaa3).
It still registers the fundamental action as `DEFINED_FAMILY_NOT_FINALIZED`.
No number of physical Theta components is assumed here; four complex
coefficients describe one algebra element, not the full physical field content.

## Run

Python 3.10 or later:

```sh
python -m pip install -r requirements.txt pytest sympy
python -m pytest -q
python scripts/verify_wave_bridge.py --check
```

```python
import sys
sys.path.insert(0, 'workspace-pydev/hyperspace_comm')
from causal_modes import QuadraticModes, damped_template, matched_filter
import numpy as np

# Explicit illustrative units, NOT selected UBT coefficients:
model = QuadraticModes([[1]], [[1]], [[0.25]])
print(model.front_speeds())  # [1] in the chosen length/time units
print(model.frequencies(2))
t = np.arange(256)/100
h = damped_template(t, frequency_hz=3, decay_rate=2)
result = matched_filter(h, h, noise_variance=1)
```

The old `FTLCommunicationLink` name is retained for compatibility, but requires
an explicitly supplied hypothetical speed and cannot calculate a physical group
velocity. Existing communication routines encode sampled waveforms only.

## Connection to the anti-Helmholtz A experiment

The companion [magnetic-field-antihemholz](https://github.com/DavJ/magnetic-field-antihemholz)
repository computes the Coulomb-gauge A of opposed toroidal fluxes. The present
extension integrates finite rectangular cores and preserves linked circulation.
Its A-gradient has zero divergence outside the cores. A separate physical
Theta mode must be derived before it can have a nonzero additional source or
pickup response. See [the common source–propagator–detector bridge](UBT_ANALYSIS.md#5-sourcepropagatordetector-bridge).

## Conditional transmission between sectors

The [intersegment-channel derivation](docs/intersegment-channel.md) develops a
stable two-mode response, exact zero-coupling and charge-selection gates,
matrix Fourier overlaps and two-step mode conversion. It builds on the
existing upstream 137/139 research without identifying prime labels, Fourier
indices and physical universes. The transmitter/receiver gains and a physical
intersector coupling remain open.

```sh
python scripts/verify_segment_channel.py --check
```

## Research status and provenance

[SUMMARY.md](SUMMARY.md) is the short ledger; [the numerical/exact check record](docs/results/wave-bridge.json)
states tool versions and scope. `COMPLETION_REPORT.md` and
`DETECTION_AND_APPLICATIONS.md` retain superseded historical text under explicit
warnings. The GNU Radio tree remains historical and is not covered by the new
Python verification. Lean formalization and experimental validation are pending.

The 2026 revision is AI-assisted working research; exhaustive human review is
not claimed. [Research-origin record](docs/research-origin-2026-10-31.json).
Original code and earlier hypotheses retain their existing Git history.

## License

**GNU General Public License**

Files in this repository can be used based on GNU public license.

**Citation Requirement:** Whenever used for a derivative work (scientific,
commercial, or non-commercial), citation of the author is required.
