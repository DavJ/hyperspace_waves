# Hyperspace Waves — UBT research and verifiable wave models

**Author:** David Jaroš · [UBT](https://github.com/UBT-Institute/unified-biquaternion-theory)

This repository studies the original complex-frequency wave proposal alongside
biquaternion algebra, signal processing and conditional effective dynamics.
The [current derivation and gap ledger](UBT_ANALYSIS.md) supersedes the 2025
completion claims.

## What we achieved

| Result | What we established | Verification |
|---|---|---|
| Passive transfer between two modes | With positive gaps and nonnegative coupling, the quadratic potential is positive away from zero. The driven response follows by eliminating one mode away from a pole. | [Lean transfer proof](HyperspaceWaves/Transfer.lean) and [time-domain controls](docs/intersegment-channel.md). |
| Compact propagation | On a specified real circle, we derived the periodic attenuation kernel and the causal front. Finite Fourier truncation can produce an apparent response before that front. | [Analytic derivation and numerical controls](docs/compact-locality.en.md) ([česky](docs/compact-locality.cs.md)). |
| Radar localization in several dimensions | A reference port and two quarter-period ports per axis determine one target from noiseless first-return times at known speed on a flat rectangular torus. We derived the point-target transfer and computed the 2D scattered impulse with later wrapped paths in the first Born approximation. | [Lean range-uniqueness proof](HyperspaceWaves/CompactGeometry.lean), [Lean point-target response](HyperspaceWaves/Transfer.lean) and [radar derivation and simulation](docs/electromagnetic-theta-coupling.en.md#radar-on-a-real-torus). |
| Phase-sensitive localization | On a translation-invariant background, two uniform ports are blind to the position of a translated target. With phase-calibrated adjacent harmonics, the mixed response carries the angle. A simulation recovers positions that give identical same-mode responses. | [Lean phase identities](HyperspaceWaves/Phase.lean), [port derivation](docs/electromagnetic-theta-coupling.en.md#a-compact-phase-reference-for-radar) and [resonance example](docs/compact-locality.en.md#a-higher-resonance-enhances-a-position-dependent-return). |
| Electromagnetic drive and readout | The geometric Maxwell coupling vanishes for a compensated carrier shift that keeps the metric fixed. In the separate 5D Einstein–Maxwell model, a prepared compact pump and localized layer give a propagating metric mode and reciprocal electromagnetic readout. | [Lean cutoff and source algebra](HyperspaceWaves/Radiation.lean), [Maxwell variation](docs/compact-einstein-maxwell-port.en.md) and [direct scattering calculation](docs/compact-radiation-layer.en.md) ([česky](docs/compact-radiation-layer.cs.md)). |

**Physical status.** The real circle and torus are assumptions of the comparison
models. A physical UBT radar still needs a constrained propagating sector, a
way to prepare and detect its modes, and a measured target echo.

## Verify the results

With [Elan](https://lean-lang.org/doc/reference/latest/Build-Tools-and-Distribution/Managing-Toolchains-with-Elan/)
installed, run

```sh
lake exe cache get
lake build
```

The Lean and Mathlib revisions are pinned. GitHub Actions builds the proofs and
audits their transitive axioms against `propext`, `Classical.choice` and
`Quot.sound`. The Lean theorems cover the hypotheses and conclusions in their
statements. The linked derivations and Python checks cover the physical
comparators and numerical examples.

The Python examples run with Python 3.10 or later.

```sh
python -m pip install -r requirements.txt pytest sympy
python -m pytest -q
python scripts/verify_wave_bridge.py --check
python scripts/verify_compact_locality.py --radar-example-2d
python scripts/verify_compact_locality.py --radar-resonance
python scripts/verify_source_matching.py --compact-radiation
```

## Original waveform and signal-processing tools

The original samples are ordinary damped sinusoids:

```math
u(t)=\exp[2\pi f(s_1+i s_2)t],\qquad t\geq0.
```

Any negative `s1` gives a decaying half-line template. The historical value
`-1/sqrt(2)` is not selected by finite energy. Complex frequency alone supplies
neither a spatial dispersion law nor a new propagation channel.

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

The initial UBT audit is pinned to
[`386b6db`](https://github.com/UBT-Institute/unified-biquaternion-theory/commit/386b6dbed7d35ea63c8fd42fd88d57ac2f9cdaa3).
It still registers the fundamental action as `DEFINED_FAMILY_NOT_FINALIZED`.
Subsequent calculations link their own upstream audits and the open
[neutrality correction](https://github.com/UBT-Institute/unified-biquaternion-theory/pull/825).
No number of physical Theta components is assumed here; four complex
coefficients describe one algebra element, not the full physical field content.

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

## Matching a source and detector from one effective action

The [source/readout calculation](docs/source-detector-matching.md) audits the
left-phase neutrality assumption and derives a scalar-loop h*F² coefficient
from a positive charged mass matrix and its neutral-mode derivatives. One
explicit neutral effective potential supplies mixing and the required mass
slopes; the same portal fixes source and Maxwell readout. This is a conditional
ordinary EFT mechanism, not an established UBT spectrum or multiverse device.

[Electromagnetic coupling and conditional radar echo](docs/electromagnetic-theta-coupling.en.md)
([česky](docs/electromagnetic-theta-coupling.cs.md)) derives the Maxwell
stress coupling through the tetrad, its flat frozen-connection cancellation,
the point-target return on the real circle, and a transmission-line bench analog.
On a real rectangular torus, a cross of ports recovers a single target from
first-return times. The two-dimensional example also calculates the scattered
impulse in the first Born approximation.

```sh
python scripts/verify_source_matching.py --check
python scripts/verify_compact_locality.py --radar-example-2d
```

## Research status and provenance

[SUMMARY.md](SUMMARY.md) is the short ledger; [the numerical/exact check record](docs/results/wave-bridge.json)
states tool versions and scope. `COMPLETION_REPORT.md` and
`DETECTION_AND_APPLICATIONS.md` retain superseded historical text under explicit
warnings. The GNU Radio tree remains historical and is not covered by the new
Python verification. Experimental validation remains open.

The 2026 revision is AI-assisted working research; exhaustive human review is
not claimed. [Research-origin record](docs/research-origin-2026-10-31.json).
Original code and earlier hypotheses retain their existing Git history.

## License

**GNU General Public License**

Files in this repository can be used based on GNU public license.

**Citation Requirement:** Whenever used for a derivative work (scientific,
commercial, or non-commercial), citation of the author is required.

## Neutral-composite response follow-up

The [computed response](docs/composite-response.en.md) ([česky](docs/composite-response.cs.md))
now includes a general Gaussian contraction, a normalized scalar-pair continuum,
exact static Bessel kernel and the ordinary scalar-QED photon-pair power-law
tail. The tail prevents extending the free massive exponential to the full
electromagnetic comparator. Its physical UBT projection and nonzero fiber
harmonic coupling remain open. These calculations define no new universe or
independent neutral field.

Run `python scripts/verify_composite_response.py --check`.

## Opposite-index pair occupation

The [occupation calculation](docs/occupation-channel.en.md) ([česky](docs/occupation-channel.cs.md))
now works out an external electric pulse, scalar pair production, ideal count
statistics and a subsequent collisionless electromagnetic readout from the
same comparator action. Direct Klein–Gordon evolution and gamma-function
matching independently check the exact pulse spectrum. Pair occupation can
change while the total psi index remains zero. Uniform pump/readout profiles
do not localize a transmitter and receiver in different psi segments.

Run `python scripts/verify_occupation_channel.py --check`.

## Localizing a source or receiver

The [localization calculation](docs/compact-locality.en.md) ([česky](docs/compact-locality.cs.md))
quantifies the interharmonic coherence needed to concentrate an amplitude;
changing diagonal occupations alone leaves its mean density uniform. An
explicitly assumed real-circle comparator supplies an exact periodic Green
function and causal front. Without a gradient or other mixing, disjoint local
copies have zero transfer. Finite Fourier truncation can fake cross response
and pre-front signals. The real circle and compact kinetic term remain
additional assumptions; neither is derived from UBT imaginary time.

The higher-mode example recovers a point target from coherent returns at two
adjacent harmonics on the real circle. Run `python scripts/verify_compact_locality.py --radar-resonance`.

Run `python scripts/verify_compact_locality.py --check`.
