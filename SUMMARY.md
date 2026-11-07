# Current research ledger — 2026-11-07

The algebra and conditional effective wave toolkit have been improved. A new
physical propagation channel is not established. The old completion/FTL claims
are superseded by [UBT_ANALYSIS.md](UBT_ANALYSIS.md).

- Fixed: complex BQ inverse/exponential; zero divisors and nilpotents tested
  against independent matrices.
- Refuted for its stated ansatz: sqrt(2)*c from the real half of a complex
  mass shell; the missing imaginary residual is nonzero.
- Added: arbitrary-dimensional reduced-mode comparator, stability checks,
  characteristic speeds, source–detector response and a normalized noise test.
- Corrected: generator exponent conventions, central theta modular identity,
  periodicity check and misleading energy-conservation terminology.
- Companion A model: finite rectangular cores, flux normalization, linked
  circulation and a traceless central A-gradient. No new div(A) source.
- Added follow-up: an exact conditional sector response, Fourier harmonic
  selection, Schur backreaction and the neutral-operator charge obstruction.
  [Derivation](docs/intersegment-channel.md); physical neighboring worlds and
  a source/receiver map remain unestablished.
- New matching result: neutral mixing and a scalar-loop EM source/readout
  coefficient follow from one specified effective comparator. A left-phase
  neutrality counterexample blocks treating the old zero-mode argument as a
  physical embedding. [Derivation](docs/source-detector-matching.md).
- Open: final UBT action, nondegenerate on-shell background, complete constraint
  reduction, physical mode count, derived couplings and actual experimental data.

Run `python -m pytest -q` and `python scripts/verify_wave_bridge.py --check`.
[Recorded checks and versions](docs/results/wave-bridge.json).
The [achievement summary](README.md#what-we-achieved) links the checked Lean
results and their scope. AI-assisted working research. Exhaustive human review
is not claimed. No experimental result or author attestation is inferred
from passing software checks.

- Composite response: the free scalar pair gives a Bessel/Yukawa continuum;
  the matched density also has an ordinary two-photon tail proportional to
  r^-7. This is a standard-QED comparator result, not a new UBT pole or a
  nonzero fiber-channel identification. [English](docs/composite-response.en.md);
  [Czech](docs/composite-response.cs.md).
- Occupation follow-up: a prescribed Sauter electric pulse produces opposite-index
  scalar pairs in the declared comparator. Their random population supplies a
  calculated state-induced Drude response; no vacuum photon mass is assigned.
  Exact/gamma/Klein–Gordon controls agree. Psi-localized sources and a physical
  UBT spectrum remain open. [English](docs/occupation-channel.en.md);
  [Czech](docs/occupation-channel.cs.md).
- Localization follow-up: exact interval concentration distinguishes coherent
  packets from uniform diagonal populations. A separate real-circle comparator
  gives periodic attenuation, the uniform near-resonance term and a causal
  arrival front; finite mode truncation can fake nonlocal signals. It does not
  supply a UBT compact kinetic term or transport of the Sauter-created pairs.
  [English](docs/compact-locality.en.md); [Czech](docs/compact-locality.cs.md).

- Electromagnetic coupling and radar echo: the Maxwell stress pulls back to
  the tetrad jet, with zero source in the flat frozen-connection vacuum control.
  A separate point-scattering comparator supplies a causal round-trip echo
  and a transmission-line calibration analog. Physical UBT port and target
  couplings remain open. [English](docs/electromagnetic-theta-coupling.en.md) ·
  [Czech](docs/electromagnetic-theta-coupling.cs.md).
- Radar localization: noiseless first returns at a cross of ports determine a
  target on the real rectangular torus. A uniform port pair is position blind
  on a translation-invariant background. Calibrated mixed harmonics carry the
  angle. [Lean geometry](HyperspaceWaves/CompactGeometry.lean) and
  [Lean phase identities](HyperspaceWaves/Phase.lean).
- Compact conversion: a specified scalar layer supplies longitudinal momentum
  for emission and readout in the 5D Einstein–Maxwell comparator. The compact
  pump is an input. [English](docs/compact-radiation-layer.en.md) ·
  [Czech](docs/compact-radiation-layer.cs.md).
- Single radar return: a passive finite-profile tidal oscillator closes the
  scattering calculation between isolated free-wave ports. The two-frequency
  ratio recovers distance and compact angle without the bare target resonance.
  [English](docs/compact-radar-echo.en.md) · [Czech](docs/compact-radar-echo.cs.md).
- Repeated radar returns: rank-one scattering preserves the calibrated ratio
  after direct background subtraction, even with mode mixing at the station.
  A passive station with nonzero readout of the target pattern gives a
  convergent loop and preserves the net flux bound.
  [English](docs/compact-radar-feedback.en.md) · [Czech](docs/compact-radar-feedback.cs.md).
