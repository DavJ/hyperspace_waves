# Deriving a conditional transmitter/receiver channel between sectors

**Working research, 2026-10-31.** AI-assisted; exhaustive human review is not
claimed. Mathematical comparator only: neighboring multiverse segments and a
laboratory coupling to them have not been established. This note derives an
operational criterion, a stable example and blocking selection rules. It does
not derive or finalize a new fundamental UBT action. **LEAN-PENDING**.

## 1. What can a segment mean?

The current UBT snapshot is
[`3eb72538fca37c917b98f5925e430700999160aa`](https://github.com/UBT-Institute/unified-biquaternion-theory/commit/3eb72538fca37c917b98f5925e430700999160aa).
Its existing sources already distinguish several notions:

| Notion | Mathematical object | Missing identification |
|---|---|---|
| Fourier/winding index n | Basis mode on a compact circle | A mode is not automatically a universe |
| Hecke/prime label p | Arithmetic sector label | No automatic p=n map or physical distance |
| Imaginary-time position psi | Coordinate/parameter in a specified theory | Physical access is not supplied by analytic continuation |
| Decohered branch | Macroscopic quantum-state decomposition | Not automatically a compact Fourier sector |
| Physical subsystems A,B | Observable algebras or reduced physical-mode subspaces | Must be derived before a transmission claim |

[The branch-selection track](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/3eb72538fca37c917b98f5925e430700999160aa/research_tracks/complex_time_branch_selection/psi_branch_selection.en.md)
explicitly leaves its multiverse interpretation speculative. If psi only
parametrizes analytic continuation of a solution, values at different psi are
not independently specified transmitters or receivers. They are constrained
by the same analytic function and evolution. A separate physical sector map
is required. Ordering two prime numbers does not define geometric adjacency.

[The existing 137/139 bridge](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/3eb72538fca37c917b98f5925e430700999160aa/speculative_extensions/bridge_137_139/G137_139_metric_bridge.tex)
already studies first-harmonic mixing and a two-step path through 138 in a
restricted scalar/2D background. We use its existence as prior research, not a
proof of a physical source or receiver. Its scalar lift, charge consistency,
physical background and action matching remain open; its numerical amplitude
and universality claims are not independently endorsed by the checks here.

## 2. Operational definition of sending and receiving

Let O_A and O_B be **physical gauge-invariant observables** of the proposed
source and receiver. Apply an externally controlled perturbation
`delta H(t)=-J_A(t)*O_A`. To first order in J_A, unitary perturbation theory gives

```math
\delta\langle O_B(t)\rangle=\int dt'\,\chi^R_{BA}(t,t')J_A(t'),\quad
\chi^R_{BA}=\frac{i}{\hbar}\theta(t-t')\langle[O_B(t),O_A(t')]\rangle.
```

The sign corresponds to the displayed minus sign in delta H. Assume suitable
operator domains/correlations or regulated observables so the perturbative
integral exists. In a classical reduced field theory the corresponding object
is its retarded response Green operator. The condition for a controllable
linear channel is a nonzero response over a usable source/receiver band, with
calibrated gains and noise. A nonzero unobserved field coordinate alone is not
a transmitted message.

If the full evolution and **all allowed source/readout operations** preserve
independent blocks, and B has zero initial retarded perturbation, no drive
confined to A changes B: the cross response is zero. A mere correlation or
common initial condition is not this driven response. A real interaction can
produce ordinary causal signalling; the no-communication theorem for isolated
subsystems does not prohibit an interaction channel.

For a derived physical reduced Hessian D^R and source/readout maps s_A,d_B,

```math
H_{BA}=d_B^\dagger P_B(D^R)^{-1}P_A s_A.
```

P_A,P_B must project onto physical modes after gauge/Dirac reduction. The formula
is a projected response; the field Theta may be a matrix of biquaternions and
its total number of physical components is not fixed here. All composite
metric/connection variations belong in D^R. The field algebra alone does not
supply the projections or hardware gains.

## 3. Stable two-mode comparator and exact transfer

Take one canonically normalized real mode amplitude in each proposed physical
subspace. They are coordinates of selected perturbations, not a statement that
Theta is scalar. For zero initial perturbations consider the supplied effective
quadratic Lagrangian

```math
L=\frac12[\dot q_A^2+\dot q_B^2-\Omega_A^2q_A^2-\Omega_B^2q_B^2
-\eta(q_A-q_B)^2]+s_AJ_Aq_A,\quad\eta\geq0.
```

Eta has units s^-2; Omega has units s^-1. This positive spring term includes its
diagonal contributions. A naked off-diagonal term between two otherwise
massless modes would instead give a negative squared frequency. Add supplied
passive damping gamma_A,gamma_B>=0 to represent an accounted-for environment:

```math
\ddot q_A+2\gamma_A\dot q_A+(\Omega_A^2+\eta)q_A-\eta q_B=s_AJ_A,
\quad
\ddot q_B+2\gamma_B\dot q_B+(\Omega_B^2+\eta)q_B-\eta q_A=0.
```

For convention exp(-i*omega*t), set

```math
D_A=\Omega_A^2+\eta-\omega^2-2i\gamma_A\omega,\qquad
D_B=\Omega_B^2+\eta-\omega^2-2i\gamma_B\omega.
```

Inverting the full 2x2 matrix, with receiver `V_B=d_B*q_B`, gives

```math
\boxed{\frac{\delta V_B}{J_A}
=\frac{d_B\,\eta\,s_A}{D_AD_B-\eta^2}.}
```

This derives the transfer **from the stated effective comparator**, not eta or
the gains from UBT. Eta=0, s_A=0 or d_B=0 eliminates this channel. Resonance
changes an allowed amplitude; it does not defeat an exact zero selection rule.
Undamped poles require a proper initial-value/retarded distribution or specified
losses; a fitted imaginary regulator is not a physical damping derivation.

The energy and its balance are

```math
E=\frac12[\dot q_A^2+\dot q_B^2+\Omega_A^2q_A^2+\Omega_B^2q_B^2
+\eta(q_A-q_B)^2],\quad
\dot E=s_AJ_A\dot q_A-2\gamma_A\dot q_A^2-2\gamma_B\dot q_B^2.
```

Energy is supplied by the source, stored in the coupled modes and dissipated
through the environment. A finite transfer despite exponential damping is
perfectly consistent. In the symmetric case the normal frequencies are Omega
and sqrt(Omega^2+2*eta). The retarded cross impulse is one half the difference
of their damped oscillator Green functions. It vanishes before the drive; the
code checks it independently by integrating the full time-domain equations.
A lumped pair supplies no spatial or multiverse distance or front speed.

To receive a controlled message from B, B would need its own independently
controlled source and a reciprocal physical map back to an A detector. With
only local access to A, the measurable self-response is instead

```math
\frac{q_A}{s_AJ_A}=\frac{1}{D_A-\eta^2/D_B}.
```

Its extra term is an A->B->A backreaction of order eta^2 away from poles. An
extra local resonance can also be an ordinary hidden mode; it does not identify
another universe or a received reply. Eliminating a resonant intermediate
mode perturbatively is invalid; retain it in the full response.

## 4. Where a mixing coefficient could come from

Only as an additional **explicit conditional model**, suppose a compact
coordinate y of length 2*pi*R and a physical reduced perturbation q(x,y) have
actually been obtained from the single action. The separate symbol y avoids
silently identifying an independent spatial coordinate with imaginary time.
Let the compact operator be

```math
L_y=-\partial_y[C(y)\partial_y]+U(y),\quad
\phi_n(y)=e^{iny/R}/\sqrt{2\pi R}.
```

C,U may be matrices on any physical component space; use periodic regular
coefficients and fields, and a positive physical kinetic pairing. Integration
by parts gives the exact block matrix

```math
(L_y)_{mn}=\frac{mn}{R^2}C_{m-n}+U_{m-n},\quad
C_r=\frac1{2\pi}\int_0^{2\pi}d\theta\,e^{-ir\theta}C(R\theta).
```

U has the same Fourier convention. Hermiticity requires C_-r=C_r^dagger and
U_-r=U_r^dagger. Nonconstant kinetic/spatial/damping coefficients similarly give
blocks `-omega^2*K_(m-n)+k^2*Cx_(m-n)-2i*omega*Gamma_(m-n)`. This is a matrix
physical-mode calculation, not a scalar replacement of Theta.

Consequences:

- Uniform coefficients preserve n at linear order. No cross-mode conversion
  is created merely by changing the laboratory carrier frequency.
- A first harmonic has Delta n=+/-1 at first order. A direct 137->139 block
  vanishes at that order. A second harmonic can connect Delta n=2 directly.
- Two first-harmonic steps can connect 137->138->139 if those physical modes
  are present. Algebraic elimination gives
  `D_eff(139,137)=-D(139,138)*D(138,138)^(-1)*D(138,137)` when the direct block
  is zero. This is order epsilon^2 away from intermediate poles, with no
  assumption that the intermediate integer labels a stable universe.
- In a localized spatial experiment an outgoing target channel may additionally
  be below its propagation threshold. A forced oscillator response alone does
  not guarantee a travelling target wave.

A Fourier off-diagonal element in an arbitrary coordinate basis is not itself
an invariant physical transition. Transform the metric, basis, sources and
readout together; test whether the mixed **observable response** survives
constraints and coordinate changes. A finite Galerkin truncation is a numerical
comparator and does not integrate out all omitted modes.

Crucially, [the existing torus action gate](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/3eb72538fca37c917b98f5925e430700999160aa/research_tracks/information_first/TORUS_THETA_CANONICAL_ACTION_BRIDGE_GATE.cs.md)
states that the compact derivative term and its physical survival have not been
derived from the declared four-dimensional canonical family. The Fourier
calculation cannot supply a missing compact kinetic term by notation alone.

## 5. Source and charge selection can block the toroid proposal

For a source profile h(y), the overlap is `s_n=int dy phi_n^*(y)*h(y)`.
A normalized uniform profile drives only n=0. A profile with a particular
harmonic can address the corresponding Fourier mode, **if** the theory gives
the apparatus physical control of that profile. Shaping coils in ordinary
x,y,z does not automatically shape an independent compact coordinate.

The earlier [F² bridge](../UBT_ANALYSIS.md#5-sourcepropagatordetector-bridge)
provides a possible gauge-invariant operator to match, not that control map.
Some UBT sources discuss winding charge; the exact electrical interpretation
must be stated before using a winding change as inter-world transmission.
If the relevant total conserved charge obeys `Q|n>=q*n|n>` and the allowed
operator O is neutral, then

```math
[Q,O]=0\quad\Longrightarrow\quad
q(m-n)\langle m|O|n\rangle=0.
```

For distinct charges the matrix element vanishes. In particular F_munu F^munu
is neutral under electromagnetic U(1). If n is exactly electric charge, a
bare neutral F² interaction cannot directly change n; a charged background,
reservoir or other explicit degree of freedom must take up the compensating
charge. This condition refers to **total** charge: a charged subsystem can
exchange charge with a reservoir while total charge remains conserved.
If n is an independent neutral label, the electrical selection rule does not
apply automatically, but the nonzero mixed overlap still must be derived.

An unsupported charged condensate cannot be introduced just to bypass the
selection rule: its equations, energy, gauge symmetry and electromagnetic
consequences must be analyzed in that same action. The existing 137/139 bridge
already flags charge consistency as blocking. Nothing in an anti-Helmholtz
A-gradient by itself supplies the missing neutral-sector coupling.

## 6. What sending and receiving would require operationally

For a candidate that passes the gates above, sending means modulating a
physical J_A(t) inside the nonzero response band, with the source normalization
and energy budget derived. Receiving means measuring the specified O_B (or a
controlled return into O_A), with independently characterized noise and ordinary
cross-talk. Matched filtering can then test the chosen waveform; it cannot
establish the existence of the underlying second sector.

There is currently no derived value of R, eta, the source/detector gains, an
operating frequency or a receiver address in another universe. Assigning a
coil current or geometry without these maps would invent a device prescription.
The useful result is a precise equation and concrete zero/nonzero gates.

**Next physical calculation:** specify the actual sector projectors and a
nondegenerate on-shell background of one proposed UBT action, then compute its
full constrained Hessian and mixed electromagnetic source/readout derivatives.
Determine whether a neutral intersector block survives. A numerical oscillator
eta, a prime index difference, or an unexplained voltage residual is not that
calculation.

A subsequent [source/readout matching calculation](source-detector-matching.md)
derives the effective eta and scalar-loop EM coefficients from a specified
neutral/charged comparator. It audits the earlier left-phase neutrality
assumption and keeps the UBT representation/constraint embedding open.

## 7. Verification and sources

`segment_channel.py` implements the stable comparator, retarded symmetric
impulse, matrix Fourier selection and exact Schur elimination. Independent tests
compare with analytic inverses, continuous periodic integrals, time-domain ODEs
and a Fourier transform of the retarded impulse. `scripts/verify_segment_channel.py`
checks the transfer, self-energy, energy balance and Schur identities exactly
with SymPy and writes [the scoped result record](results/segment-channel.json).

```sh
python -m pytest -q
python scripts/verify_segment_channel.py --check
```

The CAS proves only the encoded finite algebra. Numerical examples do not prove
operator domains, a continuum causal cone, UBT constraints, physical worlds,
source/detector existence or hardware performance. No checked Lean proof exists
here. This follow-up leaves the previous Maxwell/G null and action gap ledger
unchanged. [Research-origin event](research-origin-2026-10-31-segment-channel.json).

The oscillator and normal-mode framework is standard; see the primary
[Feynman Lectures I.48](https://www.feynmanlectures.caltech.edu/I_48.html) and
[I.49](https://www.feynmanlectures.caltech.edu/I_49.html). Those sources support
the comparator mathematics, not its proposed multiverse interpretation.
