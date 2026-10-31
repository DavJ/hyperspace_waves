# Complex-frequency waves, physical Theta modes and the toroidal A experiment

**Working research revision: 2026-10-31.** AI-assisted derivation and code;
exhaustive human review is not claimed. This supersedes the November 2025
physical conclusions without assigning new historical priority. It does not
finalize the UBT action. Lean: **LEAN-PENDING**.

## 1. Algebra and the original waveform

Use a commuting complex unit i and quaternion units e1,e2,e3, with
`e1*e2=e3`, cyclic permutations, and `ej^2=-1`. One algebra element has the form
`q=w+x*e1+y*e2+z*e3`, with complex coefficients. This does not determine the
number of algebra-valued entries of the physical UBT field Theta.

The faithful representation used for independent checks is

```math
\rho(q)=\begin{pmatrix}w+ix&y+iz\\-y+iz&w-ix\end{pmatrix},\quad
D(q)=\det\rho(q)=w^2+x^2+y^2+z^2.
```

For `D(q)!=0`, `q^-1=q^sharp/D(q)`, where sharp reverses the quaternion-vector
part. The positive coefficient norm `|w|^2+|x|^2+|y|^2+|z|^2` is different and
is not multiplicative. In particular `q=1+i*e1` is nonzero but has `D(q)=0`.
For `v=x*e1+y*e2+z*e3` and `s=x^2+y^2+z^2`,

```math
\exp(w+v)=e^w\left[\cos\sqrt{s}+\frac{\sin\sqrt{s}}{\sqrt{s}}v\right].
```

The entire-function limit at `s=0` is `e^w(1+v)`, even for nonzero nilpotent v.
The old use of Hermitian absolute squares in these two operations was incorrect
for genuinely complex vector coefficients. Matrix multiplication, inverse and
`scipy.linalg.expm` provide independent numerical checks.

The actual generator uses

```math
u(t)=e^{2\pi f(s_1+i s_2)t}=e^{-\gamma t+i\omega_d t},\quad
\gamma=-2\pi f s_1,\quad\omega_d=2\pi f s_2.
```

The 2025 text inserted an extra i before `(s1+i*s2)` and therefore interchanged
the envelope and oscillation roles. For f>0 on t>=0,

```math
\int_0^\infty |u(t)|^2dt=\frac{1}{2\gamma}\quad(\gamma>0).
```

All `s1<0` work; finite norm selects no special ratio. A two-sided exponential
on the full real time axis cannot decay in both directions. The legacy
`energy_density/total_energy` methods return a coefficient norm/sample sum,
not a derived physical Hamiltonian or a conservation test. Complexifying time
or using biquaternionic time does not by itself specify the physical evolution
operator, real evolution slice, constraints or source coupling.

## 2. The missing imaginary mass-shell equation

For this **test-field calculation only**, take Minkowski signature (+---),
coordinates `(ct,x,y,z)`, inverse-length mass `mu=mc/hbar`, and a free field
`(Box+mu^2)Psi=0`. The UBT metric convention below is separate. A complex
four-wavevector `K=a+i*b` must satisfy the full complex equation

```math
K\cdot K=\mu^2
\quad\Longleftrightarrow\quad
a\cdot a-b\cdot b=\mu^2,\qquad a\cdot b=0.
```

The old substitution `a=(omega,k)`, `b=(omega/sqrt(2),0)` in units c=1 gives

```math
K\cdot K-\mu^2=\omega^2/2-|\mathbf k|^2-\mu^2
+i\sqrt{2}\,\omega^2.
```

Setting only the real part to zero does **not** produce a solution for nonzero
omega. For omega=1, mu=0, k=(1/sqrt(2),0,0), the exact residual is i*sqrt(2).
Consequently the old `omega^2/2=k^2+mu^2` is not a derived UBT dispersion law.
If separately postulated as a real PDE, it instead changes the characteristic
cone; this would be an additional physical assumption requiring independent
justification. Complex spatial momenta can describe evanescence subject to both
equations and boundary data; that does not imply transmission through every
material or faster information transport.

## 3. Conditional dynamics with the full physical mode space

Let Theta0 be an on-shell background with nondegenerate physical tetrad, and
let `q^A` denote local real coordinates for its **physical perturbations after
constraint reduction**. Their dimension is left unspecified. For smooth fields
with compact support or boundary conditions killing the integration-by-parts
terms, consider the homogeneous, isotropic constant-coefficient comparator

```math
S_2=\frac12\int dt\,d^3x\,[\dot q^T K\dot q
-\sum_i(\partial_iq)^T C(\partial_iq)-q^T Mq]+\int q^T J.
```

K,C,M are real symmetric. `K>0`, `C>=0`, `M>=0` are **assumptions on the reduced
sector**, not results about the raw UBT pairing. For an unreduced singular K,
perform Dirac/gauge reduction first; for an indefinite physical K, report the
energy obstruction. Never discard troublesome components or replace the UBT
pairing by a Euclidean norm to manufacture positivity.

Variation gives `K*q_tt-C*Delta(q)+M*q=J`. For J=0 and vanishing boundary flux,

```math
E=\frac12\int[\dot q^TK\dot q+\sum_i(\partial_iq)^TC(\partial_iq)+q^TMq]d^3x,
\quad \dot E=0.
```

The physical angular frequencies obey the generalized eigenproblem

```math
(k^2 C+M)u=\omega^2 Ku,\qquad
v_{\rm front}^2\in\operatorname{spec}(C,K).
```

For a differentiable positive-frequency eigenbranch with `u^T K u=1`,
Hellmann–Feynman differentiation gives

```math
v_g=\frac{d\omega}{dk}=\frac{k\,u^TCu}{\omega},\qquad
0\leq v_g^2\leq u^TCu\leq\lambda_{\max}(C,K).
```

The first inequality uses `omega^2=k^2*u^TCu+u^TMu` and M>=0. At degeneracy,
diagonalize C within the degenerate K-orthonormal eigenspace; zero-frequency
branch derivatives require limits. Thus `C<=c^2*K` bounds front and these
lossless group speeds by c for this sector. A finite cone follows by the
constant-coefficient symmetric energy/domain-of-dependence argument. This is
not a theorem for arbitrary dispersive, anisotropic, nonlocal or lossy media,
and group speed alone is not a universal information-speed test.

For a single open-system channel, passive damping gives

```math
q_{tt}+2\gamma q_t-v^2\Delta q+\Omega_0^2q=J,\quad
\omega_\pm=-i\gamma\pm\sqrt{v^2k^2+\Omega_0^2-\gamma^2}
```

with convention `exp(-i*omega*t)`. Both poles are in the lower half-plane for
nonnegative parameters, and `dE/dt=-2*gamma*int q_t^2`. The principal speed
remains v. The ordinary oscillatory ringdown is underdamped; for the historical
ratio and k=0, `Omega0^2=gamma^2+omega_d^2`, damping ratio `1/sqrt(3)`.
Its loss requires an environment/energy account; damping is not a closed-action
proof. No FTL claim follows from a decaying temporal envelope.

## 4. How this connects to current UBT

The inspected upstream snapshot is
[`386b6dbed7d35ea63c8fd42fd88d57ac2f9cdaa3`](https://github.com/UBT-Institute/unified-biquaternion-theory/commit/386b6dbed7d35ea63c8fd42fd88d57ac2f9cdaa3).
Read together:

- [canonical/ACTION.en.md](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/386b6dbed7d35ea63c8fd42fd88d57ac2f9cdaa3/canonical/ACTION.en.md):
  `DEFINED_FAMILY_NOT_FINALIZED`; reduced stability and gauge kinetic terms
  are not supplied by a bare appeal to a canonical action.
- [canonical/geometry/biquaternion_metric.tex](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/386b6dbed7d35ea63c8fd42fd88d57ac2f9cdaa3/canonical/geometry/biquaternion_metric.tex):
  on the classical Lorentz slice, `E_mu=N0^(-1/2)D_mu Theta` and
  `g_munu*1=(E_mu^sharp E_nu+E_nu^sharp E_mu)/2`, with signature (-+++).
  The full ordered product also has an antisymmetric noncentral part.
- [canonical/interactions/qed.tex](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/386b6dbed7d35ea63c8fd42fd88d57ac2f9cdaa3/canonical/interactions/qed.tex):
  a declared effective U(1) sector with local phase symmetry and conserved current.
  This does not derive its microscopic embedding or an extra scalar mode.

The 2025 symmetric `g_R+i*g_I` story is not the active canonical GR metric.
No trace, real-part projection, averaging, selected component count or new
independent propagating field is used to repair it here. In curved UBT the
representation and multiplication sides of `D_mu` must be stated, and every
composite metric/connection variation retained in the Hessian. Merely writing
a biquaternion-valued plane wave does none of this.

### Neutrality correction continued upstream

[Draft UBT PR #825](https://github.com/UBT-Institute/unified-biquaternion-theory/pull/825)
repairs the left-action/conjugation swap and continues the original-carrier
analysis. Its restricted real-linear Lorentz commutant theorem does not permit
a partial neutral/charged linear split of the declared single carrier.
A neutral invariant composite exists, but is indefinite; its physical
susceptibility, isolated pole if any and electromagnetic matching still need
to be derived. See the [source/readout follow-up](docs/source-detector-matching.md#upstream-repair-and-the-original-carrier-restriction).
The effective neutral/charged comparator is not promoted to a derived UBT
spectrum by this correction, and no inter-universe channel is established.

## 5. Source–propagator–detector bridge

With supplied passive symmetric damping Gamma>=0 and convention exp(-i*omega*t),
the conditional frequency-domain response is

```math
\delta V=d^T[M+k^2C-\omega^2K-2i\omega\Gamma]^{-1}s.
```

This Fourier-channel expression assumes translation invariance. A localized
experiment needs the retarded spatial Green operator and source/detector
integrals, with material boundaries and bandwidth. It yields a voltage only
when s,d and the q normalization have been derived in physical units. The
code leaves s,d explicit and has no arbitrary detector-gain default.
An absent source or detector coupling gives no additional response. An unknown
coupling is **unknown**, not evidence of either a measured zero or a signal.

In Maxwell theory, `G=c^-2*phi_t+div(A)` changes by `-Box(chi)` under a smooth
single-valued gauge transformation. Ordinary conserved source/probe currents
cannot create or measure a pure gauge as a new local excitation. The companion
[scalar-source gate](https://github.com/DavJ/magnetic-field-antihemholz/blob/9b35e3facf93a763b5dc921f908e706cc7c2cb0e/docs/scalar-source-gate.md)
already proves the conditional zero result for its specified frameworks and
retarded zero-scalar data. This revision does not reopen it by renaming G.

A constructive **conditional matching target**, rather than another fundamental
action, is the Taylor expansion of an action-derived gauge kinetic coefficient
for an already-existing physical scalar mode of Theta:

```math
\mathcal L_{\rm eff}\supset-\frac{Z(q)}{4\mu_0}F_{\mu\nu}F^{\mu\nu},\quad
Z(q)=Z_0+z_Aq^A+\cdots.
```

For this paragraph use the SI normalization `F^2=2(B^2-E^2/c^2)`. If this
term actually follows from UBT, its scalar source is

```math
s_A=-\frac{z_A}{4\mu_0}F^2
=\frac{z_A}{2}(\epsilon_0E^2-B^2/\mu_0).
```

The coefficients z_A, the existence of such a scalar mode, and its detector
coupling remain unselected; z_A may vanish. This operator is familiar in scalar
photon EFT and is **not** claimed as a new UBT discovery. Other derived spin,
parity or derivative channels need separate matching. In particular a linear
Maxwell readout from delta Z involves a background/probe field through
`partial_mu(delta Z * F_background^munu)`; an ordinary pickup at zero such
background does not acquire an arbitrary linear-in-q gain.

For disjoint ideal opposed toroidal cores, linear fluxes can cancel in a
symmetric pickup but B^2 contributions add. At zero DC bias a sinusoidal current
has only DC and 2*omega in B^2, not omega. A DC bias adds the omega cross term.
Simultaneously reversing DC and AC currents leaves B^2 unchanged while ordinary
linear induction changes sign. These are conditional channel selection rules,
not a predicted extra voltage. Joule heating, force and electronics also
produce even harmonics; matched backgrounds remain essential.

For an explicit follow-up on the proposed neighboring multiverse segment,
see [the conditional sector-channel derivation](docs/intersegment-channel.md).
It separates physical sector projectors from Fourier/prime labels, derives the
transfer and return response, and checks the blocking charge selection rule.
No physical adjacent universe or laboratory access map is inferred.

The [action-to-source/readout follow-up](docs/source-detector-matching.md)
now computes the conditional scalar-loop coefficient in terms of a charged
mass matrix and its neutral-mode derivatives. It also identifies the exact
left-phase/conjugation mismatch in the older neutrality argument. That
matching result supplies neither the physical UBT representation nor its
constrained Hessian.

## 6. Theta functions and measurement

For central complex tau with Im(tau)>0, the scalar Jacobi theta series is
well-defined and obeys

```math
\vartheta_3(z/\tau,-1/\tau)=\sqrt{-i\tau}\,e^{i\pi z^2/\tau}\vartheta_3(z,\tau).
```

The old determinant-norm prefactor omitted the z-dependent exponential and was
not this identity. The modular helper now supports only central complex
arguments and explicitly rejects the unproved noncentral case. The matrix
series accepts a sufficient convergence condition: the Hermitian imaginary
part of rho(tau) is positive definite. Scalar Im(tau.w)>0 alone is insufficient.
A truncated sum is not an automatic tail bound for all accepted parameters.

There is also a direct obstruction to the old proposed global reconstruction:
every `theta3(f*t,tau_n)` has real period 1/f. A convergent fixed-coefficient
sum is periodic, and cannot equal a nonzero decaying exponential on the entire
half-line. Existing Gaussian weights are a historical heuristic, not computed
expansion coefficients or a physical stability spectrum. Periodic-box momenta
`k_i=2*pi*n_i/L_i` follow from an **imposed boundary condition**, not from merely
using theta notation.

The mFFT kernel is an ordinary finite spiral transform. Exponential weighting
changes its noise covariance; comparing its raw peak to a plain FFT peak is
not a transmission test. The new matched filter uses a unit-norm template. For
one prespecified template and circular complex white noise with known variance,
`|h_hat^dagger*x|^2/sigma^2` has a unit-mean exponential null. Frequency/damping,
time-window and other scans require trials correction or independent null data.
None of these waveform statistics establishes a new propagation mechanism.

## 7. Verification, remaining gaps and next calculation

`python -m pytest -q` compares BQ operations and theta sums with independent
matrix algorithms, eigenbranch slopes with finite differences, and noise-null
statistics with seeded independent samples. `scripts/verify_wave_bridge.py`
checks exact mass-shell/envelope/energy identities in SymPy and records
versions, source SHAs, assumptions and exclusions in
[docs/results/wave-bridge.json](docs/results/wave-bridge.json).

Analytical energy and domain-of-dependence statements are not proved by the
finite numerical examples. No checked Lean proof exists here; the exact scope
is algebraic CAS checks plus numerical controls. The new CI repeats those checks
on supported Python versions; it does not test the historical GNU Radio build.

| Gate | Current state | Required next evidence |
|---|---|---|
| Complex BQ arithmetic | Corrected and independently tested | Broader numerical conditioning if required |
| Original sqrt(2) argument | Refuted for the stated ansatz | A different fully specified, consistent action if proposed |
| Physical Theta spectrum | OPEN | Complete constrained Hessian about a nondegenerate on-shell background |
| Positive kinetic energy and causal cone | CONDITIONAL comparator | Derive K,C,M and reduce constraints from that same action |
| Extra toroidal source and pickup | OPEN matching; standard G null unchanged | Derive s,d with units, energy and gauge identities |
| Barrier transmission | NOT PREDICTED | Derived interaction and boundary-value calculation |
| FTL or retrocausal information | NOT ESTABLISHED | A consistent physical model and discriminating evidence |

**Next smallest informative calculation:** for an already proposed upstream
single-Theta action and background, derive the constrained quadratic operator
and its mixed electromagnetic derivatives from the same functional. Test whether
any physical eigenvector has a nonzero source/detector overlap. Do not first fit
an unexplained pickup residual and then invent a mode for it.

Primary background references: [NIST DLMF, theta transformations](https://dlmf.nist.gov/20.7.E32),
[Tong, classical field theory](https://www.damtp.cam.ac.uk/user/tong/qft/qfthtml/S1.html),
[Tong, gauge redundancy](https://www.damtp.cam.ac.uk/user/tong/qft/qfthtml/S6.html),
[Prokopec, scalar/photon effective operators, arXiv:astro-ph/0106247](https://arxiv.org/abs/astro-ph/0106247).
These supply standard frameworks, not validation of UBT-specific claims.
