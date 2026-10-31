# From a specified action to mixing, electromagnetic source and readout

**2026-10-31 · AI-assisted working research; exhaustive human review is not
claimed.** The scalar-loop matching below is derived for an explicitly stated
effective comparator. Its embedding in the physical UBT spectrum remains
open. No adjacent universe, transmitter hardware or new physical discovery is
asserted. **LEAN-PENDING**.

**Česky:** V konkrétním efektivním modelu už nemusíme nezávisle hádat vazbu,
zdroj a přijímač. Vazba neutrálních módů plyne z druhé derivace jednoho
potenciálu. Virtuální nabité módy pak vytvářejí spočtenou vazbu h*F²; stejný
člen určuje buzení i elektromagnetický odečet. Zásadně ale zbývá dokázat, že
příslušné fyzikální módy a jejich nábojová reprezentace skutečně vyplývají z
jedné akce UBT. Starší argument, že reálná složka je neutrální, protože
komutuje s fází, při deklarovaném levém působení fáze neplatí.

## 1. Current UBT input and an exact representation boundary

The inspected snapshot is
[`9b24d4b41cb486c01f2bfc4002588c85f8e691cd`](https://github.com/UBT-Institute/unified-biquaternion-theory/commit/9b24d4b41cb486c01f2bfc4002588c85f8e691cd).
The [action registry](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/9b24d4b41cb486c01f2bfc4002588c85f8e691cd/canonical/ACTION.en.md)
still gives `DEFINED_FAMILY_NOT_FINALIZED`. The latest
[full-Theta real-linear graded jet](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/9b24d4b41cb486c01f2bfc4002588c85f8e691cd/research_tracks/action_selection/full_theta_reallinear_graded_jet_2026_10.en.md)
closes a restricted mathematical representation gate while explicitly keeping
physical central U(1), matter, the Hamiltonian quotient and spectator dynamics
open. Its Hermitian/anti-Hermitian split is not automatically an electric
neutral/charged decomposition.

There is an exact mismatch between two earlier effective-sector statements:

- [qed.tex](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/9b24d4b41cb486c01f2bfc4002588c85f8e691cd/canonical/interactions/qed.tex)
  declares the **left** action `Theta -> exp(i*alpha)*Theta` and
  `D=partial+i*e*A`.
- [step1_u1_protection.tex](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/9b24d4b41cb486c01f2bfc4002588c85f8e691cd/canonical/qed_phi_const/step1_u1_protection.tex)
  argues that a nonzero real/psi-zero-mode background is neutral because it
  commutes with the central phase, and tests **conjugation** by that phase.

For a nonzero real v these are different transformations:

```math
\left.\partial_\alpha(e^{i\alpha}v)\right|_{\alpha=0}=iv\ne0,
\qquad e^{i\alpha}ve^{-i\alpha}=v.
```

A psi-independent alpha leaves Fourier index n=0 unchanged while rotating its
electromagnetic phase. Therefore psi-zero-mode status and commutation alone do
not establish electric neutrality. This is a scoped counterexample to that
argument, not a statement that every possible UBT representation lacks a neutral
sector. Other claims or formulas in those sources are not endorsed here.

For a specified unitary effective representation with Hermitian charge
operator Q and canonical positive complex-scalar kinetic term, a constant
background v gives

```math
\mathcal L\supset e^2 A_\mu A^\mu\,v^\dagger Q^2v,
\qquad m_\gamma^2=2e^2\|Qv\|^2.
```

The factor two uses the Proca convention `m_gamma^2*A^2/2` and an unrescaled
complex background v. A neutral background obeys **Qv=0**. In the left-central
representation Q=I only v=0 satisfies this. In a representation with a genuine
kernel, nonzero neutral backgrounds are possible. Such a representation must be
stated and proved compatible with the full Theta geometry; it cannot be inferred
from a commuting scalar. Central conjugation alone acts trivially on the whole
algebra and cannot simultaneously charge its other components.

## 2. What minimal coupling can generate at tree level

Suppose h_a are actual neutral real physical perturbations and chi is a heavy
complex charged physical mode, with Q*h=0 and <chi>=0. These are **conditional
mode coordinates**, not a fixed component count or a scalar replacement of the
full possibly matrix-valued Theta. On a prescribed flat background, use canonical
positive kinetic terms, natural units hbar=c=1 and the effective comparator

```math
\mathcal L=-\tfrac14F^2+\tfrac12\partial_\mu h^T\partial^\mu h-V_0(h)
+(D_\mu\chi)^\dagger D^\mu\chi-\chi^\dagger\mathcal M^2(h)\chi
-\lambda_\chi|\chi|^4,\quad D_\mu=\partial_\mu+ieA_\mu Q.
```

Here lambda_chi>=0 may bound the full scalar potential. A multiplet of heavy
complex scalars is allowed. This is a declared ordinary effective action used
for matching, **not** another object called the fundamental UBT action.

At chi=0, minimal coupling with a constant photon kinetic coefficient supplies
no tree vertex linear in h and quadratic in A. Its seagull term is
`e^2*A^2*chi^dagger*Q^2*chi`, which is quadratic in charged fluctuations.
A nonzero neutral background does not produce a photon mass. This restricted
absence of a classical h*A^2 drive does not forbid pair production above a
charged threshold, other microscopic operators or coupling through geometry.

The possible bridge is that the charged fluctuation mass matrix depends on h.
The coefficient `partial_a M^2` is a mixed derivative of this same potential,
not a guessed detector gain.

## 3. A computed scalar-loop portal

Assume canonical positive charged kinetic terms, a positive Hermitian heavy
matrix M²(h), Hermitian Q, and `[Q,M²]=[Q,partial_a M²]=0` for neutral h.
External momenta and neutral masses must be small compared with the heavy mass
spectrum, the field must be weak `|e*q*F| << M²`, and perturbation theory must
apply. A derivative expansion around the specified background is required.
These are assumptions on the **physical reduced sector**, not properties
proved for the unreduced UBT pairing.

For one complex charged scalar the Euclidean determinant is
`Gamma_E=Tr log(-D_E²+M²)`. Its connection curvature is `Omega_munu=i*e*q*F_munu`.
The flat heat-kernel term `s²*Omega_munu*Omega_munu/12`, with `(4*pi*s)^-2`,
and a mass-independent subtraction at M0 give

```math
\Delta\Gamma_E\supset
-\frac{e^2q^2}{192\pi^2}\log\frac{M^2(h)}{M_0^2}\,F_E^2.
```

The sign follows from `log P=-int ds exp(-sP)/s` and
`Omega²=-e²q²F_E²`. Wick continuation to the stated Minkowski convention gives
`Delta L_M=+e²*q²*log(M²(h)/M0²)*F²/(192*pi²)`. Equivalently,

```math
\mathcal L_{\rm eff}=-\tfrac14 Z(h)F^2+\cdots,\quad
\Delta Z(h)=-\frac{e^2}{48\pi^2}\operatorname{Tr}
\left[Q^2\bigl(\log\mathcal M^2(h)-\log\mathcal M^2(0)\bigr)\right].
```

Differentiating the trace log, valid even if the mass derivative does not
commute with the mass inside equal-charge blocks, yields

```math
\boxed{\mathcal L_{\rm eff}\supset c_a h_a F^2,\qquad
c_a=\frac{e^2}{192\pi^2}\operatorname{Tr}
\left[Q^2(\mathcal M^2)^{-1}\partial_a\mathcal M^2\right],
\qquad Z_{,a}=-4c_a.}
```

For a single mode with `M²=M0²+kappa_a*h_a+...`, this is
`c_a=e²*q²*kappa_a/(192*pi²*M0²)`. Canonically normalized h has mass dimension
one; kappa has dimension one and c dimension minus one. Opposite charge signs
add through q² when their mass derivatives agree. Neutral heavy modes, zero
mass derivatives or cancellations among species give zero at this order.

This is a standard low-energy threshold mechanism, **not new physics by itself**.
The heat-kernel coefficient is in Vassilevich, eqs. (4.25),(4.28),
[hep-th/0306138](https://arxiv.org/abs/hep-th/0306138); the mass-derivative matching
method is reviewed by Kniehl and Spira,
[hep-ph/9505225](https://arxiv.org/abs/hep-ph/9505225). The explicit restricted
scalar formula and normalization above are derived here using those inputs.

An independently allowed microscopic h*F² Wilson term adds to this answer.
Other spins/multiplicities have different weights. This comparator assumes no
bare h*F² term in its renormalizable action; it does not prove that boundary
condition for UBT or determine its spectrum/statistics. The charged loop also
corrects V0, tadpoles and neutral kinetic terms. The example below uses a tree
neutral background/propagator with leading one-loop portal vertices. A
quantum-corrected vacuum and resonant resummation need additional work; large
loop mass shifts or near-pole perturbation theory cannot be ignored.

## 4. One explicit potential supplies mixing and source derivatives

To demonstrate a complete **effective** calculation, specify

```math
V_0(h_A,h_B)=\frac\lambda4(h_A^2+h_B^2-v^2)^2
+\frac\delta2(h_A-h_B)^2,
\qquad M_\chi^2(h)=m_0^2+y_A^2h_A^2+y_B^2h_B^2.
```

Require lambda,delta,v,m0²>0, yA²,yB²>=0. The h's are declared neutral,
chi charged, and <chi>=0. At `hA=hB=v/sqrt(2)` this potential has a global
tree minimum, Mchi²>0 and no charged-condensate photon mass. Its exact Hessian is

```math
H=\begin{pmatrix}\lambda v^2+\delta&\lambda v^2-\delta\\
\lambda v^2-\delta&\lambda v^2+\delta\end{pmatrix},\qquad
m_+^2=2\lambda v^2,\quad m_-^2=2\delta.
```

In this specified hA,hB basis the earlier mixing parameter is now **derived**:
`eta=-H_AB=delta-lambda*v²`. It need not be positive in a general stable basis;
the actual positivity test is H>0. The charged mass derivatives are
`kappa_A=sqrt(2)*v*yA²`, `kappa_B=sqrt(2)*v*yB²`, fixing cA,cB by the same formula.
The parameters of this effective potential remain inputs. No embedding of
this polynomial into the classified single-biquaternion Lorentz potential,
nondegenerate UBT tetrad or finalized UBT action is claimed.

The code's example numbers are **illustrative dimensionless scales**, not UBT
constants, laboratory frequencies or operating recommendations. It supports an
arbitrary charged mass matrix for matching; the two-mode example is not a
choice of the full Theta carrier dimension.

## 5. Source, readout and energy exchange from the same term

Normalize the constant photon kinetic term to Z0=1. Varying the same effective
action, with all fields retained, gives to leading order

```math
(\Box+H)_{ab}\,\delta h_b=c_aF_{\rm pump}^2,\qquad
\partial_\mu[(1-4c_a\delta h_a)F^{\mu\nu}]=J^\nu.
```

For a specified probe/background F_probe, the linear Maxwell readout obeys

```math
\partial_\mu f^{\mu\nu}=4\partial_\mu[(c_a\delta h_a)F_{\rm probe}^{\mu\nu}].
```

Thus source and readout are constrained by the same c_a. They are not two
independent fitted gains. A classical linear pickup with zero background/probe
field has zero readout at this order. Quantum h->two-photon decays and other
nonlinear readouts are separate processes. Obtaining pickup volts still requires
the Maxwell Green function, spatial mode overlaps, probe/lead geometry, loading
and calibrated material response. The code reports delta Z, **not volts**.

In the homogeneous modal comparator, with supplied passive damping Gamma,

```math
\frac{\delta Z}{F_{\rm pump}^2}
=-4c^T[H-\omega^2 I-2i\omega\Gamma]^{-1}c.
```

For source support only on mode A and readout only on B, the corresponding
projected term is `-4*cB*(D^-1)_BA*cA`. Actual spatial support and the meaning of
A/B must be derived; two neutral modes are not automatically different worlds.
Eta=0 eliminates the scalar cross term in that specified reduced operator,
while ordinary electromagnetic cross-talk and direct local modal response can
still remain. Radiative corrections can alter a tree tuning of H_AB.

Energy does not appear for free. With `L_EM=-Z(h)F²/4`, the EM energy balance
contains exchange `+(Zdot/4)*F²`; the scalar balance contains
`-(Zdot/4)*F²`. At linear order `Zdot=-4*c.hdot`, so the two exchanges cancel.
Drive work, boundary flux and any damping environment complete the account.
Positivity also requires Z>0 inside the weak-perturbation regime.

## 6. What this says about the toroid and an actual experiment

In natural units `F²=2(B²-E²)`. A single ideal vacuum plane wave has E²=B²
and therefore gives zero leading scalar source. A magnetically dominated
near field has nonzero F². This is the action-level reason to examine a
near-field toroid rather than identifying an arbitrary emitted waveform with
an additional physical wave.

For `B=Bdc+Bac*cos(omega*t)` in a fixed direction, neglecting the electric
contribution only where justified, F² has harmonic coefficients
`4*Bdc*Bac` at omega and `Bac²` at 2*omega. With zero bias only DC and 2*omega
remain. These are conditional source selection rules. A receive-side magnetic
bias/probe supplies the analogous linear readout factor. Actual material
nonlinearities, heating, ordinary induction and feedthrough must be modeled
before a residual can be assigned to this channel.

The existing [finite-core A calculation](https://github.com/DavJ/magnetic-field-antihemholz/pull/3)
provides magnetostatic geometry and ordinary flux/energy controls. Its external
Coulomb-gauge A-gradient is not itself this source. The total local F² and
physical mode profile matter; a static finite-core A solver alone gives no
outgoing scalar power. No static A gauge change can replace the loop portal.

## 7. What is now derived, and what remains genuinely missing

| Item | Result in the stated comparator | UBT boundary |
|---|---|---|
| Neutrality | Qv=0; left-phase commutation argument fails | Derive the physical charge representation |
| Neutral mixing | Hessian of one explicit V0, eta=delta-lambda*v² | Derive the physical neutral subspace and potential |
| EM source coefficient | Calculated scalar threshold c_a | Derive the charged physical spectrum and mass derivatives |
| EM readout | Same c_a and a specified probe/background | Derive spatial detector mapping and apparatus response |
| Energy | Source/scalar/probe exchange from one action | Full microscopic constraints and geometry remain open |
| Other multiverse segment | Not inferred | Establish a physical sector map independent of labels |

The remaining gap is now more precise than an arbitrary eta or detector gain:
**derive a physical neutral mode and a charged Hessian M²(h) from one consistent
single-Theta action, with the necessary symmetry and constraint reduction.**
Until that map exists, this is a verified mechanism available to a candidate,
not a claim that UBT contains it or that a multiverse link has been built.

## 8. Verification

`source_matching.py` checks positive heavy masses, unitary charges, neutral mass
derivatives, matrix-valued scalar matching and the reciprocal delta-Z response.
Tests compare the trace formula with an independently evaluated proper-time
integral, a Feynman-parameter integral, finite differences of a matrix log and
unitary basis changes. One potential's Hessian and mass derivatives are checked
independently by finite differences. Plane-wave and harmonic source controls
are included.

`scripts/verify_source_matching.py` verifies exact neutrality counterexamples,
the potential minimum/Hessian, tree source absence, heat-kernel prefactor,
source/readout energy exchange and harmonic coefficients with SymPy. Its
[JSON record](results/source-matching.json) pins source versions, assumptions,
tool versions and limits. No checked Lean proof, full one-loop diagrammatic
calculation, continuum UBT constraint proof or experimental validation is claimed.

```sh
python -m pytest -q
python scripts/verify_source_matching.py --check
```

[Research-origin event](research-origin-2026-10-31-source-matching.json).
