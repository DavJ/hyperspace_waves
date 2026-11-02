# Pumping opposite-index pairs and reading their occupation

<!-- BILINGUAL-UNIT: occupation.scope -->

**DERIVED_WITH_ASSUMPTIONS · 2026-11-01 · working research.** The previous selection rule forbids a nonzero coherent mean of a nonzero psi harmonic under a zero-harmonic source and an invariant state. It does not forbid a neutral pair occupation. Here we calculate one possible pump and a subsequent readout in the existing scalar-QED comparator. The result uses standard pair production, not a new wave or an established UBT channel. No physical Theta component count, mass spectrum, charge assignment, new fundamental field or neighboring universe is selected. Natural units and signature (+---) are used.

<!-- BILINGUAL-UNIT: occupation.action -->

## One action supplies both stages

Assume, conditionally, that constraint reduction has already produced canonical positive-energy complex modes phi_n with masses mu_n>0 and physical charges g_n=e*q_n. This reduction is not derived here. A psi-independent ordinary photon couples diagonally in n. Antiparticles of a given mode have the opposite electromagnetic charge and opposite conserved psi index to its particles. The index is a Fourier translation label, not a universe address. Different charges or masses must be derived before numerical UBT predictions can be made.

```math
\mathcal L=-\frac14F_{\mu\nu}F^{\mu\nu}
+\sum_n\left[(D_\mu\phi_n)^*D^\mu\phi_n-\mu_n^2|\phi_n|^2\right],
\qquad D_\mu=\partial_\mu+i g_n A_\mu.
\qquad \int_0^{2\pi R}\!d\psi\,e^{i(n-m)\psi/R}=2\pi R\,\delta_{nm}.
```

<!-- BILINGUAL-UNIT: occupation.selection -->

The pump can create a particle/antiparticle pair within a constituent mode even when n is nonzero: its total conserved index is zero. The pair creation operator commutes with the index generator, and both number operators are invariant. Thus an occupation changes without producing the forbidden one-point harmonic. The zero harmonic of a density contains all constituent n. Vacuum and squeezed pair correlations are different observables from the coherent harmonic mean.

```math
Q_\psi=n(N_a-N_b),\qquad [Q_\psi,a_n^\dagger b_n^\dagger]=0,
\qquad Q_\psi|k,k\rangle=0,
\qquad \langle H_m\rangle_{A_0}=0\ (m\ne0),\quad
\langle N_a\rangle=\langle N_b\rangle=N_{\boldsymbol p}.
```

<!-- BILINGUAL-UNIT: occupation.pulse -->

## A physical electric pulse, not a constant vector potential

For one specified mode suppress n. Take a prescribed homogeneous electric pulse along z in temporal gauge. This is a local idealization per unit spatial volume, not a finite Maxwell/coil solution. In the following A_z denotes the physical three-vector component; the covariant component is A_3=-A_z. A constant C is a gauge offset. Canonical momentum p_z must be relabeled together with C; kinetic momentum p_z-g*A_z is invariant. The time-dependent field does work. A constant pure-gauge A alone creates no pairs.

```math
A_z(t)=C-E_0T\tanh(t/T),\qquad E_z(t)=E_0\operatorname{sech}^2(t/T),
\qquad \ddot u_{\boldsymbol p}+\Omega_{\boldsymbol p}(t)^2u_{\boldsymbol p}=0,
\quad \Omega_{\boldsymbol p}^2=\mu^2+p_\perp^2+[p_z-g A_z(t)]^2.
\qquad C\mapsto C+\Delta C,\quad p_z\mapsto p_z+g\Delta C.
```

<!-- BILINGUAL-UNIT: occupation.bogoliubov -->

Specify the incoming vacuum by a positive-frequency unit-Wronskian mode. At late times split the same evolved mode into positive and negative outgoing frequencies. Their mixing gives the mean number of pairs in that momentum pair. N is an expectation, not the probability of detecting a pair or a number density. Integrating N over canonical momenta with measure d³p/(2*pi)³ gives the pair density. The outgoing kinetic momenta of the two members are opposite. The Wronskian is a numerical control, never a normalization imposed on an inaccurate integration.

```math
p_0=p_z-gC,\quad
\omega_{\rm in/out}=\sqrt{\mu^2+p_\perp^2+(p_0\mp gE_0T)^2},
\qquad u_{\rm in}\sim\frac{e^{-i\omega_{\rm in}t}}{\sqrt{2\omega_{\rm in}}},
\quad u\sim\frac{\alpha e^{-i\omega_{\rm out}t}+\beta e^{+i\omega_{\rm out}t}}{\sqrt{2\omega_{\rm out}}},
\qquad |\alpha|^2-|\beta|^2=1,\quad N_{\boldsymbol p}=|\beta|^2.
```

<!-- BILINGUAL-UNIT: occupation.hypergeometric -->

## Derivation and an independent exact solution

Set x=t/T, y=(1+tanh(x))/2 and u=y^a*(1-y)^b*v(y). The transformed Klein–Gordon equation is the hypergeometric equation below. Continuing its incoming solution from y=0 to y=1 yields beta. The ratio of gamma functions is checked independently against direct mode integration and the hyperbolic expression. The scalar result has a plus sign in the numerator; the spinor expression is different and is not substituted here. For imaginary lambda the cosine continuation is real and nonnegative after combining the terms.

```math
K=gE_0T^2,\quad a=-iT\omega_{\rm in}/2,\quad b=iT\omega_{\rm out}/2,
\quad\lambda=\sqrt{K^2-\tfrac14},\quad A,B=a+b+\tfrac12\pm i\lambda,\quad c=1+2a,
\qquad y(1-y)v''+[c-(A+B+1)y]v'-ABv=0,
\quad AB=(a+b)(a+b+1)+K^2,
\qquad N_{\boldsymbol p}=\frac{\omega_{\rm out}}{\omega_{\rm in}}
\left|\frac{\Gamma(c)\Gamma(2b)}{\Gamma(A)\Gamma(B)}\right|^2
=\frac{\cosh(2\pi\lambda)+\cosh[\pi T(\omega_{\rm out}-\omega_{\rm in})]}
{2\sinh(\pi T\omega_{\rm in})\sinh(\pi T\omega_{\rm out})}.
```

<!-- BILINGUAL-UNIT: occupation.limits -->

Several limits distinguish pair pumping from an arbitrary source gain. Neutral modes or zero E_0 give exactly zero production. At fixed nonzero central parallel momentum, weak-pump production starts at E_0². At p_0=0 it starts at E_0⁴ because the linear vector-potential perturbation vanishes there; deleting the A² term would give the wrong answer. The long-pulse central-mode limit tends to the usual scalar Schwinger exponent. There is no universal hard single-photon threshold for a prescribed strong time-dependent field: multiphoton/tunneling processes and the pulse spectrum matter. Pump energy, not an added neutral field, supplies the outgoing particles.

```math
\omega_0=\sqrt{\mu^2+p_\perp^2+p_0^2},\qquad
N_{\boldsymbol p}\sim\frac{\pi^2K^2p_0^2}{\omega_0^2\sinh^2(\pi T\omega_0)}\quad(p_0\ne0,\ E_0\to0),
\qquad N_{p_0=0}\sim\frac{\pi^2K^4}{\sinh^2(\pi T\sqrt{\mu^2+p_\perp^2})},
\qquad N_{p_0=0}\longrightarrow e^{-\pi(\mu^2+p_\perp^2)/|gE_0|}\quad(T\to\infty).
```

<!-- BILINGUAL-UNIT: occupation.statistics -->

## Information resides in a random pair count

For a vacuum input, each bosonic momentum pair is a two-mode squeezed state supported on |k,k>. Pair counts have the geometric distribution, equal arm counts and positive covariance. A single arm with independent detection efficiency eta and no dark counts has the click probability below. Even a nonzero N does not guarantee one click. These are ideal count statistics, not a receiver noise budget, a channel capacity or a demonstration of communication. Near the pulse the state also contains anomalous coherence proportional to \(\alpha\beta^*\); an occupation-only description discards that phase information only after dephasing or averaging.

```math
P(k)=\frac{N^k}{(1+N)^{k+1}},\quad k=0,1,\ldots,
\quad \operatorname{Var}k=\operatorname{Cov}(N_a,N_b)=N(1+N),
\qquad G(z)=\sum_kP(k)z^k=\frac1{1+N(1-z)},
\quad P_{\rm click,one\ arm}=1-G(1-\eta)=\frac{\eta N}{1+\eta N}.
```

<!-- BILINGUAL-UNIT: occupation.readout -->

## A readout coefficient follows from the charged-particle energy

After the pulse, consider the dephased out-particle distribution f(k), per species, in kinetic momentum k; for this pulse it is N evaluated at p_0=k_z-gE_0T. Each pair also has an antiparticle at -k with the same occupation. Approximate a soft, spatially homogeneous time-dependent probe by the collisionless kinetic response. Its frequency must be small compared with the particle energies/pair gap and large compared with relevant collision rates. Keep initial occupations fixed, subtract the background current, and work before significant depletion or field backreaction, or with an external current maintaining the background. This approximation does not describe coherent near-pulse readout or the full quantum polarization at arbitrary frequency and wavevector.

```math
E_{\boldsymbol k}=\sqrt{\mu^2+\boldsymbol k^2},\qquad
\frac{\partial^2E(\boldsymbol k-g\delta\boldsymbol A)}{\partial A_i\partial A_j}\bigg|_0
=g^2\left(\frac{\delta_{ij}}{E_{\boldsymbol k}}-\frac{k_i k_j}{E_{\boldsymbol k}^3}\right),
\qquad D_{ij}=2g^2\int\frac{d^3k}{(2\pi)^3}f(\boldsymbol k)
\left(\frac{\delta_{ij}}{E_{\boldsymbol k}}-\frac{k_i k_j}{E_{\boldsymbol k}^3}\right),
\quad \delta j_i=-D_{ij}\delta A_j,\qquad
\epsilon_{\rm out}=2\int\frac{d^3k}{(2\pi)^3}E_{\boldsymbol k}f(\boldsymbol k).
```

<!-- BILINGUAL-UNIT: occupation.medium -->

Both charges contribute with g², so their response adds despite vanishing net electric charge. The factor two counts the two members of each pair, not two assumed Theta components. D is positive for a nonzero positive population and mu>0, and vanishes for an empty or neutral population. At low kinetic momentum it becomes 2*g²*n_pair/mu times the unit matrix. In the Fourier convention exp(-i*omega*t), the ideal collisionless conductivity is i*D/(omega+i0). This is a state-dependent medium response with an initial condition, not a vacuum Proca mass. A simultaneous constant gauge offset and canonical-momentum relabeling changes no observable; a physical probe is the change of A relative to its initial value, with E=-dA/dt. The isolated distribution can have a background current even though its charge is zero.

<!-- BILINGUAL-UNIT: occupation.coil -->

## Relation to the anti-Helmholtz vector potential

The companion toroidal-coil model supplies a magnetostatic Coulomb-gauge A, fluxes and linked circulation. Those quantities do not specify this electric pump. Locally curl-free static A outside a core can affect a linked quantum phase through holonomy, but does not by itself create the pairs calculated here. Driving flux in time produces a Faraday field tied to the actual geometry, return currents and Maxwell boundary conditions. Opposed fluxes can cancel parts of that field. A coil waveform, its nonzero local E and its overlap with physical charged modes must be computed before replacing the ideal Sauter pump by the apparatus. A bare A-gradient or div(A) is not the missing coupling.

<!-- BILINGUAL-UNIT: occupation.segment -->

## What this permits, and what it leaves open

There is now a conditional pump → pair occupation → electromagnetic readout calculation from one comparator action. It changes the problem from a linear vacuum response to an energized excited state; it does not remove the previously computed vacuum attenuation. A zero-harmonic pump and readout are uniform in psi. Fourier modes are delocalized, and the invariant state's local mean density stays translation invariant. Changing N_n therefore supplies no addressed transmitter in one psi segment and receiver in the next. Such a channel still requires a physical spectrum, localized source/readout profiles, propagation/transport between them, and a nonzero measured transfer distinct from ordinary electromagnetic effects. No new physical effect beyond the declared comparator, faster front, coil power or multiverse range follows.

<!-- BILINGUAL-UNIT: occupation.verification -->

## Checks and reproduction

A subsequent [localization audit](compact-locality.en.md) quantifies the required interharmonic coherence and tests a separate causal real-circle comparator. It does not derive transport of these produced pairs or identify the real circle with imaginary time.

The code independently integrates the Klein–Gordon equation and checks its Wronskian against the hyperbolic spectrum and gamma-function connection. CAS checks the transformed differential equation, energy Hessian and count generating function. Tests cover both branches of lambda, its zero, weak-pump powers, long-pulse behavior, constant-gauge relabeling, readout positivity/rotations, finite-difference energy derivatives and explicit count sums. Examples are dimensionless individual modes or one phase-space cell per unit weight; no full integrated density or hardware parameters are reported. AI-assisted derivation and translation; English is the source. Exhaustive human review is not claimed. `LEAN-PENDING`: lean/lake are unavailable; no compiled formal proof or experiment is asserted.

```sh
python -m pytest -q
python scripts/verify_occupation_channel.py --check
```

<!-- BILINGUAL-UNIT: occupation.sources -->

- [Kim, Lee and Yoon, arXiv:0807.2696, scalar Eqs. 53–68 and squeezed-state construction](https://arxiv.org/abs/0807.2696). Our constant-offset and canonical-momentum convention differs from the paper's; the frequencies and resulting spectrum are explicitly matched above.
- [Previous composite response and selection rule](composite-response.en.md).
- [Common source/readout comparator](source-detector-matching.md).
- [Anti-Helmholtz companion](https://github.com/DavJ/magnetic-field-antihemholz).
- [Computed record](results/occupation-channel.json); [research provenance](research-origin-2026-11-01-occupation-channel.json).
