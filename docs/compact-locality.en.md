# Localization, interharmonic coherence and a conditional causal circle channel

<!-- BILINGUAL-UNIT: locality.scope -->

**DERIVED_WITH_ASSUMPTIONS · 2026-11-02 · working research.** We close a mathematical gap left by the pair-occupation calculation: populating Fourier modes is different from localizing a source, state or receiver. First we give an exact finite-mode localization bound, independent of any proposed UBT action. Then we calculate propagation only in an explicitly assumed real-circle reduced-mode comparator. This does not derive a compact UBT kinetic term, transport of the previously created scalar pairs, or a new pole of the composite density. The real coordinate y is not silently identified with the imaginary part psi of tau=t+i*psi. No Theta component count or new fundamental field is selected. AI-assisted; exhaustive human review is not claimed. `LEAN-PENDING`.

<!-- BILINGUAL-UNIT: locality.basis -->

## Fourier index and position are different bases

Let a single normalized complex amplitude use a specified finite set of harmonics on a circle of circumference L=2*pi*R. The 2*N+1 labels below count basis functions, not components of Theta. Each basis function has constant intensity everywhere. The positive L² norm is an explicit mathematical choice for a reduced physical amplitude; it is not a positivity proof for the indefinite full-Theta invariant Tr(Theta^sharp*Theta†). A physical application must first derive the positive constrained mode space.

```math
\phi_n(y)=\frac{e^{2\pi iny/L}}{\sqrt L},\qquad
u(y)=\sum_{n=-N}^Na_n\phi_n(y),\qquad \sum_n|a_n|^2=1.
```

<!-- BILINGUAL-UNIT: locality.populations -->

For a positive unit-trace one-body matrix rho, its local intensity is uniform whenever rho is diagonal in the harmonic basis. Increasing the total occupation then increases that intensity uniformly. Populating many nonzero n does not localize it. Off-diagonal coherences are needed for a nonuniform one-body density; higher-point pair correlations can still contain other information. An invariant state driven only by an invariant source remains invariant, consistently with the previous all-orders selection rule. Independent phase control or a localized operation is an extra physical resource that the coil has not been shown to provide along psi.

```math
I_\rho(y)=\sum_{m,n}\rho_{nm}\phi_m(y)^*\phi_n(y),\qquad
\rho_{nm}=p_n\delta_{nm}\ \Longrightarrow\ I_\rho(y)=1/L.
```

If a physical charge superselection rule restricts these coherences, a compensating physical reference or reservoir must also be derived. The geometric concentration bound does not grant the required operations.

<!-- BILINGUAL-UNIT: locality.concentration -->

## How many accessible harmonics can localize an amplitude?

For an interval of width w and center y_0, integrate the basis products exactly. The Hermitian concentration matrix C is positive and bounded by the unit matrix. The largest eigenvalue is the exact maximal fraction attainable inside the interval by a normalized amplitude in this mode space. Its trace gives a weaker necessary bound. The optimum requires a coherent eigenvector, not the incoherent populations with the same diagonal entries. Here sinc(x)=sin(pi*x)/(pi*x).

```math
C_{mn}=\int_{y_0-w/2}^{y_0+w/2}\phi_m^*(y)\phi_n(y)\,dy
=\frac wL e^{2\pi i(n-m)y_0/L}\operatorname{sinc}\!\left(\frac{(n-m)w}{L}\right),
\qquad P_w=a^\dagger Ca\le\lambda_{\max}(C)
\le\min\!\left(1,\operatorname{Tr}C\right)
=\min\!\left(1,(2N+1)\frac wL\right).
```

For example, for an interval covering 10% of a declared circle, the checked dimensionless maxima are below. Every diagonal mixture gives exactly 10%, including mixtures occupying all these modes. These fractions are geometric concentration, not transmission efficiency, detector gain or a probability of finding another universe. Restricting the accessible mode set can only reduce the available state space; the symmetric cutoff is a declared example, not an inferred UBT spectrum.

| Cutoff N | Harmonic count | Maximum coherent fraction | Diagonal mixture |
|---:|---:|---:|---:|
| 0 | 1 | 0.100000 | 0.100000 |
| 2 | 5 | 0.468928 | 0.100000 |
| 4 | 9 | 0.736606 | 0.100000 |
| 6 | 13 | 0.888716 | 0.100000 |
| 8 | 17 | 0.958567 | 0.100000 |

<!-- BILINGUAL-UNIT: locality.action -->

## Propagation requires a separate dynamical operator

To test what localization would buy if it were physically accessible, retain one neutral reduced coordinate q from the earlier general mode comparator and explicitly postulate the following quadratic action. This q is not identified with the composite density, nor with the full Theta. The real circle, positive speed parameter v, mass gap mu and linear source are additional assumptions. Ordinary spatial dependence is suppressed: this is a 1+1-dimensional propagation calculation, not a laboratory-distance calculation. Periodic boundaries give a positive energy and the displayed source work balance. A negative gradient coefficient in this unconstrained comparator would instead make sufficiently high harmonics unstable; it cannot be used merely to rescue attenuation.

```math
S=\int dt\int_0^Ldy\left[\frac12(q_t^2-v^2q_y^2-\mu^2q^2)+Jq\right],
\qquad (\partial_t^2-v^2\partial_y^2+\mu^2)q=J,
\quad\omega_n^2=\mu^2+v^2(2\pi n/L)^2,
\qquad E=\frac12\int_0^L(q_t^2+v^2q_y^2+\mu^2q^2)dy,
\quad\dot E=\int_0^LJq_t\,dy.
```

<!-- BILINGUAL-UNIT: locality.profiles -->

Use source and receiver profiles f_A,f_B with unit L¹ integral: J(t,y)=J_A(t)*f_A(y), O_B=int(f_B*q)dy. This fixes the normalization while changing profile widths; it is not a fitted apparatus gain or an energy budget for the source. For uniform coefficients, Fourier modes do not convert into one another, yet the position-space Green function may connect different positions. A diagonal Fourier operator therefore does not imply independent position segments. For centered top-hat profiles of widths w_A,w_B, the exact response is the infinite sum below; the implementation exposes its finite Galerkin cutoff. Top-hats are ideal external profiles, not finite-gradient-energy wave states.

```math
H_{BA}(\omega)=\int dy\,dy' f_B(y)G_\omega(y-y')f_A(y')
=\frac1L\sum_{n\in\mathbb Z}
\frac{e^{2\pi in(y_B-y_A)/L}\operatorname{sinc}(nw_A/L)\operatorname{sinc}(nw_B/L)}
{\mu^2-(\omega+i0)^2+v^2(2\pi n/L)^2}.
```

<!-- BILINGUAL-UNIT: locality.zero-gradient -->

## What happens without the compact gradient?

If v=0 and there is no other nonlocal or mixing term, the assumed circle describes independent local oscillators. The full infinite Fourier sum is a periodic delta function. Nonoverlapping source and receiver supports then give exactly zero transfer at all times, not an exponentially weak channel. For overlapping profiles one gets the contact term below. This is a statement about this local-copy comparator, not a no-go theorem for all UBT actions. Importantly, finite Fourier projection replaces the delta source by a nonlocal Dirichlet kernel and can fake a nonzero cross response even when the exact disjoint response is zero.

```math
v=0:\qquad G_\omega(y-y')=\frac{\delta_L(y-y')}{\mu^2-(\omega+i0)^2},
\qquad H_{BA}(\omega)=\frac{\int f_B(y)f_A(y)dy}{\mu^2-(\omega+i0)^2}.
```

<!-- BILINGUAL-UNIT: locality.green -->

## Exact periodic response below the gap

For v>0, mu>0 and real |omega|<mu, the point Green function can be obtained independently by solving the differential equation with a derivative jump, by summing images, or by summing all Fourier modes. Let d denote the geodesic distance in [0,L/2]. Its image form is numerically stable at large kappa*L. At distances short compared with the circle size, the usual evanescent exponential remains. This kernel belongs to the explicitly postulated q operator; inserting it for the previously calculated composite bubble would be unjustified.

```math
\Delta=\mu^2-\omega^2>0,\quad\kappa=\sqrt\Delta/v,
\qquad G_\omega(d)=\sum_{\ell\in\mathbb Z}\frac{e^{-\kappa|d+\ell L|}}{2v^2\kappa}
=\frac{\cosh[\kappa(L/2-d)]}{2v^2\kappa\sinh(\kappa L/2)},
\qquad G_\omega'(0^+)-G_\omega'(L^-)=-1/v^2,
\quad \int_0^LG_\omega(y)dy=1/\Delta.
```

The derivative jump is with respect to the periodic coordinate y on [0,L], with the geodesic distance understood as a function of y.

<!-- BILINGUAL-UNIT: locality.uniform -->

If either end is uniform along y, it retains only n=0 and the transfer loses all dependence on relative position. Such a source already assumes control distributed over the circle. It cannot be called a transmitter confined to one segment. Multiplying the calculated source/readout by an ordinary coil profile in x,y,z supplies no independent compact profile. The electromagnetic overlaps and actual compact control must be derived from the same constrained physical action.

```math
f_A=1/L\ \text{or}\ f_B=1/L
\quad\Longrightarrow\quad H_{BA}=\frac1{L[\mu^2-(\omega+i0)^2]}.
```

<!-- BILINGUAL-UNIT: locality.resonance -->

## Tuning to the lowest resonance amplifies the uniform component

Taking omega to mu from below reduces the inverse decay length but makes the dominant response spatially uniform. The point ratio G(d)/G(0) tends to one. After subtracting the uniform zero mode, a finite Bernoulli-polynomial kernel remains. A differential readout cancels the divergent uniform component and keeps a finite position-dependent response; it does not inherit the zero-mode resonance gain. This is not a proof that all addressing fails near resonance: finite-time arrival information and the nonzero modes still matter. Undamped resonance itself needs an initial-value treatment, not an infinite steady amplitude or an invented damping constant.

```math
G_\omega(d)=\frac1{L\Delta}
+\frac L{2v^2}\left[(d/L)^2-d/L+\frac16\right]+O(\Delta),
\qquad \lim_{\omega\to\mu^-}[G_\omega(0)-G_\omega(d)]
=\frac{d(L-d)}{2Lv^2},\qquad \lim_{\omega\to\mu^-}\frac{G_\omega(d)}{G_\omega(0)}=1.
```

<!-- BILINGUAL-UNIT: locality.modal-radar -->

## A higher resonance enhances a position-dependent return

Add uniform damping $\gamma>0$ to the reduced real-circle model. Put a stationary point stiffness $\lambda>0$ at $x$. The energy $E$ includes the positive defect energy $\lambda q(x)^2/2$. The work balance identifies the driven source $J$ as the energy input.

```math
[\partial_t^2+2\gamma\partial_t+\mu^2-v^2\partial_y^2+\lambda\delta_L(y-x)]q=J,
\qquad \dot E=\int_0^L J\,\partial_tq\,dy-2\gamma\int_0^L(\partial_tq)^2\,dy.
```

Assume four phase-calibrated drive and readout profiles, the normalized cosine and sine of each of two adjacent harmonics $n,n+1$, with $n\ge 1$. They are distributed over the circle and measured at the same drive frequency $\omega>0$ with convention $e^{-i\omega t}$. The factor $\sqrt{2/L}$ normalizes each profile. Their values at the target are the entries of $\sqrt{2/L}\,u_j$.

```math
\theta=2\pi x/L,\qquad u_j=(\cos j\theta,\sin j\theta)^\top,
\qquad \omega_j^2=\mu^2+(2\pi v j/L)^2,
\qquad \chi_j=(\omega_j^2-\omega^2-2i\gamma\omega)^{-1}.
```

Subtract the empty-circle transfer from the transfer with the target. To compute its $2\times2$ block $B_{ab}$ from drive harmonic $b$ to readout harmonic $a$, solve the field at the defect from $q(x)=q_0(x)-\lambda G_\omega(0)q(x)$ and project the scattered field onto the readout profiles. The coincident Green function includes all circle harmonics. Choose the square root with positive real part.

```math
\kappa=\frac{\sqrt{\mu^2-\omega^2-2i\gamma\omega}}{v},\qquad
G_\omega(0)=\frac{\coth(\kappa L/2)}{2v^2\kappa},\qquad
B_{ab}=-\frac{2\lambda}{L}\frac{\chi_a\chi_b}{1+\lambda G_\omega(0)}u_a u_b^\top.
```

The same-harmonic block $B_{nn}$ is proportional to $u_nu_n^\top$. It carries the double angle $2n\theta$, with indistinguishable positions spaced by $L/(2n)$. The cross-harmonic block $C$ gives the difference angle $(n+1)\theta-n\theta$. Its trace and antisymmetric part give the cosine and sine times one common complex factor. Divide by the trace of $B_{nn}$ and correct the ratio of the empty-circle susceptibilities to recover the target modulo $L$. The Frobenius norms also give the relative size of the mixed return.

```math
e^{i\theta}
=\frac{\chi_n}{\chi_{n+1}}
 \frac{C_{cc}+C_{ss}+i(C_{sc}-C_{cs})}{\operatorname{tr}B_{nn}},
\qquad C=B_{n+1,n},\qquad
\frac{\lVert C\rVert_F}{\lVert B_{nn}\rVert_F}
=\left|\frac{\chi_{n+1}}{\chi_n}\right|.
```

At the empty-circle natural frequency $\omega_n$, $|\chi_n|=1/(2\gamma\omega_n)$. For $|\lambda G_\omega(0)|\ll1$ and an isolated resonance pair, the same-harmonic echo scales as $\gamma^{-2}$ and the mixed echo as $\gamma^{-1}$. At fixed stiffness and sufficiently small damping, the term $G_\omega(0)\sim(2/L)\chi_n$ in the exact denominator changes the scaling. The same-harmonic norm grows as $|\chi_n|$, while the mixed norm approaches the finite $|\chi_{n+1}|$. The strongest return and the return that resolves the position can thus differ substantially.

For $L=12$, $v=\mu=1$, $\gamma=0.015$, $\lambda=0.01$ and $x=9.3$, the $n=1$ block has 18.4 times the amplitude at $\omega_1$ compared with the midpoint of $\mu$ and $\omega_1$. The mixed block at $\omega_1$ has 4.11% of the resonant block norm. Its phase distinguishes $x=9.3$ from $x=3.3$. All numbers are dimensionless. This calculation assumes steady, coherent, calibrated profiles and one point target. For a UBT application the compact propagation law and physical mode overlaps must be derived for the constrained biquaternionic dynamics.

Run `python scripts/verify_compact_locality.py --radar-resonance` for the example. The test suite also compares the closed form with a direct standing-wave matrix solve.

<!-- BILINGUAL-UNIT: locality.front -->

## A nonzero steady response is not instantaneous communication

For the same operator, the retarded image solution has an exact front. At finite time only finitely many images contribute. A point perturbation cannot reach a point receiver before the shortest travel time d/v. For finite profiles, the earliest allowed time is the distance between their supports divided by v. A uniform profile has support everywhere and therefore gives no positive travel-distance lower bound. Truncating the Fourier sum can give an apparent pre-front signal: that source has spatial tails and is no longer a truly local impulse. The code records an explicit counterexample and never labels the truncated response a faster physical channel. No identification of v with a lab speed along imaginary time is made.

```math
G_R(t,d)=\frac1{2v}\sum_{\ell\in\mathbb Z}
\theta\!\left(t-\frac{|d+\ell L|}{v}\right)
J_0\!\left(\mu\sqrt{t^2-\frac{(d+\ell L)^2}{v^2}}\right),
\qquad G_R(t,d)=0\quad(t<d/v).
\qquad \int_0^\infty e^{-st}G_R(t,d)dt
=G_{\Delta=\mu^2+s^2}(d),\quad s>0.
```

<!-- BILINGUAL-UNIT: locality.ubt -->

## The remaining UBT gate is specific

The current upstream torus-action gate explicitly states that the four-dimensional canonical family does not by itself supply a compact measure, a positive compact metric or a surviving D_psi kinetic term. A higher-dimensional candidate is listed but not selected as the completed fundamental action. The branch-selection track also distinguishes an independent real circle from analytic continuation and from decohered macroscopic branches. Our q action supplies none of these missing identifications. It does not alter the tetrad construction, locked metric, number of Theta components, or two-sided canonical derivative. To continue physically, derive the compact operator and source/readout maps on a nondegenerate on-shell background after constraints; then test whether the apparatus can create interharmonic coherences or localized observable couplings. Coil geometry alone, a prime label or a guessed R is insufficient.

<!-- BILINGUAL-UNIT: locality.verification -->

## Reproduction and limits

Independent checks include interval wavefunction quadrature, exact eigenvalue/trace bounds, dephasing, source/readout convolution, periodic image/Fourier sums, the differential equation and jump, zero-mode normalization, resonance subtraction, and a numerical Laplace transform of the causal image kernel with an explicit tail bound. The Bessel interior equation and Bernoulli limit are checked by CAS. These are mathematical/comparator checks, not physical UBT validation or an experimental result. English is the translation source; exhaustive human review is not claimed. `LEAN-PENDING`: lean/lake are unavailable; no compiled formal proof.

```sh
python -m pytest -q
python scripts/verify_compact_locality.py --check
```

<!-- BILINGUAL-UNIT: locality.sources -->

- [Previous occupation calculation](occupation-channel.en.md); [sector comparator](intersegment-channel.md); [composite response](composite-response.en.md).
- [UBT torus-action gate, pinned current master](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/68c73508e239bfbccb1d8bd9266fac0e76f1839e/research_tracks/information_first/TORUS_THETA_CANONICAL_ACTION_BRIDGE_GATE.cs.md).
- [UBT branch taxonomy and continuation limits, pinned current master](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/68c73508e239bfbccb1d8bd9266fac0e76f1839e/research_tracks/complex_time_branch_selection/psi_branch_selection.en.md).
- [Dereziński and Sikorski, arXiv:2406.08327, §5.3](https://arxiv.org/abs/2406.08327): primary reference for retarded Klein–Gordon Green functions and support; our circle periodization and speed normalization are derived above.
- [NIST DLMF 24.8.1](https://dlmf.nist.gov/24.8#E1): Bernoulli Fourier series for the zero-mode-subtracted limit.
- [Computed record](results/compact-locality.json); [research provenance](research-origin-2026-11-02-compact-locality.json).
