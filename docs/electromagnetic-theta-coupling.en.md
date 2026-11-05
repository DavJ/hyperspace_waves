# Electromagnetic coupling to a Theta perturbation

**2026-11-04 · working research.** AI-assisted derivation and translation. Exhaustive human review is not claimed.

<!-- BILINGUAL-UNIT: em-coupling.action -->

Use natural units, signature \((-+++)\) and the Maxwell term written in the UBT QED sector. During the metric variation hold the covariant two-form \(F_{\mu\nu}\) fixed. The canonical geometric derivative defines \(E_\mu=N_0^{-1/2}D_\mu\Theta\), with fixed \(N_0\). The bilinear pairing below uses complex-linear quaternion sharp. The canonical metric and its admissible variations are real and central. The derivation uses the full complex biquaternion carrier subject to that reality condition. The UBT action registry records the full action as \(\texttt{DEFINED\_FAMILY\_NOT\_FINALIZED}\).

~~~math
S_{\rm EM}=-\frac14\int d^4x\,\sqrt{-g}\,F_{\mu\nu}F^{\mu\nu},
\qquad b_\sharp(X,Y)\mathbf1=\frac{X^\sharp Y+Y^\sharp X}{2},
\qquad g_{\mu\nu}=b_\sharp(E_\mu,E_\nu).
~~~

[Canonical geometry](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/ef3e52c3438012acf9010bc43afbe0802f4c4f2d/canonical/CANONICAL_DEFINITIONS.md#L167) · [Curved QED term](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/ef3e52c3438012acf9010bc43afbe0802f4c4f2d/canonical/interactions/qed.tex#L259) · [Action registry](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/ef3e52c3438012acf9010bc43afbe0802f4c4f2d/canonical/ACTION.en.md)

<!-- BILINGUAL-UNIT: em-coupling.variation -->

Write \(h_{\mu\nu}=\delta g_{\mu\nu}\), with raised indices and traces evaluated on the background. The inverse-metric and volume variations give the stress tensor and its pullback to the tetrad jet. The variation of \(D_\nu\Theta\) includes the variation of the geometric connection. Any reduced mode must use the variation allowed by the coupled constraints.

~~~math
\begin{aligned}
\delta g^{\mu\nu}&=-h^{\mu\nu},&
\delta\sqrt{-g}&=\tfrac12\sqrt{-\bar g}\,h^\rho{}_\rho,\\
T_{\rm EM}^{\mu\nu}&=F^\mu{}_\rho F^{\nu\rho}-\tfrac14\bar g^{\mu\nu}F^2,&
h_{\mu\nu}&=b_\sharp(\delta E_\mu,\bar E_\nu)+b_\sharp(\bar E_\mu,\delta E_\nu),\\
\delta S_{\rm EM}&=\tfrac12\int d^4x\,\sqrt{-\bar g}\,T_{\rm EM}^{\mu\nu}h_{\mu\nu}
=\int d^4x\,\sqrt{-\bar g}\,T_{\rm EM}^{\mu\nu}b_\sharp(\bar E_\mu,\delta E_\nu),&
\delta E_\nu&=N_0^{-1/2}\delta(D_\nu\Theta).
\end{aligned}
~~~

If a canonically normalized physical coordinate \(u\) has a local algebraic metric polarization \(h_{\mu\nu}=C_{\mu\nu}u\), its geometric source is \(J_u=C_{\mu\nu}T_{\rm EM}^{\mu\nu}/2\). For a differential polarization the source is the adjoint of the linearized metric map applied to \(T_{\rm EM}/2\), with the appropriate measure and boundary conditions. A magnetic field along \(z\) gives \(J_u=B_z^2(C_{00}+C_{xx}+C_{yy}-C_{zz})/4\) in a local inertial frame. Conformal polarizations give zero because \(T^\mu{}_\mu=0\) in four classical Maxwell dimensions.

<!-- BILINGUAL-UNIT: em-coupling.flat -->

Take an affine flat tetrad and freeze its geometric connection at zero as a control. In a constant Lorentz tetrad basis, write \(\delta\Theta/\sqrt{N_0}=\xi^a b_a\) with complex coefficients. On admissible perturbations the symmetric gradient is already real. Its imaginary part vanishes by the metric reality condition. Integrate by parts for compactly supported variations, or impose vanishing boundary terms. Source-free Maxwell fields on shell have conserved stress, hence this geometric source vanishes in the stated control. Regions containing currents require the field and source stresses together for an apparatus calculation.

~~~math
h_{\mu\nu}=2\partial_{(\mu}\operatorname{Re}\xi_{\nu)},\qquad
\delta S_{\rm EM}=-\int d^4x\,(\operatorname{Re}\xi_\nu)\,\partial_\mu T_{\rm EM}^{\mu\nu}=0,
\qquad\partial_\mu T_{\rm EM}^{\mu\nu}=0.
~~~

Here the last integral uses the real part of \(\xi\), which generates the real metric variation. The reduced coupling in a dynamical geometry requires its own connection and constraint calculation. In particular, assigning the declared left electromagnetic phase to the same covariant tetrad jet would give \(g_{\mu\nu}\mapsto e^{2i\alpha}g_{\mu\nu}\) by complex linearity of sharp. Compatibility of the physical charge action with the real metric is therefore a separate condition.

<!-- BILINGUAL-UNIT: em-coupling.loop -->

The scalar source and readout in the apparatus sketch use the previously computed low-energy matching model. Assume a physical neutral \(u\), a positive canonically normalized heavy complex scalar with electric charge \(eq\) and \(M^2(u)=M_0^2+\kappa u+\cdots\), a soft weak electromagnetic field, and zero independent bare \(uF^2\) coefficient. In the Euclidean heat kernel the connection curvature is \(ieqF_E\) and its quadratic term is \(s^2\Omega^2/12\). The complex scalar determinant and a subtraction at \(M_0\) yield the following Minkowski term.

~~~math
\Delta\mathcal L=\frac{e^2q^2}{192\pi^2}\log\!\frac{M^2(u)}{M_0^2}\,F^2,
\qquad c=\frac{e^2q^2}{192\pi^2}\frac{\kappa}{M_0^2},
\qquad \mathcal L_{\rm EM,eff}=-\tfrac14(1-4cu)F^2+\cdots.
~~~

Thus \(J_u=cF_{\rm pump}^2\) on a fixed metric in this model. Linear readout around a specified probe field comes from varying \((1-4cu)F^{\mu\nu}\) in Maxwell's equation. The same \(c\) fixes both responses. For UBT the physical charge representation, stable mode \(u\), charged mass derivative \(\kappa\), and the source and receiver overlaps along \(\psi\) remain to be derived.

[Heat-kernel coefficient, Eq. 4.28](https://arxiv.org/abs/hep-th/0306138) · [Conditional matching and energy balance](source-detector-matching.md)

<!-- BILINGUAL-UNIT: em-coupling.apparatus -->

For the scalar model the pump depends on \(F^2=2(B^2-E^2)\). A single toroid with a time-dependent current is a candidate source of the ordinary fields in the model. A pair with opposed fluxes provides an optional comparison geometry. The magnetostatic finite-core calculation supplies the ordinary \(A\), flux and symmetry controls. Calculating the driven fields and the ordinary transfer to the probe requires a dynamic field or circuit model for the chosen setup.

~~~mermaid
flowchart TD
    Drive["Modulated current"] --> Coil{"Source geometry"}
    Coil --> Single["Single toroid"]
    Coil --> Pair["Toroids with opposed fluxes"]
    Single --> Fields["Physical E and B"]
    Pair --> Fields
    Fields --> Direct["Ordinary EM coupling"]
    Direct --> Probe["Probe and reference field"]
    Fields -. "Mode overlap open" .-> Mode["Physical Theta mode"]
    Mode -. "Transfer along psi open" .-> Readout["Conditional mode response"]
    Readout -. "Detector overlap open" .-> Probe
~~~

[Finite-core toroid baseline](https://github.com/DavJ/magnetic-field-antihemholz/blob/71e3a754e86db4d4a268e66f16a94f1d05a67342/docs/ubt-wave-bridge.md) · [Separate circle propagation model](compact-locality.en.md)

<!-- BILINGUAL-UNIT: em-coupling.radar -->

## Conditional radar echo

Radar needs a return from a target in addition to a transmitter and receiver. In the separate real-circle comparator place both ports at \(y=0\) and model the target by a stationary positive point stiffness \(\lambda\delta_L(y-d)\), with shortest-arc distance \(0<d\le L/2\). The positive stiffness adds nonnegative potential energy. For real Laplace parameter \(s>0\), \(G_0\) is the inverse of \(s^2+\mu^2-v^2\partial_y^2\) on that circle. Solving the field at the target and substituting it back gives the difference between the target-present and target-absent port responses.

~~~math
\Delta G(s;0,0)
=-\frac{\lambda\,G_0(s;0,d)G_0(s;d,0)}
 {1+\lambda G_0(s;d,d)}.
~~~

The two propagators set the earliest return at \(2d/v\), as in a conventional monostatic round trip. The denominator sums repeated interactions at the target. The first Born approximation replaces it by one. An actual port response would also contain both physical overlaps with the medium. On the circle later returns can take the other arc, and the first arrival determines only the shortest arc if \(v\) is known.

For \(\mu=0\) and \(0\le t<L/v\) only the shortest paths can contribute. Use the infinite-line transform \(G_0(s;0,d)=e^{-sd/v}/(2vs)\) and \(G_0(s;d,d)=1/(2vs)\) in the resolvent difference. Partial fractions and the time shift \(2d/v\) give the exact impulse difference below, with \(r_+=\max(r,0)\). This response has a gradual onset even for a point impulse. A finite transmitted pulse convolves this response with the pulse shape.

~~~math
\Delta G_R(t;0,0)
=-\frac1{2v}\left(1-e^{-\lambda(t-2d/v)_+/(2v)}\right),
\qquad 0\le t<L/v.
~~~

The reproducible example uses dimensionless \(L=12\), \(v=1\), \(d=2\), \(\lambda=1/4\) and \(\mu=0\). It gives onset \(t=4\) and impulse difference \(-0.1105996\) at \(t=6\). This is a virtual test of echo extraction. Hardware calibration can measure conventional feedthrough and returns from a known target in ordinary space. A prediction for UBT needs the actual propagator, the two port couplings and the target scattering operator. An echo unexplained by the calibration would need those predictions to identify its origin.

[Conventional round-trip timing](https://www.ll.mit.edu/media/6941) · [Circle propagator and causal front](compact-locality.en.md)

<!-- BILINGUAL-UNIT: em-coupling.radar-2d -->

## Radar on a real torus

Use a flat rectangular torus with \(N\) periodic coordinates of periods \(L_i\) and a known common wave speed \(v\). Each port transmits and receives at the same position \(p\). For one stationary target at \(x\), the earliest return time \(T(p)\) gives its shortest round-trip distance, provided the port couples to that path and the first return is resolved. Write \(\delta_{L_i}\) for the shortest distance around coordinate \(i\).

~~~math
r_p^2=\left(\frac{vT(p)}{2}\right)^2
      =\sum_{i=1}^N \delta_{L_i}(x_i-p_i)^2,
\qquad \delta_L(z)=\min_{k\in\mathbb Z}|z+kL|.
~~~

One port gives a distance contour. Put a reference port at the origin and two ports at \(\pm a_i e_i\) on each axis, where \(a_i=L_i/4\). The differences \(r_\pm^2-r_0^2\) cancel every coordinate except \(i\). Set \(u=x_i/a_i\) using the representative in \([-2,2)\), and define \(P=(r_-^2-r_+^2)/a_i^2\) and \(Q=(r_+^2+r_-^2-2r_0^2)/a_i^2\) for that axis. Evaluating the shortest distances gives

~~~math
(P,Q)=
\begin{cases}
(4u,2), & |u|\le 1,\\
(\operatorname{sgn}(u)(8-4|u|),\,10-8|u|), & 1<|u|\le 2.
\end{cases}
~~~

For \(Q=2\), recover \(u=P/4\). For \(Q<2\), recover \(|u|=(10-Q)/8\) and use the sign of \(P\). At \(P=0,Q=-6\), both signs represent the same point modulo the period. This proves that \(2N+1\) ports at the stated locations suffice in this model. It assumes noiseless times, known port positions and speed, and a single target with correctly associated returns. In two dimensions, \(L=(12,8)\) and \(v=1\) give a reference-port return at \(2\sqrt{13}\approx7.2111\) for the target \((10,5)\). The five times together recover \((10,5)\) modulo the periods. All quantities in this example are dimensionless.

For the two-dimensional massless wave equation, sum the retarded Green function on the covering plane over the periodic images. With travel times \(a_\ell=|x-p_T+\ell L|/v\), \(b_k=|x-p_R+kL|/v\) and positive point stiffness \(\lambda\), first-order scattering gives the following impulse difference. Here \(*\) is causal time convolution and the displayed value of \(g_a\) is for \(t>a\), with zero before the front.

~~~math
g_a(t)=\frac{1}{2\pi v^2\sqrt{t^2-a^2}}\quad(t>a),
\qquad
\Delta G_R^{(1)}(t)=-\lambda\sum_{\ell,k\in\mathbb Z^2}(g_{a_\ell}*g_{b_k})(t).
~~~

Only pairs with \(a_\ell+b_k<t\) contribute away from the fronts. For positive leg times \(a,b\), set \(D=t-a-b>0\). Substituting the integration variable \(s=a+D\sin^2\theta\) into the convolution reduces it to the complete elliptic integral \(K\), with parameter \(m\) (modulus \(\sqrt m\)).

~~~math
(g_a*g_b)(t)=\frac{2K(m)}{(2\pi v^2)^2\sqrt{(D+2a)(D+2b)}},
\qquad m=1-\frac{4ab}{(D+2a)(D+2b)}.
~~~

The earliest echo arrives at the sum of the two shortest leg times. The impulse response has a tail behind each front, and later paths can wrap around either period. Repeated scattering at a point in two dimensions would require a finite target model or a regularization of the divergent same-point Green function. A UBT application requires an independently specified propagation law, target interaction, and physical couplings at the port locations. The coordinates here are real model coordinates. The localization result requires independent port positions in those coordinates.

Run `python scripts/verify_compact_locality.py --radar-example-2d` for the five times, recovered coordinates, and sampled Born impulse difference. The tests also compare the convolution with direct quadrature and cover a case with multiple shortest paths.

[Retarded Green functions by dimension](https://arxiv.org/abs/1701.00776) · [Complete elliptic integral](https://dlmf.nist.gov/19.2.E8)

<!-- BILINGUAL-UNIT: em-coupling.radar-reference -->

## A compact phase reference for radar

Let $\varphi$ be an angle of period $2\pi$ in a conditional compact model, with $\varphi=\psi/R_\psi$ only if that identification is physical. Work on a space of perturbations after constraint reduction, with translation $(T_\alpha f)(\varphi)=f(\varphi-\alpha)$ acting on the entire physical space, including any internal components. Suppose the background and the target-free operator $D_0$ are translation invariant. A single target at $\alpha$ has operator $T_\alpha V_{\rm origin}T_\alpha^{-1}$. The response difference $\Delta R_\alpha=D_\alpha^{-1}-D_0^{-1}$ uses retarded inverses.

~~~math
D_\alpha=D_0+T_\alpha V_{\rm origin}T_\alpha^{-1},
\qquad [D_0,T_\alpha]=0,\qquad
\Delta R_\alpha=T_\alpha\Delta R_{\rm origin}T_\alpha^{-1}.
~~~

For source $s_k$ and receiver $d_\ell$ in harmonics $k$ and $\ell$, use $T_\alpha f_k=e^{-ik\alpha} f_k$. All dependence on target position is in the phase associated with the harmonic difference.

~~~math
\Delta H_{\ell k}(\alpha)
=d_\ell^\dagger\Delta R_\alpha s_k
=e^{-i(\ell-k)\alpha}\Delta H_{\ell k}(0).
\qquad \Delta H_{00}(\alpha)=\Delta H_{00}(0).
~~~

Two uniform ports may detect a target through $\Delta H_{00}$, but every shifted position produces the same return. The identity includes repeated scattering and arbitrary mixing of internal components at the target. A time-dependent pump that preserves compact translation symmetry also leaves this covariance intact for the two-time retarded response. Increasing a resonance in such a setup can increase the measured amplitude while retaining the same position ambiguity.

A calibrated nonzero channel with $\ell-k=1$ yields the angle modulo $2\pi$ from its complex phase relative to a known origin response. With separately measured and calibrated nonzero differences $q$, the common remaining period is $2\pi/\gcd(|q|)$. This follows by equating their phase factors, or by taking integer combinations of the phases. The statement uses phase-resolved channels. A cosine alone still has a reflection ambiguity.

### A full-matrix candidate and its gauge test

In an explicitly assumed portal $-ZF_{\mu\nu}F^{\mu\nu}/4$ averaged over the compact angle, take $\Theta^\sharp=\operatorname{adj}\Theta$ in the $\mathrm{Mat}_2(\mathbb C)$ representation. The scalar $\mathcal Q$ is invariant under determinant-one Lorentz congruence and central phase. For a real scaling variation of the full background, the same variation of $Z$ controls drive and readout.

~~~math
\mathcal Q(\Theta)=\operatorname{Re}\operatorname{tr}(\Theta^\sharp\Theta^\dagger),
\qquad
Z=Z_0+c\,\mathcal Q(\Theta),\qquad
\left.\delta Z\right|_{\delta\Theta=u\bar\Theta}
=2c\,u\,\mathcal Q(\bar\Theta),\quad u\in\mathbb R.
~~~

A pure central winding $e^{iN\varphi}\Theta_0$ gives constant $\mathcal Q$. A different case is the existing full-Theta winding family for winding one, evaluated at the spacetime origin and with an angular phase $\beta$. Direct determinant and trace evaluation give the expressions below.

~~~math
\begin{gathered}
\Theta_\beta=\sigma_3+i\begin{pmatrix}0&r\\r&b\end{pmatrix},
\qquad r^2=1+\rho\cos(\varphi-\beta),\quad
b=\rho\sin(\varphi-\beta),\quad 0<\rho<1,\\
\det\Theta_\beta=\rho e^{i(\varphi-\beta)},
\qquad
\mathcal Q(\Theta_\beta)=-2-2r^2=-4-2\rho\cos(\varphi-\beta).
\end{gathered}
~~~

The trace has a first harmonic with phase fixed by $\beta$. If this reference were physical and held fixed, and the displayed scaling variation were physical, its coefficient would weight both the uniform and first harmonics in the same portal. Measuring the phase requires calibrated access to the two quadratures. Positions would be defined relative to the reference.

The upstream split-jet construction keeps the same tetrad and spin connection while changing the carrier in this family. The Maxwell coupling through the metric therefore sees the same geometry. The displayed $\mathcal Q$ changes at fixed geometry, so it fails the carrier-shift invariance test in the freely varied gravity-only chart. Adding a coupling to it would require a new constraint analysis of the coupled action. This leaves two physical requirements open in UBT, a compact propagator on the reduced modes and an observable, stable phase reference.

Run `python scripts/verify_source_matching.py --radar-reference` to reproduce a finite matrix example with two internal components per harmonic. Direct inversions yield a nonzero position-independent uniform return and recover the target angles from the calibrated first-harmonic return. The check also evaluates the full-Theta determinant and trace symbolically.

[Full-Theta winding and the split-jet quotient](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/672f365ad39c46149ecbbbabe2e6178d6c89e054/research_tracks/action_selection/mm_full_theta_psi_winding_decoupling_2026_11.en.md) · [Compact action gate](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/672f365ad39c46149ecbbbabe2e6178d6c89e054/research_tracks/information_first/TORUS_THETA_CANONICAL_ACTION_BRIDGE_GATE.en.md)

<!-- BILINGUAL-UNIT: em-coupling.line -->

A bench analog uses a transmission line with matched far termination and a removable shunt inductor \(L_s\) at distance \(d\) from the measured port. In the lossless line model \(L'\) and \(C'\) are inductance and capacitance per unit length. With node flux \(\Phi\) and \(V=\partial_t\Phi\), the line energy has kinetic density \(C'\Phi_t^2/2\), gradient density \(\Phi_y^2/(2L')\) and shunt energy \(\Phi(d)^2/(2L_s)\). Dividing the field equation by \(C'\) recovers the massless point-stiffness model with the following parameter map and voltage reflection coefficient at the shunt.

~~~math
v=(L'C')^{-1/2},\qquad Z_0=\sqrt{L'/C'},\qquad
\lambda=\frac1{C'L_s},\qquad
\Gamma(s)=-\frac{\lambda}{2vs+\lambda}=-\frac{Z_0}{2sL_s+Z_0}.
~~~

The port sees the factor \(e^{-2sd/v}\Gamma(s)\) for one incident wave and a matched source. Compare otherwise identical traces with and without the shunt. The ideal comparison calibrates a known target delay and reflection, while real connectors, component parasitics and the source termination affect the traces. The example supplied here is a calculation. The independent hardware measurements and the UBT parameters are open.

[Transmission-line flux and voltage, Eqs. 5–7](https://arxiv.org/abs/2307.14756)

<!-- BILINGUAL-UNIT: em-coupling.verification -->

Run `python scripts/verify_source_matching.py --check` and `python scripts/verify_compact_locality.py --check`. Use `python scripts/verify_compact_locality.py --radar-example` for the target-present and target-absent traces. The finite CAS checks cover the ten independent symmetric metric variations, trace, axial magnetic stress, complex sharp chain rule and phase weight. The echo check integrates the causal convolution at the target and compares it to the closed form. The existing scalar checks compare the threshold to proper-time and Feynman-parameter integrals. The coordinate cancellation follows analytically from stress conservation and the boundary condition. LEAN-PENDING.
