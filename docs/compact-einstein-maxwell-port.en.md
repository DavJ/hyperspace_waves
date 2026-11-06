# A geometric electromagnetic port for a compact harmonic

**2026-11-05 · working research.** Derived in the stated comparison actions. AI-assisted derivation and translation. Exhaustive human review is not claimed.

<!-- BILINGUAL-UNIT: compact-port.ubt -->

## The coupling in the existing UBT geometry

Use the freely varied split-jet chart with fixed normalization $N_0$. Here $X$ is the Lorentz-real carrier projection of the full biquaternionic $\Theta$, $Q_\mu$ is its compensating auxiliary jet and $D_\mu$ uses the fixed spin connection for this shift. The Maxwell action sees the tetrad through its metric. Its variation along the carrier-shift direction is

~~~math
E_\mu=N_0^{-1/2}(D_\mu X+Q_\mu),\qquad
\delta_\epsilon X=\epsilon,\quad \delta_\epsilon Q_\mu=-D_\mu\epsilon,\quad
\delta_\epsilon E_\mu=0,\qquad
\delta_\epsilon S_{\rm EM}
=\frac12\int d^4x\sqrt{-g}\,T^{\mu\nu}\delta_\epsilon g_{\mu\nu}=0.
~~~

The finite compensated shift preserves the same $E_\mu$ and hence the same Maxwell action. This is an exact zero for the pure carrier reference of the winding example in this chart. The four-dimensional action family also leaves the compact principal kinetic term unspecified. A physical coupling to a nonzero compact harmonic requires a mode with a geometric variation surviving the constraints and a source with the corresponding compact overlap.

[Exact carrier-shift kernel](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/672f365ad39c46149ecbbbabe2e6178d6c89e054/research_tracks/action_selection/mm_shift_gauge_dirac.en.md) · [Compact action gate](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/672f365ad39c46149ecbbbabe2e6178d6c89e054/research_tracks/information_first/TORUS_THETA_CANONICAL_ACTION_BRIDGE_GATE.en.md)

<!-- BILINGUAL-UNIT: compact-port.action -->

## Explicit nonzero coupling in a spacelike compact comparator

Take five-dimensional Einstein–Maxwell theory on a flat background with signature $(-++++)$, a real spacelike $y\sim y+2\pi R$, natural units and a positive gravitational normalization $M_5^3$. Upper-case indices include $y$ and $h_{AB}=\delta G_{AB}$. The first metric variation of Maxwell at fixed covariant $F$ gives the coupling. The compact coordinate, action and accessible compact electromagnetic pump are additional physical assumptions for this comparison.

~~~math
S_5=\int d^4x\,dy\sqrt{-G}
\left(\frac{M_5^3}{2}\mathcal R_5-\frac14F_{AB}F^{AB}\right),\qquad
\delta S_{\rm EM,5}=\frac12\int d^4x\,dy\,h_{AB}T^{AB},
\qquad T_{AB}=F_{AC}F_B{}^C-\frac14\eta_{AB}F^2.
~~~

For waves independent of the transverse coordinates $x_1,x_2$, the tensor polarization $h_{11}=h_+$, $h_{22}=-h_+$ has zero trace and divergence. Its projection $(h_{11}-h_{22})/2$ is invariant under linearized diffeomorphisms in this Fourier sector, since the transverse momentum vanishes. After dropping boundary terms the quadratic Einstein kinetic term is $-M_5^3\partial_C h_{AB}\partial^C h^{AB}/8$ on this polarization. Both diagonal entries contribute, giving the first term below. Varying $h_+$ and expanding all fields as $\sum_n f_n e^{iny/R}$ gives the compact mass $|n|/R$ and the driven wave equation.

~~~math
S_+^{(2)}+S_+^{\rm int}
=\int d^4x\,dy\left[
\frac{M_5^3}{4}\big((\partial_t h_+)^2-(\partial_z h_+)^2-(\partial_y h_+)^2\big)
+\frac12 h_+(T_{11}-T_{22})\right],\qquad
(\partial_t^2-\partial_z^2+n^2/R^2)h_{+,n}=\frac{(T_{11}-T_{22})_n}{M_5^3}.
~~~

The positive free kinetic term makes this a propagating polarization of the comparison theory. In an orthonormal tetrad the four-dimensional part is the variation $\delta E_1=h_+\bar E_1/2$ and $\delta E_2=-h_+\bar E_2/2$. The freely varied split-jet chart can lift this metric tangent at fixed carrier by $\delta Q_\mu=\sqrt{N_0}\,\delta E_\mu$. The dynamics of the five-dimensional action is an additional assumption of the comparator.

<!-- BILINGUAL-UNIT: compact-port.overlap -->

## Drive and readout from the same Maxwell term

The source on the right-hand side of the wave equation uses the Fourier coefficient of a product. A zero-harmonic signal and a pump with compact index $p$ can drive $n=p$. Varying the same Maxwell term around pump $F_{\rm p}$ gives the linear Maxwell equation for the readout field strength $f$. Index raising in both equations uses the flat background.

~~~math
\begin{aligned}
\frac{(T_{11}-T_{22})_n}{M_5^3}
 &=\frac{1}{M_5^3}\sum_{p+q=n}
 \big(F_{1C,p}F_{1,q}{}^C-F_{2C,p}F_{2,q}{}^C\big),\\
\partial_A f^{AB}
 &=\partial_A\!\left(h^A{}_C F_{\rm p}^{CB}
 +h^B{}_C F_{\rm p}^{AC}-\frac12 h^C{}_C F_{\rm p}^{AB}\right).
\end{aligned}
~~~

A readout pump with index $-m$ can return a metric signal of index $m$ to a zero-harmonic electromagnetic signal. The readout equation is expressed in field strengths. With source-free pump and signal the beat stress is conserved. A source-supported pump would require including the apparatus stress in the gravitational source.

<!-- BILINGUAL-UNIT: compact-port.example -->

## A conserved plane-wave example

Take ordinary coordinates $(t,x_1,x_2,z)$ and choose both Maxwell potentials along $x_1$. The pump travels in $y$ and the signal travels in $z$. Both momenta are null in five dimensions and transverse to the polarization. Use complex amplitudes plus their conjugates for the real field. The difference-frequency term has covector $B$ and coefficient $b$.

~~~math
\begin{gathered}
P_A=(-\Omega,0,0,0,\Omega),\qquad K_A=(-\omega,0,0,\omega,0),
\qquad \Omega=R^{-1},\quad 0<\omega<\Omega,\\
A_1=a_{\rm p} e^{iP_A x^A-i\beta}+a_{\rm s} e^{iK_A x^A}+\mathrm{c.c.},
\qquad B_A=P_A-K_A,\qquad b=a_{\rm p}a_{\rm s}^*e^{-i\beta}.
\end{gathered}
~~~

Compute $F_{AB}=\partial_A A_B-\partial_B A_A$ and keep the term proportional to $e^{iB_A x^A}$. The compact harmonic is one. The inverse retarded propagator is $D_n(\nu,k)=k^2+n^2/R^2-(\nu+i0)^2$. The last line uses the conjugate pump and gives the readout source at the conjugate signal momentum.

~~~math
\begin{aligned}
B^A(T_{AC})_{\rm beat}&=0,\qquad
(T_{11}-T_{22})_{\rm beat}=-2\Omega\omega\, b,\\
D_1(\Omega-\omega,-\omega)
&=\omega^2+\Omega^2-(\Omega-\omega+i0)^2=2\Omega\omega,\qquad
h_{+,1}^{\rm beat}=-b/M_5^3,\\
(\partial_A f^{A1})_{-K}
&=\Omega\omega\, h_{+,1}^{\rm beat}\,a_{\rm p}^*e^{i\beta}.
\end{aligned}
~~~

This is a nonzero forced metric response with reciprocal readout. For these two plane waves the beat lies off the free graviton mass shell since $B^2=2\Omega\omega>0$. It therefore supplies a coupling and a readout source. A radiation or echo calculation must specify the finite apparatus, momentum transfer and target scattering.

The [finite-layer calculation](compact-radiation-layer.en.md) derives an outgoing mode and its readout when a static medium supplies the longitudinal momentum mismatch.

<!-- BILINGUAL-UNIT: compact-port.radar -->

## What the two pump profiles contribute to radar

Drive and readout at the same compact harmonic yield a diagonal target response on a translation-invariant background, with no angle dependence. A drive pump that selects $n$ and a calibrated readout pump that selects $m$ permit a mixed target response with phase $e^{-i(m-n)\alpha}$ for a target shifted by angle $\alpha$. A difference of one gives a full-circle phase measurement if the target transition is nonzero and calibrated. Preparing and reading independent compact pump profiles are physical requirements beyond the ordinary zero-harmonic electromagnetic port. Identifying $y$ with the imaginary part of UBT time would also require its own physical derivation.

<!-- BILINGUAL-UNIT: compact-port.check -->

The command `python scripts/verify_source_matching.py --check` checks the beat stress, its conservation, the inverse propagator at the beat and the reciprocal Maxwell source by symbolic matrix calculation. The comparison gives no experimental parameters. LEAN-PENDING.

[Linearized Einstein equations and transverse traceless polarization](https://arxiv.org/abs/gr-qc/9712019) · [Einstein–Maxwell field mixing](https://doi.org/10.1140/epjc/s10052-023-12304-1)
