# A single compact-mode radar return with a passive target

**2026-11-07 · working research.** This completes the first-return calculation in the specified real-spacelike-circle comparator with isolated, calibrated ports. The calculation assumes a prepared compact pump, a single neutral target with one internal response, and separated free-wave channels between the stations and the target. We keep the first contribution with one outward and one inward conversion. Direct electromagnetic paths, spatially overlapping pump-driven fields and repeated paths between the station and the target would require a different full-system calculation. AI-assisted derivation and translation. Exhaustive human review is not claimed.

<!-- BILINGUAL-UNIT: compact-echo.target -->

## A finite tidal oscillator

Write $\varphi=y/R$ and take the planar $h_{11}=-h_{22}=h_+$ sector of the [compact Einstein–Maxwell model](compact-einstein-maxwell-port.en.md). The linearized tidal curvature $R_{0101}-R_{0202}=-\ddot h_+$ is invariant under linearized gauge transformations about the flat background. Use a real quadrupole coordinate $q$ of bare frequency $\omega_0>0$ and an averaged profile $f=w(z-d)p(\varphi-\alpha)$. The tidal term $-gq\ddot U_f$ becomes the kinetic mixing below after integrating by parts in time. The normalization of $U$ includes the compact length. Here $\langle f,U\rangle=\int fU\,d\mu$ for real fields.

~~~math
\begin{aligned}
U&=\sqrt{M_5^3(2\pi R)/2}\,h_+,\qquad U_f=\langle f,U\rangle,\qquad
d\mu=dz\,\frac{d\varphi}{2\pi},\\
L&=\tfrac12\int\!\left(\dot U^2-U_z^2-R^{-2}U_\varphi^2\right)d\mu
   +\tfrac12(\dot q^2-\omega_0^2q^2)+g\dot q\,\dot U_f,\\
2K&=\|\dot U+g\dot q f\|_2^2+(1-g^2\|f\|_2^2)\dot q^2 .
\end{aligned}
~~~

The total quadratic energy has the same kinetic square and nonnegative gradient and oscillator terms. It is passive for $g^2\|f\|_2^2<1$. The example uses a normalized box $w=1/(2a)$ on $[-a,a]$ and a real even $p$ with coefficients $p_0=1$, $p_{\pm1}=0.25$, $p_{\pm2}=0.10$. The parameter $\alpha$ translates this known angular profile and $d$ translates the box. The self-response is finite with this longitudinal average and finite harmonic support.

[Dynamical quadrupoles coupled to tidal curvature](https://arxiv.org/abs/1608.01907)

<!-- BILINGUAL-UNIT: compact-echo.scatter -->

## Exact linear response of the target

Use $e^{-i\nu t}$ with $\nu>0$, away from channel thresholds. For the retarded square root take positive real $k_j$ in open channels and positive imaginary $k_j$ in closed channels. The field and oscillator equations give the following solution. The scattered field is $\mathcal T\langle f,U_{\rm in}\rangle G_\nu f$.

~~~math
\begin{aligned}
k_j&=\sqrt{(\nu+i0)^2-j^2/R^2},\qquad G_j(s)=\frac{i}{2k_j}e^{ik_j|s|},
\qquad \Sigma=\langle f,G_\nu f\rangle,\\
U&=U_{\rm in}+g\nu^2 q\,G_\nu f,\qquad
(\omega_0^2-\nu^2)q=g\nu^2 U_f,\qquad
\mathcal T=\frac{g^2\nu^4}{\omega_0^2-\nu^2-g^2\nu^4\Sigma}.
\end{aligned}
~~~

For unit rightgoing incidence in $n$, the overlap at the target carries $p_{-n}\widehat w(-k_n)e^{ik_nd+in\alpha}$, with $\widehat w(k)=\int w(s)e^{-iks}ds$. Expand the outgoing Green function on the left to get the reflection coefficient $\mathcal R_{mn}$ of $e^{-ik_m z+im\varphi}$. The transmission coefficient $\mathcal P_{mn}$ adds the incident term $\delta_{mn}$ and replaces $k_n+k_m$ by $k_n-k_m$ and $\widehat w(-k_m)$ by $\widehat w(k_m)$ in the scattered term.

~~~math
\begin{aligned}
\mathcal R_{mn}
 &=\frac{i\mathcal T}{2k_m}\,p_m p_{-n}\,\widehat w(-k_m)\widehat w(-k_n)
       e^{i(k_m+k_n)d+i(n-m)\alpha},\\
\operatorname{Im}\Sigma
 &=\sum_{j\ {\rm open}}\frac{|p_j\widehat w(k_j)|^2}{2k_j},
\qquad
\operatorname{Im}\mathcal T=|\mathcal T|^2\operatorname{Im}\Sigma,\qquad
\sum_{m\ {\rm open}}k_m\big(|\mathcal R_{mn}|^2+|\mathcal P_{mn}|^2\big)=k_n .
\end{aligned}
~~~

Closed modes enter the real part of $\Sigma$. In open modes the imaginary Green kernel is $\cos(k_j|s|)/(2k_j)$, whose average gives the displayed squared Fourier overlap. The interference term from the forward transmitted wave is $-\operatorname{Im}\mathcal T\,|p_n\widehat w(k_n)|^2$. The sum of scattered fluxes equals its opposite by the optical identity. The target changes phases and redistributes flux with no absorption in this model.

<!-- BILINGUAL-UNIT: compact-echo.range -->

## End-to-end echo and separation of the two coordinates

At a common metric frequency $\nu>2/R$, transmit harmonic $n=1$ with ordinary frequency $\omega_1$ and use the [layer conversion coefficients](compact-radiation-layer.en.md) $\mathcal A_1$ at transmission and $\mathcal B_m$ at reception. The readout pump for $m=1,2$ has frequency $m/R$ and returns the ordinary frequency $\omega_m$. An even layer gives the same leading conversion coefficient for the opposite propagation direction at its centre. Calibration fixes the complex port factors and the known profile overlaps. Both returns must be nonzero.

~~~math
\begin{aligned}
E_{mn}&=\mathcal B_m\,\mathcal R_{mn}\,\mathcal A_n,\qquad
\omega_j=\nu-j/R,\qquad F_j=p_j\widehat w(-k_j),\\
Z(\nu)&=\frac{E_{21}/E_{11}}{K(\nu)}
       =e^{i\delta(\nu)d-i\alpha},\qquad \delta(\nu)=k_2-k_1,\qquad
K(\nu)=\frac{\mathcal B_2 k_1 F_2}{\mathcal B_1 k_2 F_1}.
\end{aligned}
~~~

The same-mode return alone is independent of $\alpha$. The ratio cancels the entire scalar resonance factor $\mathcal T(\nu)$ at each frequency, including its phase. One frequency still mixes $d$ with $\alpha$. Use two distinct frequencies $\nu_b>\nu_a>2/R$ and divide the calibrated ratios to remove $\alpha$.

~~~math
\frac{Z(\nu_b)}{Z(\nu_a)}=e^{i(\delta_b-\delta_a)d},\qquad
e^{i\alpha}=\frac{e^{i\delta_a d}}{Z(\nu_a)},\qquad
\delta'(\nu)=\nu\left(\frac1{k_2}-\frac1{k_1}\right)>0.
~~~

The distance is unique given a known interval of width less than $2\pi/(\delta_b-\delta_a)$, with one representative if the interval is half-open and exactly that width. Then $\alpha$ is fixed modulo $2\pi$. The result uses a single common target factor and known relative harmonic overlaps. An arbitrary target with multiple independent response coordinates need not have that cancellation.

<!-- BILINGUAL-UNIT: compact-echo.verify -->

## What the calculation establishes

The new result in this comparator is a passive, coherent single return with separate reconstruction of longitudinal distance and compact angle even when the target resonance is unknown. It combines tidal response, wave scattering and coherent phase calibration. The compact coordinate, compact pump preparation and target coupling have the status of assumptions for an application to UBT.

Run `python scripts/verify_compact_echo.py` for the dimensionless example $R=1$, $a=0.4$, $g=0.12$, $\nu_a=2.3$, $\nu_b=2.8$. The range period is $22.46$. The reconstruction gives $d=7.4$ and $\alpha=1.2$ for both bare resonances $\omega_0=2.45$ and $4$. The code checks the averaged Green kernel by quadrature and flux balance with both open and closed modes. [Lean](../HyperspaceWaves/Echo.lean) checks the finite-dimensional kinetic square, target response algebra, scalar optical identity and the two phase cancellations. The field model and port isolation are premises of the calculation.
