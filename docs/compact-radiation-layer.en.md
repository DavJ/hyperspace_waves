# Propagating compact-mode conversion at a finite layer

**2026-11-06 · working research.** The calculation extends the [compact Einstein–Maxwell comparison](compact-einstein-maxwell-port.en.md). It assumes a real spacelike circle of radius $R$, a prepared electromagnetic pump with compact index one, and planar fields independent of $x_1,x_2$. Work at leading weak gravity on the flat background and leading order in the electromagnetic drive amplitudes. The static layer's own curvature affects higher orders of this expansion. The outgoing coefficients refer to the flat planar problem. AI-assisted derivation and translation. Exhaustive human review is not claimed.

<!-- BILINGUAL-UNIT: compact-radiation.layer -->

## Matter that can exchange longitudinal momentum

Use a canonical real scalar $\chi$ with $v,\ell>0$ and electromagnetic factor $Z>0$. The parameter $\epsilon$ controls the small electromagnetic contrast of the layer.

~~~math
\mathcal L_{\rm matter}=-\tfrac14 Z(\chi)F_{AB}F^{AB}
-\tfrac12\partial_A\chi\,\partial^A\chi-V(\chi),\qquad
V(\chi)=\frac{(\chi^2-v^2)^2}{2\ell^2 v^2},\qquad
Z(\chi)=e^{\epsilon(1-\chi^2/v^2)}.
~~~

With the electromagnetic amplitudes set to zero, the planar kink solves the flat scalar equation. Maxwell and scalar stress exchange cancel when both matter equations are used. The scalar pressure in directions $x_1$ and $x_2$ is equal even for a scalar perturbation that depends on $t,z,y$. It therefore drops out of the difference driving $h_+$ while it accounts for momentum exchange with the medium. The steady weak-drive expansion assumes frequencies away from scalar resonances of the layer.

~~~math
\chi_0(z)=v\tanh(z/\ell),\qquad Z(z)=e^{\epsilon W(z)},\qquad
W(z)=\operatorname{sech}^2(z/\ell),\qquad
\partial^A T^{\rm EM}_{AB}=-\tfrac14F^2\partial_B Z,\quad
\partial^A T^\chi_{AB}=+\tfrac14F^2\partial_B Z.
~~~

<!-- BILINGUAL-UNIT: compact-radiation.drive -->

## Drive and momentum mismatch

Take both Maxwell potentials polarized along $x_1$ and add complex conjugates to form real fields. The pump at $\Omega=1/R$ travels along $y$ and is an exact Maxwell mode of the given planar $Z(z)$ because its longitudinal wave number vanishes. The ordinary signal enters from the left with amplitude $a_{\rm in}$ and $\omega>0$. The electromagnetic stress at the sum frequency is obtained directly from the cross term of the two field strengths.

~~~math
\begin{gathered}
A_{1,\rm p}=a_{\rm p}e^{i(y/R-\Omega t)},\qquad
A_{1,\rm s}=a(z)e^{-i\omega t},\qquad
(Za')'+Z\omega^2a=0,\\
\nu=\Omega+\omega,\qquad k=\sqrt{\nu^2-\Omega^2}=\sqrt{\omega^2+2\Omega\omega},
\qquad \Delta=k-\omega,\qquad (T_{11}-T_{22})_{\nu,1}=2\Omega\omega\, Z a_{\rm p}a.
\end{gathered}
~~~

The sum frequency is above the free compact threshold. Its rightgoing metric wave has wave number $k$, while the incident signal contributes $\omega$ along $z$. The layer can supply the difference $\Delta$. For $0<\omega<\Omega$, the difference frequency $\Omega-\omega$ stays below the free threshold in the same compact harmonic. A static layer preserves the frequency.

<!-- BILINGUAL-UNIT: compact-radiation.emission -->

## Isolating the emitted wave

Write $h(z)$ for the amplitude of $h_+$ at $(\nu,n=1)$. In the uniform regions there is a driven metric response proportional to the electromagnetic field. Subtract it by solving for $r$. The Maxwell equation for $a$ yields a localized source for $r$ exactly at the stated gravity and drive orders.

~~~math
\begin{aligned}
(-\partial_z^2-k^2)h&=2\Omega\omega\,Za_{\rm p}a/M_5^3,\qquad
r=h+a_{\rm p}a/M_5^3,\\
(-\partial_z^2-k^2)r
 &=\frac{a_{\rm p}}{M_5^3}
 \left[2\Omega\omega(Z-1)a+\frac{Z'}{Z}a'\right].
\end{aligned}
~~~

The retarded spatial inverse is $i e^{ik|z-z'|}/(2k)$, as follows from the derivative jump of $-\partial_z^2-k^2$. The coefficient $\mathcal A_+$ of $e^{ikz}$ on the right is $i/(2k)$ times the source's Fourier transform at $k$. To first order in $\epsilon$, use $a=a_{\rm in}e^{i\omega z}$ in that localized source. Integration by parts gives $[2\Omega\omega-\omega\Delta]\widehat W(\Delta)=k\Delta\widehat W(\Delta)$. The transform follows by substituting $u=(1+\tanh(z/\ell))/2$ in its integral.

~~~math
\mathcal A_+
=\frac{i\epsilon a_{\rm p}a_{\rm in}}{2M_5^3}\,\Delta\,\widehat W(\Delta)
+O(\epsilon^2),\qquad
\widehat W(q)=\int_{\mathbb R}W(z)e^{-iqz}\,dz
=\frac{\pi\ell^2 q}{\sinh(\pi\ell q/2)}.
~~~

For nonzero pump and signal, the kink gives a nonzero rightgoing amplitude to first order in $\epsilon$. The derivative term in the source comes from the response of the electromagnetic wave to the layer. Keeping both terms gives the stated coefficient.

<!-- BILINGUAL-UNIT: compact-radiation.readout -->

## Readout through the same layer

Send a rightgoing metric mode $h=H_{\rm in}e^{ikz}$ and use the conjugate pump. The Maxwell source $\partial_A(ZhF_{\rm p}^{A1*})$ has ordinary harmonic zero at frequency $\omega$. Let $b$ be that photon amplitude. Subtract its distributed driven part and take the outgoing Maxwell solution for the remainder $u$. At first order in $\epsilon$, the outgoing coefficient $\mathcal B_+$ of $e^{i\omega z}$ follows from the same Fourier integral with the sign of $\Delta$ reversed.

~~~math
\begin{aligned}
(Zb')'+Z\omega^2 b&=-\Omega\omega\,Za_{\rm p}^*h,\qquad
u=b-\tfrac12 a_{\rm p}^*h,\\
(Zu')'+Z\omega^2u&=-\tfrac12 a_{\rm p}^*Z'h',\\
\mathcal B_+&=\frac{i\epsilon a_{\rm p}^*H_{\rm in}}{4\omega}
 \,k\Delta\,\widehat W(-\Delta)+O(\epsilon^2).
\end{aligned}
~~~

For the stated real-field convention the energy flux divided by frequency is $2\omega|a|^2$ for an ordinary photon and $M_5^3 k|H|^2$ for the metric mode, per transverse area and per unit compact length. With these flux normalizations the two first-order conversion coefficients are negative complex conjugates, since $W$ is real.

<!-- BILINGUAL-UNIT: compact-radiation.check -->

## Reproduction and scope

Run `python scripts/verify_source_matching.py --compact-radiation` for the dimensionless example $R=\ell=1$, $\omega=0.5$. The predicted outgoing coefficients per unit $\epsilon$ are $0.530662i$ and $0.593298i$ with unit input and pump amplitudes and unit $M_5^3$ for the coefficients. The numerical check integrates Maxwell scattering in the actual layer, computes the two outgoing coefficients and checks the zero-contrast limit. These calculations give planar emission and readout in the assumed five-dimensional model. A radar echo still requires target scattering and compact pump preparation, and an application to UBT requires a physical identification of the compact coordinate. The [frequency cutoffs, localized source algebra and mismatch factor](../HyperspaceWaves/Radiation.lean) are checked in Lean. The outgoing integrals and Maxwell scattering use the checks above.

[Outgoing Helmholtz Green functions](https://math.mit.edu/icg/resources/teaching/18.367/notes367.pdf) · [Fourier treatment of phase matching](https://doi.org/10.1109/3.161322)
