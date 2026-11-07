# Compact-mode localization with repeated reflections

**2026-11-07 · working research.** We extend the first-return calculation in the real-circle comparator. The station and a single finite tidal oscillator are separated by propagating channels. We assume a fixed linear station at each metric frequency with calibrated transmit and receive maps. This description retains the isolated target response, including its closed-channel self-energy. Exchange between the two scatterers takes place through the open channels. The physical realization of the station and compact pump in UBT remains open. AI-assisted derivation and translation.

[Single-return derivation and tidal-target model](compact-radar-echo.en.md)

<!-- BILINGUAL-UNIT: compact-feedback.sum -->

## Exact sum of repeated returns

At fixed $\nu$ write the compact amplitudes in units of flux, by multiplying the field amplitudes in mode $j$ by $\sqrt{k_j}$. Let $s$ be the external input, $x$ the compact wave leaving the station, $y$ the wave returning to it and $e$ the external output. Include the propagation phases in the target reflection $R_f$. The maps $A,B,S,D$ describe launch, readout, internal station reflection and the direct external response, respectively.

~~~math
x=As+Sy,\qquad y=R_f x,\qquad e=Ds+By.
~~~

The target has one internal oscillator, so $R_f=\tau u\ell$ with $\tau=i\mathcal T$. Here $u$ is an outgoing pattern and $\ell$ a complex-linear functional of the incident amplitudes. With $c_j=p_j\widehat w(-k_j)/\sqrt{2k_j}$, the expression for $\mathcal R_{mn}$ in the first-return calculation gives

~~~math
u_m=c_m e^{ik_m d-im\alpha},\qquad
\ell(x)=\sum_n \frac{p_{-n}\widehat w(-k_n)}{\sqrt{2k_n}}
              e^{ik_n d+in\alpha}x_n .
~~~

Since every reflection from the target lies along $u$, write $y=a u$. The scalar equation and its solution are

~~~math
a=\tau\ell(As)+\lambda a,\qquad \lambda=\tau\ell(Su),\qquad
e-e_0=\frac{\tau\ell(As)}{1-\lambda}\,Bu,\qquad e_0=Ds .
~~~

The denominator is the sum factor for all repeated returns, $1+\lambda+\lambda^2+\cdots$ for $|\lambda|<1$. Mode mixing in $S$ changes this factor and can change the phase and amplitude even in a same-mode echo. The returned pattern remains rank one. This is a rank-one specialization of scattering-matrix composition. [Generalized star product](https://arxiv.org/abs/math-ph/0008022)

<!-- BILINGUAL-UNIT: compact-feedback.passive -->

## Why a readable passive loop converges

For the target, $\|u\|^2=\|\ell\|^2=\Gamma=\operatorname{Im}\Sigma$ by the open-channel optical identity. We take $\Gamma>0$. The dressed inverse response has imaginary part $-\Gamma$ and real bare inverse response. Thus $\|R_f\|=|\tau|\Gamma\le 1$. Station passivity for the internal incident pattern alone gives $\|Su\|^2+\|Bu\|^2\le\|u\|^2$. Cauchy–Schwarz gives the bound

~~~math
\eta_B=\frac{\|Bu\|^2}{\Gamma},\qquad
|\lambda|\le |\tau|\Gamma\sqrt{1-\eta_B}\le\sqrt{1-\eta_B}.
~~~

If $Bu\ne0$, the bound is strict even when some other internal pattern reflects perfectly at the station. A unit real-frequency loop requires a pattern that is dark to the readout. The sum can increase or decrease the echo amplitude relative to the first return. For the total external signal $e$, adding the station and target flux inequalities cancels $x$ and $y$, leaving $\|e\|^2+\|t\|^2\le\|s\|^2$, where $t$ is the wave transmitted past the target. Equality holds for lossless blocks when all their outgoing flux is counted.

<!-- BILINGUAL-UNIT: compact-feedback.range -->

## What still determines position

Subtract the coherent baseline $e_0$ for the same input. If the calibrated receivers select modes $1$ and $2$ with complex gains $B_1,B_2$, the two echo components share the entire feedback factor. For nonzero echoes and known relative profile overlaps, the calibrated ratio is

~~~math
Z(\nu)=\frac{(e-e_0)_2}{(e-e_0)_1}\frac{B_1c_1}{B_2c_2}
      =e^{i(k_2-k_1)d-i\alpha} .
~~~

Two distinct frequencies above $2/R$ give the same range period and angular reconstruction as the first return. This needs phase calibration at both frequencies, a drive with $\ell(As)\ne0$ and a single target pattern. In general, an unsubtracted coherent baseline or multiple independent target patterns changes the ratio.

<!-- BILINGUAL-UNIT: compact-feedback.check -->

## Numerical and formal checks

Run `python scripts/verify_compact_echo_feedback.py`. The dimensionless station is a lossless block with a unitary Fourier mixer and diagonal readout. One internal mode has zero readout, so the station's internal reflection has norm one. The code compares the scalar sum with a direct matrix solve and checks target, station and combined flux. At $\nu_a=2.3$ the feedback changes the amplitude by factors $1.24$ and $0.80$ for bare target frequencies $2.27$ and $2.30$. Using $\nu_b=2.8$ as the second frequency recovers $d=7.4$ and $\alpha=1.2$ in both cases. Lean checks the rank-one feedback equation, the scalar target bound, strict loop convergence under the normalized flux assumptions and cancellation of the internal flux terms. [Lean proofs](../HyperspaceWaves/Echo.lean)
