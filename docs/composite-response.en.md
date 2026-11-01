# Computed neutral-composite response and the ordinary photon-pair tail

<!-- BILINGUAL-UNIT: composite-response.scope -->

## What is calculated

**DERIVED_WITH_ASSUMPTIONS · 2026-11-01 · working research.** We calculate the neutral-composite response left open by the neutrality audit, first as a general Gaussian contraction and then for one positive-energy complex physical-mode comparator. This does not define the full Theta as a scalar, select its component count, replace the tetrad metric or add a fundamental field. The radius r is ordinary spatial separation in a local 3+1-dimensional vacuum calculation; it is not a distance between psi segments. Natural units and signature (+---) are used. The UBT constrained physical propagator and its source/readout overlaps remain open.

<!-- BILINGUAL-UNIT: composite-response.wick -->

## A dimension-independent Gaussian identity

Let z be any finite vector of real Gaussian coordinates with zero mean, let B be real symmetric, and subtract disconnected terms. Wick contraction gives the following expression. Fourier transformation makes it a momentum convolution, not the square of a propagator at the same frequency. For the original invariant H=Tr(Theta^sharp Theta†), B is its quadratic form in the chosen coordinates. Any physical reduction must be derived before inserting a propagator. Indefinite B does not by itself imply a negative spectral density: with canonical positive scalar modes the weights are squares. Positivity of the actual constrained Hilbert space remains a prerequisite, not an output of the algebra.

```math
H=z^TBz,\quad G(x)=\langle z(x)z(0)^T\rangle,\qquad
\langle H(x)H(0)\rangle_c=2\operatorname{Tr}[B G(x) B G(x)^T].
\quad \rho_H(s)=2\sum_{a,b}|B_{ab}|^2\rho_{\rm pair}(s;\mu_a,\mu_b)
.
```

In a constant-background or frozen-coefficient calculation, a nonzero full Theta background also gives the observable a linear projection. Zero-mean fluctuation coordinates do not set that full background to zero. Its connected response includes the original propagator projected by ell; possible one-particle weight from this term is not excluded by the pair calculation. The background and projection must still pass the charge and locked-metric compatibility gates. The worked scalar-density vacuum below has ell=0.

```math
H=H_0+\ell^Tz+z^TBz,\quad\ell=2Bz_0,\qquad
\langle H(x)H(0)\rangle_c=\ell^TG(x)\ell+2\operatorname{Tr}[BG(x)BG(x)^T].
```

<!-- BILINGUAL-UNIT: composite-response.spectrum -->

## The scalar-pair response with fixed normalization

Take a canonical complex mode phi in its free, phase-invariant vacuum and the Hermitian density O=phi†phi. It is an example contribution after a physical reduction, not an identification H=O. Use an external source +J O and the retarded convention below. The connected correlator has the pair continuum; it has no delta-function one-particle weight. The equal-mass threshold is twice the constituent mass. Polynomial subtraction terms are local contacts and do not change the separated-point kernels.

```math
\chi_O^R(x)=i\theta(t)\langle[O(x),O(0)]\rangle,\quad
\rho_{\rm pair}(s;a,b)=\frac{\sqrt{[s-(a+b)^2][s-(a-b)^2]}}{16\pi^2 s}\,
\mathbf1_{s\ge(a+b)^2},\qquad a,b>0.
\quad \rho_O(s)=\frac{1}{16\pi^2}\sqrt{1-\frac{4\mu^2}{s}}\,\mathbf1_{s\ge4\mu^2}.
```

<!-- BILINGUAL-UNIT: composite-response.subtraction -->

## A once-subtracted susceptibility, not an assigned pole

The Feynman-parameter result and its independent spectral-dispersion representation agree. The subtraction fixes the unknown local constant without introducing a resonance. At the pair threshold the susceptibility is finite, although its derivative is singular; a branch point is not a particle pole. A bound-state pole would require the interactions and physical constraints of a selected action.

```math
z=(\omega+i0)^2-\boldsymbol k^2,\qquad
\widehat\Pi(z)=\Pi(z)-\Pi(0)
=-\frac1{16\pi^2}\int_0^1dx\,\log\left(1-\frac{x(1-x)z}{\mu^2}\right)
=z\int_{4\mu^2}^{\infty}\frac{\rho_O(s)}{s(s-z)}\,ds.
\quad \widehat\Pi'(0)=\frac1{96\pi^2\mu^2},\qquad
\widehat\Pi(4\mu^2)=\frac1{8\pi^2}.
```

<!-- BILINGUAL-UNIT: composite-response.static -->

## Exact static response at separated points

Spatial Fourier transformation of the spectral representation gives a superposition of Yukawa kernels. The static integral has a closed Bessel form. An independent check integrates the square of the Euclidean massive propagator over Euclidean time. Thus a nonzero response below the pair threshold is allowed, but its massive free-vacuum tail is exponentially suppressed. The massless free limit is shown separately; it cannot be inserted into the massive electromagnetic matching below.

```math
\chi_O^R(\omega,r)=\frac1{4\pi r}\int_{4\mu^2}^{\infty}ds\,\rho_O(s)
 e^{-r\sqrt{s-(\omega+i0)^2}},\qquad r>0,
\quad \chi_O^R(0,r)=\frac{\mu K_1(2\mu r)}{16\pi^3r^2}
\sim\frac{\sqrt\mu}{32\pi^{5/2}r^{5/2}}e^{-2\mu r}.
\quad G_E(\tau,r)=\frac{\mu K_1(\mu\sqrt{\tau^2+r^2})}{4\pi^2\sqrt{\tau^2+r^2}},
\quad \chi_O^R(0,r)=\int_{-\infty}^{\infty}G_E(\tau,r)^2\,d\tau,
\quad \lim_{\mu\to0}\chi_O^R(0,r)=\frac1{32\pi^3r^3}.
```

<!-- BILINGUAL-UNIT: composite-response.threshold -->

## Frequency tuning changes the decay, with a nonuniform limit

At fixed positive kappa the long-distance expansion has an exponential times a power. Raising frequency toward the pair threshold reduces kappa in this comparator; lowering frequency does not reproduce the improved penetration of low-frequency radio in a conductor. Exactly at the threshold the expansion changes to a different power law and the response stays finite at separated points. This is not an isolated resonant mode. The underlying retarded propagators have light-cone support, so their spectral superposition does not supply a faster signal front.

```math
\kappa=\sqrt{4\mu^2-\omega^2}>0,\qquad
\chi_O^R(\omega,r)\sim\frac{(2\kappa)^{3/2}}{256\pi^{5/2}\mu\,r^{5/2}}e^{-\kappa r}
\quad(r\to\infty;\,\kappa>0),
\qquad \chi_O^R(2\mu,r)\sim\frac1{32\pi^3\mu\,r^4}.
\quad \chi_O^R(t,\boldsymbol r)=0\quad(t<|\boldsymbol r|).
```

<!-- BILINGUAL-UNIT: composite-response.matching -->

## Electromagnetic mass-source response from the same comparator

Now retain the existing minimally coupled scalar-QED comparator and define the renormalized density by differentiation with respect to an external mass-squared source J at fixed ultraviolet couplings. J is a probe, not a dynamical scalar. Assume no bare J F² term, weak fields and soft momenta. Differentiating the already matched scalar-loop logarithm supplies the local density response. Its negative magnetic-field sign is independently checked from the proper-time factor x/sinh(x). This mixed derivative is a local response, not a separate propagator for O. The low-energy coefficient cannot be used at frequencies near twice the same mass without computing full momentum-dependent matching.

```math
\mathcal L=|D\phi|^2-(\mu^2-J)|\phi|^2-\tfrac14F_{\mu\nu}F^{\mu\nu},\quad O=\frac{\delta\Gamma}{\delta J},
\qquad \Delta\mathcal L=\frac{e^2q^2}{192\pi^2}\log\frac{\mu^2-J}{\mu^2}F^2,
\quad O_{\rm eff}\supset c_OF^2,\qquad c_O=-\frac{e^2q^2}{192\pi^2\mu^2}.
\quad \frac{x}{\sinh x}=1-\frac{x^2}{6}+O(x^4),\qquad |p|\ll\mu.
```

<!-- BILINGUAL-UNIT: composite-response.photons -->

## The full electromagnetic comparator is not purely exponential

Once electromagnetic interactions are retained, the same density can create two ordinary massless photons. Its leading low-energy spectral density therefore starts at zero, even though the free massive-scalar pair starts at its positive threshold. Wick contraction of the canonical Euclidean Maxwell field gives the following normalization and static tail. This contribution is of order e⁴; at sufficiently large separation it dominates the exponential free matter contribution. It is a photon-pair continuum, with no new one-photon or composite pole. We do not claim the sum of these leading terms to be a fully resummed, loop-corrected vacuum response.

```math
\langle F^2(x_E)F^2(0)\rangle_c=\frac{48}{\pi^4|x_E|^8},\qquad
\rho_{F^2}(s)=\frac{s^2}{4\pi^2}\mathbf1_{s\ge0},\qquad
\rho_O^{\gamma\gamma}(s)\simeq c_O^2\frac{s^2}{4\pi^2}\quad(s\ll\mu^2),
\qquad \boxed{\chi_O^{\gamma\gamma}(0,r)\simeq\frac{15c_O^2}{\pi^3 r^7}}
\quad(\mu r\gg1).
```

<!-- BILINGUAL-UNIT: composite-response.fiber -->

## Why the power-law tail does not identify a neighboring segment

The photon tail is an ordinary electromagnetic mechanism in physical space. In a psi-translation-invariant background with only zero-harmonic ordinary photons, it couples to the zero harmonic of the composite. Two zero indices cannot supply a nonzero conserved fiber index. A nonzero source/readout profile, another photon spectrum or a translation-breaking background must be derived separately. Neither this tail nor the massive pair kernel identifies a multiverse segment, a coil voltage or a device range. The recent gravity-only MM winding nonselection result also supplies no protected channel by itself.

```math
n_{\gamma,1}=n_{\gamma,2}=0,\qquad
m=n_{\gamma,1}+n_{\gamma,2}=0
.
```

With an invariant action, measure and state and only zero-harmonic sources, translation symmetry forces the nonzero-index one-point mean to vanish to all orders. This phase is Fourier translation, not electromagnetic charge. The statement does not forbid creating opposite-index pairs or modulating neutral occupations and correlations. In particular, the zero harmonic of a density contains every constituent index n, not just the zero-index constituent. Such an occupation/correlation protocol requires its own higher-point response, physical spectrum and electromagnetic readout; it is not established by the selection rule alone.

```math
\psi\mapsto\psi+R\alpha:\quad H_m\mapsto e^{im\alpha}H_m,\quad
\langle H_m\rangle_{A_0}=e^{im\alpha}\langle H_m\rangle_{A_0}\ (\forall\alpha)
\Longrightarrow\langle H_m\rangle_{A_0}=0\ (m\ne0).
\quad H_0=L^{-1/2}\sum_n\operatorname{Tr}(\theta_n^\sharp\theta_n^\dagger),
\qquad \langle H_mH_{-m}\rangle_{A_0}\ 	ext{is not forced to vanish}.
```

<!-- BILINGUAL-UNIT: composite-response.verification -->

## Verification and next physical gate

A subsequent [conditional occupation calculation](occupation-channel.en.md) supplies an explicit scalar-QED electric pump and a dephased kinetic readout. It does not supply the UBT spectrum or psi-localized profiles required for an intersegment channel.

Derivation and code are AI-assisted. English is the translation source; exhaustive human review is not claimed. The matrix contraction, spectral dispersion, Bessel form, direct Euclidean time integral, Maxwell tensor normalization, magnetic proper-time sign and finite-difference mass-source matching are independently checked. The JSON record supplies exact assumptions, tool versions, errors and dimensionless examples. `LEAN-PENDING`: lean/lake are unavailable; no checked formal proof or experimental validation is claimed. The next UBT calculation is the action-derived physical propagator and its composite projection, followed by the fiber-resolved electromagnetic overlap. New poles require actual interactions, not an assigned free neutral field.

<!-- BILINGUAL-UNIT: composite-response.sources -->

## Sources and reproduction

- [D’Hoker, Quantum Field Theory, §§4.2, 6.2](https://www.pa.ucla.edu/faculty-websites/dhoker-lecture-notes/graduate-courses/qft-2016.pdf).
- [NIST DLMF 10.32.8](https://dlmf.nist.gov/10.32#E8).
- [Vassilevich, hep-th/0306138, §4](https://arxiv.org/abs/hep-th/0306138).
- [Scalar-loop source matching](source-detector-matching.md).
- [UBT PR #825](https://github.com/UBT-Institute/unified-biquaternion-theory/pull/825).
- [MM winding audit](https://github.com/UBT-Institute/unified-biquaternion-theory/blob/65231d02d1ceade5068ee61b12aaa168df151a9a/research_tracks/action_selection/mm_full_theta_psi_winding_decoupling_2026_11.en.md).
- [JSON](results/composite-response.json); [provenance](research-origin-2026-11-01-composite-response.json).

```sh
python -m pytest -q
python scripts/verify_composite_response.py --check
```
