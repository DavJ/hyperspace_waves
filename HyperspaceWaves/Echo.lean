module

public import HyperspaceWaves.Phase
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Tactic.FieldSimp
public import Mathlib.Tactic.LinearCombination

public section

namespace HyperspaceWaves

open scoped BigOperators

/-- Finite-dimensional kinetic square for a smeared tidal oscillator. -/
theorem targetKineticSquare {ι : Type*} [Fintype ι]
    (velocity profile : ι → ℝ) (rate coupling : ℝ) :
    (∑ i, velocity i ^ 2) + rate ^ 2 +
        2 * coupling * rate * (∑ i, profile i * velocity i) =
      (∑ i, (velocity i + coupling * rate * profile i) ^ 2) +
        (1 - coupling ^ 2 * (∑ i, profile i ^ 2)) * rate ^ 2 := by
  have expand (i : ι) :
      (velocity i + coupling * rate * profile i) ^ 2 =
        velocity i ^ 2 + (2 * coupling * rate) * (profile i * velocity i) +
        (coupling ^ 2 * rate ^ 2) * profile i ^ 2 := by ring
  simp_rw [expand, Finset.sum_add_distrib, ← Finset.mul_sum]
  ring

/-- Solve the self-consistent retarded rank-one target response. -/
theorem tidalTargetResponse {overlap drive selfEnergy coupling gap response : ℂ}
    (hfield : overlap = drive + coupling * selfEnergy * response)
    (hoscillator : gap * response = coupling * overlap)
    (hdenom : gap - coupling ^ 2 * selfEnergy ≠ 0) :
    coupling * response =
      coupling ^ 2 * drive / (gap - coupling ^ 2 * selfEnergy) := by
  apply (eq_div_iff hdenom).2
  linear_combination coupling * hoscillator + coupling ^ 2 * hfield

/-- Scalar optical-theorem identity for a real bare inverse response. -/
theorem tidalOpticalIdentity (bare : ℝ) (selfEnergy scattering : ℂ)
    (h : ((bare : ℂ) - selfEnergy) * scattering = 1) :
    scattering.im = (scattering.re ^ 2 + scattering.im ^ 2) * selfEnergy.im := by
  have hre := congrArg Complex.re h
  have him := congrArg Complex.im h
  simp only [Complex.mul_re, Complex.mul_im, Complex.sub_re, Complex.sub_im,
    Complex.ofReal_re, Complex.ofReal_im, Complex.one_re, Complex.one_im] at hre him
  linear_combination scattering.re * him - scattering.im * hre

/-- A nonzero common target factor drops out of the ratio of the two returns. -/
theorem cancelCommonEchoFactor {target mixed diagonal : ℂ}
    (htarget : target ≠ 0) (hdiagonal : diagonal ≠ 0) :
    (target * mixed) / (target * diagonal) = mixed / diagonal := by
  field_simp

/-- Taking the ratio across frequencies cancels the compact-angle phase. -/
theorem twoFrequencyEchoPhase (δa δb distance angle : ℂ) :
    Complex.exp (Complex.I * (δb * distance - angle)) /
        Complex.exp (Complex.I * (δa * distance - angle)) =
      Complex.exp (Complex.I * ((δb - δa) * distance)) := by
  rw [← Complex.exp_sub]
  congr 1
  ring

end HyperspaceWaves
