module

public import HyperspaceWaves.Phase
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Algebra.Module.LinearMap.Basic
public import Mathlib.Tactic.FieldSimp
public import Mathlib.Tactic.Linarith
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

/-- A lossless tidal oscillator has reflection norm at most one. -/
theorem tidalReflectionBound {bare : ℝ} {selfEnergy scattering : ℂ}
    (h : ((bare : ℂ) - selfEnergy) * scattering = 1) :
    ‖scattering‖ * selfEnergy.im ≤ 1 := by
  have hnorm := congrArg (fun z : ℂ => ‖z‖) h
  rw [norm_mul, norm_one] at hnorm
  have him : selfEnergy.im ≤ ‖(bare : ℂ) - selfEnergy‖ := by
    rw [← norm_neg ((bare : ℂ) - selfEnergy), neg_sub]
    simpa [Complex.sub_im] using Complex.im_le_norm (selfEnergy - (bare : ℂ))
  rw [mul_comm, ← hnorm]
  exact mul_le_mul_of_nonneg_right him (norm_nonneg _)

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

/-- Linear feedback of a rank-one return changes its scalar strength only. -/
theorem rankOneEchoFeedback {V : Type*} [AddCommGroup V] [Module ℂ V]
    (sense : V →ₗ[ℂ] ℂ) (station : V →ₗ[ℂ] V) (launch pattern returned : V)
    (strength : ℂ) (hdenom : 1 - strength * sense (station pattern) ≠ 0) :
    returned = (strength * sense (launch + station returned)) • pattern ↔
      returned = (strength * sense launch / (1 - strength * sense (station pattern))) • pattern := by
  have hsense (amplitude : ℂ) (h : returned = amplitude • pattern) :
      sense (launch + station returned) = sense launch + amplitude * sense (station pattern) := by
    simp only [h, map_add, map_smul, smul_eq_mul]
  constructor
  · intro h
    let amplitude := strength * sense (launch + station returned)
    change returned = amplitude • pattern at h
    have hloop : amplitude = strength * (sense launch + amplitude * sense (station pattern)) := by
      change strength * sense (launch + station returned) = _
      rw [hsense amplitude h]
    have hvalue : amplitude = strength * sense launch / (1 - strength * sense (station pattern)) := by
      apply (eq_div_iff hdenom).2
      linear_combination hloop
    simpa only [hvalue] using h
  · intro h
    let amplitude := strength * sense launch / (1 - strength * sense (station pattern))
    change returned = amplitude • pattern at h
    have hvalue : amplitude * (1 - strength * sense (station pattern)) = strength * sense launch :=
      (eq_div_iff hdenom).1 rfl
    have hloop : amplitude = strength * (sense launch + amplitude * sense (station pattern)) := by
      linear_combination hvalue
    rw [hsense amplitude h, ← hloop]
    exact h

/-- Nonzero readout from the returned pattern makes a passive loop strictly subunit. -/
theorem observableEchoLoop {loop : ℂ} {stationReturn readout : ℝ}
    (hloop : ‖loop‖ ≤ stationReturn) (hpassive : stationReturn ^ 2 + readout ^ 2 ≤ 1)
    (hreadout : readout ≠ 0) : ‖loop‖ < 1 ∧ 1 - loop ≠ 0 := by
  have hstrict : ‖loop‖ < 1 := by
    have hreadoutSq : 0 < readout ^ 2 := sq_pos_of_ne_zero hreadout
    by_contra! hunit
    have hreturn : 1 ≤ stationReturn := hunit.trans hloop
    nlinarith [sq_nonneg (stationReturn - 1)]
  refine ⟨hstrict, ?_⟩
  intro hzero
  have hone : loop = 1 := (sub_eq_zero.mp hzero).symm
  simpa [hone] using hstrict

/-- The internal waves cancel in the net flux balance of two passive scatterers. -/
theorem composedEchoPassivity {incident external outgoing returned through : ℝ}
    (station : external + outgoing ≤ incident + returned)
    (target : returned + through ≤ outgoing) : external + through ≤ incident := by
  linarith

end HyperspaceWaves
