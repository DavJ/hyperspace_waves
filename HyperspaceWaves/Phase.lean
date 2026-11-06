import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Data.Matrix.Mul
import Mathlib.Tactic.Ring

namespace HyperspaceWaves

theorem inverse_conjugate {G : Type*} [Group G] (shift target : G) :
    (shift * target * shift⁻¹)⁻¹ = shift * target⁻¹ * shift⁻¹ := by
  simp [mul_inv_rev, mul_assoc]

/-- The response difference transforms by conjugation if the free operator is invariant. -/
theorem translatedResolvent {A : Type*} [Ring A] (shift target free : Aˣ)
    (hfree : shift * free * shift⁻¹ = free) :
    (↑((shift * target * shift⁻¹)⁻¹) : A) - ↑free⁻¹ =
      (↑shift : A) * ((↑target⁻¹ : A) - ↑free⁻¹) * ↑shift⁻¹ := by
  have hfreeInv : shift * free⁻¹ * shift⁻¹ = free⁻¹ := by
    rw [← inverse_conjugate, hfree]
  have hcast : (↑shift : A) * ↑free⁻¹ * ↑shift⁻¹ = ↑free⁻¹ := by
    simpa only [Units.val_mul] using congrArg (fun x : Aˣ => (x : A)) hfreeInv
  rw [inverse_conjugate]
  simp only [Units.val_mul, mul_sub, sub_mul]
  rw [hcast]

/-- Diagonal phase conjugation contributes exactly the two port phases. -/
theorem translatedEntry {ι : Type*} [Fintype ι] [DecidableEq ι]
    (phase : ι → ℂ) (response : Matrix ι ι ℂ) (receive transmit : ι) :
    (Matrix.diagonal phase * response * Matrix.diagonal (fun i => star (phase i)))
        receive transmit =
      phase receive * response receive transmit * star (phase transmit) := by
  rw [Matrix.mul_diagonal, Matrix.diagonal_mul]

/-- The translated response in a uniform-to-uniform port is position independent. -/
theorem uniformPortInvariant {ι : Type*} [Fintype ι] [DecidableEq ι]
    (phase : ι → ℂ) (response : Matrix ι ι ℂ) (port : ι) (hport : phase port = 1) :
    (Matrix.diagonal phase * response * Matrix.diagonal (fun i => star (phase i)))
        port port = response port port := by
  rw [translatedEntry, hport]
  simp

/-- Trace and antisymmetric component of the adjacent-mode response recover the angle. -/
theorem adjacentModePhase (n θ : ℝ) :
    Real.cos ((n + 1) * θ) * Real.cos (n * θ) +
        Real.sin ((n + 1) * θ) * Real.sin (n * θ) = Real.cos θ ∧
    Real.sin ((n + 1) * θ) * Real.cos (n * θ) -
        Real.cos ((n + 1) * θ) * Real.sin (n * θ) = Real.sin θ := by
  have h : (n + 1) * θ - n * θ = θ := by ring
  constructor
  · rw [← Real.cos_sub, h]
  · rw [← Real.sin_sub, h]

end HyperspaceWaves
