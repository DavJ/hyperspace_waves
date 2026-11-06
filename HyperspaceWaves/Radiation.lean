import Mathlib.Data.Complex.Basic
import Mathlib.Tactic.LinearCombination
import Mathlib.Tactic.Linarith

namespace HyperspaceWaves

/-- Relative to the first compact mass, the sum propagates and the difference is below cutoff. -/
theorem sidebandCutoffs {pump signal : ℝ}
    (hpump : 0 < pump) (hsignal : 0 < signal) (hsmall : signal < pump) :
    pump ^ 2 < (pump + signal) ^ 2 ∧ (pump - signal) ^ 2 < pump ^ 2 := by
  constructor
  · nlinarith [mul_pos hpump hsignal, sq_nonneg signal]
  · have hpositive : 0 < (2 * pump - signal) * signal :=
      mul_pos (by linarith) hsignal
    nlinarith

/-- Removing the homogeneous forced response leaves the finite-layer source. -/
theorem localizedLayerSource {pump signal k2 z logDeriv c a da dda h ddh : ℂ}
    (hdisp : k2 = signal ^ 2 + 2 * pump * signal)
    (hmaxwell : dda = -(signal ^ 2) * a - logDeriv * da)
    (hmetric : -ddh - k2 * h = 2 * pump * signal * z * c * a) :
    -(ddh + c * dda) - k2 * (h + c * a) =
      c * (2 * pump * signal * (z - 1) * a + logDeriv * da) := by
  rw [hdisp] at hmetric ⊢
  rw [hmaxwell]
  linear_combination hmetric

/-- The two first-order source contributions give the phase-mismatch factor. -/
theorem emissionMismatch {pump signal k : ℝ}
    (hdisp : k ^ 2 = signal ^ 2 + 2 * pump * signal) :
    2 * pump * signal - signal * (k - signal) = k * (k - signal) := by
  linear_combination -hdisp

end HyperspaceWaves
