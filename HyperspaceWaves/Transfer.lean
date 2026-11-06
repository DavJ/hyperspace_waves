module

public import Mathlib.Basic.Complex.Basic
public import Mathlib.Tactic.LinearCombination
public import Mathlib.Tactic.Positivity

public section

namespace HyperspaceWaves

/-- Positive quadratic potential for two coupled modes with positive gaps. -/
theorem coupledPotential_pos {a b coupling x y : ℝ}
    (ha : 0 < a) (hb : 0 < b) (hc : 0 ≤ coupling) (hxy : x ≠ 0 ∨ y ≠ 0) :
    0 < a * x ^ 2 + b * y ^ 2 + coupling * (x - y) ^ 2 := by
  have hx : 0 ≤ a * x ^ 2 := mul_nonneg ha.le (sq_nonneg x)
  have hy : 0 ≤ b * y ^ 2 := mul_nonneg hb.le (sq_nonneg y)
  have hcoupling : 0 ≤ coupling * (x - y) ^ 2 := mul_nonneg hc (sq_nonneg (x - y))
  rcases hxy with hx0 | hy0
  · exact add_pos_of_pos_of_nonneg
      (add_pos_of_pos_of_nonneg (mul_pos ha (sq_pos_of_ne_zero hx0)) hy) hcoupling
  · exact add_pos_of_pos_of_nonneg
      (add_pos_of_nonneg_of_pos hx (mul_pos hb (sq_pos_of_ne_zero hy0))) hcoupling

/-- Eliminate the driven coordinate in the conditional two-mode system. -/
theorem coupledTransfer {a b coupling drive x y : ℂ}
    (hx : a * x - coupling * y = drive) (hy : b * y - coupling * x = 0)
    (hdet : a * b - coupling ^ 2 ≠ 0) :
    y = coupling * drive / (a * b - coupling ^ 2) := by
  apply (eq_div_iff hdet).2
  linear_combination coupling * hx + a * hy

/-- Point-target echo after solving the target's own Green-function response. -/
theorem pointTargetEcho {stiffness localGreen driveGreen receiveGreen drive atTarget echo : ℂ}
    (htarget : atTarget = driveGreen * drive - stiffness * localGreen * atTarget)
    (hecho : echo = -stiffness * receiveGreen * atTarget)
    (hpole : 1 + stiffness * localGreen ≠ 0) :
    echo = (-stiffness * receiveGreen * driveGreen * drive) /
      (1 + stiffness * localGreen) := by
  apply (eq_div_iff hpole).2
  linear_combination (1 + stiffness * localGreen) * hecho - stiffness * receiveGreen * htarget

end HyperspaceWaves
