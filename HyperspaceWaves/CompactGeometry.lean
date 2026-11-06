module

public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Analysis.Real.Sqrt
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.Ring

@[expose] public section

namespace HyperspaceWaves

open scoped BigOperators

noncomputable section

/-- Short arc for the relevant points, in units of a quarter circumference. -/
def normalizedDistance (u port : ℝ) : ℝ :=
  min |u - port| (4 - |u - port|)

/-- Contrasts of the squared distances from the two quarter ports and the origin. -/
def axisContrast (u : ℝ) : ℝ × ℝ :=
  (normalizedDistance u (-1) ^ 2 - normalizedDistance u 1 ^ 2,
   normalizedDistance u 1 ^ 2 + normalizedDistance u (-1) ^ 2 -
     2 * normalizedDistance u 0 ^ 2)

private theorem originDistanceSq {u : ℝ} (lo : -2 ≤ u) (hi : u ≤ 2) :
    normalizedDistance u 0 ^ 2 = u ^ 2 := by
  have bound : |u| ≤ 2 := abs_le.mpr ⟨lo, hi⟩
  have short : |u| ≤ 4 - |u| := by linarith
  simp only [normalizedDistance, sub_zero, min_eq_left short, sq_abs]

theorem axisContrast_middle {u : ℝ} (lo : -1 ≤ u) (hi : u ≤ 1) :
    axisContrast u = (4 * u, 2) := by
  have plus : normalizedDistance u 1 = 1 - u := by
    unfold normalizedDistance
    rw [abs_of_nonpos (by linarith : u - 1 ≤ 0),
      min_eq_left (by linarith : -(u - 1) ≤ 4 - -(u - 1))]
    ring
  have minus : normalizedDistance u (-1) = u + 1 := by
    unfold normalizedDistance
    rw [abs_of_nonneg (by linarith : 0 ≤ u - (-1)),
      min_eq_left (by linarith : u - (-1) ≤ 4 - (u - (-1)))]
    ring
  rw [axisContrast, plus, minus, originDistanceSq (by linarith) (by linarith)]
  exact Prod.ext (by dsimp; ring) (by dsimp; ring)

theorem axisContrast_positive {u : ℝ} (lo : 1 ≤ u) (hi : u ≤ 2) :
    axisContrast u = (8 - 4 * u, 10 - 8 * u) := by
  have plus : normalizedDistance u 1 = u - 1 := by
    unfold normalizedDistance
    rw [abs_of_nonneg (by linarith : 0 ≤ u - 1),
      min_eq_left (by linarith : u - 1 ≤ 4 - (u - 1))]
  have minus : normalizedDistance u (-1) = 3 - u := by
    unfold normalizedDistance
    rw [abs_of_nonneg (by linarith : 0 ≤ u - (-1)),
      min_eq_right (by linarith : 4 - (u - (-1)) ≤ u - (-1))]
    ring
  rw [axisContrast, plus, minus, originDistanceSq (by linarith) hi]
  exact Prod.ext (by dsimp; ring) (by dsimp; ring)

theorem axisContrast_negative {u : ℝ} (lo : -2 ≤ u) (hi : u ≤ -1) :
    axisContrast u = (-8 - 4 * u, 10 + 8 * u) := by
  have plus : normalizedDistance u 1 = 3 + u := by
    unfold normalizedDistance
    rw [abs_of_nonpos (by linarith : u - 1 ≤ 0),
      min_eq_right (by linarith : 4 - -(u - 1) ≤ -(u - 1))]
    ring
  have minus : normalizedDistance u (-1) = -u - 1 := by
    unfold normalizedDistance
    rw [abs_of_nonpos (by linarith : u - (-1) ≤ 0),
      min_eq_left (by linarith : -(u - (-1)) ≤ 4 - -(u - (-1)))]
    ring
  rw [axisContrast, plus, minus, originDistanceSq lo (by linarith)]
  exact Prod.ext (by dsimp; ring) (by dsimp; ring)

private theorem axisRegions (u : ℝ) (lo : -2 ≤ u) (hi : u < 2) :
    (u < -1 ∧ axisContrast u = (-8 - 4 * u, 10 + 8 * u)) ∨
    (-1 ≤ u ∧ u ≤ 1 ∧ axisContrast u = (4 * u, 2)) ∨
    (1 < u ∧ axisContrast u = (8 - 4 * u, 10 - 8 * u)) := by
  by_cases neg : u < -1
  · exact Or.inl ⟨neg, axisContrast_negative lo neg.le⟩
  · have lower : -1 ≤ u := le_of_not_gt neg
    by_cases middle : u ≤ 1
    · exact Or.inr (Or.inl ⟨lower, middle, axisContrast_middle lower middle⟩)
    · have upper : 1 < u := lt_of_not_ge middle
      exact Or.inr (Or.inr ⟨upper, axisContrast_positive upper.le hi.le⟩)

/-- The representative interval includes just one side of the periodic seam. -/
theorem axisContrast_injective {u w : ℝ}
    (ulo : -2 ≤ u) (uhi : u < 2) (wlo : -2 ≤ w) (whi : w < 2)
    (same : axisContrast u = axisContrast w) : u = w := by
  rcases axisRegions u ulo uhi with ⟨uneg, hu⟩ | ⟨ulower, uupper, hu⟩ | ⟨upos, hu⟩ <;>
    rcases axisRegions w wlo whi with ⟨wneg, hw⟩ | ⟨wlower, wupper, hw⟩ | ⟨wpos, hw⟩ <;>
    rw [hu, hw] at same
  all_goals
    have first := congrArg Prod.fst same
    have second := congrArg Prod.snd same
    dsimp at first second
    linarith

def referenceSquared {ι : Type*} [Fintype ι] (quarter point : ι → ℝ) : ℝ :=
  ∑ j, quarter j ^ 2 * normalizedDistance (point j) 0 ^ 2

def sideSquared {ι : Type*} [Fintype ι] [DecidableEq ι]
    (quarter point : ι → ℝ) (i : ι) (port : ℝ) : ℝ :=
  ∑ j, quarter j ^ 2 * normalizedDistance (point j) (if j = i then port else 0) ^ 2

/-- Subtracting the reference eliminates all other axes. -/
theorem sideSquared_sub_reference {ι : Type*} [Fintype ι] [DecidableEq ι]
    (quarter point : ι → ℝ) (i : ι) (port : ℝ) :
    sideSquared quarter point i port - referenceSquared quarter point =
      quarter i ^ 2 *
        (normalizedDistance (point i) port ^ 2 - normalizedDistance (point i) 0 ^ 2) := by
  rw [sideSquared, referenceSquared, ← Finset.sum_sub_distrib, Finset.sum_eq_single i]
  · simp <;> ring
  · intro j _ different
    simp [different]
  · simp

/-- One reference and two quarter ports per axis suffice for noiseless squared ranges. -/
theorem crossPorts_injective {ι : Type*} [Fintype ι] [DecidableEq ι]
    (quarter u w : ι → ℝ) (hquarter : ∀ i, 0 < quarter i)
    (hu : ∀ i, -2 ≤ u i ∧ u i < 2) (hw : ∀ i, -2 ≤ w i ∧ w i < 2)
    (href : referenceSquared quarter u = referenceSquared quarter w)
    (hplus : ∀ i, sideSquared quarter u i 1 = sideSquared quarter w i 1)
    (hminus : ∀ i, sideSquared quarter u i (-1) = sideSquared quarter w i (-1)) :
    u = w := by
  funext i
  have leg (port : ℝ) (hs : sideSquared quarter u i port = sideSquared quarter w i port) :
      normalizedDistance (u i) port ^ 2 - normalizedDistance (u i) 0 ^ 2 =
      normalizedDistance (w i) port ^ 2 - normalizedDistance (w i) 0 ^ 2 := by
    apply mul_left_cancel₀ (pow_ne_zero 2 (hquarter i).ne')
    rw [← sideSquared_sub_reference quarter u i port,
      ← sideSquared_sub_reference quarter w i port, hs, href]
  have plus := leg 1 (hplus i)
  have minus := leg (-1) (hminus i)
  apply axisContrast_injective (hu i).1 (hu i).2 (hw i).1 (hw i).2
  unfold axisContrast
  exact Prod.ext (by dsimp; linarith) (by dsimp; linarith)

/-- The assumed shortest round-trip time in a medium of known speed. -/
def roundTripTime (speed squaredRange : ℝ) : ℝ :=
  2 * Real.sqrt squaredRange / speed

theorem roundTripTime_injective {speed x y : ℝ}
    (hspeed : 0 < speed) (hx : 0 ≤ x) (hy : 0 ≤ y)
    (h : roundTripTime speed x = roundTripTime speed y) : x = y := by
  have scaled := congrArg (fun t : ℝ => t * speed) h
  simp only [roundTripTime, div_mul_cancel₀ _ hspeed.ne'] at scaled
  have roots : Real.sqrt x = Real.sqrt y := by linarith
  calc
    x = (Real.sqrt x) ^ 2 := (Real.sq_sqrt hx).symm
    _ = (Real.sqrt y) ^ 2 := by rw [roots]
    _ = y := Real.sq_sqrt hy

/-- Equality of the first returns at the cross ports determines the target. -/
theorem crossPortTimes_injective {ι : Type*} [Fintype ι] [DecidableEq ι]
    (quarter u w : ι → ℝ) (hquarter : ∀ i, 0 < quarter i)
    (hu : ∀ i, -2 ≤ u i ∧ u i < 2) (hw : ∀ i, -2 ≤ w i ∧ w i < 2)
    (speed : ℝ) (hspeed : 0 < speed)
    (href : roundTripTime speed (referenceSquared quarter u) =
      roundTripTime speed (referenceSquared quarter w))
    (hplus : ∀ i, roundTripTime speed (sideSquared quarter u i 1) =
      roundTripTime speed (sideSquared quarter w i 1))
    (hminus : ∀ i, roundTripTime speed (sideSquared quarter u i (-1)) =
      roundTripTime speed (sideSquared quarter w i (-1))) :
    u = w := by
  have range_nonneg (point port : ι → ℝ) :
      0 ≤ ∑ j, quarter j ^ 2 * normalizedDistance (point j) (port j) ^ 2 :=
    Finset.sum_nonneg fun j _ =>
      mul_nonneg (sq_nonneg _) (sq_nonneg _)
  apply crossPorts_injective quarter u w hquarter hu hw
  · exact roundTripTime_injective hspeed (range_nonneg u fun _ => 0)
      (range_nonneg w fun _ => 0) href
  · intro i
    exact roundTripTime_injective hspeed (range_nonneg u fun j => if j = i then 1 else 0)
      (range_nonneg w fun j => if j = i then 1 else 0) (hplus i)
  · intro i
    exact roundTripTime_injective hspeed (range_nonneg u fun j => if j = i then -1 else 0)
      (range_nonneg w fun j => if j = i then -1 else 0) (hminus i)

end

end HyperspaceWaves
