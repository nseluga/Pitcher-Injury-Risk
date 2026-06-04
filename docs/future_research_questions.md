# Future Research Questions

This document catalogs the research questions that the Pitcher Injury Risk+ platform is designed to eventually answer. Questions are organized by theme and annotated with the modeling or simulation components needed to address them.

---

## Pitcher Role and Usage Structure

### Are traditional starters inherently less healthy than hybrid starters?
- **Approach:** Matched cohort comparison controlling for age, workload, and pitch mix.
  Use `usage_strategy_simulator.compare_starter_vs_hybrid()`.
- **Hypothesis:** Hybrid starters (e.g., 4-inning openers or "bulk" relievers) may
  accumulate workload more gradually, reducing peak-game stress at the cost of
  more frequent activation.
- **Required data:** Role classification, IL history, pitch-level Statcast.

### What is the optimal pitch count for different pitcher archetypes?
- **Approach:** For each archetype group, use `workload_simulator.find_optimal_pitch_count()`
  to identify the pitch count that minimizes Injury Risk+. Will vary by age, injury history,
  pitch mix, and physical profile.
- **Hypothesis:** Optimal pitch count is not universal — a power fastball/slider reliever
  likely has a lower optimal threshold than a contact-inducing sinker-baller.

### Should some pitchers transition between starting and relieving?
- **Approach:** For starters with persistently high Injury Risk+, simulate a role transition
  using `usage_strategy_simulator.simulate_role_transition()` and estimate the expected
  risk reduction.
- **Hypothesis:** Some pitchers' mechanics or pitch profiles are better suited to short,
  high-intensity bursts (relief) rather than extended appearances.

---

## Pitch Mix and Arm Health

### Can certain pitch mixes reduce injury risk?
- **Approach:** Use `pitch_mix_simulator.find_risk_minimizing_pitch_mix()` for individual
  pitchers and `pitch_mix_simulator.compute_pitch_type_risk_sensitivity()` at the population level.
- **Hypothesis:** Pitch mixes with lower mechanical stress (e.g., sinker-heavy vs.
  slider-heavy) are associated with lower elbow injury rates.

### Does reducing slider usage lower long-term injury risk?
- **Approach:** Intervention analysis — compare injury rates before and after pitchers
  significantly reduced slider usage. Also use
  `pitch_mix_simulator.simulate_slider_reduction()` for counterfactuals.
- **Hypothesis:** Sliders generate high valgus torque at the elbow. Reducing usage
  should lower UCL stress, particularly in heavy slider users.
- **Confound:** Pitchers may reduce slider usage *because* they are already feeling
  discomfort — selection bias must be addressed.

### Are velocity spikes more dangerous than sustained high velocity?
- **Approach:** Compare injury rates in the 30 days following a velocity spike vs. 30 days
  following matched high-velocity appearances without a spike. Use `velocity_spike` feature
  in SHAP analysis and partial dependence plots.
- **Hypothesis:** An unusual spike relative to a pitcher's own baseline is a stronger
  injury signal than absolute velocity, because it suggests a change in mechanics or effort.

---

## Workload and Scheduling

### Is there an optimal rest schedule?
- **Approach:** Use `workload_simulator.simulate_rest_schedule()` across a grid of rest
  values and compare predicted injury risk. Segment by archetype (starters may benefit
  from more rest than relievers).
- **Hypothesis:** The optimal rest interval is non-linear — both too little rest (fatigue)
  and too much rest (deconditioning) may increase risk.

### Can workload be distributed differently across a pitching staff to reduce total injury burden?
- **Approach:** `usage_strategy_simulator.simulate_staff_workload_distribution()` with
  risk-weighted redistribution strategy.
- **Hypothesis:** Teams that concentrate workload on 1–2 high-usage relievers may be
  increasing total staff injury risk. A more even distribution with targeted use of
  the lowest-risk arms may reduce aggregate IL days.

---

## Mechanical and Biometric Factors

### Does release point inconsistency predict injury?
- **Approach:** Include `release_x_std` and `release_z_std` as features in the injury
  prediction model. Analyze SHAP values and partial dependence.
- **Hypothesis:** Mechanical inconsistency (as proxied by release point variability)
  reflects muscular fatigue or compensatory movement patterns that precede injury.

### Does spin rate decline predict injury before other signals appear?
- **Approach:** Analyze `spin_rate_delta_fb` in the weeks before confirmed IL placements.
  Build a lead-time analysis: how many days before injury does spin rate begin to drop?
- **Hypothesis:** Spin rate is particularly sensitive to grip fatigue and finger/hand
  health, and may decline before velocity does.

---

## Longitudinal and Career Effects

### How does injury risk evolve across a pitcher's career?
- **Approach:** Plot age-adjusted Injury Risk+ trajectories. Segment by initial archetype,
  injury history, and physical build.
- **Hypothesis:** Risk is not monotonically increasing with age. Young pitchers face
  elevated risk from workload inexperience; veterans face elevated risk from cumulative
  wear. Mid-career (ages 26–30) may be the lowest-risk window.

### Do pitchers who had Tommy John Surgery face permanently elevated risk?
- **Approach:** Compare post-TJS injury rates against a matched cohort of pitchers
  without TJS, controlling for time since return and workload.
- **Hypothesis:** Post-TJS pitchers who return to pre-surgery velocity may face
  elevated stress elsewhere (shoulder compensation) even after full elbow recovery.

---

## Methodological Questions

### What is the minimum feature set needed for a reliable Injury Risk+ score?
- **Approach:** Recursive feature elimination or SHAP-based pruning. Identify the
  smallest feature set that retains ≥ 95% of full-model predictive performance.
- **Motivation:** A simpler model is easier to deploy in real-time, requires less
  data, and is more interpretable for front offices.

### How far in advance can injury be reliably predicted?
- **Approach:** Train separate models for 14-day, 30-day, 60-day, and 90-day horizons.
  Compare calibration and discrimination metrics across horizons.
- **Hypothesis:** Short horizons (14–30 days) will have substantially better calibration
  than long horizons (60–90 days), where uncertainty dominates.
