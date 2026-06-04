# Injury Risk+ Design Document

## Concept

Injury Risk+ is a single, normalized composite score that expresses a pitcher's current injury risk relative to the league average. It is modeled after other "plus" metrics in baseball analytics (ERA+, FIP+, wRC+) which normalize raw statistics so that 100 always represents the population mean.

**Scale:**
- **100** = League-average injury risk for that pitcher's archetype and era
- **> 100** = Riskier than average (e.g., 130 = 30% riskier than average)
- **< 100** = Safer than average (e.g., 70 = 30% safer than average)

---

## Why a Composite Score?

Injury risk is not captured by any single metric. A pitcher can have manageable pitch counts but dangerous mechanics, or pristine recent health but a fragile injury history. A composite score integrates:

1. **Probability of injury** — How likely is this pitcher to be injured in the next 30/60/90 days?
2. **Severity expectation** — If injured, how many days is he likely to miss?
3. **Time-to-event signal** — How imminent is the risk (hazard rate from survival model)?

Each component is weighted and blended into a single raw score, which is then normalized to the 100-scale.

---

## Score Construction

### Step 1: Compute Component Predictions

Run the multi-task model to generate three outputs per pitcher-game:

| Component | Model | Output |
|-----------|-------|--------|
| `injury_prob_30d` | Multi-task classifier | P(injury within 30 days) |
| `expected_days_lost` | Multi-task regressor | E[days lost \| injured] |
| `hazard_rate` | Survival model | Instantaneous hazard h(t) |

### Step 2: Blend into Raw Risk Score

```
raw_risk_score = (
    w_prob * injury_prob_30d
    + w_severity * expected_days_lost_normalized
    + w_hazard * hazard_rate_normalized
)
```

Default weights (subject to calibration):
- `w_prob = 0.50`
- `w_severity = 0.30`
- `w_hazard = 0.20`

Weights are optimized via cross-validated grid search in `score_calibration.py`.

### Step 3: Normalize to 100-Scale

For each season-archetype group, divide by the group mean and multiply by 100:

```
injury_risk_plus = (raw_risk_score / group_mean_raw_score) * 100
```

Group means are stored in the normalization reference table produced by
`score_calibration.build_normalization_reference()`.

### Step 4: Probability Calibration

Before blending, `injury_prob_30d` is passed through an isotonic regression
calibration model to ensure that predicted probabilities correspond to observed
injury rates. See `score_calibration.calibrate_probabilities()`.

---

## Normalization Groups

Scores are normalized within:

1. **Season** — Prevents era drift (a 130 in 2018 = a 130 in 2024)
2. **Pitcher archetype** — Starters and relievers have systematically different
   baseline risk profiles; comparing them on the same scale without adjustment
   would be misleading

Archetype groups (see `risk_factor_features.encode_pitcher_archetype`):
- `starter`
- `hybrid_starter`
- `long_reliever`
- `middle_reliever`
- `setup`
- `closer`

---

## Score Interpretation

| Range | Interpretation |
|-------|---------------|
| < 70 | Very low risk — pitcher is substantially safer than peers |
| 70–90 | Below-average risk |
| 90–110 | Average risk (± ~10% of league mean) |
| 110–130 | Above-average risk |
| 130–160 | High risk — warrants attention |
| > 160 | Extreme risk — significant injury concern |

---

## Limitations and Known Issues

1. **Data availability lag** — Statcast data from the current season may have
   a 24-hour or longer lag. Scores should be labeled with their data cutoff date.

2. **Small sample early in season** — Rolling features require sufficient history.
   Scores computed in April of a debut season are less reliable than mid-season scores.

3. **Injury type heterogeneity** — The score does not distinguish between
   elbow injuries (high impact, often long-term) and finger blisters (low impact).
   A severity-weighted variant is planned.

4. **Missing mechanical data** — Biomechanical data (arm stress, valgus torque)
   from high-speed cameras is not publicly available. This is the single biggest
   gap in the feature set.

5. **Survivorship bias** — Pitchers who are already injured are not at risk.
   The dataset inherently reflects pitchers healthy enough to pitch.

---

## Future Enhancements

- Separate Injury Risk+ subscores by body region (elbow vs. shoulder vs. back)
- Confidence intervals around each score
- Career trajectory version: aggregate Injury Risk+ across seasons
- Real-time score updates after each appearance
- Integration with biomechanics data if ever made public
