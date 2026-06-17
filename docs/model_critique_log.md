# Model Critique Log

One entry per notebook, appended after each Phase 2 critique session.
Format: research findings → decisions critiqued → improvements implemented → verification.

<!-- Entries will be appended below as each notebook is critiqued -->

## [2026-06-16] NB05 — Feature Engineering: Critique & Improvements

### Research Findings

1. **[Logue et al. 2021, PubMed 34189147]** Analyzed 223 MLB pitchers pre-Tommy John surgery. Found significant velocity *decreases* in 4-seam FBs, 2-seam FBs, and sliders in the ~15 games before surgery, plus a significant *negative* spin rate trend on 4-seam fastballs. This directly motivates (a) a 14-day velocity rolling window and (b) slider-specific spin rate tracking.

2. **[Logue et al. 2025, PMC12717397]** Acute UCL injuries study — tracked 5 Statcast metrics: velocity, spin rate, release extension, arm angle, and acceleration magnitude. Identified per-pitch baseline comparison as the most sensitive detection method. Confirms that release point tracking (extension, arm angle) and spin rate are the strongest leading indicators.

3. **[ACWR review, PMC7534929]** Baseball-specific ACWR review confirms the 7-day acute / 28-day chronic window is the industry standard. Threshold for elevated risk is ACWR ≥1.27 (not 1.5 as in field sports). Athletes with ACWR outside the 0.7–1.3 range are ~8× more likely to sustain a throwing-overuse injury.

4. **[Velocity/pitch type UCL paper, PubMed 26995458]** Sliders with higher spin rate and fastballs thrown at higher velocity were independently associated with UCL surgery risk. This motivated adding `sl_spin_mean` and `sl_spin_delta_30d` as explicit features, which were missing from the original implementation.

5. **[Release point variability, PMC11608975 / arXiv 2603.04864]** Release point range and CV features contribute 33–39% of total model importance in injury risk models, consistently outranking mean values. Our within-game `rel_x_std` / `rel_z_std` captures this, but repeated exposure to extreme mechanics (top-decile pitches) may be an even stronger signal. Not implemented now (requires per-pitch percentile — future work).

### Decisions Critiqued

- **ACWR window (7:28):** Confirmed as the correct industry standard for baseball. **Verdict: no change — current approach matches literature.**

- **ACWR denominator (pitches_28d / 4):** Divides by 4 to express chronic load as a weekly average. This is the standard rolling-ACWR formulation. Exponentially-weighted ACWR (ewACWR) is an alternative endorsed by some researchers but adds complexity without strong evidence of superiority in baseball-specific studies. **Verdict: no change for now; ewACWR flagged as future enhancement.**

- **`intragame_velo_drop` computation:** Used `df[inning_col].max()` globally to identify "late innings," so the "late" window was innings ≥ (global_max − 1). For starters exiting in inning 5–7, the global max (typically 9) meant zero pitches qualified as "late," making this feature NaN for most rows. **Verdict: BUG — fixed by computing per-game max inning.**

- **Velocity rolling windows (7/30/90 days):** The Logue et al. pre-surgery window is ~15 games for starters (~14 calendar days). The 7d window is too short for starters on 5-day rest; 30d is too long to isolate the acute decline. A 14-day window fills this gap. **Verdict: added 14-day window.**

- **Slider spin rate feature:** Literature directly links slider spin rate to UCL injury risk (Logue et al. 2021, PubMed 26995458). Movement module only tracked `fb_spin_mean`. **Verdict: added `sl_spin_mean`, `sl_spin_mean_30d_avg`, `sl_spin_delta_30d`.**

- **Pitch mix entropy:** Shannon entropy of pitch distribution is present. No direct validation in the peer-reviewed literature for entropy as a standalone predictor, but pitch mix change is validated (curveball % increases before TJ surgery per Logue et al.). The entropy feature captures total mix variance; individual pitch-type delta features (`sl_pct_delta_30d`, etc.) are more interpretable. **Verdict: keep entropy; the per-type delta features already exist.**

- **Injury history features:** Prior IL count, days since last injury, days lost prior — all well-supported by literature as strongest-known predictors. **Verdict: no change — current approach matches literature.**

### Improvements Implemented

1. **Bug fix — `intragame_velo_drop`** (`src/features/velocity_features.py`): Changed `df[inning_col].max()` (global) to a per-pitcher-game max inning join. Pitchers who exit in inning 6 now get a valid late-inning reading instead of NaN. This affects all downstream models that use this feature.

2. **New feature — 14-day velocity window** (`src/features/velocity_features.py`): Added `fb_velo_14d_avg` window to `build_velocity_features()`. Directly aligned with the Logue et al. pre-surgery analysis window.

3. **New features — slider spin rate** (`src/features/movement_features.py`): Added `sl_spin_mean`, `sl_spin_mean_30d_avg`, and `sl_spin_delta_30d`. `SLIDER_TYPES = {"SL", "ST"}` constant added. Pitchers who do not throw sliders will have NaN — downstream imputation handles this correctly.

### Verified

- Smoke test `.scratch/test_nb05_changes.py` passed: `intragame_velo_drop` returns non-NaN values, `fb_velo_14d_avg` present, `sl_spin_mean` / `sl_spin_delta_30d` present.
- Full notebook run: `python run_notebooks.py --only 05 --fail-fast` — PASS.
- `python scripts/verify_outputs.py --only 05` — PASS.
