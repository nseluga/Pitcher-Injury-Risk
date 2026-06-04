# Data Dictionary

This document defines all columns in the core datasets produced by the Pitcher Injury Risk+ pipeline.

---

## Raw Data Sources

### Statcast (pitch-level)

| Column | Type | Source | Description |
|--------|------|---------|-------------|
| `pitcher` | int | Statcast | MLB player ID (MLBAM ID) |
| `player_name` | str | Statcast | Pitcher full name |
| `game_date` | date | Statcast | Date of the game |
| `game_pk` | int | Statcast | Unique game ID |
| `pitch_type` | str | Statcast | Pitch type code (FF, SL, CH, CU, SI, FC, etc.) |
| `release_speed` | float | Statcast | Pitch velocity at release point (mph) |
| `release_spin_rate` | float | Statcast | Spin rate at release (rpm) |
| `release_pos_x` | float | Statcast | Horizontal release position (ft, catcher's perspective) |
| `release_pos_z` | float | Statcast | Vertical release position (ft) |
| `release_extension` | float | Statcast | Extension toward plate at release (ft) |
| `pfx_x` | float | Statcast | Horizontal movement (in, vs. spinless trajectory) |
| `pfx_z` | float | Statcast | Vertical movement (in, vs. spinless trajectory) |
| `plate_x` | float | Statcast | Horizontal plate location (ft) |
| `plate_z` | float | Statcast | Vertical plate location (ft) |
| `balls` | int | Statcast | Ball count at pitch delivery |
| `strikes` | int | Statcast | Strike count at pitch delivery |
| `outs_when_up` | int | Statcast | Outs when pitch was thrown |
| `inning` | int | Statcast | Inning number |
| `inning_topbot` | str | Statcast | 'Top' or 'Bot' |

### Injury / IL Transactions

| Column | Type | Source | Description |
|--------|------|---------|-------------|
| `player_id` | int | MLB Transactions | MLBAM player ID |
| `player_name` | str | MLB Transactions | Player full name |
| `team` | str | MLB Transactions | Team abbreviation at time of transaction |
| `transaction_type` | str | MLB Transactions | 'IL_placement', 'IL_activation', 'IL_transfer' |
| `transaction_date` | date | MLB Transactions | Date of the IL action |
| `il_days` | int | MLB Transactions | IL list type (10, 15, 60 days) |
| `injury_type` | str | Parsed | Normalized injury category (see Injury Category Codes) |
| `injury_notes` | str | MLB Transactions | Raw notes from the transaction record |
| `activation_date` | date | Derived | Date of corresponding IL activation (if found) |
| `days_lost` | int | Derived | Days between placement and activation |
| `season_ending` | bool | Derived | True if no activation found in same season |

### Transactions (broader)

| Column | Type | Source | Description |
|--------|------|---------|-------------|
| `player_id` | int | MLB Transactions | MLBAM player ID |
| `team_from` | str | MLB Transactions | Originating team |
| `team_to` | str | MLB Transactions | Destination team |
| `transaction_type` | str | MLB Transactions | DFA, trade, option, outrighted, signed, released |
| `transaction_date` | date | MLB Transactions | Date of transaction |
| `is_rehab` | bool | Derived | True if a minor-league rehab assignment |

### Player Metadata

| Column | Type | Source | Description |
|--------|------|---------|-------------|
| `player_id` | int | MLB Stats API / pybaseball | MLBAM player ID |
| `player_name` | str | MLB Stats API | Full name |
| `birth_date` | date | MLB Stats API | Date of birth |
| `height_in` | int | MLB Stats API | Height in inches |
| `weight_lbs` | int | MLB Stats API | Weight in pounds |
| `throws` | str | MLB Stats API | Throwing hand ('R' or 'L') |
| `mlb_debut` | date | MLB Stats API | MLB debut date |
| `position` | str | MLB Stats API | Primary position code |

---

## Processed / Feature Columns

### Workload Features

| Column | Description |
|--------|-------------|
| `pitches_last_7d` | Total pitches thrown in the prior 7 calendar days |
| `pitches_last_15d` | Total pitches thrown in the prior 15 calendar days |
| `pitches_last_30d` | Total pitches thrown in the prior 30 calendar days |
| `pitches_last_90d` | Total pitches thrown in the prior 90 calendar days |
| `rest_days` | Calendar days since previous appearance |
| `season_pitches_to_date` | Cumulative pitches thrown in the current season (pre-game) |
| `season_ip_to_date` | Cumulative innings pitched in the current season (pre-game) |
| `high_leverage_pitches` | Pitches thrown in 3-2 count or inning 7+ |
| `high_leverage_pct` | high_leverage_pitches / pitch_count |
| `workload_zscore_30d` | Z-score of pitches_last_30d vs. pitcher's career distribution |

### Velocity Features

| Column | Description |
|--------|-------------|
| `velo_fb_mean` | Mean 4-seam fastball velocity this game (mph) |
| `velo_fb_max` | Max fastball velocity this game |
| `velo_fb_std` | Std dev of fastball velocity this game |
| `velo_fb_mean_last_7d` | Rolling 7-day mean fastball velocity |
| `velo_fb_mean_last_15d` | Rolling 15-day mean fastball velocity |
| `velo_fb_mean_last_30d` | Rolling 30-day mean fastball velocity |
| `velo_delta_vs_season_avg` | Fastball velo this game minus season-to-date average |
| `velocity_spike` | Boolean: velo exceeded 30-day rolling mean by ≥ 2 mph |
| `intragame_velo_drop` | Mean first-inning velo minus mean last-inning velo |

### Pitch Mix Features

| Column | Description |
|--------|-------------|
| `usage_FF` | Four-seam fastball usage rate this game |
| `usage_SL` | Slider usage rate this game |
| `usage_CH` | Changeup usage rate this game |
| `usage_CU` | Curveball usage rate this game |
| `usage_SI` | Sinker usage rate this game |
| `usage_FC` | Cutter usage rate this game |
| `pitch_mix_entropy` | Shannon entropy of pitch mix (higher = more diverse) |
| `slider_heavy` | Boolean: slider usage ≥ 35% |
| `usage_SL_delta_30d` | Slider rate minus 30-day rolling baseline |

### Injury History Features

| Column | Description |
|--------|-------------|
| `prior_il_stints_total` | Career IL stints before this game |
| `prior_il_elbow` | Prior elbow-related IL stints |
| `prior_il_shoulder` | Prior shoulder-related IL stints |
| `prior_il_forearm` | Prior forearm-related IL stints |
| `days_since_last_injury` | Days since most recent IL placement (NaN if no prior injury) |
| `prior_days_lost` | Days lost in most recent IL stint |
| `recurring_elbow_risk` | Boolean: pitcher has a prior elbow injury |
| `injury_frequency_rate` | IL stints per 162 games (career-to-date) |

### Label Columns (Modeling Targets)

| Column | Description |
|--------|-------------|
| `injured_within_30d` | Boolean: pitcher goes on IL within 30 days of this game |
| `injured_within_60d` | Boolean: pitcher goes on IL within 60 days |
| `injured_within_90d` | Boolean: pitcher goes on IL within 90 days |
| `days_until_next_injury` | Days from this game until next IL placement (NaN if none) |
| `next_injury_type` | Category of next injury (NaN if none in window) |
| `next_days_lost` | Days lost in next IL stint (NaN if none in window) |

---

## Injury Category Codes

| Code | Description |
|------|-------------|
| `elbow` | Elbow (UCL, flexor, medial epicondyle, olecranon) |
| `shoulder` | Shoulder (rotator cuff, labrum, AC joint, biceps tendon) |
| `forearm` | Forearm (flexor-pronator mass, strain) |
| `back` | Back, lumbar, thoracic |
| `oblique` | Oblique, side |
| `hamstring` | Hamstring |
| `hip` | Hip, groin |
| `knee` | Knee |
| `finger_hand` | Finger, hand, blister |
| `illness` | Non-musculoskeletal illness |
| `other` | Does not fit another category |
