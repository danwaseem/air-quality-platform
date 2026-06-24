# Air Quality GBM Model — Geographic Bias Audit

> **Scope:** GBM (HistGradientBoostingRegressor) predictions on the held-out test set,
> using the same time-based 80/20 split as training (no temporal leakage).
> 10 US states, 2023, hourly cadence.
> This document measures and documents disparities; it does not claim they have been eliminated.

## Summary of Findings

| Pollutant | Overall MAE | State MAE range | Ratio | Worst state | Flagged |
|-----------|-------------|-----------------|-------|-------------|---------|
| pm25 (µg/m³) | 4.3781 | 2.0564 – 5.9061 | 2.87× | Arizona | none |
| no2 (ppb) | 5.8871 | 4.4037 – 8.1142 | 1.84× | New York | none |
| ozone (ppm) | 0.0072 | 0.0065 – 0.0082 | 1.26× | New York | none |

---

## PM25 (µg/m³)

**Overall test MAE:** 4.3781 · RMSE: 8.0367 · R²: 0.1131 · n=158,193

### By Census Region

| Region | MAE | RMSE | R² | n_test | n_stations | States |
|--------|-----|------|----|--------|------------|--------|
| South | 3.6718 | 6.8712 | 0.0760 | 49,810 | 35 | Florida, Georgia, North Carolina, Texas |
| Midwest | 3.9657 | 5.8060 | 0.0628 | 31,164 | 20 | Illinois, Ohio |
| Northeast | 4.6673 | 7.2132 | 0.0362 | 25,361 | 17 | New York, Pennsylvania |
| West | 5.1629 | 10.2946 | 0.1349 | 51,858 | 36 | Arizona, California |

### By State

| State | Region | MAE | RMSE | R² | n_test | n_stations | Flagged |
|-------|--------|-----|------|----|--------|------------|---------|
| Florida | South | 2.0564 | 6.6215 | 0.0295 | 15,253 | 10 | no |
| Georgia | South | 3.4327 | 4.9698 | 0.2321 | 1,582 | 1 | no |
| Illinois | Midwest | 3.7242 | 5.2061 | -0.0064 | 7,900 | 5 | no |
| Ohio | Midwest | 4.0477 | 5.9961 | 0.0723 | 23,264 | 15 | no |
| Texas | South | 4.4078 | 7.2058 | 0.0341 | 26,045 | 19 | no |
| North Carolina | South | 4.5158 | 6.4857 | 0.1400 | 6,930 | 5 | no |
| New York | Northeast | 4.5580 | 6.3920 | -0.0022 | 6,821 | 5 | no |
| Pennsylvania | Northeast | 4.7075 | 7.4927 | 0.0384 | 18,540 | 12 | no |
| California | West | 4.7616 | 7.5457 | 0.2138 | 33,673 | 24 | no |
| Arizona | West | 5.9061 | 14.0281 | 0.0807 | 18,185 | 12 | no |

### Disparity Analysis

- **Best state:** Florida (MAE = 2.0564)
- **Worst state:** Arizona (MAE = 5.9061)
- **State MAE spread ratio (worst/best):** 2.87×
- **Best region:** South (MAE = 3.6718)
- **Worst region:** West (MAE = 5.1629)
- **Region MAE spread ratio:** 1.41×
- **Flagged states:** none
- **Spearman ρ(n\_test, MAE) across states:** 0.382 — moderate positive correlation (not significant at p<0.05)

## NO2 (ppb)

**Overall test MAE:** 5.8871 · RMSE: 8.0122 · R²: 0.4008 · n=162,171

### By Census Region

| Region | MAE | RMSE | R² | n_test | n_stations | States |
|--------|-----|------|----|--------|------------|--------|
| South | 5.2598 | 7.3080 | 0.3452 | 58,933 | 39 | Florida, Georgia, North Carolina, Texas |
| Midwest | 6.0280 | 8.0100 | 0.2516 | 17,395 | 11 | Illinois, Ohio |
| West | 6.1756 | 8.2900 | 0.4430 | 69,629 | 43 | Arizona, California |
| Northeast | 6.7773 | 9.1653 | 0.2028 | 16,214 | 10 | New York, Pennsylvania |

### By State

| State | Region | MAE | RMSE | R² | n_test | n_stations | Flagged |
|-------|--------|-----|------|----|--------|------------|---------|
| Florida | South | 4.4037 | 6.4419 | 0.3494 | 10,721 | 7 | no |
| North Carolina | South | 5.1226 | 6.8208 | 0.3186 | 6,553 | 4 | no |
| Georgia | South | 5.4245 | 7.0242 | 0.3205 | 4,493 | 3 | no |
| Ohio | Midwest | 5.4312 | 7.2741 | 0.2902 | 9,303 | 6 | no |
| Texas | South | 5.5110 | 7.6516 | 0.3348 | 37,166 | 25 | no |
| Pennsylvania | Northeast | 5.9643 | 7.9100 | 0.2403 | 10,083 | 6 | no |
| Arizona | West | 6.1338 | 8.2390 | 0.4487 | 13,731 | 9 | no |
| California | West | 6.1858 | 8.3025 | 0.4386 | 55,898 | 34 | no |
| Illinois | Midwest | 6.7141 | 8.7801 | 0.0861 | 8,092 | 5 | no |
| New York | Northeast | 8.1142 | 10.9203 | 0.0048 | 6,131 | 4 | no |

### Disparity Analysis

- **Best state:** Florida (MAE = 4.4037)
- **Worst state:** New York (MAE = 8.1142)
- **State MAE spread ratio (worst/best):** 1.84×
- **Best region:** South (MAE = 5.2598)
- **Worst region:** Northeast (MAE = 6.7773)
- **Region MAE spread ratio:** 1.29×
- **Flagged states:** none
- **Spearman ρ(n\_test, MAE) across states:** 0.079 — weak positive correlation (not significant at p<0.05)

## OZONE (ppm)

**Overall test MAE:** 0.0072 · RMSE: 0.0093 · R²: 0.5990 · n=223,694

### By Census Region

| Region | MAE | RMSE | R² | n_test | n_stations | States |
|--------|-----|------|----|--------|------------|--------|
| West | 0.0069 | 0.0091 | 0.6834 | 113,288 | 64 | Arizona, California |
| South | 0.0075 | 0.0095 | 0.4931 | 83,249 | 48 | Florida, Georgia, North Carolina, Texas |
| Northeast | 0.0077 | 0.0095 | 0.2580 | 15,186 | 10 | New York, Pennsylvania |
| Midwest | 0.0078 | 0.0098 | 0.1567 | 11,971 | 20 | Illinois, Ohio |

### By State

| State | Region | MAE | RMSE | R² | n_test | n_stations | Flagged |
|-------|--------|-----|------|----|--------|------------|---------|
| Arizona | West | 0.0065 | 0.0087 | 0.6868 | 55,380 | 31 | no |
| Florida | South | 0.0071 | 0.0089 | 0.3838 | 21,181 | 12 | no |
| California | West | 0.0073 | 0.0095 | 0.6724 | 57,908 | 33 | no |
| Georgia | South | 0.0075 | 0.0094 | 0.5781 | 2,238 | 2 | no |
| Pennsylvania | Northeast | 0.0075 | 0.0092 | 0.3255 | 10,823 | 6 | no |
| North Carolina | South | 0.0077 | 0.0096 | 0.5221 | 4,016 | 3 | no |
| Texas | South | 0.0077 | 0.0097 | 0.4427 | 55,814 | 31 | no |
| Illinois | Midwest | 0.0078 | 0.0096 | 0.0975 | 5,664 | 11 | no |
| Ohio | Midwest | 0.0079 | 0.0099 | 0.1819 | 6,307 | 9 | no |
| New York | Northeast | 0.0082 | 0.0100 | 0.0158 | 4,363 | 4 | no |

### Disparity Analysis

- **Best state:** Arizona (MAE = 0.0065)
- **Worst state:** New York (MAE = 0.0082)
- **State MAE spread ratio (worst/best):** 1.26×
- **Best region:** West (MAE = 0.0069)
- **Worst region:** Midwest (MAE = 0.0078)
- **Region MAE spread ratio:** 1.13×
- **Flagged states:** none
- **Spearman ρ(n\_test, MAE) across states:** -0.494 — moderate negative correlation (not significant at p<0.05)

---

## Equity Interpretation

### What this audit measures
Geographic disparity in model accuracy is a real equity concern: communities in
regions where the model performs poorly receive less reliable air quality forecasts,
which can affect health-protective decisions (outdoor activity, ventilation, alerts).

### Data sparsity and its role
The Spearman ρ values above test whether states with more test data tend to have
lower error.  A significant negative correlation (ρ < 0, p < 0.05) would confirm
that data-sparse states are systematically underserved.  The values above should
be interpreted in this light.

### Likely causes of regional disparity
- **Pollution regime differences:** Ozone in the West (CA, AZ) peaks in summer
  and is driven by photochemistry + topography — patterns that may generalise
  poorly to Southern or Midwestern ozone dynamics.
- **PM2.5 episodic events:** California wildfire smoke creates extreme PM2.5
  spikes that a 24h-ahead model trained mostly on background conditions will
  underpredict.  This inflates CA error.
- **NO₂ urban vs suburban mix:** NO₂ is highly local (traffic/industrial point
  sources).  Dense urban networks (NY, IL) provide tighter spatial coverage;
  sparser states have higher station-to-station variance that the model cannot
  resolve with lat/lon alone.
- **Sample size imbalance:** States like California (many stations, many rows)
  may dominate model training, leading to better fit in CA and relatively worse
  fit in states with fewer training examples.

### Limitations of this audit
- Only 10 states are covered; findings do not generalise to unrepresented regions.
- Census regions group states with very different pollution climates
  (e.g. West includes urban CA and rural AZ).
- Station-level disparity (urban vs rural within a state) is not captured here.
- A 1-year dataset (2023) may not represent long-run regional patterns.

### What was not done
- No demographic overlay (EJ communities, income, race) — this would be a
  necessary next step for a full environmental justice audit.
- No temporal breakdown (does accuracy drop in wildfire season?  winter inversions?).

*Generated by scripts/bias_audit.py · model: GBM weather-augmented (20 features) · data: clean_hourly_weather.parquet · test fraction: 0.2 · flag threshold: 150%× overall MAE*