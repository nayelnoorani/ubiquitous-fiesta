# Q1S5 — Weekend Effect: Statistical Analysis

*Run: 2026-04-15 12:28 | Elapsed: 0.0s*

## 1. Day-of-Week Summary

| Day | N | Mean |spread| | Median |spread| | Std | Persist rate |
|---|---|---|---|---|---|
| Monday | 165 | 1.5729% | 0.7593% | 3.1075% | 81.2% |
| Tuesday | 165 | 1.528% | 0.8656% | 2.6005% | 78.2% |
| Wednesday | 166 | 1.3123% | 0.8526% | 1.8916% | 80.1% |
| Thursday | 166 | 1.695% | 0.9453% | 2.7556% | 76.5% |
| Friday | 166 | 1.4056% | 0.8115% | 2.1748% | 76.5% |
| Saturday | 165 | 1.1071% | 0.7391% | 1.6324% | 85.5% |
| Sunday | 165 | 1.4265% | 0.8203% | 4.4974% | 79.4% |

## 2. Mann-Whitney U — Weekend vs Weekday |Spread|

| Metric | Value |
|---|---|
| Weekend mean |spread| | 1.2668% |
| Weekday mean |spread| | 1.501% |
| Mann-Whitney U | 123710.0 |
| p-value (two-sided) | 0.011 |
| Significant (p < 0.05) | Yes ✓ |
| N (weekend / weekday) | 330 / 829 |

## 3. Cohen's d — Weekend vs Weekday Effect Size

Cohen's d = **-0.0876**

Interpretation: |d| < 0.2 = negligible, 0.2–0.5 = small, 0.5–0.8 = medium, > 0.8 = large


## 4. Friday → Monday Persistence

| Metric | Value |
|---|---|
| Friday → Monday pairs | 165 |
| Fri→Mon persist rate | 68.5% |
| Tue→Wed pairs | 166 |
| Tue→Wed persist rate | 80.1% |

---

# Q1S5 — Weekend Effect: ML Models

*Run: 2026-04-15 12:28 | Elapsed: 0.4s*

## Run Details

- Training: 2023-02-07 → 2025-08-23 (925 days)
- Test:     2025-08-24 → 2026-04-12 (232 days)
- Classifier target: spread persists same sign to next day
- Regression target: next-day spread change

## Class Balance

| | Train | Test |
|---|---|---|
| Persists (1) | 79.8% | 78.9% |
| Flips    (0) | 20.2% | 21.1% |

## Linear Regression — Day-of-Week Effect on Spread Change

MAE: 0.737%  |  Direction accuracy: 52.6%

| Feature | Coefficient |
|---|---|
| `spread_vs_net_rolling_mean_7d` | -0.362079 |
| `spread_vs_net_rolling_std_7d` | -0.132368 |
| `spread_vs_net_lag_1d` | 0.031513 |
| `dow_1` (dummy) | 0.320741 |
| `dow_2` (dummy) | 0.907399 |
| `dow_3` (dummy) | 0.047948 |
| `dow_4` (dummy) | 0.770055 |
| `dow_5` (dummy) | 0.903479 |
| `dow_6` (dummy) | 0.329538 |
| intercept | -0.11684 |

## Classifier Performance

| Model | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic regression | 0.5562 | 0.7864 | 0.8852 | 0.8329 |
| XGBoost classifier | 0.6395 | 0.7936 | 0.9454 | 0.8628 |

**Best classifier by AUC:** XGBoost classifier (AUC=0.6395)

## XGBoost Feature Importance

| Feature | Importance |
|---|---|
| `spread_vs_net_rolling_std_7d` | 0.1427 |
| `dow_5` | 0.1235 |
| `dow_6` | 0.1204 |
| `dow_2` | 0.1043 |
| `spread_vs_net_rolling_mean_7d` | 0.1034 |
| `spread_vs_net_lag_1d` | 0.0966 |
| `dow_1` | 0.0859 |
| `day_of_week` | 0.0785 |
| `dow_3` | 0.0769 |
| `dow_4` | 0.0677 |

## Interpretation

### Day-of-week summary — Saturday is the most stable day

Saturday has the lowest mean |spread| (1.11%) and the highest persistence rate (85.5%). Friday and Thursday have the lowest persistence rates (both 76.5%) and above-average spread levels. The weekend is the most stable part of the week, not the least. Sunday's standard deviation (4.50%) is notably high — occasional large moves occur on Sundays even though the median is normal.

### Mann-Whitney U — significant but negligible effect size

Weekend |spread| is statistically lower than weekday (1.27% vs 1.50%, p=0.011). Cohen's d = −0.088 — negligible by any standard classification. The effect exists but is not practically meaningful. With 1,159 observations, even tiny differences become statistically significant.

### Friday → Monday persistence — the hypothesis is wrong

Fri→Mon persist rate is 68.5% vs 80.1% for Tue→Wed. The spread is less likely to persist its sign across the weekend than across a typical mid-week transition. Friday spreads do not get locked in over the weekend — the weekend allows more reversion. DeFi protocol rates update continuously including over weekends, so there is no trading halt to lock in Friday's spread. Lower weekend borrowing activity is consistent with the smaller mean |spread| on weekends.

### Linear regression dummies — Wednesday and Saturday have large coefficients

`dow_2` (Wednesday) and `dow_5` (Saturday) both have coefficients near +0.90 — the largest among all day dummies relative to the Monday reference. All days except Monday show positive coefficients, meaning the spread tends to widen relative to Monday on every other day of the week. Direction accuracy of 52.6% means these coefficients improve mean predictions but do not reliably predict direction.

### ML classifiers — weekend dummies dominate XGBoost importance

XGBoost AUC=0.639. The two weekend dummies (`dow_5` and `dow_6`) account for 24.4% of combined feature importance — more than any individual spread feature. Day-of-week carries genuine predictive value for the persistence target, even if the magnitude of the weekend effect is small. The 80% class imbalance (spread persists on 80% of days) inflates F1 and recall; AUC is the reliable metric.

### Overall

The weekend effect is real but opposite to the hypothesis. Spreads are smaller and more stable on weekends, and Friday spreads are less persistent into Monday than mid-week transitions. `day_of_week` adds modest signal to predictive models but is not a dominant driver of spread behaviour.
