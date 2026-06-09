# Model Card - Churn Risk Classifier

**Version:** `20260609T144919518246-892b26fac7`
**Generated:** 2026-06-09T14:49:19.772555+00:00
**Training data:** `synthetic-v2`
**Model type:** `hist_gbdt` (calibration: `isotonic`)

## Intended use
Estimate the probability that a B2B SaaS account churns within the renewal
window, to prioritise customer-success outreach and in-product interventions.
**Not** intended for automated, irreversible actions (e.g. cancelling accounts)
without a human in the loop.

## Performance (held-out test set)
| Metric | Value |
|---|---|
| model_type | hist_gbdt |
| calibration | isotonic |
| n_train | 1200 |
| n_test | 300 |
| positive_rate | 0.264 |
| roc_auc | 0.7643 |
| pr_auc | 0.5748 |
| cv_roc_auc | 0.7194 |
| brier | 0.1556 |
| log_loss | 0.4877 |
| threshold | 0.225 |
| f1 | 0.5487 |
| precision | 0.4218 |
| recall | 0.7848 |
| accuracy | 0.66 |
| f1_at_0.5 | 0.4364 |

- **Operating threshold:** `0.22499999999999998` (chosen to maximise F1 on
  out-of-fold predictions; compare `f1` vs `f1_at_0.5`).
- **Risk bands:** medium ≥ `0.2578`,
  high ≥ `0.4341` (score quantiles).

## Calibration (reliability curve)
| Mean predicted | Observed frequency |
|---|---|
| 0.0464 | 0.1333 |
| 0.0985 | 0.0645 |
| 0.1288 | 0.1034 |
| 0.1622 | 0.2 |
| 0.2005 | 0.0667 |
| 0.2385 | 0.2333 |
| 0.2878 | 0.2333 |
| 0.3414 | 0.3667 |
| 0.441 | 0.4333 |
| 0.6728 | 0.8 |

## Fairness / segment performance (ROC-AUC)
- **company_size**: enterprise=0.8649, mid_market=0.7175, smb=0.7458
- **region**: apac=0.7786, emea=0.7229, latam=0.773, na=0.7895

A material AUC gap across segments is a signal to revisit features or collect
more representative data before relying on the model for that segment.

## Limitations
- Trained on synthetic telemetry; replace with warehouse extracts for production.
- Probabilities reflect historical patterns and can drift; see drift monit