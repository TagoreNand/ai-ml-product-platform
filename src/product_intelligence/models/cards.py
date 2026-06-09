"""Generate a Markdown model card from a training run.

A model card documents intended use, data, metrics, calibration, fairness
caveats and operating points so reviewers (DS, PM, risk) can sign off before a
model is promoted. We render it straight from the ``TrainingResult`` so the doc
can never silently drift from the artifact it describes.
"""

from __future__ import annotations

from datetime import datetime, timezone

from product_intelligence.models.train import TrainingResult


def _metrics_table(metrics: dict) -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in metrics.items())
    return f"| Metric | Value |\n|---|---|\n{rows}"


def render_churn_model_card(result: TrainingResult, *, version: str, data_version: str) -> str:
    seg = result.segment_metrics or {}
    seg_lines = []
    for dim, values in seg.items():
        inner = ", ".join(f"{k}={v}" for k, v in values.items())
        seg_lines.append(f"- **{dim}**: {inner}")
    seg_block = (
        "\n".join(seg_lines) if seg_lines else "- Not enough per-segment positives to report."
    )

    calib = result.calibration_curve or {"prob_pred": [], "prob_true": []}
    calib_rows = "\n".join(
        f"| {p} | {t} |" for p, t in zip(calib["prob_pred"], calib["prob_true"], strict=False)
    )

    return f"""# Model Card - Churn Risk Classifier

**Version:** `{version}`
**Generated:** {datetime.now(timezone.utc).isoformat()}
**Training data:** `{data_version}`
**Model type:** `{result.metrics.get("model_type")}` (calibration: `{result.metrics.get("calibration")}`)

## Intended use
Estimate the probability that a B2B SaaS account churns within the renewal
window, to prioritise customer-success outreach and in-product interventions.
**Not** intended for automated, irreversible actions (e.g. cancelling accounts)
without a human in the loop.

## Performance (held-out test set)
{_metrics_table(result.metrics)}

- **Operating threshold:** `{result.threshold}` (chosen to maximise F1 on
  out-of-fold predictions; compare `f1` vs `f1_at_0.5`).
- **Risk bands:** medium ≥ `{(result.band_thresholds or {}).get("medium")}`,
  high ≥ `{(result.band_thresholds or {}).get("high")}` (score quantiles).

## Calibration (reliability curve)
| Mean predicted | Observed frequency |
|---|---|
{calib_rows}

## Fairness / segment performance (ROC-AUC)
{seg_block}

A material AUC gap across segments is a signal to revisit features or collect
more representative data before relying on the model for that segment.

## Limitations
- Trained on synthetic telemetry; replace with warehouse extracts for production.
- Probabilities reflect historical patterns and can drift; see drift monitoring.
- Explanations are local attributions, not causal claims.
"""
