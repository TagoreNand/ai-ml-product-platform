# Data Science Experimentation Plan

## Business questions
1. Which accounts are most likely to churn in the next renewal window?
2. Which feature should be recommended next to maximize retention and expansion?
3. Can modeled signals improve roadmap decisions relative to intuition-only planning?

## Offline evaluation
- churn: ROC-AUC, F1, calibration review, segment breakdowns
- recommendation: macro F1, top-k hit rate, adoption lift after exposure
- prioritization: retrospective rank correlation between priority score and realized business impact

## Online evaluation
- A/B test in-product feature prompts
- holdout accounts for CSM intervention playbooks
- measure retention, adoption, ticket rate, and revenue retention

## Risks
- selection bias in intervention data
- drift after pricing or packaging changes
- confounding between support burden and contract segment
