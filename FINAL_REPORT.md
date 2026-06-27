# Paper 1 Final Semantic-Schema Correction Report

## Correction Applied

The Mistral prompt allowed sentiment labels:

- `Positive`
- `Neutral`
- `Negative`

`Satisfaction` is valid only in the emotion field. The archived Mistral feature file contained 270 rows with `sentiment=Satisfaction`.

The final locked run uses:

`data/llm_features_ollama_cleaned.csv`

Correction:

`Satisfaction -> Positive`

Unknown sentiment values are mapped to `Neutral`.

## Schema Audit

Audit file:

`model/final_locked_run/sentiment_schema_audit.csv`

Summary:

- `Satisfaction -> Positive`: 270 records
- Unknown sentiment mapped to `Neutral`: 0 records
- Final sentiment labels: `Neutral` 1478, `Positive` 1181, `Negative` 674

## Acceptance Checks

- No remaining `Satisfaction` values in the final sentiment column: passed
- `Satisfaction` remains available in the emotion column: passed
- All five pipeline prediction files have 667 rows: passed
- All five pipelines use identical held-out test records: passed
- No missing prediction probabilities: passed
- AUC recalculated from each prediction file matches `metrics_final.csv`: passed

## Metrics

| Pipeline | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| Structured only | 0.909097 | 0.851908 | 0.810811 |
| Structured + VADER | 0.922572 | 0.860048 | 0.810811 |
| Structured + TF-IDF + SVD | 0.988660 | 0.961894 | 0.906250 |
| Structured + RoBERTa sentiment | 0.972689 | 0.924100 | 0.851282 |
| Structured + Mistral 7B cleaned semantic features | 0.969343 | 0.907792 | 0.814433 |

## Comparison With Prior Locked Run

Metrics, prediction probabilities, and SHAP ranking are unchanged except for negligible floating-point representation differences on the order of `1e-16`.

Reason: the previous evaluation encoded both `Positive` and out-of-schema `Satisfaction` sentiment as `0`; the cleaned workflow maps `Satisfaction` to `Positive`, which is also encoded as `0`.

Detailed comparison:

`model/final_locked_run/prior_locked_run_comparison.csv`
