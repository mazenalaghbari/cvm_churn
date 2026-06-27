# Paper 1 Submission Bundle

This folder contains the code, data copies, locked model outputs, figures, and supplementary material needed for the Paper 1 BDCC submission.

## Paper

Privacy-Aware Semantic Feature Engineering for Telecom Churn Prediction

## Contents

- `scripts/`: scripts used for Mistral feature extraction and final locked baseline evaluation.
- `data/raw_sources/churn_dataset.csv`: raw ChurnKB-style telecom churn dataset used in the run.
- `data/features/llm_features_ollama.csv`: existing Mistral 7B/Ollama semantic features. This file was not regenerated.
- `data/llm_features_ollama_cleaned.csv`: schema-corrected Mistral semantic features used in the final locked run.
- `data/cache/roberta_features_cached.csv`: cached RoBERTa sentiment probability features used in the final locked run.
- `model/final_locked_run/`: internally consistent metrics, prediction files, figures, SHAP outputs, schema audit, and run metadata.
- `supplementary/`: Supplementary Material S1 with the Mistral prompt, JSON schema, allowed labels, and fallback rules.
- `github_repo_draft/`: compact draft contents for a public GitHub repository.

## Final Results Folder

Use only:

`model/final_locked_run/`

for manuscript tables and figures. That folder contains:

- `metrics_final.csv`
- `paper1_ablation.csv`
- `paper1_shap_top10.csv`
- `json_fallback_audit_final.csv`
- `sentiment_schema_audit.csv`
- `prior_locked_run_comparison.csv`
- five per-pipeline prediction files
- `all_pipeline_predictions.csv`
- `figures/roc_pr_curves.png`
- `figures/calibration_curve.png`
- `figures/lift_chart.png`
- `figures/shap_importance.png`
- `final_run_log.txt`
- `run_metadata.json`

## Reproducibility

The final locked run used:

- `test_size=0.20`
- `random_state=42`
- stratified train/test split
- threshold `0.40` for predicted class
- paired bootstrap AUC comparison with 1,000 resamples

The Mistral/Ollama extraction was not rerun for this bundle.

## Semantic Schema Correction

The Mistral prompt allows sentiment values `Positive`, `Neutral`, and `Negative`. `Satisfaction` is valid only as an emotion label. The archived Mistral feature file contained 270 out-of-schema `sentiment=Satisfaction` rows.

The final locked run uses `data/llm_features_ollama_cleaned.csv`, where:

- `Satisfaction` sentiment is mapped to `Positive`.
- Unknown sentiment values are mapped to `Neutral`.
- Final sentiment labels contain only `Positive`, `Neutral`, and `Negative`.
- `Satisfaction` remains available in the emotion field.

The schema audit is saved at `model/final_locked_run/sentiment_schema_audit.csv`.

## GitHub Repository URL Needed

BDCC expects a live public repository URL at submission. Create a public GitHub repository from `github_repo_draft/` or this entire `paper1/` folder, then replace the manuscript placeholder with the final URL.
