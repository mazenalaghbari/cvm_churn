# Paper 1 Bundle Manifest

## Main Folder

`paper1/`

This folder is a standalone submission/handoff bundle. It was created by copying files from the existing project and adding documentation. It should be safe to use for a separate GitHub repository later.

## Required Submission Files

### Final locked outputs

Path:

`model/final_locked_run/`

Contains:

- `run_metadata.json`
- `metrics_final.csv`
- `paper1_ablation.csv`
- `paper1_shap_top10.csv`
- `json_fallback_audit_final.csv`
- `sentiment_schema_audit.csv`
- `prior_locked_run_comparison.csv`
- `all_pipeline_predictions.csv`
- `Structured_only_predictions.csv`
- `VADER_predictions.csv`
- `TFIDF_SVD_predictions.csv`
- `RoBERTa_sentiment_predictions.csv`
- `Mistral_7B_predictions.csv`
- `figures/roc_pr_curves.png`
- `figures/calibration_curve.png`
- `figures/lift_chart.png`
- `figures/shap_importance.png`
- `final_run_log.txt`

### Supplementary Material S1

Path:

`supplementary/`

Files:

- `Supplementary_Material_S1_Mistral_Prompt_JSON_Schema.docx`
- `Supplementary_Material_S1_Mistral_Prompt_JSON_Schema.md`

Use the `.docx` file for BDCC supplementary upload.

### Code

Path:

`scripts/`

Files:

- `cvm_bert_baseline.py`
- `cvm_feature_extraction_ollama.py`

### Data copies

Path:

`data/`

Files:

- `raw_sources/churn_dataset.csv`
- `features/llm_features_ollama.csv`
- `llm_features_ollama_cleaned.csv`
- `cache/roberta_features_cached.csv`
- `README_data.md`

## GitHub Blocking Item

The repository URL is still required before BDCC submission.

Create a public GitHub repository from either:

- the full `paper1/` folder, or
- the smaller `paper1/github_repo_draft/` folder.

Then update the manuscript Data Availability Statement with the public URL.
