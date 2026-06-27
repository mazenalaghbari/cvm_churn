# ChurnKB NLP Benchmark for Paper 1

This repository draft contains reproducibility materials for the BDCC Paper 1 submission:

Privacy-Aware Semantic Feature Engineering for Telecom Churn Prediction

## Included

- `01_extract_mistral_features.py`: local Mistral/Ollama semantic feature extraction script.
- `02_final_locked_baseline.py`: final locked evaluation script for the five NLP configurations.
- `metrics_final.csv`: final locked model comparison table.
- `sample_or_full_churn_dataset.csv`: dataset copy for reproducibility, subject to sharing policy.

## Final Evaluation Configurations

1. Structured only
2. Structured + VADER
3. Structured + TF-IDF + SVD
4. Structured + RoBERTa sentiment pipeline
5. Structured + Mistral 7B cleaned semantic features

## Reproduction Notes

The published final run used the files in `../model/final_locked_run/`.

The cleaned Mistral semantic file maps out-of-schema `sentiment=Satisfaction` values to `Positive`; `Satisfaction` remains valid only as an emotion label.

Random seed: `42`

Test split: `20%`, stratified by churn.

Threshold for predicted class: `0.40`.

AUC comparison method: paired bootstrap AUC comparison with 1,000 resamples.

## Data Availability Placeholder

Replace this section after the public GitHub repository is created:

`Repository URL: TO_BE_ADDED_BEFORE_SUBMISSION`
