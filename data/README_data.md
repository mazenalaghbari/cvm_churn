# Data Notes

## Files

- `raw_sources/churn_dataset.csv`: telecom churn dataset used for the final locked run.
- `features/llm_features_ollama.csv`: previously generated Mistral 7B semantic features.
- `llm_features_ollama_cleaned.csv`: schema-corrected Mistral 7B semantic features used by the final locked run.
- `cache/roberta_features_cached.csv`: cached RoBERTa sentiment features.

## Dataset Citation

Dataset: ChurnKB, Shahabikargar et al. (2025), DOI: `10.3390/a18040238`.

## Customer IDs

The raw CSV used in this project does not contain a persistent `customer_id` or `customerID` field. For reproducible prediction exports, the final script creates `customer_id` from the raw file row index and aligns it with `llm_features_ollama.csv`, whose `customer_id` values were generated from the same raw row order.

This limitation is documented in `model/final_locked_run/run_metadata.json`.

## Sentiment Schema Correction

The Mistral extraction prompt allowed only `Positive`, `Neutral`, and `Negative` as sentiment labels. The archived feature file contained 270 rows with `sentiment=Satisfaction`, which is valid only for the emotion field.

The cleaned feature file maps:

`Satisfaction -> Positive`

Unknown sentiment values are mapped to `Neutral`.

The final cleaned sentiment field contains no `Satisfaction` values.
