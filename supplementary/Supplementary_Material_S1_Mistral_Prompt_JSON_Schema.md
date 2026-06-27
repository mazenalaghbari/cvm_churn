# Supplementary Material S1

## Mistral 7B Prompt and JSON Schema for Semantic Feature Extraction

This supplementary material documents the exact local Mistral/Ollama semantic extraction prompt, output schema, allowed labels, and fallback rules used to generate `llm_features_ollama.csv`.

For the final locked run, the archived Mistral output was schema-corrected into `data/llm_features_ollama_cleaned.csv`. The correction maps out-of-schema `sentiment=Satisfaction` values to `Positive`, because `Satisfaction` is valid only under the emotion feature.

## Source Script

`scripts/cvm_feature_extraction_ollama.py`

## Model

Default local model:

`mistral`

The script reads the model name from:

`OLLAMA_MODEL`

If no environment variable is set, it defaults to `mistral`.

## User Prompt Template

```text
Analyze the following customer service chat log and extract the attributes in JSON format.
Only return the JSON object, no markdown, no code blocks, no explanations.

Chat Log:
"{chat_log}"

Attributes to extract:
1. sentiment: One of ["Positive", "Neutral", "Negative"]
2. emotion: One of ["Anger", "Frustration", "Confusion", "Satisfaction", "Gratitude", "Neutral"]
3. urgency: A score from 0.0 (Low) to 1.0 (Critical)
4. topic: One of ["Billing", "Technical Support", "Service Information", "Complaint", "Other"]
5. risk_indicator: Boolean (true/false)

JSON Output:
```

## System Prompt

No separate system prompt is defined in the source script. The extraction instruction is sent as a single user message to the local Ollama chat endpoint.

## Chat Log Truncation Rule

The script truncates each customer interaction to the first 2,000 characters before prompt formatting:

```python
prompt = PROMPT_TEMPLATE.format(chat_log=chat_log[:2000])
```

## Allowed Labels

### sentiment

- `Positive`
- `Neutral`
- `Negative`

### emotion

- `Anger`
- `Frustration`
- `Confusion`
- `Satisfaction`
- `Gratitude`
- `Neutral`

### urgency

Numeric score from `0.0` to `1.0`.

- `0.0`: Low
- `1.0`: Critical

### topic

- `Billing`
- `Technical Support`
- `Service Information`
- `Complaint`
- `Other`

### risk_indicator

Boolean:

- `true`
- `false`

## JSON Schema

```json
{
  "sentiment": "Positive | Neutral | Negative",
  "emotion": "Anger | Frustration | Confusion | Satisfaction | Gratitude | Neutral",
  "urgency": 0.0,
  "topic": "Billing | Technical Support | Service Information | Complaint | Other",
  "risk_indicator": false
}
```

## Expected Output Example

```json
{
  "sentiment": "Negative",
  "emotion": "Frustration",
  "urgency": 0.8,
  "topic": "Complaint",
  "risk_indicator": true
}
```

## Response Cleaning and JSON Parsing

The script removes Markdown code fences if present:

```python
text = content.replace('```json', '').replace('```', '').strip()
```

If the model returns extra text around the JSON object, the script extracts the substring between the first `{` and the last `}`:

```python
if '{' in text and '}' in text:
    start = text.find('{')
    end = text.rfind('}') + 1
    text = text[start:end]
```

The remaining text is parsed with:

```python
data = json.loads(text)
```

## Fallback Rules

If the Ollama call fails, the model response is not valid JSON, or any exception occurs during extraction, the script logs the error and returns the following fallback object:

```json
{
  "sentiment": "Neutral",
  "emotion": "Neutral",
  "urgency": 0.0,
  "topic": "Other",
  "risk_indicator": false
}
```

## Final Locked-Run Sentiment Normalization

The final locked run applies the following sentiment normalization before model training:

```python
SENTIMENT_NORMALIZATION = {
    "Positive": "Positive",
    "Neutral": "Neutral",
    "Negative": "Negative",
    "Satisfaction": "Positive"
}
```

Unknown sentiment values are mapped to:

```text
Neutral
```

The final sentiment field must contain only:

- `Positive`
- `Neutral`
- `Negative`

The final locked run audit found:

```text
Satisfaction -> Positive: 270 records
Unknown sentiment -> Neutral: 0 records
```

## Final Locked-Run Strict Schema Validation

The cleaned feature file enforces:

- sentiment: `Positive`, `Neutral`, `Negative`
- emotion: `Anger`, `Frustration`, `Confusion`, `Satisfaction`, `Gratitude`, `Neutral`
- topic: `Billing`, `Technical Support`, `Service Information`, `Complaint`, `Other`
- urgency: numeric value clipped to `[0.0, 1.0]`
- risk_indicator: boolean `true` or `false`

Out-of-schema emotion values are mapped to `Neutral`.

Out-of-schema topic values are mapped to `Other`.

## Resume and Save Logic

The script resumes from an existing `llm_features_ollama.csv` file when available. Already processed rows are skipped using `customer_id`.

The script writes incremental checkpoints every 25 records and performs a final save at completion.

## Generated Feature File

Output file:

`data/llm_features_ollama.csv`

Columns:

- `sentiment`
- `emotion`
- `urgency`
- `topic`
- `risk_indicator`
- `customer_id`
