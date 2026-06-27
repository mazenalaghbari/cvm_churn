import ollama
import pandas as pd
import os
import json
import time
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = '../data'
INPUT_FILE = 'churn_dataset.csv'
OUTPUT_FILE = 'llm_features_ollama.csv'
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'mistral') # Default to mistral, user can change

SENTIMENT_NORMALIZATION = {
    "Positive": "Positive",
    "Neutral": "Neutral",
    "Negative": "Negative",
    "Satisfaction": "Positive",
}
VALID_EMOTIONS = {"Anger", "Frustration", "Confusion", "Satisfaction", "Gratitude", "Neutral"}
VALID_TOPICS = {"Billing", "Technical Support", "Service Information", "Complaint", "Other"}

PROMPT_TEMPLATE = """
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
"""

def normalize_and_validate_features(data):
    original_sentiment = data.get("sentiment", "Neutral")
    normalized_sentiment = SENTIMENT_NORMALIZATION.get(original_sentiment, "Neutral")
    if original_sentiment not in SENTIMENT_NORMALIZATION:
        logger.warning(f"Unknown sentiment value mapped to Neutral: {original_sentiment}")

    emotion = data.get("emotion", "Neutral")
    if emotion not in VALID_EMOTIONS:
        logger.warning(f"Unknown emotion value mapped to Neutral: {emotion}")
        emotion = "Neutral"

    topic = data.get("topic", "Other")
    if topic not in VALID_TOPICS:
        logger.warning(f"Unknown topic value mapped to Other: {topic}")
        topic = "Other"

    try:
        urgency = float(data.get("urgency", 0.0))
    except (TypeError, ValueError):
        logger.warning(f"Invalid urgency value mapped to 0.0: {data.get('urgency')}")
        urgency = 0.0
    urgency = min(1.0, max(0.0, urgency))

    risk = data.get("risk_indicator", False)
    if isinstance(risk, str):
        risk = risk.strip().lower() in {"true", "1", "yes"}
    else:
        risk = bool(risk)

    return {
        "sentiment": normalized_sentiment,
        "emotion": emotion,
        "urgency": urgency,
        "topic": topic,
        "risk_indicator": risk,
    }

def extract_features(chat_log, model):
    try:
        prompt = PROMPT_TEMPLATE.format(chat_log=chat_log[:2000])
        response = ollama.chat(model=model, messages=[
            {'role': 'user', 'content': prompt},
        ])
        
        content = response['message']['content']
        # Clean response
        text = content.replace('```json', '').replace('```', '').strip()
        # Find start and end of JSON if extra text exists
        if '{' in text and '}' in text:
            start = text.find('{')
            end = text.rfind('}') + 1
            text = text[start:end]
            
        data = json.loads(text)
        return normalize_and_validate_features(data)
    except Exception as e:
        logger.error(f"Error extraction: {e}")
        return normalize_and_validate_features({
            "sentiment": "Neutral",
            "emotion": "Neutral",
            "urgency": 0.0,
            "topic": "Other",
            "risk_indicator": False
        })

if __name__ == "__main__":
    # Check if model is available (basic check)
    try:
        models = [m['name'] for m in ollama.list()['models']]
        # Normalize model names (remove :latest if needed for comparison, but keep simple)
        logger.info(f"Available Ollama models: {models}")
        if OLLAMA_MODEL not in models and f"{OLLAMA_MODEL}:latest" not in models:
             logger.warning(f"Model {OLLAMA_MODEL} might not be pulled. Try 'ollama pull {OLLAMA_MODEL}'")
    except Exception as e:
        logger.warning(f"Could not connect to Ollama: {e}. Ensure it is running.")

    # Load Data
    data_path = os.path.join(DATA_DIR, 'raw_sources', INPUT_FILE)
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        exit(1)
        
    df = pd.read_csv(data_path)
    # Process all records
    TARGET_COUNT = len(df)
    print(f"Targeting Full Dataset: {TARGET_COUNT} records")
    logger.info(f"Loaded {len(df)} records. Processing all {TARGET_COUNT}...")
    df_subset = df.head(TARGET_COUNT)
    
    # Check for existing output to resume
    output_path = os.path.join(DATA_DIR, OUTPUT_FILE)
    results = []
    processed_ids = set()
    
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            results = existing_df.to_dict('records')
            processed_ids = set(existing_df['customer_id'].astype(int))
            logger.info(f"Found {len(results)} already processed records. Resuming...")
        except Exception as e:
            logger.warning(f"Could not read existing file to resume: {e}. Starting fresh.")

    logger.info(f"Starting Extraction with local model: {OLLAMA_MODEL}...")
    
    for idx, row in tqdm(df_subset.iterrows(), total=len(df_subset)):
        # Resume logic: Skip if already processed
        if idx in processed_ids:
            continue

        chat_log = row.get('chat_log', '')
        if not isinstance(chat_log, str):
            chat_log = ""
            
        features = extract_features(chat_log, OLLAMA_MODEL)
        features['customer_id'] = idx
        results.append(features)
        
        # Incremental save every 25 records to prevent data loss
        if len(results) % 25 == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)
        
        # Local model might accept requests faster or slower, no API rate limit sleep needed usually
        # but good to be nice to the CPU/GPU
        # time.sleep(0.1) 

    # Final Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved extracted features to {output_path}")
