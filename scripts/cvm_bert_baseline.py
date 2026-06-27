"""Paper 1 final locked run with cleaned Mistral semantic schema."""

from __future__ import annotations

import json
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    auc as sk_auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw_sources"
FEATURE_DIR = DATA_DIR / "features"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = PROJECT_ROOT / "model"
FINAL_DIR = MODEL_DIR / "final_locked_run"
PREDICTION_DIR = FINAL_DIR / "pipeline_predictions"
FIGURE_DIR = FINAL_DIR / "figures"
PRIOR_LOCKED_DIR = PROJECT_ROOT / "results" / "final_locked_run"

RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD = 0.40
N_BOOTSTRAPS = 1000
BOOTSTRAP_LABEL = "Paired bootstrap AUC comparison (1,000 resamples)"

SENTIMENT_NORMALIZATION = {
    "Positive": "Positive",
    "Neutral": "Neutral",
    "Negative": "Negative",
    "Satisfaction": "Positive",
}
VALID_SENTIMENTS = {"Positive", "Neutral", "Negative"}
VALID_EMOTIONS = {"Anger", "Frustration", "Confusion", "Satisfaction", "Gratitude", "Neutral"}
VALID_TOPICS = {"Billing", "Technical Support", "Service Information", "Complaint", "Other"}

LLM_SENTIMENT_MAP = {"Positive": 0, "Neutral": 1, "Negative": 2}
LLM_EMOTION_MAP = {
    "Gratitude": 0,
    "Satisfaction": 1,
    "Neutral": 2,
    "Confusion": 3,
    "Frustration": 4,
    "Anger": 5,
}
LLM_TOPIC_MAP = {
    "Service Information": 0,
    "Billing": 1,
    "Technical Support": 2,
    "Complaint": 3,
    "Other": 4,
}
PIPELINE_IDS = {
    "Structured only": "Structured_only",
    "Structured + VADER": "VADER",
    "Structured + TF-IDF+SVD": "TFIDF_SVD",
    "Structured + RoBERTa sentiment": "RoBERTa_sentiment",
    "Structured + Mistral 7B cleaned semantic features": "Mistral_7B",
}
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "scale_pos_weight": 5.9,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
}


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def reset_output_dirs() -> None:
    if FINAL_DIR.exists():
        import shutil

        shutil.rmtree(FINAL_DIR)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_churn(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0})
        .fillna(0)
        .astype(int)
    )


def _to_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def create_cleaned_llm_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_path = FEATURE_DIR / "llm_features_ollama.csv"
    cleaned_path = DATA_DIR / "llm_features_ollama_cleaned.csv"
    llm = pd.read_csv(raw_path).reset_index(drop=True)
    original = llm.copy()

    unknown_sentiment_mask = ~llm["sentiment"].isin(SENTIMENT_NORMALIZATION)
    llm["sentiment"] = llm["sentiment"].map(SENTIMENT_NORMALIZATION).fillna("Neutral")

    unknown_emotion_mask = ~llm["emotion"].isin(VALID_EMOTIONS)
    llm.loc[unknown_emotion_mask, "emotion"] = "Neutral"

    unknown_topic_mask = ~llm["topic"].isin(VALID_TOPICS)
    llm.loc[unknown_topic_mask, "topic"] = "Other"

    llm["urgency"] = pd.to_numeric(llm["urgency"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    llm["risk_indicator"] = llm["risk_indicator"].map(_to_bool)

    assert set(llm["sentiment"].unique()).issubset(VALID_SENTIMENTS)
    assert "Satisfaction" not in set(llm["sentiment"].unique())
    assert set(llm["emotion"].unique()).issubset(VALID_EMOTIONS)
    assert "Satisfaction" in set(llm["emotion"].unique())
    assert set(llm["topic"].unique()).issubset(VALID_TOPICS)
    assert llm["urgency"].between(0.0, 1.0).all()
    assert llm["risk_indicator"].map(type).eq(bool).all()

    llm = llm[["sentiment", "emotion", "urgency", "topic", "risk_indicator", "customer_id"]]
    llm.to_csv(cleaned_path, index=False)
    (FEATURE_DIR / "llm_features_ollama_cleaned.csv").write_text(cleaned_path.read_text(encoding="utf-8"), encoding="utf-8")

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for label, count in original["sentiment"].value_counts(dropna=False).items():
        rows.append({"section": "original_sentiment_counts", "label": str(label), "count": int(count)})
    rows.extend(
        [
            {
                "section": "mapped_satisfaction_to_positive",
                "label": "Satisfaction -> Positive",
                "count": int((original["sentiment"] == "Satisfaction").sum()),
            },
            {
                "section": "unknown_sentiment_mapped_to_neutral",
                "label": "Unknown -> Neutral",
                "count": int(unknown_sentiment_mask.sum()),
            },
        ]
    )
    for label, count in llm["sentiment"].value_counts(dropna=False).items():
        rows.append({"section": "final_sentiment_counts", "label": str(label), "count": int(count)})
    rows.extend(
        [
            {"section": "unknown_emotion_mapped_to_neutral", "label": "Unknown -> Neutral", "count": int(unknown_emotion_mask.sum())},
            {"section": "unknown_topic_mapped_to_other", "label": "Unknown -> Other", "count": int(unknown_topic_mask.sum())},
        ]
    )
    audit = pd.DataFrame(rows)
    audit["run_datetime_utc"] = now
    audit["script_version"] = git_commit_hash()
    audit.to_csv(FINAL_DIR / "sentiment_schema_audit.csv", index=False)
    return llm, audit


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, list[str], pd.DataFrame, pd.Series, pd.DataFrame]:
    raw = pd.read_csv(RAW_DIR / "churn_dataset.csv").reset_index(drop=True)
    if "customer_id" not in raw.columns and "customerID" not in raw.columns:
        raw["customer_id"] = np.arange(len(raw))
    id_column = "customer_id" if "customer_id" in raw.columns else "customerID"
    customer_ids = raw[id_column].astype(str)

    llm, schema_audit = create_cleaned_llm_features()
    if id_column in llm.columns:
        llm[id_column] = llm[id_column].astype(raw[id_column].dtype, copy=False)
        llm = raw[[id_column]].merge(llm, on=id_column, how="left", validate="one_to_one")
    else:
        assert len(llm) == len(raw), "LLM features lack customer ID and row count differs from raw data"
    assert len(llm) == len(raw)
    assert llm["sentiment"].notna().all()

    y = normalize_churn(raw["churn"])
    chat_logs = raw["chat_log"].fillna("").astype(str).tolist()
    drop_cols = ["churn", "customer_id", "customerID", "chat_log", "phone_number", "state"]
    static_raw = raw.drop(columns=[c for c in drop_cols if c in raw.columns], errors="ignore")
    X_static = pd.get_dummies(static_raw, drop_first=True).apply(pd.to_numeric, errors="coerce").fillna(0)
    X_llm = pd.DataFrame(
        {
            "sentiment_enc": llm["sentiment"].map(LLM_SENTIMENT_MAP),
            "emotion_enc": llm["emotion"].map(LLM_EMOTION_MAP),
            "urgency": llm["urgency"],
            "topic_enc": llm["topic"].map(LLM_TOPIC_MAP),
            "risk_enc": llm["risk_indicator"].astype(int),
        }
    )
    assert X_llm.notna().all().all(), "Cleaned LLM encoding produced missing values"
    return X_static, X_llm, y, chat_logs, llm, customer_ids, schema_audit


def extract_vader(chat_logs: list[str]) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    scores = [analyzer.polarity_scores(text)["compound"] for text in tqdm(chat_logs, desc="VADER")]
    return pd.DataFrame({"vader_compound": scores})


def extract_tfidf_svd_features(chat_logs: list[str], train_idx: np.ndarray) -> pd.DataFrame:
    train_texts = [chat_logs[i] for i in train_idx]
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X_text_train = vectorizer.fit_transform(train_texts)
    X_text_all = vectorizer.transform(chat_logs)
    svd = TruncatedSVD(n_components=50, random_state=RANDOM_STATE)
    svd.fit(X_text_train)
    return pd.DataFrame(svd.transform(X_text_all), columns=[f"tfidf_svd_{i:02d}" for i in range(50)])


def normalize_roberta_labels(features: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=features.index)
    for label in ["negative", "neutral", "positive"]:
        matches = [c for c in features.columns if c.lower().endswith(label) or c.lower() in {f"bert_{label}", f"roberta_{label}"}]
        out[f"roberta_{label}"] = features[matches[0]] if matches else 1.0 / 3.0
    return out.div(out.sum(axis=1).replace(0, 1), axis=0)


def extract_roberta(chat_logs: list[str]) -> pd.DataFrame:
    for path in [CACHE_DIR / "roberta_features_cached.csv", MODEL_DIR / "bert_features.csv"]:
        if path.exists():
            print(f"Using cached RoBERTa sentiment features: {path}")
            return normalize_roberta_labels(pd.read_csv(path))
    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device,
        top_k=None,
        truncation=True,
        max_length=512,
    )
    rows = []
    for start_idx in tqdm(range(0, len(chat_logs), 32), desc="RoBERTa"):
        batch = chat_logs[start_idx : start_idx + 32]
        predictions = pipe(batch)
        for pred_list in predictions:
            rows.append({f"roberta_{item['label'].lower()}": float(item["score"]) for item in pred_list})
    features = normalize_roberta_labels(pd.DataFrame(rows).fillna(1 / 3))
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    features.to_csv(CACHE_DIR / "roberta_features_cached.csv", index=False)
    return features


def bootstrap_auc_ci(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    rng = np.random.RandomState(RANDOM_STATE)
    aucs = []
    for _ in range(N_BOOTSTRAPS):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) > 1:
            aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return tuple(np.percentile(aucs, [2.5, 97.5]))


def paired_bootstrap_auc_test(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> tuple[float, float]:
    rng = np.random.RandomState(RANDOM_STATE)
    diffs = []
    for _ in range(N_BOOTSTRAPS):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) > 1:
            diffs.append(roc_auc_score(y_true[idx], pred1[idx]) - roc_auc_score(y_true[idx], pred2[idx]))
    se = np.std(diffs)
    auc1 = roc_auc_score(y_true, pred1)
    auc2 = roc_auc_score(y_true, pred2)
    z = (auc1 - auc2) / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


def compute_metrics(name: str, y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | str]:
    y_pred = (y_prob >= THRESHOLD).astype(int)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ci_lower, ci_upper = bootstrap_auc_ci(y_true, y_prob)
    return {
        "Configuration": name,
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": sk_auc(recall, precision),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Brier": brier_score_loss(y_true, y_prob),
        "CI_lower": ci_lower,
        "CI_upper": ci_upper,
    }


def fit_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    return model


def train_evaluate_configs(
    configs: list[tuple[str, pd.DataFrame]],
    y: pd.Series,
    customer_ids: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    export_predictions: bool = False,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, xgb.XGBClassifier], dict[str, pd.DataFrame]]:
    results, probabilities, models, test_matrices, prediction_frames = [], {}, {}, {}, []
    y_test = y.iloc[test_idx].to_numpy()
    for name, X in configs:
        X_clean = X.apply(pd.to_numeric, errors="coerce").fillna(0)
        X_train, X_test = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        model = fit_xgboost(X_train, y.iloc[train_idx])
        probs = model.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(name, y_test, probs)
        results.append(metrics)
        probabilities[name] = probs
        models[name] = model
        test_matrices[name] = X_test
        print(f"{name}: ROC-AUC={metrics['ROC-AUC']:.4f}, PR-AUC={metrics['PR-AUC']:.4f}, F1={metrics['F1']:.4f}")
        if export_predictions:
            pipeline_id = PIPELINE_IDS[name]
            pred_df = pd.DataFrame(
                {
                    "customer_id": customer_ids.iloc[test_idx].values,
                    "actual_churn": y_test,
                    "predicted_probability": probs,
                    "predicted_class": (probs >= THRESHOLD).astype(int),
                    "pipeline": pipeline_id,
                }
            )
            auc_from_file = roc_auc_score(pred_df["actual_churn"], pred_df["predicted_probability"])
            assert np.isclose(auc_from_file, metrics["ROC-AUC"], atol=1e-10)
            pred_df.to_csv(PREDICTION_DIR / f"{pipeline_id}_predictions.csv", index=False)
            prediction_frames.append(pred_df)

    proposed_name = "Structured + Mistral 7B cleaned semantic features"
    proposed_probs = probabilities[proposed_name] if proposed_name in probabilities else probabilities[configs[-1][0]]
    for row in results:
        if row["Configuration"] == proposed_name:
            row["Delta_AUC_vs_proposed"] = 0.0
            row["Paired_bootstrap_z_vs_proposed"] = 0.0
            row["Paired_bootstrap_p_vs_proposed"] = 1.0
            continue
        probs = probabilities[str(row["Configuration"])]
        z, p = paired_bootstrap_auc_test(y_test, proposed_probs, probs)
        row["Delta_AUC_vs_proposed"] = roc_auc_score(y_test, proposed_probs) - roc_auc_score(y_test, probs)
        row["Paired_bootstrap_z_vs_proposed"] = z
        row["Paired_bootstrap_p_vs_proposed"] = p

    if export_predictions:
        combined = pd.concat(prediction_frames, ignore_index=True)
        combined.to_csv(FINAL_DIR / "all_pipeline_predictions.csv", index=False)
    return pd.DataFrame(results), probabilities, models, test_matrices


def audit_json_fallbacks(llm: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    fallback = (
        (llm["sentiment"].astype(str) == "Neutral")
        & (pd.to_numeric(llm["urgency"], errors="coerce").fillna(0) == 0.0)
        & (llm["topic"].astype(str) == "Other")
        & (~llm["risk_indicator"].astype(bool))
    )
    audit = pd.DataFrame({"churn": y, "invalid_json": fallback})
    out = pd.DataFrame(
        [
            {
                "Group": "Retained",
                "Total": int((audit["churn"] == 0).sum()),
                "Invalid_JSON_Count": int(audit.loc[audit["churn"] == 0, "invalid_json"].sum()),
                "Invalid_JSON_Pct": round(100 * audit.loc[audit["churn"] == 0, "invalid_json"].mean(), 4),
            },
            {
                "Group": "Churned",
                "Total": int((audit["churn"] == 1).sum()),
                "Invalid_JSON_Count": int(audit.loc[audit["churn"] == 1, "invalid_json"].sum()),
                "Invalid_JSON_Pct": round(100 * audit.loc[audit["churn"] == 1, "invalid_json"].mean(), 4),
            },
        ]
    )
    out["Persistent_Fallback_Pct"] = out["Invalid_JSON_Pct"]
    out.to_csv(FINAL_DIR / "json_fallback_audit_final.csv", index=False)
    return out


def run_paper1_ablation(X_static, X_llm, y, customer_ids, train_idx, test_idx) -> pd.DataFrame:
    configs = [
        ("Structured only", X_static),
        ("Structured + LLM only", pd.concat([X_static, X_llm], axis=1)),
    ]
    df, probs, _, _ = train_evaluate_configs(configs, y, customer_ids, train_idx, test_idx)
    y_test = y.iloc[test_idx].to_numpy()
    z, p = paired_bootstrap_auc_test(y_test, probs["Structured + LLM only"], probs["Structured only"])
    base_auc = roc_auc_score(y_test, probs["Structured only"])
    df["Delta_AUC"] = df["ROC-AUC"] - base_auc
    df["Paired_bootstrap_z"] = ["", z]
    df["Paired_bootstrap_p"] = ["", p]
    df.to_csv(FINAL_DIR / "paper1_ablation.csv", index=False)
    return df


def plot_workflow() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")
    boxes = [
        ("ChurnKB Dataset", 0.08, 0.5),
        ("Mistral 7B\nLocal LLM", 0.30, 0.5),
        ("Cleaned Semantic\nFeatures", 0.50, 0.5),
        ("XGBoost\nClassifier", 0.70, 0.5),
        ("Churn Prediction\n+ SHAP", 0.90, 0.5),
    ]
    for text, x, y in boxes:
        ax.text(x, y, text, ha="center", va="center", fontsize=13, bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f7f7", "edgecolor": "#444444"})
    for x1, x2 in [(0.16, 0.23), (0.38, 0.43), (0.57, 0.63), (0.77, 0.83)]:
        ax.annotate("", xy=(x2, 0.5), xytext=(x1, 0.5), arrowprops={"arrowstyle": "->", "lw": 1.8})
    fig.savefig(FIGURE_DIR / "workflow.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_semantic_churn_rates(llm: pd.DataFrame, y: pd.Series) -> None:
    df = llm.copy()
    df["churn"] = y.values
    sent = df.groupby("sentiment")["churn"].mean().mul(100).reindex(["Negative", "Neutral", "Positive"]).dropna()
    emo = df.groupby("emotion")["churn"].mean().mul(100).sort_values(ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    sent.plot(kind="bar", ax=axes[0], color=["#d62728", "#ff9800", "#2ca02c"])
    axes[0].set_title("Churn rate by cleaned sentiment")
    axes[0].set_ylabel("Churn rate (%)")
    emo.plot(kind="bar", ax=axes[1], color="#777777")
    axes[1].set_title("Churn rate by emotion")
    axes[1].set_ylabel("Churn rate (%)")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "semantic_churn_rates.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline_metrics(metrics: pd.DataFrame) -> None:
    plot_df = metrics.set_index("Configuration")[["ROC-AUC", "PR-AUC", "F1"]]
    fig, ax = plt.subplots(figsize=(11, 5))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Pipeline metrics")
    ax.set_ylabel("Score")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "pipeline_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr(y_test: np.ndarray, probabilities: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for name, probs in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        precision, recall, _ = precision_recall_curve(y_test, probs)
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, probs):.3f})")
        axes[1].plot(recall, precision, label=f"{name} ({sk_auc(recall, precision):.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    for ax, title, xlabel, ylabel in [
        (axes[0], "ROC curves", "False positive rate", "True positive rate"),
        (axes[1], "Precision-recall curves", "Recall", "Precision"),
    ]:
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "roc_pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_shap(model: xgb.XGBClassifier, X_test: pd.DataFrame) -> pd.DataFrame:
    values = shap.TreeExplainer(model).shap_values(X_test)
    top = (
        pd.DataFrame({"Feature": X_test.columns, "Mean_SHAP": np.abs(values).mean(axis=0)})
        .sort_values("Mean_SHAP", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    llm_features = {"sentiment_enc", "emotion_enc", "urgency", "topic_enc", "risk_enc"}
    top["Type"] = np.where(top["Feature"].isin(llm_features), "LLM", "Structured")
    top["Source"] = np.where(top["Type"] == "LLM", "Mistral 7B cleaned", "ChurnKB")
    top.insert(0, "Rank", np.arange(1, len(top) + 1))
    top.to_csv(FINAL_DIR / "paper1_shap_top10.csv", index=False)
    plot_df = top.sort_values("Mean_SHAP")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(plot_df["Feature"], plot_df["Mean_SHAP"], color=np.where(plot_df["Type"] == "LLM", "#1f77b4", "#777777"))
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title("Global SHAP importance")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "shap_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return top


def plot_confusion_calibration(y_test: np.ndarray, probabilities: dict[str, np.ndarray]) -> None:
    proposed = probabilities["Structured + Mistral 7B cleaned semantic features"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.heatmap(
        confusion_matrix(y_test, (proposed >= THRESHOLD).astype(int)),
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
        ax=axes[0],
    )
    axes[0].set_title(f"Confusion matrix (threshold={THRESHOLD:.2f})")
    for label, probs in probabilities.items():
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
        axes[1].plot(mean_pred, frac_pos, marker="o", label=label)
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].set_title("Calibration curve")
    axes[1].set_xlabel("Mean predicted probability")
    axes[1].set_ylabel("Fraction of positives")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "calibration_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_lift_chart(y_test: np.ndarray, static_probs: np.ndarray, proposed_probs: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    base_rate = y_test.mean()
    for probs, label in [(static_probs, "Structured only"), (proposed_probs, "Mistral 7B cleaned")]:
        order = np.argsort(probs)[::-1]
        sorted_y = y_test[order]
        lift = (np.cumsum(sorted_y) / np.arange(1, len(sorted_y) + 1)) / base_rate
        ax.plot(np.arange(1, len(sorted_y) + 1) / len(sorted_y), lift, label=label, lw=2)
    ax.axhline(1.0, color="black", linestyle="--", lw=1)
    ax.set_xlabel("Fraction of test customers targeted")
    ax.set_ylabel("Lift over base churn rate")
    ax.set_title("Lift chart")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "lift_chart.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def compare_with_prior(metrics: pd.DataFrame) -> pd.DataFrame:
    prior_path = PRIOR_LOCKED_DIR / "metrics_final.csv"
    rows = []
    if prior_path.exists():
        prior = pd.read_csv(prior_path)
        name_map = {"Structured + Mistral 7B (proposed)": "Structured + Mistral 7B cleaned semantic features"}
        prior["Configuration"] = prior["Configuration"].replace(name_map)
        merged = metrics.merge(prior, on="Configuration", suffixes=("_new", "_prior"))
        for _, row in merged.iterrows():
            rows.append(
                {
                    "Configuration": row["Configuration"],
                    "ROC_AUC_new": row["ROC-AUC_new"],
                    "ROC_AUC_prior": row["ROC-AUC_prior"],
                    "ROC_AUC_delta": row["ROC-AUC_new"] - row["ROC-AUC_prior"],
                    "PR_AUC_delta": row["PR-AUC_new"] - row["PR-AUC_prior"],
                    "F1_delta": row["F1_new"] - row["F1_prior"],
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(FINAL_DIR / "prior_locked_run_comparison.csv", index=False)
    return out


def verify_final_outputs(metrics: pd.DataFrame, cleaned_llm: pd.DataFrame) -> dict[str, object]:
    expected = [
        "Structured_only_predictions.csv",
        "VADER_predictions.csv",
        "TFIDF_SVD_predictions.csv",
        "RoBERTa_sentiment_predictions.csv",
        "Mistral_7B_predictions.csv",
    ]
    ids_reference = None
    checks = {
        "no_satisfaction_in_sentiment": "Satisfaction" not in set(cleaned_llm["sentiment"]),
        "satisfaction_available_in_emotion": "Satisfaction" in set(cleaned_llm["emotion"]),
        "prediction_rows_each": 667,
        "identical_test_records": True,
        "missing_prediction_probabilities": 0,
        "auc_files_match_metrics": True,
    }
    for fname in expected:
        pred = pd.read_csv(PREDICTION_DIR / fname)
        assert len(pred) == 667
        checks["missing_prediction_probabilities"] += int(pred["predicted_probability"].isna().sum())
        ids = tuple(pred["customer_id"].astype(str))
        if ids_reference is None:
            ids_reference = ids
        elif ids != ids_reference:
            checks["identical_test_records"] = False
        pipeline = pred["pipeline"].iloc[0]
        config = next(name for name, pipe in PIPELINE_IDS.items() if pipe == pipeline)
        metric_auc = float(metrics.loc[metrics["Configuration"] == config, "ROC-AUC"].iloc[0])
        file_auc = roc_auc_score(pred["actual_churn"], pred["predicted_probability"])
        if not np.isclose(file_auc, metric_auc, atol=1e-10):
            checks["auc_files_match_metrics"] = False
    assert all(
        [
            checks["no_satisfaction_in_sentiment"],
            checks["satisfaction_available_in_emotion"],
            checks["identical_test_records"],
            checks["missing_prediction_probabilities"] == 0,
            checks["auc_files_match_metrics"],
        ]
    )
    return checks


def write_metadata(n_records: int, train_idx: np.ndarray, test_idx: np.ndarray, checks: dict[str, object]) -> None:
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "script_version_git_commit": git_commit_hash(),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "threshold": THRESHOLD,
        "n_records": int(n_records),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "mistral_features_regenerated": False,
        "semantic_schema_correction": "Satisfaction sentiment mapped to Positive; unknown sentiment mapped to Neutral.",
        "auc_comparison_method": BOOTSTRAP_LABEL,
        "acceptance_checks": checks,
    }
    (FINAL_DIR / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def write_summary(metrics, ablation, shap_top, fallback_audit, schema_audit, prior_comparison, checks) -> None:
    lines = [
        "=== FINAL LOCKED RUN REPORT ===",
        "Semantic schema correction: sentiment Satisfaction -> Positive.",
        "Final sentiment labels: Positive, Neutral, Negative.",
        "Satisfaction remains valid only in emotion.",
        "",
        "=== ACCEPTANCE CHECKS ===",
    ]
    lines.extend([f"{key}: {value}" for key, value in checks.items()])
    lines.extend(["", "=== METRICS FINAL ===", metrics.to_string(index=False)])
    lines.extend(["", "=== PRIOR LOCKED RUN COMPARISON ===", prior_comparison.to_string(index=False)])
    lines.extend(["", "=== ABLATION ===", ablation.to_string(index=False)])
    lines.extend(["", "=== SHAP TOP 10 ===", shap_top.to_string(index=False)])
    lines.extend(["", "=== SENTIMENT SCHEMA AUDIT ===", schema_audit.to_string(index=False)])
    lines.extend(["", "=== JSON FALLBACK AUDIT ===", fallback_audit.to_string(index=False)])
    (FINAL_DIR / "final_run_log.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    reset_output_dirs()
    X_static, X_llm, y, chat_logs, cleaned_llm, customer_ids, schema_audit = load_inputs()
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    X_vader = extract_vader(chat_logs)
    X_tfidf_svd = extract_tfidf_svd_features(chat_logs, train_idx)
    X_roberta = extract_roberta(chat_logs)
    configs = [
        ("Structured only", X_static),
        ("Structured + VADER", pd.concat([X_static, X_vader], axis=1)),
        ("Structured + TF-IDF+SVD", pd.concat([X_static, X_tfidf_svd], axis=1)),
        ("Structured + RoBERTa sentiment", pd.concat([X_static, X_roberta], axis=1)),
        ("Structured + Mistral 7B cleaned semantic features", pd.concat([X_static, X_llm], axis=1)),
    ]
    metrics, probabilities, models, test_matrices = train_evaluate_configs(configs, y, customer_ids, train_idx, test_idx, True)
    metrics.to_csv(FINAL_DIR / "metrics_final.csv", index=False)
    ablation = run_paper1_ablation(X_static, X_llm, y, customer_ids, train_idx, test_idx)
    fallback_audit = audit_json_fallbacks(cleaned_llm, y)
    y_test = y.iloc[test_idx].to_numpy()

    plot_workflow()
    plot_semantic_churn_rates(cleaned_llm, y)
    plot_pipeline_metrics(metrics)
    plot_roc_pr(y_test, probabilities)
    proposed_name = "Structured + Mistral 7B cleaned semantic features"
    shap_top = plot_shap(models[proposed_name], test_matrices[proposed_name])
    plot_confusion_calibration(y_test, probabilities)
    plot_lift_chart(y_test, probabilities["Structured only"], probabilities[proposed_name])

    prior_comparison = compare_with_prior(metrics)
    checks = verify_final_outputs(metrics, cleaned_llm)
    write_metadata(len(y), train_idx, test_idx, checks)
    write_summary(metrics, ablation, shap_top, fallback_audit, schema_audit, prior_comparison, checks)
    print(f"Done. Final cleaned locked outputs written to {FINAL_DIR}.")


if __name__ == "__main__":
    main()
