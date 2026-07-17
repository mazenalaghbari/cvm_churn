#!/usr/bin/env python3
"""Leakage-free static knowledge pipeline for the KBS paper revision.

The source data contain one observation per customer. This script therefore does
not create synthetic monthly sequences and does not use an LSTM. It builds an
outcome-independent anomaly score from training predictors only, adds transparent
knowledge-guided telecom features, trains XGBoost, evaluates on a locked holdout,
and exports reproducible results for the revised manuscript.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_BOOTSTRAP = 1000


@dataclass
class Metrics:
    configuration: str
    roc_auc: float
    pr_auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    brier: float
    tn: int
    fp: int
    fn: int
    tp: int


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )
    aliases = {
        "customer_service_calls": "number_customer_service_calls",
        "number_customer_service_call": "number_customer_service_calls",
        "voice_mail_plan": "voice_mail_plan",
        "voicemail_plan": "voice_mail_plan",
    }
    out = out.rename(columns={k: v for k, v in aliases.items() if k in out.columns})
    return out


def encode_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)
    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1,
        "true": 1,
        "1": 1,
        "churn": 1,
        "no": 0,
        "false": 0,
        "0": 0,
        "retained": 0,
    }
    encoded = text.map(mapping)
    if encoded.isna().any():
        bad = sorted(text[encoded.isna()].unique().tolist())
        raise ValueError(f"Unsupported churn labels: {bad}")
    return encoded.astype(int)


def yes_no_to_float(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1.0, "true": 1.0, "1": 1.0, "no": 0.0, "false": 0.0, "0": 0.0})
        .fillna(0.0)
    )


def add_knowledge_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create domain features from observed predictors only.

    The churn target is never accessed. All formulas are deterministic and can be
    computed at prediction time from the single customer observation.
    """
    out = df.copy()
    eps = 1e-6

    def col(name: str) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(0.0)
        return pd.Series(0.0, index=out.index)

    day_min = col("total_day_minutes")
    eve_min = col("total_eve_minutes")
    night_min = col("total_night_minutes")
    intl_min = col("total_intl_minutes")
    day_calls = col("total_day_calls")
    eve_calls = col("total_eve_calls")
    night_calls = col("total_night_calls")
    intl_calls = col("total_intl_calls")
    day_charge = col("total_day_charge")
    eve_charge = col("total_eve_charge")
    night_charge = col("total_night_charge")
    intl_charge = col("total_intl_charge")
    service_calls = col("number_customer_service_calls")
    account_length = col("account_length")
    vmail_messages = col("number_vmail_messages")

    total_minutes = day_min + eve_min + night_min + intl_min
    total_calls = day_calls + eve_calls + night_calls + intl_calls
    total_charge = day_charge + eve_charge + night_charge + intl_charge
    domestic_minutes = day_min + eve_min + night_min

    out["kg_total_minutes"] = total_minutes
    out["kg_total_calls"] = total_calls
    out["kg_total_charge"] = total_charge
    out["kg_avg_minutes_per_call"] = total_minutes / (total_calls + eps)
    out["kg_charge_per_minute"] = total_charge / (total_minutes + eps)
    out["kg_international_usage_share"] = intl_min / (total_minutes + eps)
    out["kg_day_usage_share"] = day_min / (domestic_minutes + eps)
    out["kg_night_usage_share"] = night_min / (domestic_minutes + eps)
    out["kg_service_contact_intensity"] = service_calls / (account_length + 1.0)
    out["kg_usage_per_tenure_month"] = total_minutes / (account_length + 1.0)
    out["kg_call_dispersion"] = pd.concat(
        [day_min, eve_min, night_min], axis=1
    ).std(axis=1) / (pd.concat([day_min, eve_min, night_min], axis=1).mean(axis=1) + eps)

    intl_plan = yes_no_to_float(out["international_plan"]) if "international_plan" in out else 0.0
    vmail_plan = yes_no_to_float(out["voice_mail_plan"]) if "voice_mail_plan" in out else 0.0
    out["kg_intl_plan_low_use_mismatch"] = intl_plan * (intl_min < intl_min.median()).astype(float)
    out["kg_vmail_plan_low_use_mismatch"] = vmail_plan * (vmail_messages <= 1).astype(float)
    out["kg_high_service_calls"] = (service_calls >= 4).astype(float)
    return out


def numeric_behavior_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "account_length",
        "number_vmail_messages",
        "total_day_minutes",
        "total_day_calls",
        "total_eve_minutes",
        "total_eve_calls",
        "total_night_minutes",
        "total_night_calls",
        "total_intl_minutes",
        "total_intl_calls",
        "number_customer_service_calls",
    ]
    return [c for c in preferred if c in df.columns]


def fit_anomaly_features(
    train_df: pd.DataFrame,
    apply_frames: Iterable[pd.DataFrame],
) -> tuple[list[pd.DataFrame], dict[str, float]]:
    """Fit the scaler and Isolation Forest on training predictors only."""
    cols = numeric_behavior_columns(train_df)
    if len(cols) < 3:
        raise ValueError("At least three numeric behavioural columns are required")

    scaler = StandardScaler()
    train_numeric = train_df[cols].apply(pd.to_numeric, errors="coerce")
    medians = train_numeric.median(numeric_only=True)
    train_numeric = train_numeric.fillna(medians)
    train_scaled = scaler.fit_transform(train_numeric)

    detector = IsolationForest(
        n_estimators=500,
        contamination="auto",
        max_samples="auto",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    detector.fit(train_scaled)  # no churn label is supplied

    train_scores = -detector.decision_function(train_scaled)
    threshold = float(np.percentile(train_scores, 95))

    outputs: list[pd.DataFrame] = []
    for frame in apply_frames:
        transformed = frame.copy()
        numeric = transformed[cols].apply(pd.to_numeric, errors="coerce").fillna(medians)
        scaled = scaler.transform(numeric)
        scores = -detector.decision_function(scaled)
        transformed["anomaly_score"] = scores
        transformed["anomaly_flag"] = (scores >= threshold).astype(int)
        outputs.append(transformed)

    metadata = {
        "threshold_95th_training_percentile": threshold,
        "training_score_mean": float(np.mean(train_scores)),
        "training_score_sd": float(np.std(train_scores, ddof=1)),
        "behavioral_columns_count": len(cols),
    }
    return outputs, metadata


def make_preprocessor(df: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical = [
        c
        for c in df.columns
        if df[c].dtype == "object" or str(df[c].dtype).startswith("category")
    ]
    numeric = [c for c in df.columns if c not in categorical]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric, categorical


def make_model(y_train: pd.Series) -> XGBClassifier:
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / max(positives, 1)
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=350,
        max_depth=5,
        learning_rate=0.04,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.10,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


def evaluate(configuration: str, y_true: pd.Series, probability: np.ndarray) -> Metrics:
    prediction = (probability >= 0.50).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return Metrics(
        configuration=configuration,
        roc_auc=float(roc_auc_score(y_true, probability)),
        pr_auc=float(average_precision_score(y_true, probability)),
        accuracy=float(accuracy_score(y_true, prediction)),
        precision=float(precision_score(y_true, prediction, zero_division=0)),
        recall=float(recall_score(y_true, prediction, zero_division=0)),
        f1=float(f1_score(y_true, prediction, zero_division=0)),
        brier=float(brier_score_loss(y_true, probability)),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
    )


def prepare_configuration(
    configuration: str,
    train_raw: pd.DataFrame,
    eval_raw: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    base_drop = [c for c in ["chat_log"] if c in train_raw.columns]
    train = train_raw.drop(columns=base_drop).copy()
    evaluation = eval_raw.drop(columns=base_drop).copy()
    anomaly_meta: dict[str, float] = {}

    if configuration in {"Knowledge-guided", "Knowledge + anomaly"}:
        train = add_knowledge_features(train)
        evaluation = add_knowledge_features(evaluation)

    if configuration == "Knowledge + anomaly":
        (train, evaluation), anomaly_meta = fit_anomaly_features(train, [train, evaluation])

    return train, evaluation, anomaly_meta


def fit_configuration(
    configuration: str,
    train_raw: pd.DataFrame,
    y_train: pd.Series,
    eval_raw: pd.DataFrame,
) -> tuple[np.ndarray, XGBClassifier, ColumnTransformer, pd.DataFrame, dict[str, float]]:
    train, evaluation, anomaly_meta = prepare_configuration(configuration, train_raw, eval_raw)
    preprocessor, _, _ = make_preprocessor(train)
    x_train = preprocessor.fit_transform(train)
    x_eval = preprocessor.transform(evaluation)
    model = make_model(y_train)
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_eval)[:, 1]
    return probability, model, preprocessor, evaluation, anomaly_meta


def cross_validate(
    configuration: str,
    x_raw: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict[str, float | int | str]] = []
    for fold, (train_idx, valid_idx) in enumerate(cv.split(x_raw, y), start=1):
        x_train = x_raw.iloc[train_idx].reset_index(drop=True)
        x_valid = x_raw.iloc[valid_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        y_valid = y.iloc[valid_idx].reset_index(drop=True)
        probability, _, _, _, _ = fit_configuration(
            configuration, x_train, y_train, x_valid
        )
        rows.append(
            {
                "configuration": configuration,
                "fold": fold,
                "roc_auc": float(roc_auc_score(y_valid, probability)),
                "pr_auc": float(average_precision_score(y_valid, probability)),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_auc(
    y_true: np.ndarray,
    proposed: np.ndarray,
    baseline: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
) -> dict[str, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(y_true)
    diffs: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_sample = y_true[idx]
        if np.unique(y_sample).size < 2:
            continue
        diffs.append(
            roc_auc_score(y_sample, proposed[idx])
            - roc_auc_score(y_sample, baseline[idx])
        )
    arr = np.asarray(diffs, dtype=float)
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1))
    z = mean / sd if sd > 0 else math.inf
    p = float(2 * norm.sf(abs(z))) if np.isfinite(z) else 0.0
    return {
        "mean_auc_difference": mean,
        "ci_lower": float(np.percentile(arr, 2.5)),
        "ci_upper": float(np.percentile(arr, 97.5)),
        "bootstrap_sd": sd,
        "z_approximation": float(z),
        "p_approximation": p,
        "valid_resamples": int(len(arr)),
    }


def create_rule_routes(
    original: pd.DataFrame,
    enriched: pd.DataFrame,
    probability: np.ndarray,
) -> pd.DataFrame:
    routes: list[str] = []
    reasons: list[str] = []
    for i in range(len(original)):
        row = original.iloc[i]
        e = enriched.iloc[i]
        service_calls = float(row.get("number_customer_service_calls", 0) or 0)
        intl_plan = str(row.get("international_plan", "no")).lower() in {"yes", "true", "1"}
        intl_minutes = float(row.get("total_intl_minutes", 0) or 0)
        vmail_plan = str(row.get("voice_mail_plan", "no")).lower() in {"yes", "true", "1"}
        vmail_messages = float(row.get("number_vmail_messages", 0) or 0)
        anomaly = int(e.get("anomaly_flag", 0) or 0)
        risk = float(probability[i])

        if risk < 0.50:
            route, reason = "Standard monitoring", "Predicted risk below operational threshold"
        elif service_calls >= 4:
            route, reason = "Service recovery", "Repeated customer-service contact"
        elif intl_plan and intl_minutes < 8:
            route, reason = "International plan review", "Plan-to-usage mismatch"
        elif vmail_plan and vmail_messages <= 1:
            route, reason = "Voicemail plan review", "Subscribed feature has little observed use"
        elif anomaly == 1:
            route, reason = "Proactive account check-in", "Outcome-independent behavioural anomaly"
        else:
            route, reason = "Value and tariff review", "Elevated risk without a dominant service rule"
        routes.append(route)
        reasons.append(reason)

    return pd.DataFrame(
        {
            "predicted_probability": probability,
            "route": routes,
            "rule_reason": reasons,
        }
    )


def save_plots(
    y_test: pd.Series,
    probabilities: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for name, probability in probabilities.items():
        fpr, tpr, _ = roc_curve(y_test, probability)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, probability):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve comparison")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_comparison.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for name, probability in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_test, probability)
        ax.plot(recall, precision, label=f"{name} (AP={average_precision_score(y_test, probability):.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve comparison")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "precision_recall_comparison.png", dpi=300)
    plt.close(fig)


def save_shap(
    model: XGBClassifier,
    preprocessor: ColumnTransformer,
    evaluation: pd.DataFrame,
    out_dir: Path,
) -> None:
    try:
        import shap

        transformed = preprocessor.transform(evaluation)
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(transformed)
        if isinstance(values, list):
            values = values[-1]
        names = preprocessor.get_feature_names_out()
        importance = np.abs(np.asarray(values)).mean(axis=0)
        table = (
            pd.DataFrame({"feature": names, "mean_abs_shap": importance})
            .sort_values("mean_abs_shap", ascending=False)
            .head(20)
        )
        table.to_csv(out_dir / "shap_top20.csv", index=False)

        top = table.sort_values("mean_abs_shap", ascending=True)
        fig, ax = plt.subplots(figsize=(8.0, 6.2))
        ax.barh(top["feature"], top["mean_abs_shap"])
        ax.set_xlabel("Mean absolute SHAP value")
        ax.set_title("Global feature importance: corrected proposed model")
        fig.tight_layout()
        fig.savefig(out_dir / "shap_top20.png", dpi=300)
        plt.close(fig)
    except Exception as exc:  # SHAP is useful but should not invalidate the run
        (out_dir / "shap_error.txt").write_text(str(exc), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="data/raw_sources/churn_dataset.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--out",
        default="kbs_static_fix/outputs",
        help="Output directory",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = normalize_columns(pd.read_csv(data_path))
    if "churn" not in df.columns:
        raise ValueError("The input CSV must contain a churn column")

    y = encode_target(df.pop("churn"))
    x = df.copy()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    x_train = x_train.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    configurations = ["Structured baseline", "Knowledge-guided", "Knowledge + anomaly"]
    probabilities: dict[str, np.ndarray] = {}
    metrics: list[Metrics] = []
    models: dict[str, XGBClassifier] = {}
    preprocessors: dict[str, ColumnTransformer] = {}
    eval_frames: dict[str, pd.DataFrame] = {}
    anomaly_metadata: dict[str, dict[str, float]] = {}

    for configuration in configurations:
        probability, model, preprocessor, evaluation, meta = fit_configuration(
            configuration, x_train, y_train, x_test
        )
        probabilities[configuration] = probability
        metrics.append(evaluate(configuration, y_test, probability))
        models[configuration] = model
        preprocessors[configuration] = preprocessor
        eval_frames[configuration] = evaluation
        if meta:
            anomaly_metadata[configuration] = meta

    metrics_df = pd.DataFrame([asdict(m) for m in metrics])
    metrics_df.to_csv(out_dir / "holdout_metrics.csv", index=False)

    cv_frames = [cross_validate(c, x, y) for c in configurations]
    cv_df = pd.concat(cv_frames, ignore_index=True)
    cv_df.to_csv(out_dir / "five_fold_cv_metrics.csv", index=False)
    cv_summary = (
        cv_df.groupby("configuration")
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_sd=("roc_auc", "std"),
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_sd=("pr_auc", "std"),
        )
        .reset_index()
    )
    cv_summary.to_csv(out_dir / "five_fold_cv_summary.csv", index=False)

    bootstrap = paired_bootstrap_auc(
        y_test.to_numpy(),
        probabilities["Knowledge + anomaly"],
        probabilities["Structured baseline"],
    )
    (out_dir / "paired_bootstrap_auc.json").write_text(
        json.dumps(bootstrap, indent=2), encoding="utf-8"
    )

    route_df = create_rule_routes(
        x_test,
        eval_frames["Knowledge + anomaly"],
        probabilities["Knowledge + anomaly"],
    )
    predictions = x_test.copy()
    predictions.insert(0, "actual_churn", y_test.to_numpy())
    predictions.insert(1, "predicted_probability", probabilities["Knowledge + anomaly"])
    predictions.insert(2, "predicted_class", (probabilities["Knowledge + anomaly"] >= 0.50).astype(int))
    predictions["knowledge_route"] = route_df["route"]
    predictions["route_reason"] = route_df["rule_reason"]
    predictions.to_csv(out_dir / "locked_holdout_predictions.csv", index=False)
    route_df["actual_churn"] = y_test.to_numpy()
    route_summary = (
        route_df.groupby("route")
        .agg(
            records=("route", "size"),
            mean_predicted_risk=("predicted_probability", "mean"),
            observed_churn_rate=("actual_churn", "mean"),
        )
        .reset_index()
        .sort_values("mean_predicted_risk", ascending=False)
    )
    route_summary.to_csv(out_dir / "knowledge_route_summary.csv", index=False)

    save_plots(y_test, probabilities, out_dir)
    save_shap(
        models["Knowledge + anomaly"],
        preprocessors["Knowledge + anomaly"],
        eval_frames["Knowledge + anomaly"],
        out_dir,
    )

    summary = {
        "data_path": str(data_path),
        "records": int(len(df)),
        "predictors_before_engineering": int(df.shape[1]),
        "churn_records": int(y.sum()),
        "churn_rate": float(y.mean()),
        "train_records": int(len(x_train)),
        "test_records": int(len(x_test)),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "outcome_used_in_feature_generation": False,
        "synthetic_monthly_sequences_created": False,
        "chat_log_used_in_primary_models": False,
        "anomaly_metadata": anomaly_metadata,
        "bootstrap_proposed_vs_baseline": bootstrap,
        "software": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(metrics_df.to_string(index=False))
    print("\nFive-fold CV summary")
    print(cv_summary.to_string(index=False))
    print("\nPaired bootstrap")
    print(json.dumps(bootstrap, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
