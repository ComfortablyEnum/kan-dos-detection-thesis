"""
Comparative analysis module for DoS detection baselines.
Trains lightweight but academically valid baseline models and compares them to KAN.

Author: Samuele Scaffidi
University: eCampus
Year: 2026
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
import warnings
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier

from train import prepare_dos_data, train_kan_model

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import torch
except ImportError:
    torch = None

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


RANDOM_SEED = 42


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def dataset_to_numpy(dataset: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_train = dataset["train_input"].detach().cpu().numpy()
    y_train = dataset["train_label"].detach().cpu().numpy().ravel().astype(int)
    X_test = dataset["test_input"].detach().cpu().numpy()
    y_test = dataset["test_label"].detach().cpu().numpy().ravel().astype(int)
    return X_train, y_train, X_test, y_test


def resolve_data_path(requested_path: Path) -> Path:
    if requested_path.exists():
        return requested_path

    candidate_name = requested_path.name
    for candidate in Path(".").rglob(candidate_name):
        if ".venv" not in candidate.parts:
            return candidate

    raise FileNotFoundError(
        f"Dataset not found at '{requested_path}'. "
        "Place the CSV in that path or keep the same filename somewhere inside the project."
    )


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    train_time_sec: float,
    inference_time_sec: float,
    notes: str = "",
) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "train_time_sec": train_time_sec,
        "inference_time_sec": inference_time_sec,
        "notes": notes,
    }


def train_kan_baseline(dataset: dict, input_dim: int, epochs: int) -> dict:
    start = time.time()
    model, history = train_kan_model(dataset, input_dim=input_dim, epochs=epochs)
    train_time_sec = time.time() - start

    inference_start = time.time()
    test_logits = model(dataset["test_input"]).detach().cpu().numpy().ravel()
    inference_time_sec = time.time() - inference_start

    test_probs = 1 / (1 + np.exp(-test_logits))
    y_pred = (test_probs > 0.5).astype(int)
    y_true = dataset["test_label"].detach().cpu().numpy().ravel().astype(int)

    metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        train_time_sec=train_time_sec,
        inference_time_sec=inference_time_sec,
        notes=f"KAN trained for {epochs} epochs",
    )
    metrics["history"] = history
    return metrics


def train_sklearn_model(name: str, model, X_train, y_train, X_test, y_test) -> dict:
    start = time.time()
    model.fit(X_train, y_train)
    train_time_sec = time.time() - start

    inference_start = time.time()
    y_pred = model.predict(X_test)
    inference_time_sec = time.time() - inference_start

    metrics = compute_metrics(
        y_true=y_test,
        y_pred=y_pred,
        train_time_sec=train_time_sec,
        inference_time_sec=inference_time_sec,
        notes=f"{name} baseline",
    )
    return metrics


def build_models() -> dict:
    models = {
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            batch_size=256,
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            random_state=RANDOM_SEED,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=RANDOM_SEED,
        )

    return models


def plot_metric_comparison(results_df: pd.DataFrame, output_dir: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1"]
    plot_df = results_df.set_index("model")[metrics]

    ax = plot_df.plot(kind="bar", figsize=(12, 6), width=0.8)
    ax.set_title("Comparative Analysis of DoS Detection Models")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_thesis_tables(results_df: pd.DataFrame, output_dir: Path) -> None:
    thesis_table_main = results_df[
        ["model", "accuracy", "precision", "recall", "f1"]
    ].copy()

    thesis_table_extended = results_df[
        ["model", "accuracy", "precision", "recall", "f1", "train_time_sec", "inference_time_sec"]
    ].copy()

    thesis_table_main.to_csv(output_dir / "thesis_comparison_table.csv", index=False)
    thesis_table_extended.to_csv(output_dir / "thesis_comparison_table_extended.csv", index=False)


def write_report(results_df: pd.DataFrame, output_dir: Path, dataset_info: dict) -> None:
    thesis_table = results_df[
        ["model", "accuracy", "precision", "recall", "f1", "train_time_sec", "inference_time_sec"]
    ].copy()

    thesis_table = thesis_table.rename(
        columns={
            "model": "Model",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "train_time_sec": "Train time (s)",
            "inference_time_sec": "Inference time (s)",
        }
    )

    for column in ["Accuracy", "Precision", "Recall", "F1"]:
        thesis_table[column] = thesis_table[column].map(lambda value: f"{value:.4f}")

    for column in ["Train time (s)", "Inference time (s)"]:
        thesis_table[column] = thesis_table[column].map(lambda value: f"{value:.6f}")

    best_model = results_df.sort_values("f1", ascending=False).iloc[0]

    xgboost_note = (
        ""
        if "XGBoost" in results_df["model"].values
        else "- XGBoost was skipped because the `xgboost` package is not installed.\n"
    )

    report = (
        "# Comparative Analysis Report\n\n"
        "## Experimental setup\n"
        f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- Attack type: {dataset_info['attack_type']}\n"
        f"- Samples per class: {dataset_info['max_samples_per_class']}\n"
        f"- KAN epochs: {dataset_info['kan_epochs']}\n"
        f"- Data path: `{dataset_info['data_path']}`\n"
        f"- Random seed: {dataset_info['random_seed']}\n\n"
        "## Thesis-ready comparison table\n"
        f"{thesis_table.to_markdown(index=False)}\n\n"
        "## Best model in this run\n"
        f"- Model: {best_model['model']}\n"
        f"- Accuracy: {best_model['accuracy']:.4f}\n"
        f"- Precision: {best_model['precision']:.4f}\n"
        f"- Recall: {best_model['recall']:.4f}\n"
        f"- F1-score: {best_model['f1']:.4f}\n"
        f"- Training time (s): {best_model['train_time_sec']:.6f}\n"
        f"- Inference time (s): {best_model['inference_time_sec']:.6f}\n\n"
        "## Notes\n"
        "- This script is designed for a fast comparative baseline suitable for thesis discussion.\n"
        "- All models use the same train/test split produced by `prepare_dos_data`.\n"
        "- Inference time is measured on the full test set and is included as a complementary efficiency indicator.\n"
        f"{xgboost_note}"
    )

    with open(output_dir / "comparison_report.md", "w", encoding="utf-8") as handle:
        handle.write(report)


def main() -> None:
    set_global_seed(RANDOM_SEED)

    data_path = resolve_data_path(Path("data/Wednesday-workingHours.pcap_ISCX.csv"))
    attack_type = "DoS Hulk"
    max_samples_per_class = 30000
    kan_epochs = 75
    output_dir = Path("experiment_data_comparison")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing dataset for comparative analysis...")
    dataset, scaler, features = prepare_dos_data(
        data_path,
        attack_type=attack_type,
        max_samples_per_class=max_samples_per_class,
    )
    X_train, y_train, X_test, y_test = dataset_to_numpy(dataset)

    results = []

    print("\nTraining KAN baseline...")
    kan_metrics = train_kan_baseline(
        dataset=dataset,
        input_dim=len(features),
        epochs=kan_epochs,
    )
    results.append(
        {
            "model": "KAN",
            "accuracy": kan_metrics["accuracy"],
            "precision": kan_metrics["precision"],
            "recall": kan_metrics["recall"],
            "f1": kan_metrics["f1"],
            "train_time_sec": kan_metrics["train_time_sec"],
            "inference_time_sec": kan_metrics["inference_time_sec"],
            "notes": kan_metrics["notes"],
        }
    )

    for model_name, model in build_models().items():
        print(f"\nTraining {model_name}...")
        metrics = train_sklearn_model(
            model_name,
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )
        results.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "train_time_sec": metrics["train_time_sec"],
                "inference_time_sec": metrics["inference_time_sec"],
                "notes": metrics["notes"],
            }
        )

    results_df = pd.DataFrame(results).sort_values(by="f1", ascending=False)
    results_df.to_csv(output_dir / "comparison_results.csv", index=False)

    save_thesis_tables(results_df, output_dir)
    plot_metric_comparison(results_df, output_dir)

    dataset_info = {
        "attack_type": attack_type,
        "max_samples_per_class": max_samples_per_class,
        "kan_epochs": kan_epochs,
        "data_path": data_path,
        "random_seed": RANDOM_SEED,
    }
    write_report(results_df, output_dir, dataset_info)

    print("\nComparative analysis completed.")
    print(f"Results CSV: {output_dir / 'comparison_results.csv'}")
    print(f"Thesis table CSV: {output_dir / 'thesis_comparison_table.csv'}")
    print(f"Extended thesis table CSV: {output_dir / 'thesis_comparison_table_extended.csv'}")
    print(f"Plot: {output_dir / 'comparison_metrics.png'}")
    print(f"Report: {output_dir / 'comparison_report.md'}")

    if not XGBOOST_AVAILABLE:
        print("Note: XGBoost was skipped because the package is not installed.")


if __name__ == "__main__":
    main()