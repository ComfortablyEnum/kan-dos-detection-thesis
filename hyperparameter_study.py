from pathlib import Path
from datetime import datetime
import time
import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from kan import KAN
from train import prepare_dos_data


def train_custom_kan(
    dataset,
    input_dim,
    width,
    grid=5,
    k=3,
    epochs=15,
    batch_size=512,
    lr=0.001,
    seed=42,
):
    torch.manual_seed(seed)

    model = KAN(
        width=width,
        grid=grid,
        k=k,
        seed=seed,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    Xtr = dataset["train_input"].float()
    ytr = dataset["train_label"].float()
    Xte = dataset["test_input"].float()
    yte = dataset["test_label"].float()

    train_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    test_loader = DataLoader(
        TensorDataset(Xte, yte),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "epochs": [],
    }

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += yb.numel()

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in test_loader:
                logits = model(xb)
                loss = criterion(logits, yb)

                running_loss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.numel()

        test_loss = running_loss / len(test_loader.dataset)
        test_acc = correct / total

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["epochs"].append(epoch + 1)

        if (epoch + 1) % 5 == 0:
            print(
                f"[{width} | grid={grid} | k={k}] "
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Acc: {train_acc:.4f} - Test Acc: {test_acc:.4f}"
            )

    elapsed = time.time() - start_time

    model.eval()
    with torch.no_grad():
        logits = model(dataset["test_input"].float())
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        preds = (probs > 0.5).astype(int)

    y_true = dataset["test_label"].cpu().numpy().ravel().astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "train_time_sec": elapsed,
        "best_test_acc": max(history["test_acc"]) if history["test_acc"] else 0.0,
        "final_test_acc": history["test_acc"][-1] if history["test_acc"] else 0.0,
    }

    return model, history, metrics


def save_history_plot(all_histories, save_path):
    plt.figure(figsize=(10, 6))
    for label, history in all_histories.items():
        plt.plot(history["epochs"], history["test_acc"], label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Test Accuracy")
    plt.title("Hyperparameter Study - Test Accuracy Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def write_markdown_report(results_df, output_dir, configs, dataset_info):
    best_row = results_df.sort_values(by="f1", ascending=False).iloc[0]

    report = (
        "# Hyperparameter Study Report\n\n"
        "## Study setup\n"
        f"- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"- Attack type: {dataset_info['attack_type']}\n"
        f"- Samples per class: {dataset_info['max_samples_per_class']}\n"
        f"- Epochs per configuration: {dataset_info['epochs']}\n"
        f"- Batch size: {dataset_info['batch_size']}\n\n"
        "## Tested configurations\n"
        f"{json.dumps(configs, indent=2)}\n\n"
        "## Best configuration\n"
        f"- Width: {best_row['width']}\n"
        f"- Grid: {best_row['grid']}\n"
        f"- k: {best_row['k']}\n"
        f"- Accuracy: {best_row['accuracy']:.4f}\n"
        f"- Precision: {best_row['precision']:.4f}\n"
        f"- Recall: {best_row['recall']:.4f}\n"
        f"- F1-score: {best_row['f1']:.4f}\n"
    )

    with open(output_dir / "hyperparameter_report.md", "w", encoding="utf-8") as f:
        f.write(report)


def main():
    data_path = Path("data/Wednesday-workingHours.pcap_ISCX.csv")
    attack_type = "DoS Hulk"
    max_samples_per_class = 10000
    epochs = 20
    batch_size = 512

    output_dir = Path("experiment_data_hyper")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing reduced dataset...")
    dataset, scaler, features = prepare_dos_data(
        data_path,
        attack_type=attack_type,
        max_samples_per_class=max_samples_per_class,
    )
    input_dim = len(features)

    with open(output_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    with open(output_dir / "features.pkl", "wb") as f:
        pickle.dump(features, f)

    configs = [
        {"width": [input_dim, 16, 1], "grid": 5, "k": 3},
        {"width": [input_dim, 32, 16, 1], "grid": 5, "k": 3},
        {"width": [input_dim, 64, 32, 1], "grid": 5, "k": 3},
    ]

    results = []
    all_histories = {}

    for i, cfg in enumerate(configs, start=1):
        print(f"\n=== Running configuration {i}/{len(configs)}: {cfg} ===")

        model, history, metrics = train_custom_kan(
            dataset=dataset,
            input_dim=input_dim,
            width=cfg["width"],
            grid=cfg["grid"],
            k=cfg["k"],
            epochs=epochs,
            batch_size=batch_size,
            lr=0.001,
            seed=42,
        )

        run_name = f"run_{i}"
        run_dir = output_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "history": history,
                "architecture": {
                    "width": cfg["width"],
                    "grid": cfg["grid"],
                    "k": cfg["k"],
                },
                "timestamp": datetime.now().isoformat(),
            },
            run_dir / "trained_model.pt",
        )

        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        row = {
            "run": run_name,
            "width": str(cfg["width"]),
            "grid": cfg["grid"],
            "k": cfg["k"],
            **metrics,
        }
        results.append(row)
        all_histories[run_name] = history

    results_df = pd.DataFrame(results).sort_values(by="f1", ascending=False)
    results_df.to_csv(output_dir / "hyperparameter_results.csv", index=False)

    save_history_plot(
        all_histories=all_histories,
        save_path=output_dir / "hyperparameter_test_accuracy.png",
    )

    dataset_info = {
        "attack_type": attack_type,
        "max_samples_per_class": max_samples_per_class,
        "epochs": epochs,
        "batch_size": batch_size,
    }

    write_markdown_report(results_df, output_dir, configs, dataset_info)

    print("\nDone.")
    print(f"Results CSV: {output_dir / 'hyperparameter_results.csv'}")
    print(f"Comparison plot: {output_dir / 'hyperparameter_test_accuracy.png'}")
    print(f"Report: {output_dir / 'hyperparameter_report.md'}")


if __name__ == "__main__":
    main()