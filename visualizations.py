import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.inspection import DecisionBoundaryDisplay


def plot_accuracy_vs_k(cv_df: pd.DataFrame, metric: str, weights: str, manual_k: int, dataset_name: str):
    """Demonstration 1: Cross-Validation Accuracy vs K (K = 1 to 15)."""
    filtered = cv_df[
        (cv_df["param_knn__metric"] == metric) &
        (cv_df["param_knn__weights"] == weights)
    ].sort_values("param_knn__n_neighbors")

    k_list = filtered["param_knn__n_neighbors"].astype(int).tolist()
    scores = (filtered["mean_test_score"] * 100).tolist()

    best_idx = int(np.argmax(scores))
    best_k = k_list[best_idx]
    best_score = scores[best_idx]

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(k_list, scores, marker="o", linewidth=2.2, color="#1f77b4", label=f"5-Fold CV Accuracy ({metric}, {weights})")
    ax.scatter(best_k, best_score, color="#d62728", s=170, marker="*", zorder=5, label=f"Best K = {best_k} ({best_score:.2f}%)")
    ax.axvline(manual_k, color="#ff7f0e", linestyle="--", linewidth=1.5, label=f"Selected Slider K = {manual_k}")

    ax.set_xticks(range(1, 16))
    ax.set_xlabel("Number of Neighbors (K)")
    ax.set_ylabel("5-Fold Stratified CV Accuracy (%)")
    ax.set_title(f"KNN Validation Accuracy vs K Value — {dataset_name} Dataset")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    return fig, filtered


def plot_decision_boundaries(X, y, idx_x: int, idx_y: int, feat_x: str, feat_y: str, manual_cfg: dict, best_cfg: dict):
    """Demonstration 2: 2D Decision Boundary Comparison (Manual K vs Optimal AutoML K)."""
    X_2d = X[:, [idx_x, idx_y]]
    X_tr2d, X_te2d, y_tr2d, y_te2d = train_test_split(
        X_2d, y, test_size=0.20, random_state=42, stratify=y
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    configs = [
        (f"Manual KNN (K={manual_cfg['k']}, {manual_cfg['metric']}, {manual_cfg['weights']})", manual_cfg),
        (f"AutoML Optimal KNN (K={best_cfg['k']}, {best_cfg['metric']}, {best_cfg['weights']})", best_cfg)
    ]

    for ax, (title_text, cfg) in zip(axes, configs):
        clf_2d = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=cfg["k"], metric=cfg["metric"], weights=cfg["weights"]))
        ])
        clf_2d.fit(X_tr2d, y_tr2d)
        acc_2d = accuracy_score(y_te2d, clf_2d.predict(X_te2d)) * 100

        DecisionBoundaryDisplay.from_estimator(
            clf_2d, X_2d, response_method="predict",
            alpha=0.55, ax=ax, xlabel=feat_x, ylabel=feat_y
        )
        ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, edgecolor="k", s=35)
        ax.set_title(f"{title_text}\n2D Held-Out Test Accuracy: {acc_2d:.2f}%")

    plt.tight_layout()
    return fig


def plot_confusion_matrix_heatmap(cm, target_names, title: str):
    """Demonstration 3: Confusion Matrix Heatmap."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=target_names, yticklabels=target_names,
        cbar=False, ax=ax, annot_kws={"size": 12, "weight": "bold"}
    )
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_before_vs_after(base_scores: list, opt_scores: list, best_k: int, dataset_name: str):
    """Demonstration 4: Before vs After Hyperparameter Optimization Bar Chart."""
    labels = ["5-Fold CV Accuracy (%)", "Test Set Accuracy (%)", "Test Macro-F1 (%)"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.2))
    r1 = ax.bar(x - width / 2, base_scores, width, label="Before: Baseline KNN (K=5)", color="#6c757d")
    r2 = ax.bar(x + width / 2, opt_scores, width, label=f"After: Optimized KNN (K={best_k})", color="#2b8cbe")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(max(0, min(base_scores + opt_scores) - 8), 103)
    ax.set_ylabel("Score (%)")
    ax.set_title(f"Before vs After Hyperparameter Optimization — {dataset_name} Dataset")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for rect in r1 + r2:
        h = rect.get_height()
        ax.annotate(
            f"{h:.2f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
    plt.tight_layout()
    return fig


def plot_multi_dataset_comparison(multi_df: pd.DataFrame):
    """Demonstration 5: Multi-Dataset Comparison Bar Chart."""
    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    x = np.arange(len(multi_df))
    width = 0.35

    r1 = ax.bar(x - width / 2, multi_df["Baseline Test Acc (%)"], width, label="Baseline KNN (K=5)", color="#adb5bd")
    r2 = ax.bar(x + width / 2, multi_df["Optimized Test Acc (%)"], width, label="AutoML Optimized KNN", color="#198754")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['Dataset']}\n(Best K={r['Best K']}, {r['Best Metric']})" for _, r in multi_df.iterrows()])
    ax.set_ylim(max(0, multi_df["Baseline Test Acc (%)"].min() - 8), 103)
    ax.set_ylabel("Held-Out Test Accuracy (%)")
    ax.set_title("Multi-Dataset Benchmark: Baseline vs AutoML Optimized KNN")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    for rect in r1 + r2:
        h = rect.get_height()
        ax.annotate(
            f"{h:.2f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )
    plt.tight_layout()
    return fig
