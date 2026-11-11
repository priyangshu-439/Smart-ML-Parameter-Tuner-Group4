import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay

print("GENERATING MANDATORY VISUAL DEMONSTRATIONS")
print("==========================================")

# Load Iris and split exactly as in Step 4
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
    "knn__n_neighbors": range(1, 16),
    "knn__metric": ["euclidean", "manhattan"],
    "knn__weights": ["uniform", "distance"]
}

# ==========================================================
# DEMO 1: CONFUSION MATRIX HEATMAP (confusion_matrix.png)
# ==========================================================
print("\n1. Generating Confusion Matrix Heatmap...")
pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,
    cbar=False, annot_kws={"size": 14, "weight": "bold"}
)
plt.title("Confusion Matrix (Optimized KNN: K=3, Euclidean, Uniform)")
plt.xlabel("Predicted Species")
plt.ylabel("Actual Species")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()
print("   -> Saved: confusion_matrix.png")

# ==========================================================
# DEMO 2: DECISION BOUNDARY VISUALIZATION (decision_boundary.png)
# ==========================================================
print("2. Generating 2D Decision Boundary Comparison...")
# Use Petal Length (index 2) and Petal Width (index 3) for 2D visualization
X_2d = X[:, [2, 3]]
X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size=0.20, random_state=42, stratify=y
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
k_values_to_compare = [1, grid.best_params_["knn__n_neighbors"], 15]

for ax, k in zip(axes, k_values_to_compare):
    clf_2d = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="uniform"))
    ])
    clf_2d.fit(X_train_2d, y_train_2d)
    acc_2d = accuracy_score(y_test_2d, clf_2d.predict(X_test_2d)) * 100

    DecisionBoundaryDisplay.from_estimator(
        clf_2d, X_2d, response_method="predict",
        cmap="Pastel1", alpha=0.8, ax=ax,
        xlabel=iris.feature_names[2], ylabel=iris.feature_names[3]
    )
    scatter = ax.scatter(
        X_2d[:, 0], X_2d[:, 1], c=y, cmap="Set1", edgecolor="k", s=40
    )
    title_tag = " (Optimal)" if k == grid.best_params_["knn__n_neighbors"] else ""
    ax.set_title(f"K = {k}{title_tag}\n2D Test Acc: {acc_2d:.1f}%")

plt.suptitle("KNN Decision Boundaries on Iris (Petal Length vs Petal Width)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("decision_boundary.png", dpi=300, bbox_inches="tight")
plt.close()
print("   -> Saved: decision_boundary.png")

# ==========================================================
# DEMO 3: BEFORE VS AFTER OPTIMIZATION (before_vs_after.png)
# ==========================================================
print("3. Generating Before vs After Optimization Chart...")
# Baseline KNN (Default K=5, uniform, euclidean)
base_pipe = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
base_pipe.fit(X_train, y_train)
base_test_acc = accuracy_score(y_test, base_pipe.predict(X_test)) * 100
base_test_f1 = f1_score(y_test, base_pipe.predict(X_test), average="macro") * 100

# Extract baseline 5-fold CV score from GridSearch results
cv_df = pd.DataFrame(grid.cv_results_)
base_cv_row = cv_df[
    (cv_df["param_knn__n_neighbors"] == 5) &
    (cv_df["param_knn__metric"] == "euclidean") &
    (cv_df["param_knn__weights"] == "uniform")
]
base_cv_acc = base_cv_row["mean_test_score"].values[0] * 100

opt_cv_acc = grid.best_score_ * 100
opt_test_acc = accuracy_score(y_test, y_pred) * 100
opt_test_f1 = f1_score(y_test, y_pred, average="macro") * 100

metrics_labels = ["5-Fold CV Accuracy (%)", "Test Set Accuracy (%)", "Test Macro-F1 (%)"]
before_scores = [base_cv_acc, base_test_acc, base_test_f1]
after_scores = [opt_cv_acc, opt_test_acc, opt_test_f1]

x_pos = np.arange(len(metrics_labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
rects1 = ax.bar(x_pos - width/2, before_scores, width, label="Baseline KNN (K=5)", color="#6c757d")
rects2 = ax.bar(x_pos + width/2, after_scores, width, label=f"Optimized KNN (K={grid.best_params_['knn__n_neighbors']})", color="#2b8cbe")

ax.set_ylabel("Score (%)")
ax.set_title("Before vs After Hyperparameter Optimization (Iris Dataset)")
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics_labels)
ax.set_ylim(85, 102)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.6)

for rect in rects1 + rects2:
    h = rect.get_height()
    ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("before_vs_after.png", dpi=300)
plt.close()
print("   -> Saved: before_vs_after.png")

# ==========================================================
# DEMO 4: MULTI-DATASET COMPARISON (multi_dataset_comparison.png)
# ==========================================================
print("4. Running Multi-Dataset AutoML Comparison (Iris, Wine, Breast Cancer, Digits)...")
datasets = {
    "Iris": load_iris(),
    "Wine": load_wine(),
    "Breast Cancer": load_breast_cancer(),
    "Digits": load_digits()
}

multi_results = []

for name, data in datasets.items():
    Xd_train, Xd_test, yd_train, yd_test = train_test_split(
        data.data, data.target, test_size=0.20, random_state=42, stratify=data.target
    )
    
    # Baseline
    b_model = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())])
    b_model.fit(Xd_train, yd_train)
    b_acc = accuracy_score(yd_test, b_model.predict(Xd_test)) * 100
    b_f1 = f1_score(yd_test, b_model.predict(Xd_test), average="macro") * 100
    
    # GridSearch Optimized
    g_search = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier())]),
        param_grid, cv=cv, scoring="accuracy", n_jobs=-1
    )
    g_search.fit(Xd_train, yd_train)
    o_pred = g_search.best_estimator_.predict(Xd_test)
    o_acc = accuracy_score(yd_test, o_pred) * 100
    o_f1 = f1_score(yd_test, o_pred, average="macro") * 100
    
    multi_results.append({
        "Dataset": name,
        "Samples": len(data.data),
        "Features": data.data.shape[1],
        "Classes": len(data.target_names),
        "Best K": g_search.best_params_["knn__n_neighbors"],
        "Best Metric": g_search.best_params_["knn__metric"],
        "Best Weights": g_search.best_params_["knn__weights"],
        "CV Accuracy (%)": round(g_search.best_score_ * 100, 2),
        "Baseline Test Acc (%)": round(b_acc, 2),
        "Optimized Test Acc (%)": round(o_acc, 2),
        "Optimized Macro-F1 (%)": round(o_f1, 2)
    })

multi_df = pd.DataFrame(multi_results)
multi_df.to_csv("multi_dataset_results.csv", index=False)

print("\nMULTI-DATASET COMPARISON TABLE:")
print(multi_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5))
x_idx = np.arange(len(multi_df))
b1 = ax.bar(x_idx - width/2, multi_df["Baseline Test Acc (%)"], width, label="Baseline KNN (K=5)", color="#adb5bd")
b2 = ax.bar(x_idx + width/2, multi_df["Optimized Test Acc (%)"], width, label="Optimized KNN (AutoML)", color="#198754")

ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Multi-Dataset Performance: Baseline vs AutoML Optimized KNN")
ax.set_xticks(x_idx)
ax.set_xticklabels([f"{r['Dataset']}\n(Best K={r['Best K']}, {r['Best Metric']})" for _, r in multi_df.iterrows()])
ax.set_ylim(85, 103)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.6)

for rect in b1 + b2:
    h = rect.get_height()
    ax.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("multi_dataset_comparison.png", dpi=300)
plt.close()
print("\n   -> Saved: multi_dataset_comparison.png & multi_dataset_results.csv")
print("\nAll 5 mandatory visual demonstrations are now saved in your project folder!")