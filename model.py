import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

SUPPORTED_DATASETS = ["Iris", "Wine", "Breast Cancer", "Digits", "Titanic"]


def load_dataset_by_name(name: str):
    """
    Loads one of the 5 faculty-recommended datasets (Iris, Wine, Breast Cancer, Digits, Titanic)
    and returns (X, y, feature_names, target_names, df).
    """
    if name == "Iris":
        data = load_iris()
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        target_names = [str(t) for t in data.target_names]

    elif name == "Wine":
        data = load_wine()
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        target_names = [str(t) for t in data.target_names]

    elif name == "Breast Cancer":
        data = load_breast_cancer()
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        target_names = [str(t) for t in data.target_names]

    elif name == "Digits":
        data = load_digits()
        X, y = data.data, data.target
        feature_names = list(data.feature_names)
        target_names = [f"Digit {t}" for t in data.target_names]

    elif name == "Titanic":
        # Load built-in Titanic dataset and preprocess numerical/encoded features
        raw = sns.load_dataset("titanic")
        cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "survived"]
        clean = raw[cols].dropna().copy()
        clean["sex"] = clean["sex"].map({"male": 0, "female": 1}).astype(float)

        feature_names = ["Passenger Class (pclass)", "Sex (0=Male, 1=Female)", "Age", "Siblings/Spouses", "Parents/Children", "Fare"]
        X = clean[["pclass", "sex", "age", "sibsp", "parch", "fare"]].values.astype(float)
        y = clean["survived"].values.astype(int)
        target_names = ["Did Not Survive (0)", "Survived (1)"]
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y
    df["class_name"] = [target_names[i] for i in y]
    return X, y, feature_names, target_names, df


def run_automl_pipeline(name: str):
    """
    Performs an 80/20 stratified train-test split, trains a Baseline KNN (K=5, euclidean, uniform),
    and executes 5-fold Stratified GridSearchCV (K=1..15, euclidean/manhattan, uniform/distance)
    strictly on the training split to prevent data leakage.
    """
    X, y, feature_names, target_names, df = load_dataset_by_name(name)

    # 80/20 Stratified Train/Test Split (Test set kept strictly untouched)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Baseline KNN Model (Default K=5, Euclidean, Uniform)
    baseline_model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform"))
    ])
    baseline_model.fit(X_train, y_train)

    # 2. Automated Hyperparameter Optimization via GridSearchCV
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])

    param_grid = {
        "knn__n_neighbors": list(range(1, 16)),
        "knn__metric": ["euclidean", "manhattan"],
        "knn__weights": ["uniform", "distance"]
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="accuracy",
        n_jobs=-1,
        refit=True
    )
    grid_search.fit(X_train, y_train)

    return {
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "target_names": target_names,
        "df": df,
        "baseline_model": baseline_model,
        "grid_search": grid_search,
        "best_model": grid_search.best_estimator_,
        "best_params": grid_search.best_params_
    }


def evaluate_model(model, X_test, y_test, target_names):
    """Calculates Accuracy, Macro Precision, Macro Recall, Macro F1, Confusion Matrix, and Report."""
    y_pred = model.predict(X_test)
    return {
        "y_pred": y_pred,
        "accuracy": accuracy_score(y_test, y_pred) * 100,
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0) * 100,
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0) * 100,
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0) * 100,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report_dict": classification_report(y_test, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    }
