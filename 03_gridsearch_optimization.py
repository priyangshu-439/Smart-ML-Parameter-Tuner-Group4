
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================
# 1. LOAD DATASET
# =====================================

iris = load_iris()

X = iris.data
y = iris.target

print("SMART ML PARAMETER TUNER")
print("========================")

# =====================================
# 2. TRAIN-TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# =====================================
# 3. CREATE MACHINE LEARNING PIPELINE
# =====================================

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier())
])

# =====================================
# 4. DEFINE PARAMETERS
# =====================================

parameters = {
    "knn__n_neighbors": range(1, 16),
    "knn__metric": ["euclidean", "manhattan"],
    "knn__weights": ["uniform", "distance"]
}

# =====================================
# 5. STRATIFIED CROSS VALIDATION
# =====================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# =====================================
# 6. GRID SEARCH OPTIMIZATION
# =====================================

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=parameters,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
    refit=True
)

print("\nRunning GridSearchCV...")
grid.fit(X_train, y_train)

# =====================================
# 7. DISPLAY BEST PARAMETERS
# =====================================

print("\nOPTIMIZATION RESULTS")
print("====================")

print("\nBest parameters:")
print(grid.best_params_)

print("\nBest cross-validation accuracy:")
print(f"{grid.best_score_ * 100:.2f}%")

# =====================================
# 8. FINAL TEST EVALUATION
# =====================================

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

print("\nFINAL TEST RESULTS")
print("==================")

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Macro F1-score: {macro_f1:.4f}")

# =====================================
# 9. CONFUSION MATRIX
# =====================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# =====================================
# 10. CLASSIFICATION REPORT
# =====================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

# =====================================
# 11. SAVE RESULTS
# =====================================

results = pd.DataFrame(grid.cv_results_)

results.to_csv(
    "gridsearch_results.csv",
    index=False
)

print("\nGrid search results saved successfully!")

print("\nBest model is ready!")
