
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# 2. Split into training and untouched test data
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 3. Split training data into training and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    random_state=42,
    stratify=y_train_full
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

# 4. Test K values from 1 to 15
results = []

for k in range(1, 16):

    # Create a Machine Learning pipeline
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Predict validation data
    predictions = model.predict(X_val)

    # Calculate accuracy
    accuracy = accuracy_score(y_val, predictions)

    results.append({
        "K": k,
        "Accuracy": accuracy * 100
    })

    print(f"K = {k:2d} | Accuracy = {accuracy * 100:.2f}%")

# 5. Convert results into DataFrame
results_df = pd.DataFrame(results)

# 6. Find best K
best_index = results_df["Accuracy"].idxmax()

best_k = results_df.loc[best_index, "K"]
best_accuracy = results_df.loc[best_index, "Accuracy"]

print("\nBest K:", best_k)
print("Best validation accuracy:", best_accuracy, "%")

# 7. Plot accuracy versus K
plt.figure(figsize=(10, 5))

plt.plot(
    results_df["K"],
    results_df["Accuracy"],
    marker="o",
    linewidth=2
)

plt.scatter(
    best_k,
    best_accuracy,
    s=150,
    marker="*",
    label=f"Best K = {best_k}"
)

plt.xlabel("K Value")
plt.ylabel("Validation Accuracy (%)")
plt.title("KNN Accuracy vs K Value")

plt.xticks(range(1, 16))
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("accuracy_vs_k.png", dpi=300)
plt.show()
