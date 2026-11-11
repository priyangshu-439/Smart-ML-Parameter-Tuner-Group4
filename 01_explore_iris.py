
import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()

# Store features and target
X = iris.data
y = iris.target

# Convert into a DataFrame
df = pd.DataFrame(
    X,
    columns=iris.feature_names
)

# Add flower species
df["species"] = [
    iris.target_names[i] for i in y
]

# Display dataset information
print("SMART ML PARAMETER TUNER")
print("========================")

print("\nFirst 5 records:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nFlower categories:")
print(df["species"].unique())

print("\nDataset information:")
print(df.info())

print("\nStatistical description:")
print(df.describe())
