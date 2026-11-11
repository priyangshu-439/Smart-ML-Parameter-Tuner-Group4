
from model_config import get_model_config

algorithms = [
    "KNN",
    "Decision Tree",
    "Random Forest",
    "SVM"
]

for algorithm in algorithms:

    pipeline, param_grid = get_model_config(algorithm)

    print("\nAlgorithm:", algorithm)
    print("Pipeline:", pipeline)
    print("Parameters:", param_grid.keys())

print("\nAll algorithm configurations loaded successfully!")
