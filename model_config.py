
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def get_model_config(algorithm, random_state=42):
    """
    Returns an ML pipeline and its hyperparameter search space.
    """

    if algorithm == "KNN":

        model = KNeighborsClassifier()

        param_grid = {
            "classifier__n_neighbors": list(range(1, 16)),
            "classifier__metric": ["euclidean", "manhattan"],
            "classifier__weights": ["uniform", "distance"]
        }

    elif algorithm == "Decision Tree":

        model = DecisionTreeClassifier(
            random_state=random_state
        )

        param_grid = {
            "classifier__criterion": ["gini", "entropy"],
            "classifier__max_depth": [None, 3, 5, 10, 15],
            "classifier__min_samples_split": [2, 5, 10]
        }

    elif algorithm == "Random Forest":

        model = RandomForestClassifier(
            random_state=random_state,
            n_jobs=-1
        )

        param_grid = {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [None, 5, 10, 15],
            "classifier__min_samples_split": [2, 5],
            "classifier__max_features": ["sqrt", "log2"]
        }

    elif algorithm == "SVM":

        model = SVC()

        param_grid = {
            "classifier__C": [0.1, 1, 10, 100],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__gamma": ["scale", "auto"]
        }

    else:
        raise ValueError(
            f"Unsupported algorithm: {algorithm}"
        )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", model)
    ])

    return pipeline, param_grid
