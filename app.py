import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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
from sklearn.inspection import DecisionBoundaryDisplay

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Smart ML Parameter Tuner – AutoML Mini Lab",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Smart ML Parameter Tuner – AutoML Mini Lab")
st.markdown("**Group 4 · Hybrid Optimization + AI (KNN Hyperparameter Tuning & Visual Analytics)**")
st.divider()

# ==========================================
# 1. DATASET LOADER
# ==========================================
@st.cache_data
def get_dataset(name):
    if name == "Iris":
        data = load_iris()
    elif name == "Wine":
        data = load_wine()
    elif name == "Breast Cancer":
        data = load_breast_cancer()
    else:
        data = load_digits()
    
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df["class_name"] = [str(data.target_names[i]) for i in data.target]
    return data, df

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Control Panel")

# Control 1: Dataset Selector
dataset_name = st.sidebar.selectbox(
    "1. Select Dataset",
    ["Iris", "Wine", "Breast Cancer", "Digits"],
    index=0
)

data, df = get_dataset(dataset_name)
X = data.data
y = data.target
feature_names = list(data.feature_names)
target_names = [str(t) for t in data.target_names]

st.sidebar.subheader("Manual KNN Settings")

# Control 2: K-value Slider
manual_k = st.sidebar.slider("2. Select K Value (Neighbors)", min_value=1, max_value=15, value=5, step=1)

# Control 3: Distance Metric Selector
manual_metric = st.sidebar.selectbox("3. Distance Metric", ["euclidean", "manhattan"], index=0)

# Weighting Selector
manual_weights = st.sidebar.selectbox("4. Weighting Method", ["uniform", "distance"], index=0)

st.sidebar.divider()
st.sidebar.subheader("AutoML Optimization")

# Control 4: Run Optimization Button
run_automl = st.sidebar.button("🚀 Run AutoML GridSearchCV", type="primary", use_container_width=True)

# Keep optimization state active once clicked
if "automl_ran" not in st.session_state:
    st.session_state.automl_ran = True

if run_automl:
    st.session_state.automl_ran = True
    st.sidebar.success("AutoML Optimization Completed!")

# ==========================================
# CORE ML PIPELINE & OPTIMIZATION (CACHED)
# ==========================================
@st.cache_resource
def train_and_optimize(dataset_key):
    raw_data, _ = get_dataset(dataset_key)
    X_d, y_d = raw_data.data, raw_data.target

    # Stratified 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_d, y_d, test_size=0.20, random_state=42, stratify=y_d
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline KNN (K=5, euclidean, uniform)
    base_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform"))
    ])
    base_pipe.fit(X_train, y_train)

    # GridSearchCV AutoML
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier())
    ])
    param_grid = {
        "knn__n_neighbors": list(range(1, 16)),
        "knn__metric": ["euclidean", "manhattan"],
        "knn__weights": ["uniform", "distance"]
    }
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, refit=True)
    grid.fit(X_train, y_train)

    return X_train, X_test, y_train, y_test, base_pipe, grid

X_train, X_test, y_train, y_test, base_model, grid_search = train_and_optimize(dataset_name)

# Train manual model based on sidebar controls
manual_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=manual_k, metric=manual_metric, weights=manual_weights))
])
manual_model.fit(X_train, y_train)
y_pred_manual = manual_model.predict(X_test)

# Optimized model predictions
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
y_pred_opt = best_model.predict(X_test)

# Baseline model predictions
y_pred_base = base_model.predict(X_test)

# ==========================================
# TOP KPI PERFORMANCE CARDS
# ==========================================
st.subheader(f"📊 Model Performance Summary — {dataset_name} Dataset")

col1, col2, col3, col4, col5 = st.columns(5)

opt_acc = accuracy_score(y_test, y_pred_opt) * 100
opt_prec = precision_score(y_test, y_pred_opt, average="macro", zero_division=0) * 100
opt_rec = recall_score(y_test, y_pred_opt, average="macro", zero_division=0) * 100
opt_f1 = f1_score(y_test, y_pred_opt, average="macro", zero_division=0) * 100
base_acc = accuracy_score(y_test, y_pred_base) * 100

col1.metric("Best AutoML K", f"K = {best_params['knn__n_neighbors']}", f"{best_params['knn__metric']} ({best_params['knn__weights']})")
col2.metric("5-Fold CV Accuracy", f"{grid_search.best_score_ * 100:.2f}%")
col3.metric("Test Set Accuracy", f"{opt_acc:.2f}%", f"{opt_acc - base_acc:+.2f}% vs Baseline")
col4.metric("Test Macro-F1", f"{opt_f1:.2f}%")
col5.metric("Manual Model (K=" + str(manual_k) + ")", f"{accuracy_score(y_test, y_pred_manual)*100:.2f}% Test Acc")

st.divider()

# ==========================================
# LIVE SAMPLE PREDICTION WIDGET
# ==========================================
with st.expander("🌸 Interactive Live Class Predictor (Test a Sample Input)", expanded=(dataset_name == "Iris")):
    st.write(f"Adjust feature values below to predict the **{dataset_name}** class using the **AutoML Optimized KNN Model**:")
    input_cols = st.columns(min(4, X.shape[1]))
    user_input = []
    for i in range(X.shape[1]):
        f_min = float(np.min(X[:, i]))
        f_max = float(np.max(X[:, i]))
        f_mean = float(np.mean(X[:, i]))
        if i < 4:
            if f_min < f_max:
                val = input_cols[i].slider(
                    f"{feature_names[i]}", f_min, f_max, f_mean
                )
            else:
                input_cols[i].caption(f"{feature_names[i]}: constant ({f_min:.1f})")
                val = f_mean
        else:
            val = f_mean
        user_input.append(val)
    
    pred_class_idx = best_model.predict([user_input])[0]
    pred_probs = best_model.predict_proba([user_input])[0]
    st.success(f"**Predicted Class:** `{target_names[pred_class_idx].upper()}` (Confidence: `{np.max(pred_probs)*100:.1f}%`)")
    
# ==========================================
# 5 MANDATORY DEMONSTRATION TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 1. Accuracy vs K",
    "🗺️ 2. Decision Boundary",
    "🎯 3. Confusion Matrix & Metrics",
    "⚖️ 4. Before vs After Optimization",
    "🌍 5. Multi-Dataset Comparison"
])

# ------------------------------------------
# TAB 1: ACCURACY VS K
# ------------------------------------------
with tab1:
    st.subheader("Demonstration 1: Cross-Validation Accuracy vs. K (K = 1 to 15)")
    cv_df = pd.DataFrame(grid_search.cv_results_)
    
    # Filter CV results for currently selected metric & weights
    filtered_cv = cv_df[
        (cv_df["param_knn__metric"] == manual_metric) &
        (cv_df["param_knn__weights"] == manual_weights)
    ].sort_values("param_knn__n_neighbors")

    k_list = filtered_cv["param_knn__n_neighbors"].astype(int).tolist()
    scores_list = (filtered_cv["mean_test_score"] * 100).tolist()

    best_idx_sub = int(np.argmax(scores_list))
    best_k_sub = k_list[best_idx_sub]
    best_score_sub = scores_list[best_idx_sub]

    fig1, ax1 = plt.subplots(figsize=(9, 4.2))
    ax1.plot(k_list, scores_list, marker="o", linewidth=2, label=f"5-Fold CV Accuracy ({manual_metric}, {manual_weights})")
    ax1.scatter(best_k_sub, best_score_sub, color="red", s=160, marker="*", zorder=5, label=f"Best K = {best_k_sub} ({best_score_sub:.2f}%)")
    ax1.axvline(manual_k, color="orange", linestyle="--", label=f"Current Slider K = {manual_k}")
    ax1.set_xticks(range(1, 16))
    ax1.set_xlabel("Number of Neighbors (K)")
    ax1.set_ylabel("5-Fold Stratified CV Accuracy (%)")
    ax1.set_title(f"KNN Accuracy vs K on {dataset_name} Dataset")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    st.pyplot(fig1)

    st.dataframe(
        filtered_cv[["param_knn__n_neighbors", "param_knn__metric", "param_knn__weights", "mean_test_score", "std_test_score", "rank_test_score"]].rename(
            columns={
                "param_knn__n_neighbors": "K",
                "param_knn__metric": "Metric",
                "param_knn__weights": "Weights",
                "mean_test_score": "Mean CV Accuracy",
                "std_test_score": "Std CV Accuracy",
                "rank_test_score": "Overall Grid Rank"
            }
        ),
        use_container_width=True
    )

# ------------------------------------------
# TAB 2: DECISION BOUNDARY VISUALIZATION
# ------------------------------------------
with tab2:
    st.subheader("Demonstration 2: 2D Decision Boundary Visualization")
    c_feat1, c_feat2 = st.columns(2)
    default_f1 = 2 if len(feature_names) >= 4 else 0
    default_f2 = 3 if len(feature_names) >= 4 else 1
    feat_x = c_feat1.selectbox("Select X-Axis Feature", feature_names, index=default_f1)
    feat_y = c_feat2.selectbox("Select Y-Axis Feature", feature_names, index=default_f2)

    idx_x = feature_names.index(feat_x)
    idx_y = feature_names.index(feat_y)

    if idx_x == idx_y:
        st.warning("Please select two different features to plot the 2D decision boundary.")
    else:
        X_2d = X[:, [idx_x, idx_y]]
        X_tr2d, X_te2d, y_tr2d, y_te2d = train_test_split(
            X_2d, y, test_size=0.20, random_state=42, stratify=y
        )

        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.8))
        models_2d = [
            (f"Manual KNN (K={manual_k}, {manual_metric}, {manual_weights})", manual_k, manual_metric, manual_weights),
            (f"AutoML Best KNN (K={best_params['knn__n_neighbors']}, {best_params['knn__metric']}, {best_params['knn__weights']})",
             best_params["knn__n_neighbors"], best_params["knn__metric"], best_params["knn__weights"])
        ]

        for ax, (title_label, k_val, m_val, w_val) in zip(axes2, models_2d):
            clf_2d = Pipeline([
                ("scaler", StandardScaler()),
                ("knn", KNeighborsClassifier(n_neighbors=k_val, metric=m_val, weights=w_val))
            ])
            clf_2d.fit(X_tr2d, y_tr2d)
            acc_2d = accuracy_score(y_te2d, clf_2d.predict(X_te2d)) * 100

            DecisionBoundaryDisplay.from_estimator(
                clf_2d, X_2d, response_method="predict",
                alpha=0.6, ax=ax, xlabel=feat_x, ylabel=feat_y
            )
            ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, edgecolor="k", s=35)
            ax.set_title(f"{title_label}\n2D Test Accuracy: {acc_2d:.2f}%")

        plt.tight_layout()
        st.pyplot(fig2)

# ------------------------------------------
# TAB 3: CONFUSION MATRIX & METRICS
# ------------------------------------------
with tab3:
    st.subheader("Demonstration 3: Confusion Matrix & Detailed Classification Metrics")
    cm_col1, cm_col2 = st.columns(2)

    with cm_col1:
        st.markdown(f"**AutoML Optimized Model** (`K={best_params['knn__n_neighbors']}`, `{best_params['knn__metric']}`, `{best_params['knn__weights']}`)")
        cm_opt = confusion_matrix(y_test, y_pred_opt)
        fig3, ax3 = plt.subplots(figsize=(5.5, 4.5))
        sns.heatmap(
            cm_opt, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names, cbar=False, ax=ax3
        )
        ax3.set_xlabel("Predicted Label")
        ax3.set_ylabel("True Label")
        ax3.set_title("Confusion Matrix (Held-Out Test Set)")
        st.pyplot(fig3)

    with cm_col2:
        st.markdown("**Per-Class Precision, Recall, and F1-Score Report**")
        report_dict = classification_report(y_test, y_pred_opt, target_names=target_names, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format("{:.4f}"), use_container_width=True)

        st.info(
            f"**Summary Metrics on Held-Out Test Set:**\n"
            f"- **Accuracy:** {opt_acc:.2f}%\n"
            f"- **Macro Precision:** {opt_prec:.2f}%\n"
            f"- **Macro Recall:** {opt_rec:.2f}%\n"
            f"- **Macro F1-Score:** {opt_f1:.2f}%"
        )

# ------------------------------------------
# TAB 4: BEFORE VS AFTER OPTIMIZATION
# ------------------------------------------
with tab4:
    st.subheader("Demonstration 4: Before vs. After Optimization Comparison")
    
    # Extract baseline CV score
    base_cv_row = cv_df[
        (cv_df["param_knn__n_neighbors"] == 5) &
        (cv_df["param_knn__metric"] == "euclidean") &
        (cv_df["param_knn__weights"] == "uniform")
    ]
    base_cv_score = base_cv_row["mean_test_score"].values[0] * 100
    base_f1_score = f1_score(y_test, y_pred_base, average="macro") * 100

    comp_df = pd.DataFrame({
        "Model": ["Baseline KNN (Default)", "AutoML Optimized KNN"],
        "Hyperparameters": [
            "K=5, euclidean, uniform",
            f"K={best_params['knn__n_neighbors']}, {best_params['knn__metric']}, {best_params['knn__weights']}"
        ],
        "5-Fold CV Accuracy (%)": [round(base_cv_score, 2), round(grid_search.best_score_ * 100, 2)],
        "Test Set Accuracy (%)": [round(base_acc, 2), round(opt_acc, 2)],
        "Test Macro-F1 (%)": [round(base_f1_score, 2), round(opt_f1, 2)]
    })
    st.table(comp_df)

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    labels4 = ["5-Fold CV Accuracy (%)", "Test Accuracy (%)", "Test Macro-F1 (%)"]
    b_vals = [base_cv_score, base_acc, base_f1_score]
    o_vals = [grid_search.best_score_ * 100, opt_acc, opt_f1]
    x4 = np.arange(len(labels4))
    w4 = 0.35

    r1 = ax4.bar(x4 - w4/2, b_vals, w4, label="Baseline KNN (K=5)", color="#6c757d")
    r2 = ax4.bar(x4 + w4/2, o_vals, w4, label=f"Optimized KNN (K={best_params['knn__n_neighbors']})", color="#2b8cbe")
    ax4.set_xticks(x4)
    ax4.set_xticklabels(labels4)
    ax4.set_ylim(max(0, min(b_vals + o_vals) - 8), 102)
    ax4.set_ylabel("Percentage (%)")
    ax4.set_title(f"Before vs After Optimization ({dataset_name} Dataset)")
    ax4.legend()
    ax4.grid(axis="y", linestyle="--", alpha=0.6)

    for rect in r1 + r2:
        h = rect.get_height()
        ax4.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    st.pyplot(fig4)

# ------------------------------------------
# TAB 5: MULTI-DATASET COMPARISON
# ------------------------------------------
with tab5:
    st.subheader("Demonstration 5: Multi-Dataset AutoML Benchmark")
    st.write("Comparing Baseline KNN vs. AutoML GridSearchCV across **Iris, Wine, Breast Cancer, and Digits** datasets:")

    @st.cache_data
    def compute_multi_dataset_summary():
        rows = []
        for d_name in ["Iris", "Wine", "Breast Cancer", "Digits"]:
            raw_d, _ = get_dataset(d_name)
            _, X_te, _, y_te, b_mod, g_mod = train_and_optimize(d_name)
            b_pred = b_mod.predict(X_te)
            o_pred = g_mod.best_estimator_.predict(X_te)
            rows.append({
                "Dataset": d_name,
                "Samples": len(raw_d.data),
                "Features": raw_d.data.shape[1],
                "Classes": len(raw_d.target_names),
                "Best K": g_mod.best_params_["knn__n_neighbors"],
                "Best Metric": g_mod.best_params_["knn__metric"],
                "Best Weights": g_mod.best_params_["knn__weights"],
                "5-Fold CV Acc (%)": round(g_mod.best_score_ * 100, 2),
                "Baseline Test Acc (%)": round(accuracy_score(y_te, b_pred) * 100, 2),
                "Optimized Test Acc (%)": round(accuracy_score(y_te, o_pred) * 100, 2),
                "Optimized Macro-F1 (%)": round(f1_score(y_te, o_pred, average="macro") * 100, 2)
            })
        return pd.DataFrame(rows)

    multi_summary_df = compute_multi_dataset_summary()
    st.dataframe(multi_summary_df, use_container_width=True)

    fig5, ax5 = plt.subplots(figsize=(9, 4.2))
    x5 = np.arange(len(multi_summary_df))
    w5 = 0.35
    rb1 = ax5.bar(x5 - w5/2, multi_summary_df["Baseline Test Acc (%)"], w5, label="Baseline KNN (K=5)", color="#adb5bd")
    rb2 = ax5.bar(x5 + w5/2, multi_summary_df["Optimized Test Acc (%)"], w5, label="Optimized KNN (AutoML)", color="#198754")
    ax5.set_xticks(x5)
    ax5.set_xticklabels([f"{r['Dataset']}\n(Best K={r['Best K']}, {r['Best Metric']})" for _, r in multi_summary_df.iterrows()])
    ax5.set_ylim(85, 103)
    ax5.set_ylabel("Test Accuracy (%)")
    ax5.set_title("Multi-Dataset Benchmark: Baseline vs AutoML Optimized KNN")
    ax5.legend()
    ax5.grid(axis="y", linestyle="--", alpha=0.6)

    for rect in rb1 + rb2:
        h = rect.get_height()
        ax5.annotate(f"{h:.2f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
    st.pyplot(fig5)