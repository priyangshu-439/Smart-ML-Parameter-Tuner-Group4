import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

from model import (
    SUPPORTED_DATASETS,
    run_automl_pipeline,
    evaluate_model
)
from visualizations import (
    plot_accuracy_vs_k,
    plot_decision_boundaries,
    plot_confusion_matrix_heatmap,
    plot_before_vs_after,
    plot_multi_dataset_comparison
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Smart ML Parameter Tuner – AutoML Mini Lab",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Smart ML Parameter Tuner – AutoML Mini Lab")
st.markdown(
    "**Group 4 · Hybrid Optimization + AI** | Automated KNN Hyperparameter Optimization & Visual Analytics Lab"
)
st.divider()

# ==========================================
# SIDEBAR INTERACTIVE CONTROLS (PART A)
# ==========================================
st.sidebar.header("⚙️ Control Panel")

# 1. Dataset selector
dataset_name = st.sidebar.selectbox(
    "1. Select Dataset",
    SUPPORTED_DATASETS,
    index=0
)

st.sidebar.subheader("Manual KNN Settings")

# 2. K-value slider
manual_k = st.sidebar.slider(
    "2. Select K Value (Neighbors)",
    min_value=1, max_value=15, value=5, step=1
)

# 3. Distance metric selector
manual_metric = st.sidebar.selectbox(
    "3. Distance Metric",
    ["euclidean", "manhattan"],
    index=0
)

# Weighting method selector
manual_weights = st.sidebar.selectbox(
    "4. Weighting Method",
    ["uniform", "distance"],
    index=0
)

st.sidebar.divider()
st.sidebar.subheader("AutoML Optimization")

# 4. Run Optimization button
if st.sidebar.button("🚀 Run Optimization", type="primary", use_container_width=True):
    st.sidebar.success(f"GridSearchCV completed on {dataset_name}!")

# ==========================================
# CACHED AUTOML EXECUTION
# ==========================================
@st.cache_resource
def get_cached_automl(ds_name: str):
    return run_automl_pipeline(ds_name)

bundle = get_cached_automl(dataset_name)

X = bundle["X"]
y = bundle["y"]
X_train, X_test = bundle["X_train"], bundle["X_test"]
y_train, y_test = bundle["y_train"], bundle["y_test"]
feature_names = bundle["feature_names"]
target_names = bundle["target_names"]
grid_search = bundle["grid_search"]
best_model = bundle["best_model"]
best_params = bundle["best_params"]
baseline_model = bundle["baseline_model"]

# Evaluate Baseline, Optimized, and Manual models on the untouched held-out test set
base_eval = evaluate_model(baseline_model, X_test, y_test, target_names)
opt_eval = evaluate_model(best_model, X_test, y_test, target_names)

manual_model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=manual_k, metric=manual_metric, weights=manual_weights))
])
manual_model.fit(X_train, y_train)
manual_eval = evaluate_model(manual_model, X_test, y_test, target_names)

# ==========================================
# 6. MODEL PERFORMANCE CARDS
# ==========================================
st.subheader(f"📊 Model Performance Cards — {dataset_name} Dataset")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(
    "Best AutoML K",
    f"K = {best_params['knn__n_neighbors']}",
    f"{best_params['knn__metric']} ({best_params['knn__weights']})"
)
c2.metric("5-Fold CV Accuracy", f"{grid_search.best_score_ * 100:.2f}%")
c3.metric(
    "Optimized Test Accuracy",
    f"{opt_eval['accuracy']:.2f}%",
    f"{opt_eval['accuracy'] - base_eval['accuracy']:+.2f}% vs Baseline"
)
c4.metric("Optimized Macro-F1", f"{opt_eval['macro_f1']:.2f}%")
c5.metric(f"Manual KNN (K={manual_k})", f"{manual_eval['accuracy']:.2f}% Test Acc")

st.divider()

# ==========================================
# 5. PREDICTED FLOWER / SAMPLE CLASS WIDGET
# ==========================================
with st.expander("🌸 Interactive Sample Class Predictor (Live Prediction)", expanded=(dataset_name == "Iris")):
    st.write(f"Adjust feature measurements below to see the **Predicted Class** from the **Optimized KNN Model**:")
    input_cols = st.columns(min(4, X.shape[1]))
    user_input = []
    for i in range(X.shape[1]):
        f_min = float(np.min(X[:, i]))
        f_max = float(np.max(X[:, i]))
        f_mean = float(np.mean(X[:, i]))
        if i < 4:
            if f_min < f_max:
                val = input_cols[i].slider(f"{feature_names[i]}", f_min, f_max, f_mean)
            else:
                input_cols[i].caption(f"{feature_names[i]}: constant ({f_min:.1f})")
                val = f_mean
        else:
            val = f_mean
        user_input.append(val)

    pred_idx = best_model.predict([user_input])[0]
    pred_probs = best_model.predict_proba([user_input])[0]
    st.success(
        f"**Predicted Class:** `{target_names[pred_idx].upper()}` "
        f"(Model Confidence: `{np.max(pred_probs) * 100:.1f}%`)"
    )

# ==========================================
# PART B: FIVE MANDATORY DEMONSTRATIONS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 1. Accuracy vs K",
    "🗺️ 2. Decision Boundary",
    "🎯 3. Confusion Matrix & Metrics",
    "⚖️ 4. Before vs After Optimization",
    "🌍 5. Multi-Dataset Comparison"
])

cv_df = pd.DataFrame(grid_search.cv_results_)

# ------------------------------------------
# TAB 1: ACCURACY VERSUS K
# ------------------------------------------
with tab1:
    st.subheader("1. Accuracy vs. K (K = 1 to 15)")
    fig1, filtered_cv = plot_accuracy_vs_k(cv_df, manual_metric, manual_weights, manual_k, dataset_name)
    st.pyplot(fig1)

    st.dataframe(
        filtered_cv[[
            "param_knn__n_neighbors", "param_knn__metric", "param_knn__weights",
            "mean_test_score", "std_test_score", "rank_test_score"
        ]].rename(columns={
            "param_knn__n_neighbors": "K Value",
            "param_knn__metric": "Distance Metric",
            "param_knn__weights": "Weighting",
            "mean_test_score": "5-Fold CV Accuracy",
            "std_test_score": "Fold Std Dev",
            "rank_test_score": "Grid Rank"
        }),
        use_container_width=True
    )

# ------------------------------------------
# TAB 2: DECISION BOUNDARY VISUALIZATION
# ------------------------------------------
with tab2:
    st.subheader("2. 2D Decision Boundary Visualization")
    col_x, col_y = st.columns(2)
    def_x = 2 if len(feature_names) >= 4 else 0
    def_y = 3 if len(feature_names) >= 4 else 1
    feat_x = col_x.selectbox("Select X-Axis Feature", feature_names, index=def_x)
    feat_y = col_y.selectbox("Select Y-Axis Feature", feature_names, index=def_y)

    idx_x = feature_names.index(feat_x)
    idx_y = feature_names.index(feat_y)

    if idx_x == idx_y:
        st.warning("Please choose two distinct features to plot the 2D decision regions.")
    else:
        fig2 = plot_decision_boundaries(
            X, y, idx_x, idx_y, feat_x, feat_y,
            manual_cfg={"k": manual_k, "metric": manual_metric, "weights": manual_weights},
            best_cfg={
                "k": best_params["knn__n_neighbors"],
                "metric": best_params["knn__metric"],
                "weights": best_params["knn__weights"]
            }
        )
        st.pyplot(fig2)

# ------------------------------------------
# TAB 3: CONFUSION MATRIX & PERFORMANCE METRICS
# ------------------------------------------
with tab3:
    st.subheader("3. Confusion Matrix and Performance Metrics (Accuracy, Precision, Recall, Macro-F1)")
    cm_left, cm_right = st.columns(2)

    with cm_left:
        fig3 = plot_confusion_matrix_heatmap(
            opt_eval["confusion_matrix"],
            target_names,
            f"Optimized KNN (K={best_params['knn__n_neighbors']}, {best_params['knn__metric']}, {best_params['knn__weights']})"
        )
        st.pyplot(fig3)

    with cm_right:
        st.markdown("**Classification Report (Held-Out Test Set)**")
        rep_df = pd.DataFrame(opt_eval["report_dict"]).transpose()
        st.dataframe(rep_df.style.format("{:.4f}"), use_container_width=True)

        st.info(
            f"**Four Mandatory Evaluation Metrics (Test Set):**\n"
            f"- **Accuracy:** `{opt_eval['accuracy']:.2f}%`\n"
            f"- **Macro Precision:** `{opt_eval['precision']:.2f}%`\n"
            f"- **Macro Recall:** `{opt_eval['recall']:.2f}%`\n"
            f"- **Macro F1-Score:** `{opt_eval['macro_f1']:.2f}%`"
        )

# ------------------------------------------
# TAB 4: BEFORE VS AFTER OPTIMIZATION
# ------------------------------------------
with tab4:
    st.subheader("4. Before vs. After Optimization (Model Comparison View)")
    base_cv_row = cv_df[
        (cv_df["param_knn__n_neighbors"] == 5) &
        (cv_df["param_knn__metric"] == "euclidean") &
        (cv_df["param_knn__weights"] == "uniform")
    ]
    base_cv_acc = float(base_cv_row["mean_test_score"].values[0]) * 100
    opt_cv_acc = float(grid_search.best_score_) * 100

    comp_table = pd.DataFrame({
        "Model Stage": ["Before: Default / Baseline KNN", "After: AutoML Optimized KNN"],
        "Hyperparameters": [
            "K = 5, metric = euclidean, weights = uniform",
            f"K = {best_params['knn__n_neighbors']}, metric = {best_params['knn__metric']}, weights = {best_params['knn__weights']}"
        ],
        "5-Fold CV Accuracy (%)": [round(base_cv_acc, 2), round(opt_cv_acc, 2)],
        "Held-Out Test Accuracy (%)": [round(base_eval["accuracy"], 2), round(opt_eval["accuracy"], 2)],
        "Held-Out Test Macro-F1 (%)": [round(base_eval["macro_f1"], 2), round(opt_eval["macro_f1"], 2)]
    })
    st.table(comp_table)

    fig4 = plot_before_vs_after(
        [base_cv_acc, base_eval["accuracy"], base_eval["macro_f1"]],
        [opt_cv_acc, opt_eval["accuracy"], opt_eval["macro_f1"]],
        best_params["knn__n_neighbors"],
        dataset_name
    )
    st.pyplot(fig4)

# ------------------------------------------
# TAB 5: MULTI-DATASET COMPARISON
# ------------------------------------------
with tab5:
    st.subheader("5. Multi-Dataset AutoML Comparison (Iris, Wine, Breast Cancer, Digits, Titanic)")

    @st.cache_data
    def get_multi_dataset_table():
        rows = []
        for d_name in SUPPORTED_DATASETS:
            res = run_automl_pipeline(d_name)
            b_ev = evaluate_model(res["baseline_model"], res["X_test"], res["y_test"], res["target_names"])
            o_ev = evaluate_model(res["best_model"], res["X_test"], res["y_test"], res["target_names"])
            bp = res["best_params"]
            rows.append({
                "Dataset": d_name,
                "Samples": len(res["X"]),
                "Features": res["X"].shape[1],
                "Classes": len(res["target_names"]),
                "Best K": bp["knn__n_neighbors"],
                "Best Metric": bp["knn__metric"],
                "Best Weights": bp["knn__weights"],
                "5-Fold CV Acc (%)": round(res["grid_search"].best_score_ * 100, 2),
                "Baseline Test Acc (%)": round(b_ev["accuracy"], 2),
                "Optimized Test Acc (%)": round(o_ev["accuracy"], 2),
                "Optimized Macro-F1 (%)": round(o_ev["macro_f1"], 2)
            })
        return pd.DataFrame(rows)

    multi_df = get_multi_dataset_table()
    st.dataframe(multi_df, use_container_width=True)

    fig5 = plot_multi_dataset_comparison(multi_df)
    st.pyplot(fig5)
