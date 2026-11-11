# 🧠 Smart ML Parameter Tuner – AutoML Mini Lab
**Group 4 · Hybrid Optimization + AI**  
*University of Engineering & Management (UEM), Kolkata — B.Tech CSE (IoT, CS, BT)*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://smart-ml-parameter-tuner.streamlit.app/)

🔗 **Live Deployed Web Application:** [https://smart-ml-parameter-tuner.streamlit.app/](https://smart-ml-parameter-tuner.streamlit.app/)  
📂 **GitHub Repository:** [https://github.com/priyangshu-439/Smart-ML-Parameter-Tuner-Group4](https://github.com/priyangshu-439/Smart-ML-Parameter-Tuner-Group4)

---

## 📌 1. Project Overview
The **Smart ML Parameter Tuner – AutoML Mini Lab** is an interactive Automated Machine Learning (AutoML) web application built with **Python**, **Scikit-learn**, and **Streamlit**. It automates hyperparameter optimization for the **K-Nearest Neighbors (KNN)** classifier using **5-Fold Stratified Cross-Validation (`GridSearchCV`)** inside a leakage-free `StandardScaler` pipeline and visualizes model behavior across five mandatory experimental demonstrations.

---

## ⚙️ 2. Key Features & Faculty Requirements Implemented

### Part A: Interactive Dashboard Controls
1. **Dataset Selector:** Switch dynamically between **Iris**, **Wine**, **Breast Cancer Wisconsin**, **Digits**, and **Titanic**.
2. **K-Value Slider:** Test manual neighbor values ($K = 1$ to $15$).
3. **Distance Metric Selector:** Toggle between **Euclidean ($L_2$)** and **Manhattan ($L_1$)** distances.
4. **Weighting Method Selector:** Toggle between **Uniform** and **Distance** weighting.
5. **Run Optimization Button:** Triggers 5-Fold Stratified `GridSearchCV` across all 60 hyperparameter combinations ($15 \times 2 \times 2 = 60$ candidates, $300$ fits).
6. **Live Sample Class Predictor:** Real-time class prediction with confidence scores.
7. **Model Performance Cards:** Top-level KPI cards displaying Best $K$, 5-Fold CV Accuracy, Held-Out Test Accuracy, and Macro-F1.
8. **Model Comparison View:** Side-by-side comparison of Baseline KNN vs. AutoML Optimized KNN.

### Part B: Five Mandatory Visual Demonstrations
1. **Accuracy vs. $K$ Graph:** Plots 5-Fold Stratified Validation Accuracy across $K = 1\text{–}15$ and highlights the optimal $K$.
2. **2D Decision Boundary Visualization:** Compares classification regions between manual settings and AutoML optimal settings across any two selected features.
3. **Confusion Matrix & Performance Metrics:** Heatmap and per-class report covering **Accuracy, Precision, Recall, and Macro F1-Score** on the held-out 20% test set.
4. **Before vs. After Optimization:** Evaluates Default Baseline KNN ($K=5$, Euclidean, Uniform) against Optimized KNN on the untouched test dataset.
5. **Multi-Dataset Comparison:** Benchmarks AutoML optimization across **Iris, Wine, Breast Cancer, Digits, and Titanic**.

---

## 🗂️ 3. Repository Structure

```text
Smart-ML-Parameter-Tuner-Group4/
├── app.py                          # Main Streamlit web application (Dashboard & 5 Tabs)
├── model.py                        # Dataset loader, StandardScaler Pipeline & Stratified GridSearchCV
├── visualizations.py               # Matplotlib/Seaborn plotting functions for the 5 demonstrations
├── 01_explore_iris.py              # Step 1: Iris dataset exploration & statistical summary
├── 02_knn_accuracy.py              # Step 2: Validation accuracy across K = 1 to 15
├── 03_gridsearch_optimization.py   # Step 3: 5-Fold Stratified GridSearchCV & CSV export
├── 04_visual_demonstrations.py     # Step 4: Script to export high-res PNG charts for report
├── accuracy_vs_k.png               # Generated Plot 1: Validation Accuracy vs K
├── decision_boundary.png           # Generated Plot 2: 2D KNN Decision Boundaries (K=1, 3, 15)
├── confusion_matrix.png            # Generated Plot 3: Test Set Confusion Matrix Heatmap
├── before_vs_after.png             # Generated Plot 4: Baseline vs Optimized KNN Comparison
├── multi_dataset_comparison.png    # Generated Plot 5: Multi-Dataset AutoML Benchmark
├── gridsearch_results.csv          # Complete 60-combination 5-fold CV table
├── multi_dataset_results.csv       # Multi-dataset benchmark summary metrics
├── requirements.txt                # Required Python dependencies for Streamlit Cloud
└── README.md                       # Project documentation
