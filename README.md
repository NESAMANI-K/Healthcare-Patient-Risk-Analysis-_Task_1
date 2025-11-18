Healthcare Data Analysis & Machine Learning Pipeline

A complete, end-to-end healthcare data science project built using Python. The pipeline covers data exploration, predictive modeling, anomaly detection, and AI-driven doctor-style recommendation generation.

🚀 Project Overview

This project analyzes a real-world healthcare dataset (Kaggle) and builds a multi-stage ML pipeline with automated outputs, visualizations, and an LLM-powered guidance system.

✔️ Task 1 — Exploratory Data Analysis (EDA)

The project begins with a detailed exploratory analysis to understand trends, distributions, and patterns in patient data.

Key EDA Components

Distribution analysis using:

Boxplots

KDE plots

Violin plots

Log-transformed histograms

Frequency visualizations for:

Medical Conditions

Admission Types

Medications

Outlier detection using IQR and visual markers

Missing value heatmaps and correlation matrices

Automated EDA summary saved in the outputs/ directory

This step provides a complete statistical understanding of patient demographics, billing patterns, and disease frequency.

✔️ Task 2 — Supervised Machine Learning
🎯 Prediction Target:

“Test Results” → Normal, Abnormal, or Inconclusive

Techniques Used

Feature Engineering

Patient stay duration (Discharge – Admission)

Medication count per patient

Billing Amount bucketization

Handling High Cardinality Fields

Doctor and Hospital fields grouped using frequency encoding

Primary ML Model

CatBoostClassifier
(handles categorical features natively)

Fallback Model

RandomForestClassifier + OneHotEncoder

Model Outputs

Accuracy, Precision, Recall, F1-score

Confusion matrix and performance report

Actual vs Predicted comparison plots

Feature importance ranked file

Exported CatBoost Model (.cbm format)

Predictions saved as CSV

This supervised step predicts the clinical test outcome for each patient based on their attributes.

✔️ Task 3 — Unsupervised Learning (Anomaly Detection)
🔍 Goal: Identify abnormal billing patterns that may indicate errors, fraud, or unusual medical cases.
Techniques Used

Z-score analysis for statistical outlier detection

IsolationForest for robust anomaly detection in billing data

Generated Outputs

Dataset with anomaly flags (0 = normal, 1 = anomaly)

Top anomalies saved as CSV

Text-based interpretation explaining:

High billing spikes

Unusual low billing cases

Potential hospital/doctor behavior patterns

This task ensures deeper financial insights and highlights unusual records.

✔️ Task 4 — AI-Generated Doctor Recommendation (LLM-Style)

An integrated AI module uses model predictions and patient metadata to generate clinical-style suggestions.

Input Parameters

Age

Medical Condition

Current Medication

Model-predicted test result

Severity indicators

AI Outputs

Short, doctor-style recommendation

Follow-up instructions and risk assessment

Helpful insights tailored to the patient's condition

Saved final recommendation as a .txt file

This enhances the ML pipeline with natural-language intelligence similar to modern healthcare AI assistants.
