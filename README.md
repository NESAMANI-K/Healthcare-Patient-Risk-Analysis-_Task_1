Healthcare Patient Risk Analysis – Full ML/AI Pipeline 

The Healthcare Patient Risk Analysis project implements a complete machine learning and AI workflow starting from raw data exploration to predictive modeling and ending with an AI-powered clinical recommendation module. It simulates how hospitals, clinics, and insurance companies use data-driven systems for early detection, anomaly monitoring, and patient care enhancement.

1. Exploratory Data Analysis (EDA)

EDA is the foundation of any ML project. The purpose is to understand the dataset and identify patterns, correlations, and anomalies.

a. Numerical Feature Analysis

Three major numerical columns were examined:

Age – demographic indicator

Billing Amount – financial/medical cost indicator

Room Number – hospital infrastructure indicator

Statistical measures such as mean, median, standard deviation, min-max range, and quartiles were computed using df.describe().

b. Distribution Visualization

Histograms with KDE curves were plotted using Seaborn to understand:

Whether Age follows a normal distribution

Whether Billing Amount shows skewness (often right-skewed in healthcare datasets)

How Room assignments are spread (random vs structured)

c. Categorical Feature Analysis

Frequency plots were generated for:

Medical Condition (e.g., Diabetes, Cancer, Obesity …)

Admission Type (Emergency, Urgent, Elective)

Medication (Aspirin, Ibuprofen, Paracetamol …)

These countplots help determine the prevalence of diseases and common treatment profiles.

2. Supervised Learning – Predicting Test Results

The objective is to predict Test Results, which has three classes:

Normal

Abnormal

Inconclusive

a. Feature Preparation

Non-essential columns (Name, Doctor, Hospital, Dates) were removed.

Categorical features like Gender, Blood Type, Medical Condition, etc., were one-hot encoded using:

pd.get_dummies(..., drop_first=False)

b. Train–Test Split

Data was split into:

80% Training (X_train, y_train)

20% Testing (X_test, y_test)

Stratification ensured balanced class distribution.

c. Scaling

Numerical features (Age, Billing Amount, Room Number) were standardized using StandardScaler.

d. Model Training

A Logistic Regression classifier was trained with:

model = LogisticRegression(max_iter=1000)


Even though accuracy was modest (~34%), this is common for:

Multi-class problems

High categorical cardinality

Few strong predictors

e. Evaluation Metrics

Accuracy Score

Classification Report (precision, recall, F1)

Confusion Matrix

These help understand misclassifications and class imbalance.

3. Unsupervised Learning – Anomaly Detection

An Isolation Forest was used to detect unusual billing patterns:

iso = IsolationForest(contamination=0.05)

a. What It Detects

Extremely high billing → costly treatments

Sudden spikes → long stays or rare procedures

Very low billing → incomplete billing or clerical errors

b. Visualization

Scatter plots marked:

Blue = normal

Red = anomalous regions

This provides insights for fraud detection, insurance audits, and hospital budgeting.

4. AI Task – LLM-Powered Clinical Recommendation

The final component integrates the ML model output with a Large Language Model (LLM) to generate human-like medical recommendations.

a. Components Used

OpenRouter/OpenAI API

A custom recommendation function

Patient metadata + ML prediction

b. What the LLM Does

After the ML model predicts “Normal,” “Abnormal,” or “Inconclusive,” the LLM generates:

Risk explanation

Follow-up evaluation

Medication advice

Specialist referral suggestions

Severity considerations

This mimics real medical writing and decision support systems.

c. Example Output

The AI generates structured medical-style recommendations like:

“Findings suggest an abnormal test result — recommend prompt specialist review. Arrange further diagnostics and seek urgent care if symptoms worsen.”

This creates a realistic AI-assisted clinical decision support workflow.
