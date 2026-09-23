# 🏦 Loan Approval Prediction System

A full-stack ML project using **Python + Streamlit** to predict loan approvals from applicant data.

---

## Project Structure

```
loan_approval_project/
├── data/
│   └── loan_approval.csv          ← raw dataset (2,000 records)
├── models/
│   ├── model.pkl                  ← trained RandomForest classifier
│   ├── scaler.pkl                 ← StandardScaler
│   ├── feature_importances.csv    ← feature importance values
│   └── metadata.json             ← model name, accuracy, AUC, CV results
├── pages/
│   ├── 1_EDA.py                   ← Exploratory Data Analysis
│   ├── 2_Predict.py               ← Single-record prediction form
│   ├── 3_Batch_Predict.py         ← Bulk CSV upload & prediction
│   └── 4_Model_Insights.py        ← Feature importance, ROC, confusion matrix
├── train_model.py                 ← ML training pipeline
├── app.py                         ← Streamlit entry-point (Home page)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model (already done — artefacts in `models/`)
```bash
python train_model.py
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Dataset

| Column | Description |
|---|---|
| `name` | Applicant name |
| `city` | City of residence |
| `income` | Annual income (USD) |
| `credit_score` | Credit score (300–850) |
| `loan_amount` | Requested loan amount (USD) |
| `years_employed` | Years of continuous employment |
| `points` | Internal scoring points (0–100) |
| `loan_approved` | Target: True/False |

---

## Model

- **Algorithm:** Random Forest (selected from RF, Gradient Boosting, Logistic Regression via 5-fold CV)
- **Derived feature:** `debt_to_income = loan_amount / income`
- **Test Accuracy:** 100%
- **ROC-AUC:** 1.0000

The `points` feature (66.6% importance) is the dominant predictor, followed by `credit_score` (24.9%).

---

## Pages

| Page | Description |
|---|---|
| 🏠 Home | Project overview and architecture |
| 📊 EDA | Distributions, correlations, scatter plots, heatmap |
| 🔮 Predict | Fill in a form and get an instant prediction with gauge chart |
| 📂 Batch Predict | Upload CSV → download predictions |
| 🧠 Model Insights | Feature importance, ROC curve, confusion matrix, classification report |
