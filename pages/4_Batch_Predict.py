import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sidebar_style import apply_sidebar

st.set_page_config(page_title="Batch Predict", page_icon="📂", layout="wide")
apply_sidebar()

# ── Load model artefacts ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_model():
    model  = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        meta = json.load(f)
    return model, scaler, meta

model, scaler, meta = load_model()
FEATURES = meta["features"]

# ── Page header ─────────────────────────────────────────────────────────────
st.title("📂 Batch Loan Prediction")
st.markdown(
    "Upload a CSV file containing multiple applicants and get predictions for all of them at once."
)
st.markdown("---")

# ── Expected columns info ────────────────────────────────────────────────────
with st.expander("ℹ️ Expected CSV Format"):
    st.markdown(
        """
        Your CSV must contain these columns (column names are case-sensitive):

        | Column | Type | Example |
        |---|---|---|
        | `name` | string | John Doe |
        | `income` | int/float | 75000 |
        | `credit_score` | int | 680 |
        | `loan_amount` | int/float | 20000 |
        | `years_employed` | int | 5 |
        | `points` | float | 65.0 |

        The `debt_to_income` column is auto-derived; no need to include it.
        """
    )
    # Sample download
    sample = pd.DataFrame({
        "name":           ["Alice Smith", "Bob Jones", "Carol White"],
        "income":         [80000, 45000, 120000],
        "credit_score":   [720, 480, 800],
        "loan_amount":    [20000, 35000, 15000],
        "years_employed": [10, 2, 20],
        "points":         [75.0, 30.0, 90.0],
    })
    csv_sample = sample.to_csv(index=False).encode()
    st.download_button(
        "Download Sample CSV",
        data=csv_sample,
        file_name="sample_applicants.csv",
        mime="text/csv",
    )

# ── File uploader ────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
        st.success(f"Loaded {len(df_raw):,} rows from **{uploaded.name}**")

        # Validate required columns
        required = ["income", "credit_score", "loan_amount", "years_employed", "points"]
        missing  = [c for c in required if c not in df_raw.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        # Feature engineering
        df_raw["debt_to_income"] = df_raw["loan_amount"] / (df_raw["income"] + 1)
        X_scaled  = scaler.transform(df_raw[FEATURES])

        # Predictions
        preds      = model.predict(X_scaled)
        probs      = model.predict_proba(X_scaled)[:, 1]

        df_result  = df_raw.copy()
        df_result["prediction"]         = ["Approved" if p == 1 else "Rejected" for p in preds]
        df_result["approval_probability"] = (probs * 100).round(2)

        st.markdown("---")
        st.subheader("Batch Prediction Results")

        # ── Summary metrics ────────────────────────────────────────────────
        n_approved = (preds == 1).sum()
        n_rejected = (preds == 0).sum()
        approval_rate = n_approved / len(preds) * 100

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Applications", f"{len(df_result):,}")
        c2.metric("Approved",  f"{n_approved:,}", f"{approval_rate:.1f}%")
        c3.metric("Rejected",  f"{n_rejected:,}", f"-{100 - approval_rate:.1f}%")

        st.markdown("---")

        # ── Charts ─────────────────────────────────────────────────────────
        col_pie, col_hist = st.columns(2)

        with col_pie:
            fig_pie = px.pie(
                names=["Approved", "Rejected"],
                values=[n_approved, n_rejected],
                color=["Approved", "Rejected"],
                color_discrete_map={"Approved": "#22c55e", "Rejected": "#ef4444"},
                hole=0.4,
                title="Approval Distribution",
            )
            fig_pie.update_traces(textinfo="label+percent")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_hist:
            fig_hist = px.histogram(
                df_result, x="approval_probability",
                color="prediction",
                nbins=30,
                color_discrete_map={"Approved": "#22c55e", "Rejected": "#ef4444"},
                title="Approval Probability Distribution",
                labels={"approval_probability": "Probability (%)", "prediction": "Decision"},
                opacity=0.75,
            )
            fig_hist.update_layout(bargap=0.05)
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Results table ──────────────────────────────────────────────────
        st.subheader("Full Results Table")

        def highlight_decision(val):
            if val == "Approved":
                return "background-color: #dcfce7; color: #166534; font-weight: bold"
            elif val == "Rejected":
                return "background-color: #fee2e2; color: #991b1b; font-weight: bold"
            return ""

        display_cols = [c for c in df_result.columns]
        st.dataframe(
            df_result[display_cols].style.applymap(
                highlight_decision, subset=["prediction"]
            ),
            use_container_width=True,
            height=420,
        )

        # ── Download results ───────────────────────────────────────────────
        csv_out = df_result.to_csv(index=False).encode()
        st.download_button(
            "Download Predictions CSV",
            data=csv_out,
            file_name="loan_predictions.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to get batch predictions.")
