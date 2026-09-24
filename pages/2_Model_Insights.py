import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report, accuracy_score
)
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sidebar_style import apply_sidebar

st.set_page_config(page_title="Model Insights", page_icon="🧠", layout="wide")
apply_sidebar()

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_approval.csv")

# ── All heavy work cached — runs once, reused on every visit ─────────────────
@st.cache_data
def get_eval_data():
    model  = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    meta   = json.load(open(os.path.join(MODEL_DIR, "metadata.json")))
    df = pd.read_csv(DATA_PATH)
    df["loan_approved"] = df["loan_approved"].map(
        {"True": 1, "False": 0, True: 1, False: 0}
    )
    df["debt_to_income"] = df["loan_amount"] / (df["income"] + 1)
    FEATURES = meta["features"]
    X = df[FEATURES]
    y = df["loan_approved"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_s    = scaler.transform(X_test)
    y_pred      = model.predict(X_test_s)
    y_pred_prob = model.predict_proba(X_test_s)[:, 1]
    fi = None
    if hasattr(model, "feature_importances_"):
        fi_path = os.path.join(MODEL_DIR, "feature_importances.csv")
        if os.path.exists(fi_path):
            fi = pd.read_csv(fi_path, header=None, names=["Feature", "Importance"])
        else:
            fi = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_})
    return meta, y_test, y_pred, y_pred_prob, fi

meta, y_test, y_pred, y_pred_prob, fi_df = get_eval_data()

# ── Page header ───────────────────────────────────────────────────────────────
st.title("🧠 Model Insights")
st.markdown("Performance metrics, feature importance, ROC curve, and confusion matrix.")
st.markdown("---")

# ── Top metrics ───────────────────────────────────────────────────────────────
acc     = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model",     meta["model_name"])
c2.metric("Accuracy",  f"{acc*100:.2f}%")
c3.metric("ROC-AUC",   f"{roc_auc:.4f}")
c4.metric("Test Size", f"{len(y_test):,} records")

st.markdown("---")

# ── CV results ────────────────────────────────────────────────────────────────
st.subheader("Cross-Validation AUC Comparison")
cv_df = pd.DataFrame(
    list(meta["cv_results"].items()), columns=["Model", "CV AUC"]
).sort_values("CV AUC", ascending=False)
fig_cv = px.bar(
    cv_df, x="Model", y="CV AUC",
    color="CV AUC", color_continuous_scale="Viridis",
    text=cv_df["CV AUC"].map(lambda v: f"{v:.4f}"),
    range_y=[0, 1.05],
)
fig_cv.update_traces(textposition="outside")
fig_cv.update_layout(height=320, coloraxis_showscale=False, showlegend=False)
st.plotly_chart(fig_cv, use_container_width=True)

st.markdown("---")

# ── Feature importance ────────────────────────────────────────────────────────
if fi_df is not None:
    st.subheader("Feature Importances")
    fi_sorted = fi_df.sort_values("Importance", ascending=True)
    col_fi, col_fi_text = st.columns([2, 1])
    with col_fi:
        fig_fi = px.bar(
            fi_sorted, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Blues",
            text=fi_sorted["Importance"].map(lambda v: f"{v:.4f}"),
        )
        fig_fi.update_traces(textposition="outside")
        fig_fi.update_layout(height=320, coloraxis_showscale=False, yaxis_title="")
        st.plotly_chart(fig_fi, use_container_width=True)
    with col_fi_text:
        st.markdown("**Key Drivers:**")
        for _, row in fi_df.sort_values("Importance", ascending=False).iterrows():
            pct = row["Importance"] * 100
            bar = "█" * int(pct // 5)
            st.markdown(f"**{row['Feature']}** — `{pct:.1f}%`  \n`{bar}`")

st.markdown("---")

# ── ROC Curve ─────────────────────────────────────────────────────────────────
st.subheader("ROC Curve")
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(
    x=fpr, y=tpr, mode="lines",
    name=f"ROC (AUC = {roc_auc:.4f})",
    line=dict(color="#3b82f6", width=2),
))
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines",
    name="Random classifier",
    line=dict(color="#94a3b8", dash="dash"),
))
fig_roc.update_layout(
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate",
    height=380, legend=dict(x=0.6, y=0.1),
)
st.plotly_chart(fig_roc, use_container_width=True)

st.markdown("---")

# ── Confusion matrix ──────────────────────────────────────────────────────────
st.subheader("Confusion Matrix")
labels = ["Rejected", "Approved"]
cm = confusion_matrix(y_test, y_pred)
fig_cm = px.imshow(
    cm, text_auto=True, x=labels, y=labels,
    color_continuous_scale="Blues",
    labels=dict(x="Predicted", y="Actual", color="Count"),
    aspect="equal",
)
fig_cm.update_layout(height=360)
st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("---")

# ── Classification report ─────────────────────────────────────────────────────
st.subheader("Classification Report")
report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
report_df = pd.DataFrame(report).T.drop(columns=["support"], errors="ignore")
st.dataframe(
    report_df.style.format("{:.4f}").background_gradient(cmap="YlGn", axis=None),
    use_container_width=True,
)

st.markdown("---")

# ── Probability distribution ──────────────────────────────────────────────────
st.subheader("Predicted Probability Distribution (Test Set)")
prob_df = pd.DataFrame({
    "Approval Probability (%)": y_pred_prob * 100,
    "Actual": y_test.map({1: "Approved", 0: "Rejected"}).values,
})
fig_prob = px.histogram(
    prob_df, x="Approval Probability (%)", color="Actual",
    nbins=40, barmode="overlay", opacity=0.7,
    color_discrete_map={"Approved": "#22c55e", "Rejected": "#ef4444"},
)
fig_prob.update_layout(height=340, bargap=0.05)
st.plotly_chart(fig_prob, use_container_width=True)
