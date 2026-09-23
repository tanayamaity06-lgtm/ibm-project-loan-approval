import os
import sys
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sidebar_style import apply_sidebar

st.set_page_config(page_title="Predict", page_icon="🔮", layout="wide")
apply_sidebar()

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_model():
    model  = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    meta   = json.load(open(os.path.join(MODEL_DIR, "metadata.json")))
    return model, scaler, meta

model, scaler, meta = load_model()
FEATURES = meta["features"]

EXAMPLES = {
    "Select an example...": None,
    "✅ Approved — Strong Applicant": {
        "name": "Sarah Johnson", "income": 95000, "credit_score": 780,
        "loan_amount": 18000, "years_employed": 12, "points": 85,
    },
    "❌ Rejected — Weak Applicant": {
        "name": "Mike Turner", "income": 28000, "credit_score": 340,
        "loan_amount": 46000, "years_employed": 1, "points": 15,
    },
    "❌ Rejected — High Debt, Low Points": {
        "name": "Lisa Ray", "income": 35000, "credit_score": 420,
        "loan_amount": 50000, "years_employed": 2, "points": 20,
    },
    "❌ Rejected — Poor Credit Score": {
        "name": "David Chen", "income": 60000, "credit_score": 310,
        "loan_amount": 30000, "years_employed": 4, "points": 30,
    },
}

st.title("🔮 Loan Approval Prediction")
st.markdown("Fill in the applicant details below — or pick a preset example.")
st.markdown("---")

selected = st.selectbox("Load a preset example", list(EXAMPLES.keys()))
example  = EXAMPLES[selected]
defaults = example if example else {
    "name": "John Doe", "income": 75000, "credit_score": 650,
    "loan_amount": 25000, "years_employed": 5, "points": 65,
}

if example:
    if "Rejected" in selected:
        st.error(
            f"Rejection example — Points: {defaults['points']}, "
            f"Credit Score: {defaults['credit_score']}",
            icon="❌",
        )
    else:
        st.success(
            f"Approval example — Points: {defaults['points']}, "
            f"Credit Score: {defaults['credit_score']}",
            icon="✅",
        )

st.markdown("---")

with st.form("predict_form"):
    st.subheader("Applicant Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        name           = st.text_input("Applicant Name", value=defaults["name"])
        income         = st.number_input("Annual Income (USD)", min_value=1000, max_value=500000,
                                         value=defaults["income"], step=1000)
        credit_score   = st.slider("Credit Score", 300, 850, value=defaults["credit_score"])

    with col2:
        loan_amount    = st.number_input("Loan Amount (USD)", min_value=500, max_value=200000,
                                         value=defaults["loan_amount"], step=500)
        years_employed = st.slider("Years Employed", 0, 50, value=defaults["years_employed"])

    with col3:
        points = st.slider("Scoring Points", 0, 100, value=defaults["points"],
                           help="Internal credit scoring points (0–100)")

    submitted = st.form_submit_button("Predict Approval", use_container_width=True)

if submitted:
    dti = loan_amount / (income + 1)
    row = pd.DataFrame([{
        "income": income, "credit_score": credit_score,
        "loan_amount": loan_amount, "years_employed": years_employed,
        "points": points, "debt_to_income": dti,
    }])
    X     = scaler.transform(row[FEATURES])
    pred  = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    prob_approved  = probs[1] * 100
    prob_rejected  = probs[0] * 100
    result_label   = "APPROVED" if pred == 1 else "REJECTED"
    result_color   = "#22c55e" if pred == 1 else "#ef4444"

    st.markdown("---")
    st.subheader(f"Result for **{name}**")

    if pred == 1:
        st.success("### LOAN APPROVED", icon="✅")
    else:
        st.error("### LOAN REJECTED", icon="❌")
        reasons = []
        if points < 40:
            reasons.append(f"Scoring points too low ({points}/100)")
        if credit_score < 500:
            reasons.append(f"Poor credit score ({credit_score})")
        if dti > 0.5:
            reasons.append(f"High debt-to-income ratio ({dti:.2f})")
        if years_employed < 2:
            reasons.append(f"Insufficient employment history ({years_employed} yr)")
        if reasons:
            st.warning("**Reasons:**\n" + "\n".join(f"- {r}" for r in reasons))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Decision",              result_label)
    m2.metric("Approval Probability",  f"{prob_approved:.1f}%")
    m3.metric("Rejection Probability", f"{prob_rejected:.1f}%")
    m4.metric("Debt-to-Income",        f"{dti:.3f}")

    st.markdown("---")
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Approval Gauge")
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_approved,
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": result_color},
                "steps": [
                    {"range": [0,  40],  "color": "#fee2e2"},
                    {"range": [40, 60],  "color": "#fef9c3"},
                    {"range": [60, 100], "color": "#dcfce7"},
                ],
                "threshold": {"line": {"color": "black", "width": 3},
                              "thickness": 0.75, "value": 50},
            },
            title={"text": f"{name}"},
        ))
        fig_g.update_layout(height=300)
        st.plotly_chart(fig_g, use_container_width=True)

    with col_b:
        st.subheader("Probability Breakdown")
        fig_b = go.Figure(go.Bar(
            x=["Rejected", "Approved"],
            y=[prob_rejected, prob_approved],
            marker_color=["#ef4444", "#22c55e"],
            text=[f"{prob_rejected:.1f}%", f"{prob_approved:.1f}%"],
            textposition="outside",
        ))
        fig_b.update_layout(yaxis=dict(range=[0, 115], title="Probability (%)"),
                            height=300, showlegend=False)
        st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("---")
    st.subheader("Application Summary")
    summary = {
        "Name": name, "Annual Income": f"${income:,}",
        "Credit Score": credit_score, "Loan Amount": f"${loan_amount:,}",
        "Years Employed": years_employed, "Points": points,
        "Debt-to-Income": f"{dti:.4f}", "Decision": result_label,
        "Approval Probability": f"{prob_approved:.1f}%",
    }
    st.table(pd.DataFrame(list(summary.items()), columns=["Field", "Value"]))
