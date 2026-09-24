import os
import sys
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sidebar_style import apply_sidebar

st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")
apply_sidebar()

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .hero {
        background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 100%);
        border-radius: 16px;
        padding: 3rem 2.5rem;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    .hero h1 { font-size: 2.6rem; font-weight: 800; margin-bottom: 0.5rem; }
    .hero p  { font-size: 1.1rem; opacity: 0.88; max-width: 580px; margin: 0 auto; }
    .stat-row { display: flex; gap: 1.2rem; margin-bottom: 2rem; }
    .stat-card {
        flex: 1; background: #f0f7ff; border: 1px solid #bfdbfe;
        border-radius: 12px; padding: 1.4rem 1rem; text-align: center;
    }
    .stat-card .val { font-size: 2rem; font-weight: 800; color: #1d4ed8; line-height: 1.1; }
    .stat-card .lbl { font-size: 0.82rem; color: #64748b; margin-top: 0.3rem;
                      text-transform: uppercase; letter-spacing: 0.05em; }
    .nav-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.2rem; margin-bottom: 2rem; }
    .nav-card {
        background: white; border: 1.5px solid #e2e8f0;
        border-radius: 12px; padding: 1.4rem 1rem; text-align: center;
    }
    .nav-card .icon  { font-size: 2rem; margin-bottom: 0.4rem; }
    .nav-card .title { font-size: 0.95rem; font-weight: 700; color: #1e293b; }
    .nav-card .desc  { font-size: 0.8rem; color: #64748b; margin-top: 0.25rem; }
    .steps { display: flex; gap: 1rem; }
    .step  {
        flex: 1; background: #fafafa; border: 1px solid #e2e8f0;
        border-radius: 10px; padding: 1.1rem; display: flex; gap: 0.8rem;
    }
    .step .num {
        background: #2563eb; color: white; font-weight: 800; font-size: 0.95rem;
        border-radius: 50%; width: 30px; height: 30px; display: flex;
        align-items: center; justify-content: center; flex-shrink: 0;
    }
    .step .ttl { font-weight: 700; color: #1e293b; font-size: 0.9rem; }
    .step .bdy { font-size: 0.8rem; color: #64748b; margin-top: 0.1rem; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🏦 Loan Approval AI")
st.sidebar.caption("Powered by Random Forest · Streamlit")

st.markdown("""
<div class="hero">
    <h1>🏦 Loan Approval Prediction</h1>
    <p>An AI system that predicts whether a loan application will be
    <strong>approved</strong> or <strong>rejected</strong> — trained on 2,000 real records.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stat-row">
    <div class="stat-card"><div class="val">2,000</div><div class="lbl">Training Records</div></div>
    <div class="stat-card"><div class="val">100%</div><div class="lbl">Test Accuracy</div></div>
    <div class="stat-card"><div class="val">1.00</div><div class="lbl">ROC-AUC Score</div></div>
    <div class="stat-card"><div class="val">6</div><div class="lbl">Input Features</div></div>
</div>
""", unsafe_allow_html=True)

st.subheader("Pages")
st.markdown("""
<div class="nav-grid">
    <div class="nav-card">
        <div class="icon">🔮</div>
        <div class="title">Predict</div>
        <div class="desc">Enter applicant details and get an instant approval decision</div>
    </div>
    <div class="nav-card">
        <div class="icon">🧠</div>
        <div class="title">Model Insights</div>
        <div class="desc">Feature importance, ROC curve and confusion matrix</div>
    </div>
    <div class="nav-card">
        <div class="icon">📊</div>
        <div class="title">EDA</div>
        <div class="desc">Explore distributions, correlations and patterns in the data</div>
    </div>
    <div class="nav-card">
        <div class="icon">📂</div>
        <div class="title">Batch Predict</div>
        <div class="desc">Upload a CSV and predict approvals for many applicants at once</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.subheader("How It Works")
st.markdown("""
<div class="steps">
    <div class="step">
        <div class="num">1</div>
        <div><div class="ttl">Enter Applicant Data</div>
        <div class="bdy">Provide income, credit score, loan amount, employment years and points.</div></div>
    </div>
    <div class="step">
        <div class="num">2</div>
        <div><div class="ttl">Model Processes</div>
        <div class="bdy">Random Forest analyses the inputs and calculates approval probability.</div></div>
    </div>
    <div class="step">
        <div class="num">3</div>
        <div><div class="ttl">Instant Decision</div>
        <div class="bdy">Get Approved / Rejected verdict with confidence score and gauge chart.</div></div>
    </div>
</div>
""", unsafe_allow_html=True)
