import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sidebar_style import apply_sidebar

st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
apply_sidebar()

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "loan_approval.csv")

NUM_COLS = ["income", "credit_score", "loan_amount", "years_employed", "points", "debt_to_income"]

# ── All data loading and prep cached ──────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["loan_approved"]  = df["loan_approved"].map({"True": 1, "False": 0, True: 1, False: 0})
    df["approval_label"] = df["loan_approved"].map({1: "Approved", 0: "Rejected"})
    df["debt_to_income"] = df["loan_amount"] / (df["income"] + 1)
    return df

@st.cache_data
def get_city_data(df):
    city = (
        df.groupby("city")["loan_approved"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "approved", "count": "total"})
    )
    city["rate"] = city["approved"] / city["total"]
    return city.nlargest(15, "total").reset_index()

@st.cache_data
def get_corr(df):
    return df[NUM_COLS + ["loan_approved"]].corr().round(2)

df         = load_data()
top_cities = get_city_data(df)
corr       = get_corr(df)

# ── Page header ────────────────────────────────────────────────────────────────
st.title("📊 Exploratory Data Analysis")
st.markdown("Explore the loan approval dataset — distributions, correlations, and patterns.")
st.markdown("---")

# ── Raw data preview ───────────────────────────────────────────────────────────
with st.expander("Raw Data Preview", expanded=False):
    st.dataframe(df.head(100), use_container_width=True)
    st.caption(f"Showing 100 of {len(df):,} rows · {df.shape[1]} columns")

# ── Summary stats ──────────────────────────────────────────────────────────────
st.subheader("Summary Statistics")
st.dataframe(df[NUM_COLS].describe().T.style.format("{:.2f}"), use_container_width=True)

st.markdown("---")

# ── Class distribution + City chart ────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Loan Approval Distribution")
    counts = df["approval_label"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    fig_pie = px.pie(
        counts, names="Status", values="Count",
        color="Status",
        color_discrete_map={"Approved": "#22c55e", "Rejected": "#ef4444"},
        hole=0.4,
    )
    fig_pie.update_traces(textinfo="label+percent")
    fig_pie.update_layout(showlegend=True, height=360)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("Top 15 Cities by Applications")
    fig_city = px.bar(
        top_cities, x="city", y=["approved", "total"],
        barmode="overlay",
        labels={"value": "Count", "city": "City"},
        color_discrete_sequence=["#22c55e", "#94a3b8"],
    )
    fig_city.update_layout(height=360, xaxis_tickangle=-40, legend_title="")
    st.plotly_chart(fig_city, use_container_width=True)

st.markdown("---")

# ── Feature distribution ────────────────────────────────────────────────────────
st.subheader("Feature Distributions by Approval Status")
feat_choice = st.selectbox("Select feature", NUM_COLS, index=1)

col3, col4 = st.columns(2)

with col3:
    fig_hist = px.histogram(
        df, x=feat_choice, color="approval_label",
        barmode="overlay", nbins=50,
        color_discrete_map={"Approved": "#22c55e", "Rejected": "#ef4444"},
        opacity=0.7,
        labels={"approval_label": "Status"},
    )
    fig_hist.update_layout(height=340, bargap=0.05)
    st.plotly_chart(fig_hist, use_container_width=True)

with col4:
    fig_box = px.box(
        df, x="approval_label", y=feat_choice,
        color="approval_label",
        color_discrete_map={"Approved": "#22c55e", "Rejected": "#ef4444"},
        points="outliers",
        labels={"approval_label": "Status",
                feat_choice: feat_choice.replace("_", " ").title()},
    )
    fig_box.update_layout(height=340, showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

st.markdown("---")

# ── Scatter: Credit Score vs Points ────────────────────────────────────────────
st.subheader("Credit Score vs Points")
fig_scatter = px.scatter(
    df, x="credit_score", y="points",
    color="approval_label",
    color_discrete_map={"Approved": "#22c55e", "Rejected": "#ef4444"},
    opacity=0.5,
    labels={"credit_score": "Credit Score", "points": "Points", "approval_label": "Status"},
)
fig_scatter.update_layout(height=380)
st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")

# ── Correlation heatmap ─────────────────────────────────────────────────────────
st.subheader("Correlation Heatmap")
fig_corr = px.imshow(
    corr, text_auto=True,
    color_continuous_scale="RdBu_r",
    aspect="auto", zmin=-1, zmax=1,
)
fig_corr.update_layout(height=460)
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# ── Income vs Loan Amount ───────────────────────────────────────────────────────
st.subheader("Income vs Loan Amount")
fig_inc = px.scatter(
    df, x="income", y="loan_amount",
    color="approval_label",
    color_discrete_map={"Approved": "#22c55e", "Rejected": "#ef4444"},
    opacity=0.45,
    labels={"income": "Annual Income (USD)", "loan_amount": "Loan Amount (USD)",
            "approval_label": "Status"},
)
fig_inc.update_layout(height=380)
st.plotly_chart(fig_inc, use_container_width=True)
