import streamlit as st

def apply_sidebar():
    st.markdown("""
<style>
/* ── Sidebar background ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%) !important;
    padding: 0 !important;
}

/* ── Hide default page nav label ("navigation") ── */
[data-testid="stSidebarNav"]::before {
    display: none;
}

/* ── Logo / brand block ── */
[data-testid="stSidebar"]::before {
    content: "";
    display: block;
    height: 4px;
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    border-radius: 0 0 4px 4px;
}

/* ── Nav link items ── */
[data-testid="stSidebarNav"] a {
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
    padding: 0.6rem 1.2rem !important;
    margin: 2px 10px !important;
    border-radius: 8px !important;
    font-size: 0.92rem !important;
    font-weight: 500 !important;
    color: #cbd5e1 !important;
    text-decoration: none !important;
    transition: background 0.18s, color 0.18s !important;
}

[data-testid="stSidebarNav"] a:hover {
    background: rgba(99,102,241,0.18) !important;
    color: #ffffff !important;
}

[data-testid="stSidebarNav"] a[aria-current="page"] {
    background: linear-gradient(90deg,#2563eb22,#7c3aed22) !important;
    color: #818cf8 !important;
    font-weight: 700 !important;
    border-left: 3px solid #6366f1 !important;
}

/* ── Sidebar text (st.sidebar.markdown etc.) ── */
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown li,
[data-testid="stSidebar"] .stMarkdown span {
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
}

/* ── Sidebar title / header ── */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    padding: 0 1.2rem !important;
}

/* ── Divider ── */
[data-testid="stSidebar"] hr {
    border-color: #334155 !important;
    margin: 0.5rem 1.2rem !important;
}

/* ── Scrollbar ── */
[data-testid="stSidebar"]::-webkit-scrollbar { width: 4px; }
[data-testid="stSidebar"]::-webkit-scrollbar-thumb {
    background: #334155; border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

    # Brand header inside sidebar
    st.sidebar.markdown("""
<div style="
    padding: 1.4rem 1.2rem 0.6rem;
    border-bottom: 1px solid #1e3a5f;
    margin-bottom: 0.5rem;
">
    <div style="font-size:1.5rem; font-weight:800; color:#f1f5f9; letter-spacing:-0.5px;">
        🏦 LoanAI
    </div>
    <div style="font-size:0.75rem; color:#64748b; margin-top:2px;">
        Loan Approval Prediction System
    </div>
</div>
""", unsafe_allow_html=True)

    st.sidebar.markdown("""
<div style="padding: 0.4rem 1.2rem 0.2rem; font-size:0.7rem;
            text-transform:uppercase; letter-spacing:0.1em; color:#475569; font-weight:600;">
    Navigation
</div>
""", unsafe_allow_html=True)

    st.sidebar.markdown("""
<div style="padding: 1rem 1.2rem 0; margin-top: auto; border-top: 1px solid #1e3a5f;">
    <div style="font-size:0.72rem; color:#334155; text-align:center; padding-top:0.5rem;">
        Powered by Random Forest · Streamlit
    </div>
</div>
""", unsafe_allow_html=True)
