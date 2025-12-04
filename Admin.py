import os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "user_data.csv"


def render_admin_panel(user_collection=None):
    st.title("Admin Dashboard – Resume Analytics")

    if not DATA_PATH.exists():
        st.info("No user data found yet. Ask users to analyse resumes first.")
        return

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"Could not read data file: {e}")
        return

    if df.empty:
        st.info("Data file is empty.")
        return

    # Collapse to latest run per Email_ID
    if "Email_ID" in df.columns and "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.sort_values("Timestamp").groupby("Email_ID", as_index=False).tail(1)

    if "resume_score" in df.columns:
        df["resume_score"] = pd.to_numeric(df["resume_score"], errors="coerce")

    if "matching_score" in df.columns:
        df["matching_score_num"] = (
            df["matching_score"].astype(str).str.replace("%", "", regex=False)
        )
        df["matching_score_num"] = pd.to_numeric(
            df["matching_score_num"], errors="coerce"
        )

    # ----- Intro Text -----
    st.markdown("""
    Welcome to the **Admin Dashboard**.

    This dashboard helps you:
    - Track user resume analysis data  
    - Monitor score trends  
    - Explore skill gaps by role  
    - Filter and download data  
    """)

    # ---- KPIs ----
    total_users = len(df)
    avg_resume_score = df["resume_score"].mean() if "resume_score" in df.columns else None
    avg_match = df["matching_score_num"].mean() if "matching_score_num" in df.columns else None
    unique_roles = df["Predicted_Field"].nunique() if "Predicted_Field" in df.columns else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Unique Users", total_users)
    with kpi2:
        st.metric("Avg Resume Score", f"{avg_resume_score:.1f}" if avg_resume_score else "N/A")
    with kpi3:
        st.metric("Avg Match %", f"{avg_match:.1f}%" if avg_match else "N/A")
    with kpi4:
        st.metric("Unique Roles", unique_roles)

    # ========== FILTERS ==========
    st.markdown("---")
    st.subheader("Filters")

    roles = df["Predicted_Field"].dropna().unique().tolist() if "Predicted_Field" in df.columns else []
    levels = df["User_level"].dropna().unique().tolist() if "User_level" in df.columns else []

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_roles = st.multiselect("Filter by Role", options=roles, default=roles)
    with col_f2:
        selected_levels = st.multiselect("Filter by Experience Level", options=levels, default=levels)

    filtered = df.copy()
    if selected_roles:
        filtered = filtered[filtered["Predicted_Field"].isin(selected_roles)]
    if selected_levels:
        filtered = filtered[filtered["User_level"].isin(selected_levels)]

    filtered = filtered.sort_values(["resume_score", "matching_score_num"], ascending=False)

    # ========== TABS ==========
    tab1, tab2, tab3 = st.tabs(["📋 Data Table", "📊 Charts", "🔥 Skill Gap Insights"])

    with tab1:
        st.subheader("User Summary Table")
        display_cols = [
            "Name", "Email_ID", "Predicted_Field", "User_level",
            "resume_score", "matching_score", "Timestamp"
        ]
        display_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[display_cols], use_container_width=True)

        csv = filtered.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download CSV", csv, "admin_resume_data.csv", "text/csv")

    with tab2:
        st.subheader("Score Distribution")
        if "resume_score" in filtered.columns:
            fig1 = px.histogram(filtered, x="resume_score", nbins=10,
                                title="Distribution of Resume Scores")
            st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Average Match % by Role")
        if "matching_score_num" in filtered.columns and "Predicted_Field" in filtered.columns:
            agg = filtered.groupby("Predicted_Field")["matching_score_num"].mean().reset_index()
            fig2 = px.bar(agg, x="matching_score_num", y="Predicted_Field",
                          orientation="h", title="Avg Skill Match by Role",
                          labels={"matching_score_num": "Match %", "Predicted_Field": "Role"})
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Top 5 Missing Skills (All Users)")
        if "Recommended_skills" in df.columns and "Actual_skills" in df.columns:
            from collections import Counter

            all_missing = []
            for i, row in df.iterrows():
                actual = set(str(row["Actual_skills"]).strip("[]").replace("'", "").split(","))
                recommended = set(str(row["Recommended_skills"]).strip("[]").replace("'", "").split(","))
                missing = recommended - actual
                all_missing.extend([skill.strip() for skill in missing if skill.strip()])

            top_missing = Counter(all_missing).most_common(5)
            for skill, count in top_missing:
                st.markdown(f"- **{skill}** — {count} users")

        else:
            st.info("Missing skill data not available.")

