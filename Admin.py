import os
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "user_data.csv"


def render_admin_panel(user_collection=None):
    """
    Admin dashboard reading from local CSV `user_data.csv`.

    Logic:
    - Load all analyses from CSV
    - Collapse multiple runs per Email_ID to the LATEST run (by Timestamp)
    - Show KPIs, filters, ranked table, and a couple of plots
    """
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

    # Debug info (only if you enabled debug in session_state)
    if st.session_state.get("debug"):
        st.caption(f"(debug) Loaded {len(df)} rows from {DATA_PATH}")

    # --- Collapse to latest run per user (Email_ID) ---
    if "Email_ID" in df.columns and "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        # sort by time, then keep the last entry per Email_ID
        df = (
            df.sort_values("Timestamp")
              .groupby("Email_ID", as_index=False)
              .tail(1)
        )

    # ---- Type cleaning ----
    if "resume_score" in df.columns:
        df["resume_score"] = pd.to_numeric(df["resume_score"], errors="coerce")

    if "matching_score" in df.columns:
        df["matching_score_num"] = (
            df["matching_score"].astype(str).str.replace("%", "", regex=False)
        )
        df["matching_score_num"] = pd.to_numeric(
            df["matching_score_num"], errors="coerce"
        )

    # ---- KPIs ----
    total_users = len(df)  # now unique per Email_ID (latest only)
    avg_resume_score = df["resume_score"].mean() if "resume_score" in df.columns else None
    avg_match = df["matching_score_num"].mean() if "matching_score_num" in df.columns else None
    unique_roles = df["Predicted_Field"].nunique() if "Predicted_Field" in df.columns else 0

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Unique Users (latest run)", total_users)
    with kpi2:
        if avg_resume_score is not None:
            st.metric("Avg Resume Score", f"{avg_resume_score:.1f}")
    with kpi3:
        if avg_match is not None:
            st.metric("Avg Match %", f"{avg_match:.1f}%")
    with kpi4:
        st.metric("Unique Target Roles", unique_roles)

    st.markdown("---")

    # ---- Filters ----
    roles = (
        sorted(df["Predicted_Field"].dropna().unique().tolist())
        if "Predicted_Field" in df.columns
        else []
    )
    levels = (
        sorted(df["User_level"].dropna().unique().tolist())
        if "User_level" in df.columns
        else []
    )

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        selected_roles = st.multiselect(
            "Filter by Target Role",
            options=roles,
            default=roles if roles else None,
        )
    with col_f2:
        selected_levels = st.multiselect(
            "Filter by Experience Level",
            options=levels,
            default=levels if levels else None,
        )

    filtered = df.copy()

    if roles and selected_roles:
        filtered = filtered[filtered["Predicted_Field"].isin(selected_roles)]

    if levels and selected_levels:
        filtered = filtered[filtered["User_level"].isin(selected_levels)]

    # Sort by resume_score desc, then matching_score_num desc
    sort_cols = []
    if "resume_score" in filtered.columns:
        sort_cols.append("resume_score")
    if "matching_score_num" in filtered.columns:
        sort_cols.append("matching_score_num")
    if sort_cols:
        filtered = filtered.sort_values(sort_cols, ascending=[False] * len(sort_cols))

    # ---- Table view ----
    st.markdown("---")
    st.subheader("User Summary (latest run per Email)")

    display_cols = [
        "Name",
        "Email_ID",
        "Predicted_Field",
        "User_level",
        "resume_score",
        "matching_score",
        "Timestamp",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    st.dataframe(filtered[display_cols], use_container_width=True)

    # Download filtered data
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv,
        file_name="admin_resume_data.csv",
        mime="text/csv",
    )

    # ---- Visualizations ----
    st.markdown("---")
    st.subheader("Score Distribution (Unique Users)")

    if "resume_score" in filtered.columns:
        fig1 = px.histogram(
            filtered,
            x="resume_score",
            nbins=10,
            title="Distribution of Resume Scores",
        )
        fig1.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("No numeric resume_score field available for distribution plot.")

    st.subheader("Average Match % by Role")
    if "matching_score_num" in filtered.columns and "Predicted_Field" in filtered.columns:
        agg = (
            filtered
            .groupby("Predicted_Field")["matching_score_num"]
            .mean()
            .reset_index()
        )
        fig2 = px.bar(
            agg,
            x="matching_score_num",
            y="Predicted_Field",
            orientation="h",
            title="Average Skill Match by Role",
            labels={"matching_score_num": "Average Match %", "Predicted_Field": "Role"},
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Not enough matching_score data to build role-wise averages.")
