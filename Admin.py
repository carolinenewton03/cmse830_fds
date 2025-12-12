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
        df_all = df.copy()
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
            st.subheader("📈 Resume Analyses Over Time")
            st.markdown("""
            This chart shows how many resume analyses were performed each day using the `Timestamp` field.
            It helps track user engagement and activity trends over time.
            """)
            if "Timestamp" in df_all.columns:
                ts = df_all.copy()
                ts["Timestamp"] = pd.to_datetime(ts["Timestamp"], errors="coerce")
                ts = ts.dropna(subset=["Timestamp"])

                if not ts.empty:
                    ts["Date"] = ts["Timestamp"].dt.date

                    daily_counts = (
                        ts.groupby("Date")
                        .size()
                        .reset_index(name="Analyses")
                        .sort_values("Date")
                    )

                    fig_ts = px.line(
                        daily_counts,
                        x="Date",
                        y="Analyses",
                        markers=True,
                        title="Number of Resume Analyses per Day"
                    )
                    st.plotly_chart(fig_ts, use_container_width=True)
                    if not daily_counts.empty:
                        first = int(daily_counts["Analyses"].iloc[0])
                        last = int(daily_counts["Analyses"].iloc[-1])
                        peak_row = daily_counts.loc[daily_counts["Analyses"].idxmax()]
                        peak_day = peak_row["Date"]
                        peak_val = int(peak_row["Analyses"])
                        st.markdown(f"""
                    **Data-driven conclusion:**
                    - Daily activity changed from **{first}** to **{last}** analyses (Δ = **{last-first}**).
                    - Peak usage was on **{peak_day}** with **{peak_val}** analyses.
                    """)

                else:
                    st.info("Not enough valid timestamp data to plot time series.")
            else:
                st.info("Timestamp column not found.")

            st.subheader("Score Distribution")
            st.markdown("""
            This histogram shows the distribution of `resume_score` across the filtered users.
            It helps identify the typical score range and spot low/high outliers.
            """)

            if "resume_score" in filtered.columns:
                fig1 = px.histogram(
                    filtered,
                    x="resume_score",
                    nbins=10,
                    title="Distribution of Resume Scores"
                )
                st.plotly_chart(fig1, use_container_width=True)
                q25 = filtered["resume_score"].quantile(0.25)
                med = filtered["resume_score"].median()
                q75 = filtered["resume_score"].quantile(0.75)
                st.markdown(f"""
                **Data-driven conclusion:**
                - Median resume score is **{med:.1f}**.
                - Most users fall between **{q25:.1f}** and **{q75:.1f}** (IQR range).
                """)

            else:
                st.info("resume_score column not found.")

            st.subheader("Average Match % by Role")
            st.markdown("""
            This bar chart compares the **average skill match percentage** across roles.
            It highlights which roles users align with best and where the biggest skill gaps exist.
            """)

            if "matching_score_num" in filtered.columns and "Predicted_Field" in filtered.columns:
                agg = filtered.groupby("Predicted_Field")["matching_score_num"].mean().reset_index()
                fig2 = px.bar(
                    agg,
                    x="matching_score_num",
                    y="Predicted_Field",
                    orientation="h",
                    title="Avg Skill Match by Role",
                    labels={"matching_score_num": "Match %", "Predicted_Field": "Role"}
                )
                st.plotly_chart(fig2, use_container_width=True)
                if not agg.empty:
                    top = agg.sort_values("matching_score_num", ascending=False).iloc[0]
                    bot = agg.sort_values("matching_score_num", ascending=True).iloc[0]
                    st.markdown(f"""
                **Data-driven conclusion:**
                - Highest average match: **{top['Predicted_Field']}** (~**{top['matching_score_num']:.1f}%**)
                - Lowest average match: **{bot['Predicted_Field']}** (~**{bot['matching_score_num']:.1f}%**)
                """)

            else:
                st.info("Required columns not found for role-wise match plot.")

            st.subheader("📦 Resume Score by Experience Level")
            st.markdown("""
            This box plot compares resume score distributions across experience levels.
            It shows medians, spread, and outliers to understand how resume quality varies by seniority.
            """)

            if "resume_score" in filtered.columns and "User_level" in filtered.columns:
                fig_box = px.box(
                    filtered,
                    x="User_level",
                    y="resume_score",
                    points="all",
                    title="Resume Score Distribution Across Experience Levels",
                    labels={"User_level": "Experience Level", "resume_score": "Resume Score"},
                )
                st.plotly_chart(fig_box, use_container_width=True)
                med_by_level = filtered.groupby("User_level")["resume_score"].median().sort_values(ascending=False)
                best_level = med_by_level.index[0]
                best_val = med_by_level.iloc[0]
                st.markdown(f"""
                **Data-driven conclusion:**
                - Highest median resume score is for **{best_level}** (**{best_val:.1f}**).
                """)

            else:
                st.info("Required columns not found for box plot.")


    with tab3:
        st.subheader("Top 5 Missing Skills (All Users)")
        st.markdown("""
        This section aggregates missing skills across users by comparing `Recommended_skills` vs `Actual_skills`.
        It helps identify the most common training gaps.
        """)

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
            total_missing = len(all_missing)
            st.markdown(f"**Data-driven conclusion:** Total missing-skill instances found across users: **{total_missing}**.")


        else:
            st.info("Missing skill data not available.")

