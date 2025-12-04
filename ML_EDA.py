# ML_EDA.py

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = "user_data.csv"


# -----------------------------
# Data loading & feature engineering
# -----------------------------
@st.cache_data
def load_user_data(path: str = DATA_PATH) -> pd.DataFrame | None:
    """Load user_data.csv with simple error handling."""
    try:
        df = pd.read_csv(path)
        if df.empty:
            st.error("user_data.csv is empty. Add some records first.")
            return None
        return df
    except FileNotFoundError:
        st.error(f"Could not find {path}. Please upload user_data.csv first.")
        return None
    except Exception as e:
        st.error(f"Error while reading {path}: {e}")
        return None


def _compute_skill_count_col(
    df: pd.DataFrame, col_candidates: list[str], new_name: str
) -> pd.Series | None:
    """Helper to build a skill-count column from a text column with comma-separated skills."""
    for c in col_candidates:
        if c in df.columns:
            return (
                df[c]
                .fillna("")
                .astype(str)
                .apply(lambda x: sum(1 for s in x.split(",") if s.strip()))
                .rename(new_name)
            )
    return None


def engineer_features(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Create ML-friendly features from raw dataframe.

    Features we *try* to build (only if the underlying columns exist):
    - resume_score (numeric)
    - matching_score_num (numeric version of "85%" style matching_score)
    - actual_skills_count (how many skills we extracted from the resume)
    - recommended_skills_count (how many skills we recommended)
    - email provider dummies (gmail / yahoo / outlook / other)

    If some inputs are missing, we fall back to 'all numeric columns except targets'.
    """
    df = df_raw.copy()

    # --- Clean resume_score ---
    if "resume_score" in df.columns:
        df["resume_score"] = pd.to_numeric(df["resume_score"], errors="coerce")

    # --- matching_score -> matching_score_num (strip %) ---
    if "matching_score" in df.columns and "matching_score_num" not in df.columns:
        df["matching_score_num"] = (
            df["matching_score"]
            .astype(str)
            .str.strip()
            .str.rstrip("%")
            .replace("", np.nan)
        )
        df["matching_score_num"] = pd.to_numeric(
            df["matching_score_num"], errors="coerce"
        )

    # --- skill counts from possible columns ---
    actual_skills_series = _compute_skill_count_col(
        df,
        col_candidates=[
            "Actual_skills",
            "Actual Skills",
            "actual_skills",
            "actual_skills_extracted",
        ],
        new_name="actual_skills_count",
    )
    if actual_skills_series is not None:
        df["actual_skills_count"] = actual_skills_series

    recommended_skills_series = _compute_skill_count_col(
        df,
        col_candidates=[
            "Recommended_skills",
            "Recommended Skills",
            "recommended_skills",
        ],
        new_name="recommended_skills_count",
    )
    if recommended_skills_series is not None:
        df["recommended_skills_count"] = recommended_skills_series

    # --- email provider dummies (gmail / yahoo / outlook / other) ---
    email_provider_cols: list[str] = []
    if "Email" in df.columns:
        emails = df["Email"].astype(str).str.lower()
        df["email_provider"] = emails.str.extract(r"@([^>]+)", expand=False)
        df["email_provider"] = df["email_provider"].fillna("other")

        top_providers = ["gmail.com", "yahoo.com", "outlook.com"]
        for prov in top_providers:
            col_name = f"email_{prov.split('.')[0]}"
            df[col_name] = (df["email_provider"] == prov).astype(int)
            email_provider_cols.append(col_name)

        # All non-top providers grouped as "other"
        df["email_other"] = (~df["email_provider"].isin(top_providers)).astype(int)
        email_provider_cols.append("email_other")

    # Collect preferred feature columns if they exist
    preferred_features = [
        "resume_score",
        "matching_score_num",
        "actual_skills_count",
        "recommended_skills_count",
    ]
    feature_cols = [c for c in preferred_features if c in df.columns]
    feature_cols.extend(email_provider_cols)

    # Fallback: if we still have nothing, use all numeric columns except obvious targets / IDs
    if not feature_cols:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {"Page_no", "User_level", "Predicted_Field"}
        feature_cols = [c for c in numeric_cols if c not in exclude]

    return df, feature_cols


# -----------------------------
# EDA section
# -----------------------------
def render_eda_section(df: pd.DataFrame):
    st.header("🔍 Exploratory Data Analysis on user_data.csv")

    st.markdown(
        """
        This section helps us understand **who is using the system** and **how their resumes look**
        before we ever touch machine learning.

        We focus on:
        - Typical ranges of scores and counts (summary table)
        - How key metrics evolve over time (multi-line timeline)
        - How many skills are usually detected per resume
        - Which roles and experience levels are most common
        - How the numeric features relate to each other (correlation heatmap)
        """
    )

    # ---- Basic numeric summary ----
    st.subheader("📋 Basic Summary of Numeric Features")
    st.markdown(
        """
        This table shows basic statistics for all numeric features in the dataset:
        - **Count** of records
        - **Typical values** (mean, median via percentiles)
        - **Spread** of the data (standard deviation, min, max)

        We use this to quickly catch impossible values, extreme outliers,
        or features that barely vary across users.
        """
    )

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Drop Page_no here as well, since it's basically constant and not informative
    if "Page_no" in numeric_cols:
        numeric_cols.remove("Page_no")

    if numeric_cols:
        desc = df[numeric_cols].describe().T
        st.dataframe(desc)

        # --- Data-driven conclusion for summary ---
        conclusions = []
        if "resume_score" in desc.index:
            rs_row = desc.loc["resume_score"]
            conclusions.append(
                f"Typical resume scores are around **{rs_row['mean']:.1f}** "
                f"(median ≈ {rs_row['50%']:.1f}), ranging from {rs_row['min']:.1f} to {rs_row['max']:.1f}."
            )
        if "matching_score_num" in desc.index:
            ms_row = desc.loc["matching_score_num"]
            conclusions.append(
                f"Matching scores center around **{ms_row['mean']:.1f}%**, "
                f"with most users between roughly {ms_row['25%']:.1f}% and {ms_row['75%']:.1f}%."
            )
        if "actual_skills_count" in desc.index:
            sc_row = desc.loc["actual_skills_count"]
            conclusions.append(
                f"On average, resumes expose **{sc_row['mean']:.1f} skills**, "
                f"with most users between {sc_row['25%']:.1f} and {sc_row['75%']:.1f} skills."
            )

        if conclusions:
            st.markdown(
                "**Data-driven conclusion:**<br>"
                + "<br>".join(f"- {c}" for c in conclusions),
                unsafe_allow_html=True,
            )
    else:
        st.info("No numeric columns available for summary stats.")
    st.markdown("---")

    # ---- 1. Multi-line timeline: Resume Score + Matching Score + Skill Count ----
    if "Timestamp" in df.columns:
        st.subheader(
            "📈 Trends Over Time – Resume Score, Matching Score, Skill Count (Daily Avg)"
        )

        st.markdown(
            """
            This chart shows how **three key metrics** change over time:

            - `resume_score` – overall quality of the resume
            - `matching_score_num` – how well the resume matches the selected role
            - `actual_skills_count` – how many skills are actually detected from the resume

            For each day, we compute the **average** of each metric and plot them together.
            """
        )

        df_ts = df.copy()
        df_ts["Timestamp"] = pd.to_datetime(df_ts["Timestamp"], errors="coerce")
        df_ts = df_ts.dropna(subset=["Timestamp"])

        if not df_ts.empty:
            df_ts["day_label"] = df_ts["Timestamp"].dt.strftime("%b %d")

            metric_candidates = [
                "resume_score",
                "matching_score_num",
                "actual_skills_count",
            ]
            metrics = [m for m in metric_candidates if m in df_ts.columns]

            if metrics:
                daily = df_ts.groupby("day_label", as_index=False)[metrics].mean()

                long_df = daily.melt(
                    id_vars="day_label",
                    value_vars=metrics,
                    var_name="Metric",
                    value_name="Value",
                )

                fig_multi = px.line(
                    long_df,
                    x="day_label",
                    y="Value",
                    color="Metric",
                    markers=True,
                    labels={"day_label": "Day", "Value": "Average Value"},
                )
                fig_multi.update_layout(xaxis={"type": "category"})
                st.plotly_chart(fig_multi, use_container_width=True)

                # --- Data-driven conclusion for timeline ---
                trend_lines = []
                for m in metrics:
                    if len(daily) >= 2:
                        start = daily[m].iloc[0]
                        end = daily[m].iloc[-1]
                        diff = end - start
                        if abs(diff) < 0.5:
                            direction = "roughly **stable**"
                        elif diff > 0:
                            direction = f"**increasing** by about {diff:.1f} points"
                        else:
                            direction = f"**decreasing** by about {abs(diff):.1f} points"
                        trend_lines.append(f"- `{m}` is {direction} from the first to the latest day.")
                if trend_lines:
                    st.markdown(
                        "**Data-driven conclusion:**<br>"
                        + "<br>".join(trend_lines),
                        unsafe_allow_html=True,
                    )
            else:
                st.info(
                    "No metric columns (resume_score / matching_score_num / actual_skills_count) "
                    "available to plot over time."
                )
        else:
            st.info("Not enough valid timestamp data to plot trends over time.")
    else:
        st.info("Timestamp column not found, so time trends cannot be plotted.")

    st.markdown("---")

    # ---- 2. Distribution of Extracted Skill Counts (violin / box) ----
    skill_count_col = None
    for candidate in ["actual_skills_count", "Actual_skills_count", "Extracted_skill_count"]:
        if candidate in df.columns:
            skill_count_col = candidate
            break

    if skill_count_col:
        st.subheader("🧠 Distribution of Extracted Skill Counts")

        st.markdown(
            f"""
            This chart shows how many skills are **detected per resume** based on `{skill_count_col}`.

            - Wider sections of the violin indicate more resumes with that skill count.
            - Long tails indicate some resumes have **very few** or **very many** skills.
            """
        )

        fig_violin = px.violin(
            df,
            y=skill_count_col,
            box=True,
            points="all",
            labels={skill_count_col: "Number of Extracted Skills"},
        )
        st.plotly_chart(fig_violin, use_container_width=True)

        # --- Data-driven conclusion for skills distribution ---
        sc = df[skill_count_col].dropna()
        if not sc.empty:
            q1, q3 = sc.quantile([0.25, 0.75])
            st.markdown(
                f"**Data-driven conclusion:**<br>"
                f"- Most resumes expose between **{q1:.0f} and {q3:.0f} skills**.<br>"
                f"- The minimum is {sc.min():.0f} and the maximum is {sc.max():.0f}, "
                f"showing {'a wide' if sc.max()-sc.min()>5 else 'a fairly tight'} spread in skill density.",
                unsafe_allow_html=True,
            )
    else:
        st.info(
            "No skill-count column found (e.g., actual_skills_count) for the distribution chart."
        )

    st.markdown("---")

    # ---- 3. Role Distribution (Predicted_Field) ----
    if "Predicted_Field" in df.columns:
        st.subheader("📌 Role Distribution (Predicted_Field)")

        st.markdown(
            """
            This bar chart shows how many users are mapped to each **predicted job role**
            (e.g., Data Scientist, Data Analyst, ML Engineer).
            """
        )

        role_counts = df["Predicted_Field"].value_counts().reset_index()
        role_counts.columns = ["Predicted_Field", "Count"]

        fig_role = px.bar(
            role_counts,
            x="Predicted_Field",
            y="Count",
            labels={"Predicted_Field": "Predicted Role", "Count": "Number of Users"},
        )
        fig_role.update_layout(xaxis={"categoryorder": "total descending"})
        st.plotly_chart(fig_role, use_container_width=True)

        # --- Data-driven conclusion for role distribution ---
        total_users = role_counts["Count"].sum()
        top_role = role_counts.iloc[0]
        pct = 100 * top_role["Count"] / total_users if total_users > 0 else 0
        st.markdown(
            f"**Data-driven conclusion:**<br>"
            f"- The most common predicted role is **{top_role['Predicted_Field']}**, "
            f"covering about **{pct:.1f}%** of users.<br>"
            f"- This indicates that most users are currently targeting this role or have skills aligned to it.",
            unsafe_allow_html=True,
        )
    else:
        st.info("Column 'Predicted_Field' not found, so role distribution cannot be shown.")

    st.markdown("---")

    # ---- 4. Experience vs Role (User_level x Predicted_Field) ----
    if "Predicted_Field" in df.columns and "User_level" in df.columns:
        st.subheader("📊 Experience Level vs Predicted Role")

        st.markdown(
            """
            This grouped bar chart compares **experience level** and **predicted role**.
            """
        )

        cross_tab = (
            df.groupby(["Predicted_Field", "User_level"])
            .size()
            .reset_index(name="Count")
        )

        fig_exp_role = px.bar(
            cross_tab,
            x="Predicted_Field",
            y="Count",
            color="User_level",
            barmode="group",
            labels={
                "Predicted_Field": "Predicted Role",
                "User_level": "Experience Level",
                "Count": "Number of Users",
            },
        )
        st.plotly_chart(fig_exp_role, use_container_width=True)

        # --- Data-driven conclusion for experience vs role ---
        lvl_counts = df["User_level"].value_counts()
        main_level = lvl_counts.idxmax()
        main_pct = 100 * lvl_counts.max() / lvl_counts.sum()
        st.markdown(
            f"**Data-driven conclusion:**<br>"
            f"- Overall, **{main_level}** is the most common experience level "
            f"(about **{main_pct:.1f}%** of users).<br>"
            f"- Any roles where Beginners dominate may need clearer skill thresholds, "
            f"while roles dominated by senior levels may indicate more advanced users.",
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "Columns 'Predicted_Field' and/or 'User_level' not found for experience vs role chart."
        )

    st.markdown("---")

    # ---- 5. Correlation Heatmap ----
    st.subheader("🧬 Correlation Between Numeric Features")

    st.markdown(
        """
        This heatmap shows the **correlation** between numeric variables such as scores,
        skill counts, and other numeric features.
        """
    )

    corr_df = df.select_dtypes(include=[np.number]).copy()

    # Drop obviously useless ones if present
    for col in ["Page_no"]:
        if col in corr_df.columns:
            corr_df = corr_df.drop(columns=[col])

    # Drop columns that are entirely NaN
    corr_df = corr_df.dropna(axis=1, how="all")

    if corr_df.shape[1] >= 2:
        corr = corr_df.corr(numeric_only=True)

        fig_corr = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            labels={"x": "Features", "y": "Features", "color": "Correlation"},
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        # --- Data-driven conclusion for correlations ---
        msgs = []
        def corr_msg(a: str, b: str):
            if a in corr.index and b in corr.columns:
                val = corr.loc[a, b]
                strength = "weak"
                if abs(val) >= 0.7:
                    strength = "strong"
                elif abs(val) >= 0.4:
                    strength = "moderate"
                direction = "positive" if val >= 0 else "negative"
                return f"- `{a}` and `{b}` have a **{strength} {direction} correlation** (~{val:.2f})."
            return None

        for pair in [
            ("resume_score", "matching_score_num"),
            ("resume_score", "actual_skills_count"),
            ("matching_score_num", "actual_skills_count"),
        ]:
            msg = corr_msg(*pair)
            if msg:
                msgs.append(msg)

        if msgs:
            st.markdown(
                "**Data-driven conclusion:**<br>" + "<br>".join(msgs),
                unsafe_allow_html=True,
            )
    else:
        st.info("Not enough numeric columns to compute a useful correlation heatmap.")


# -----------------------------
# ML models section
# -----------------------------
def train_and_evaluate_models(
    df_ml: pd.DataFrame, feature_cols: list[str], target_col: str
):
    """
    Train Logistic Regression and Random Forest models and display evaluation metrics.
    """
    st.markdown(
        f"""
        ### 🔧 Training ML Models (Target = `{target_col}`)

        We use the engineered features as **inputs (X)** and `{target_col}` as the **label (y)**.

        Two models are trained and compared:
        - **Logistic Regression** – a simple linear baseline model.
        - **Random Forest** – a tree-based ensemble model that can capture non-linear patterns.

        We split the data into **train** and **test** sets,
        train on the train set, and evaluate on the unseen test set.
        """
    )

    data = df_ml.dropna(subset=feature_cols + [target_col]).copy()

    if data.empty:
        st.error("No valid rows after dropping NaNs for selected features and target.")
        return

    X = data[feature_cols]
    y = data[target_col]

    # Encode target if it's categorical text
    if y.dtype == "object":
        y = y.astype("category")
        class_names = list(y.cat.categories)  # full list of possible classes
        y_encoded = y.cat.codes  # 0,1,2,...
    else:
        y_encoded = y
        class_names = sorted(list(np.unique(y_encoded)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # ---- Logistic Regression ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000, multi_class="auto")
    log_reg.fit(X_train_scaled, y_train)

    y_pred_lr = log_reg.predict(X_test_scaled)
    acc_lr = (y_pred_lr == y_test).mean()

    # ---- Random Forest ----
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    acc_rf = (y_pred_rf == y_test).mean()

    # ---- Show accuracy comparison ----
    st.markdown("#### 📊 Model Accuracy Comparison")
    acc_df = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Random Forest"],
            "Accuracy (test set)": [round(acc_lr, 3), round(acc_rf, 3)],
        }
    )
    st.dataframe(acc_df, use_container_width=True)

    better_model = "Random Forest" if acc_rf >= acc_lr else "Logistic Regression"

    st.markdown(
        f"""
        **Data-driven conclusion:**
        - Logistic Regression accuracy: **{acc_lr:.3f}**
        - Random Forest accuracy: **{acc_rf:.3f}**
        - Based on this dataset, the better overall model is **{better_model}**.
        """
    )

    # ---- Prepare labels/target names (avoid mismatch bug) ----
    present_classes = np.unique(y_test)  # only classes that actually appear in test set

    if class_names:
        # Map encoded int -> original category name, but only for present classes
        present_names = [str(class_names[i]) for i in present_classes]
    else:
        present_names = [str(c) for c in present_classes]

    # ---- Logistic Regression report (tabulated) ----
    st.markdown("#### 📄 Detailed Classification Report – Logistic Regression")

    lr_report_dict = classification_report(
        y_test,
        y_pred_lr,
        labels=present_classes,
        target_names=present_names,
        zero_division=0,
        output_dict=True,
    )
    df_lr_report = (
        pd.DataFrame(lr_report_dict)
        .T.reset_index()
        .rename(
            columns={
                "index": "Class",
                "precision": "Precision",
                "recall": "Recall",
                "f1-score": "F1-score",
                "support": "Support",
            }
        )
    )
    st.dataframe(df_lr_report, use_container_width=True)

    # data-driven text for LR
    try:
        lr_macro_f1 = float(
            df_lr_report.loc[df_lr_report["Class"] == "macro avg", "F1-score"].iloc[0]
        )
    except Exception:
        lr_macro_f1 = np.nan
    worst_lr_row = (
        df_lr_report[df_lr_report["Class"].isin(present_names)]
        .sort_values("F1-score")
        .iloc[0]
        if not df_lr_report.empty
        else None
    )

    lr_conclusions = []
    if not np.isnan(lr_macro_f1):
        lr_conclusions.append(
            f"- Overall macro F1 for Logistic Regression is **{lr_macro_f1:.2f}**, "
            f"which summarizes balance across all classes."
        )
    if worst_lr_row is not None:
        lr_conclusions.append(
            f"- The weakest class for Logistic Regression is **{worst_lr_row['Class']}** "
            f"(F1 ≈ {float(worst_lr_row['F1-score']):.2f}); this class is harder to predict."
        )
    if lr_conclusions:
        st.markdown(
            "**Data-driven conclusion (Logistic Regression):**<br>"
            + "<br>".join(lr_conclusions),
            unsafe_allow_html=True,
        )

    # ---- Random Forest report (tabulated) ----
    st.markdown("#### 📄 Detailed Classification Report – Random Forest")

    rf_report_dict = classification_report(
        y_test,
        y_pred_rf,
        labels=present_classes,
        target_names=present_names,
        zero_division=0,
        output_dict=True,
    )
    df_rf_report = (
        pd.DataFrame(rf_report_dict)
        .T.reset_index()
        .rename(
            columns={
                "index": "Class",
                "precision": "Precision",
                "recall": "Recall",
                "f1-score": "F1-score",
                "support": "Support",
            }
        )
    )
    st.dataframe(df_rf_report, use_container_width=True)

    # data-driven text for RF
    try:
        rf_macro_f1 = float(
            df_rf_report.loc[df_rf_report["Class"] == "macro avg", "F1-score"].iloc[0]
        )
    except Exception:
        rf_macro_f1 = np.nan
    worst_rf_row = (
        df_rf_report[df_rf_report["Class"].isin(present_names)]
        .sort_values("F1-score")
        .iloc[0]
        if not df_rf_report.empty
        else None
    )

    rf_conclusions = []
    if not np.isnan(rf_macro_f1):
        rf_conclusions.append(
            f"- Overall macro F1 for Random Forest is **{rf_macro_f1:.2f}**."
        )
    if worst_rf_row is not None:
        rf_conclusions.append(
            f"- The weakest class for Random Forest is **{worst_rf_row['Class']}** "
            f"(F1 ≈ {float(worst_rf_row['F1-score']):.2f})."
        )
    if not np.isnan(lr_macro_f1) and not np.isnan(rf_macro_f1):
        better = "Random Forest" if rf_macro_f1 >= lr_macro_f1 else "Logistic Regression"
        rf_conclusions.append(
            f"- Comparing macro F1, **{better}** handles class balance better on this dataset."
        )
    if rf_conclusions:
        st.markdown(
            "**Data-driven conclusion (Random Forest):**<br>"
            + "<br>".join(rf_conclusions),
            unsafe_allow_html=True,
        )

    # ---- Confusion Matrix for the better model ----
    better_model_name = "Random Forest" if acc_rf >= acc_lr else "Logistic Regression"
    st.markdown(f"#### 🔀 Confusion Matrix – Best Model ({better_model_name})")

    if better_model_name == "Random Forest":
        cm = confusion_matrix(y_test, y_pred_rf, labels=present_classes)
    else:
        cm = confusion_matrix(y_test, y_pred_lr, labels=present_classes)

    cm_df = pd.DataFrame(
        cm,
        index=present_names,
        columns=present_names,
    )

    fig_cm = px.imshow(
        cm_df,
        text_auto=True,
        labels={"x": "Predicted label", "y": "True label", "color": "Count"},
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    total = cm.sum()
    correct = np.trace(cm)
    acc_cm = correct / total if total > 0 else 0.0

    st.markdown(
        f"""
        **Data-driven conclusion (Confusion Matrix):**
        - The best model correctly classifies about **{acc_cm:.2%}** of examples (diagonal entries).
        - Any large off-diagonal cells indicate systematic confusions between specific classes
          that may need more data or clearer feature separation.
        """
    )

    # ---- Feature importances from Random Forest ----
    st.markdown("#### 🧩 Feature Importance (Random Forest)")

    importances = rf.feature_importances_
    imp_df = pd.DataFrame(
        {"Feature": feature_cols, "Importance": importances}
    ).sort_values("Importance", ascending=False)

    fig_imp = px.bar(
        imp_df,
        x="Feature",
        y="Importance",
        labels={"Importance": "Relative Importance"},
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    top_feats = imp_df.head(3)
    feat_lines = [
        f"- **{row['Feature']}** (importance ~ {row['Importance']:.2f})"
        for _, row in top_feats.iterrows()
    ]

    st.markdown(
        "**Data-driven conclusion (Feature Importance):**<br>"
        + "<br>".join(feat_lines)
        + "<br>- These features drive most of the model's decisions; "
          "features with near-zero importance could be dropped without much impact.",
        unsafe_allow_html=True,
    )


def render_ml_section(df_ml: pd.DataFrame, feature_cols: list[str]):
    st.header("🤖 ML Models – Predicting User Level / Role")

    st.markdown(
        """
        In this section, we turn the cleaned features into **predictions**.

        Goal:
        - Use scores, skill counts, and other features to predict either:
          - the user's **experience level** (`User_level`), or
          - the **predicted job role** (`Predicted_Field`).
        """
    )

    # Choose which target to predict
    possible_targets: list[str] = []

    if "User_level" in df_ml.columns:
        possible_targets.append("User_level")
    if "Predicted_Field" in df_ml.columns:
        possible_targets.append("Predicted_Field")

    if not possible_targets:
        st.error("No suitable target columns (User_level / Predicted_Field) found.")
        return

    st.markdown(
        """
        **Step 1 – Choose the Target**

        This decides what the ML model is trying to learn:
        - `User_level`: classify users as Beginner / Intermediate / Expert (or similar).
        - `Predicted_Field`: classify users into roles like Data Scientist, Data Analyst, etc.
        """
    )
    target_col = st.selectbox("Target variable", options=possible_targets)

    st.markdown(
        """
        **Step 2 – Features Going into the Model**

        These are the columns we give to the model as inputs (`X`). 
        They might include:
        - Resume score and matching score
        - How many skills were detected / recommended
        - Email provider flags (gmail / yahoo / outlook / other)
        """
    )
    st.write(feature_cols)

    st.markdown(
        """
        **Step 3 – Train & Evaluate**

        When you click the button below:
        1. The data is split into **train** and **test** sets.
        2. Two models (Logistic Regression & Random Forest) are trained on the train set.
        3. We evaluate them on the unseen test set and compare:
           - Overall accuracy
           - Per-class precision/recall/F1
           - Confusion matrix
           - Feature importances (Random Forest)
        """
    )

    if st.button("🚀 Train & Evaluate Models"):
        train_and_evaluate_models(df_ml, feature_cols, target_col)


# -----------------------------
# Main entry for the ML + EDA page
# -----------------------------
def render_ml_eda_page():
    """Main entry for the ML + EDA page."""
    st.title("📊 ML & EDA – Learning from user_data.csv")

    df_raw = load_user_data()
    if df_raw is None:
        return

    df_ml, feature_cols = engineer_features(df_raw)

    tab_eda, tab_ml = st.tabs(["📊 EDA", "🤖 ML Models"])

    # Use engineered df for EDA so we get matching_score_num, skill counts, etc.
    with tab_eda:
        render_eda_section(df_ml)

    with tab_ml:
        render_ml_section(df_ml, feature_cols)
