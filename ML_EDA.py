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


def load_user_data():
    """Load user_data.csv and do basic cleaning / type handling."""
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error("user_data.csv not found. Run some resume analyses first.")
        return None
    except Exception as e:
        st.error(f"Error reading {DATA_PATH}: {e}")
        return None

    if df.empty:
        st.warning("user_data.csv is empty. No analyses to learn from yet.")
        return None

    # Clean numeric fields
    if "resume_score" in df.columns:
        df["resume_score"] = pd.to_numeric(df["resume_score"], errors="coerce")

    if "matching_score" in df.columns:
        # Strip "%" and convert to float
        df["matching_score_num"] = (
            df["matching_score"].astype(str).str.replace("%", "", regex=False)
        )
        df["matching_score_num"] = pd.to_numeric(
            df["matching_score_num"], errors="coerce"
        )

    return df


def engineer_features(df: pd.DataFrame):
    """
    Add engineered features for ML:
    - actual_skills_count
    - recommended_skills_count
    - email provider one-hot
    """
    df = df.copy()

    # Count skills from list-like columns stored as strings
    def count_list_like(x):
        if pd.isna(x):
            return 0
        s = str(x).strip()
        # handle things like "['Python', 'SQL']" or "Python, SQL"
        s = s.strip("[]")
        if not s:
            return 0
        return len([t for t in s.split(",") if t.strip()])

    if "Actual_skills" in df.columns:
        df["actual_skills_count"] = df["Actual_skills"].apply(count_list_like)
    else:
        df["actual_skills_count"] = 0

    if "Recommended_skills" in df.columns:
        df["recommended_skills_count"] = df["Recommended_skills"].apply(count_list_like)
    else:
        df["recommended_skills_count"] = 0

    # Email provider (gmail, yahoo, edu, etc.) – basic feature
    if "Email_ID" in df.columns:
        df["email_provider"] = (
            df["Email_ID"]
            .astype(str)
            .str.extract(r"@([\w\.-]+)", expand=False)
            .str.lower()
        )
        # One-hot encode top providers, lump others as "other"
        top_providers = df["email_provider"].value_counts().nlargest(3).index.tolist()
        df["email_provider"] = df["email_provider"].apply(
            lambda x: x if x in top_providers else "other"
        )
        provider_dummies = pd.get_dummies(df["email_provider"], prefix="email")
        df = pd.concat([df, provider_dummies], axis=1)
    else:
        provider_dummies = pd.DataFrame()

    # Numeric features for ML
    feature_cols = [
        "resume_score",
        "matching_score_num",
        "actual_skills_count",
        "recommended_skills_count",
    ]

    # Add any email_* dummy columns to feature list
    feature_cols.extend([c for c in df.columns if c.startswith("email_")])

    # Drop rows with missing core numeric features
    df_ml = df.dropna(subset=["resume_score", "matching_score_num"]).copy()

    return df_ml, feature_cols


def render_eda_section(df: pd.DataFrame):
    st.header("Exploratory Data Analysis on user_data.csv")

    # ---- Basic numeric summary ----
    st.subheader("Basic Summary of Numeric Features")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.dataframe(df[numeric_cols].describe().T)
    else:
        st.info("No numeric columns available for summary stats.")
    st.markdown("---")

    # ---- 1. Average Resume Score per Day (line chart, 'Dec 01' style) ----
    if "Timestamp" in df.columns and "resume_score" in df.columns:
        st.subheader("Average Resume Score per Day")

        temp = df.copy()
        temp["Timestamp"] = pd.to_datetime(temp["Timestamp"], errors="coerce")
        temp = temp.dropna(subset=["Timestamp"])

        if not temp.empty:
            # Format as 'Dec 01', 'Dec 02', etc.
            temp["Date"] = temp["Timestamp"].dt.strftime("%b %d")
            daily_scores = (
                temp.groupby("Date")["resume_score"]
                .mean()
                .reset_index()
            )

            fig = px.line(
                daily_scores,
                x="Date",
                y="resume_score",
                markers=True,
                labels={"Date": "Day", "resume_score": "Avg Resume Score"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid timestamps available to compute daily scores.")

    st.markdown("---")

    # ---- 2. Violin plot of Extracted Skill Counts (no bar chart) ----
    if "actual_skills_count" in df.columns:
        st.subheader("Distribution of Extracted Skill Counts")
        # Single-category violin + jitter
        fig = px.violin(
            df,
            y="actual_skills_count",
            box=True,
            points="all",
            labels={"actual_skills_count": "Number of Extracted Skills"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---- 3. Role Distribution (Predicted_Field) – bar chart (one place only) ----
    if "Predicted_Field" in df.columns:
        st.subheader("Role Distribution (Predicted_Field)")
        role_counts = (
            df["Predicted_Field"]
            .value_counts()
            .reset_index()
        )
        # Force column names to be consistent
        role_counts.columns = ["Role", "Count"]

        fig = px.bar(
            role_counts,
            x="Role",
            y="Count",
            labels={"Role": "Role", "Count": "User Count"},
        )
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---- 4. Role × Experience Level – grouped bar instead of heatmap ----
    if "Predicted_Field" in df.columns and "User_level" in df.columns:
        st.subheader("Role vs Experience Level (Grouped Bar)")

        role_level_counts = (
            df.groupby(["Predicted_Field", "User_level"])
              .size()
              .reset_index(name="count")
        )

        if not role_level_counts.empty:
            fig = px.bar(
                role_level_counts,
                x="Predicted_Field",
                y="count",
                color="User_level",
                barmode="group",
                labels={
                    "Predicted_Field": "Role",
                    "count": "User Count",
                    "User_level": "Experience Level",
                },
            )
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---- 5. Correlation Heatmap (drop Page_no / constant cols) ----
    numeric_df = df.select_dtypes(include=[np.number]).copy()

    # Explicitly drop page-no like columns if present
    for col in ["Page_no", "page_no", "Pages", "pages"]:
        if col in numeric_df.columns:
            numeric_df.drop(columns=[col], inplace=True, errors="ignore")

    # Drop columns that are constant (no variation)
    constant_cols = [c for c in numeric_df.columns if numeric_df[c].nunique() <= 1]
    numeric_df.drop(columns=constant_cols, inplace=True, errors="ignore")

    if numeric_df.shape[1] >= 2:
        st.subheader("Correlation Heatmap (Numeric Features)")
        corr = numeric_df.corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            origin="lower",
        )
        st.plotly_chart(fig, use_container_width=True)


def train_and_evaluate_models(df: pd.DataFrame, feature_cols, target_col: str):
    """
    Train Logistic Regression and RandomForest on selected target_col.
    Show metrics, comparison, confusion matrix, and feature importance.
    """
    df = df.copy()
    df = df.dropna(subset=[target_col])

    # Filter out rare classes (only 1 sample) to avoid stratify errors
    class_counts = df[target_col].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df[target_col].isin(valid_classes)]

    if df[target_col].nunique() < 2:
        st.warning(
            f"Not enough classes in {target_col} to train a model. "
            f"Need at least 2 classes with ≥ 2 samples each."
        )
        return
    
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df[target_col].astype(str)

    if len(df) < 20:
        st.warning(
            f"Only {len(df)} usable rows for ML. "
            f"Models may not be reliable but will still be trained for demonstration."
        )

    test_size = 0.2 if len(df) >= 10 else 0.3

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=y if y.nunique() > 1 else None,
        )
    except ValueError as e:
        st.error(f"Train/test split failed: {e}")
        return

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model 1: Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, multi_class="auto")
    log_reg.fit(X_train_scaled, y_train)
    y_pred_lr = log_reg.predict(X_test_scaled)

    # Model 2: Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # Metrics
    st.subheader(f"Model Evaluation – Target: {target_col}")

    st.markdown("### Logistic Regression (baseline)")
    st.text(classification_report(y_test, y_pred_lr))

    st.markdown("### Random Forest (non-linear model)")
    st.text(classification_report(y_test, y_pred_rf))

    # Comparison table (accuracy only, for quick view)
    acc_lr = (y_pred_lr == y_test).mean()
    acc_rf = (y_pred_rf == y_test).mean()

    comp_df = pd.DataFrame(
        {
            "Model": ["Logistic Regression", "Random Forest"],
            "Accuracy": [acc_lr, acc_rf],
        }
    )
    st.markdown("### Model Comparison (Accuracy)")
    st.dataframe(comp_df)

    # Confusion matrix for best model
    best_model_name = "Random Forest" if acc_rf >= acc_lr else "Logistic Regression"
    best_pred = y_pred_rf if best_model_name == "Random Forest" else y_pred_lr

    st.markdown(f"### Confusion Matrix – {best_model_name}")
    labels = sorted(y_test.unique().tolist())
    cm = confusion_matrix(y_test, best_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    fig = px.imshow(
        cm_df,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual", color="Count"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Feature importance for RF
    st.markdown("### Random Forest Feature Importance")
    importances = rf.feature_importances_
    fi_df = pd.DataFrame(
        {"feature": feature_cols, "importance": importances}
    ).sort_values("importance", ascending=False)

    fig_imp = px.bar(
        fi_df,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance (Random Forest)",
    )
    st.plotly_chart(fig_imp, use_container_width=True)


def render_ml_eda_page():
    """Main entry for the ML + EDA page."""
    st.title("ML & EDA – Learning from user_data.csv")

    df_raw = load_user_data()
    if df_raw is None:
        return

    df_ml, feature_cols = engineer_features(df_raw)

    tab_eda, tab_ml = st.tabs(["📊 EDA", "🤖 ML Models"])

    with tab_eda:
        render_eda_section(df_ml)

    with tab_ml:
        st.subheader("Train ML Models on Aggregated User Data")

        st.write(
            "Choose what you want to predict. "
            "This satisfies the final project requirement for model development & evaluation."
        )

        possible_targets = []
        if "User_level" in df_ml.columns:
            possible_targets.append("User_level")
        if "Predicted_Field" in df_ml.columns:
            possible_targets.append("Predicted_Field")

        if not possible_targets:
            st.error("No suitable target columns (User_level / Predicted_Field) found.")
            return

        target_col = st.selectbox("Target variable", options=possible_targets)

        st.markdown("**Features used for modeling:**")
        st.write(feature_cols)

        if st.button("Train & Evaluate Models"):
            train_and_evaluate_models(df_ml, feature_cols, target_col)
