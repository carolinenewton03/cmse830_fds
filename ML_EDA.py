import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.model_selection import GridSearchCV
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
    - missing_skills_count + skill_coverage_ratio (advanced FE)
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

    # --- advanced FE: skill gap features ---
    if "actual_skills_count" in df.columns and "recommended_skills_count" in df.columns:
        df["missing_skills_count"] = (
            df["recommended_skills_count"] - df["actual_skills_count"]
        ).clip(lower=0)

        df["skill_coverage_ratio"] = (
            df["actual_skills_count"]
            / df["recommended_skills_count"].replace(0, np.nan)
        )

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

        df["email_other"] = (~df["email_provider"].isin(top_providers)).astype(int)
        email_provider_cols.append("email_other")

    preferred_features = [
        "resume_score",
        "matching_score_num",
        "actual_skills_count",
        "recommended_skills_count",
        "missing_skills_count",
        "skill_coverage_ratio",
    ]

    feature_cols = [c for c in preferred_features if c in df.columns]
    feature_cols.extend(email_provider_cols)

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

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "Page_no" in numeric_cols:
        numeric_cols.remove("Page_no")

    # ---- Summary table ----
    st.subheader("📋 Basic Summary of Numeric Features")
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        st.dataframe(desc, use_container_width=True)

        if "resume_score" in desc.index:
            med = desc.loc["resume_score", "50%"]
            q25 = desc.loc["resume_score", "25%"]
            q75 = desc.loc["resume_score", "75%"]
            st.markdown(
                f"**Data-driven conclusion:** Typical `resume_score` is around **{med:.1f}** "
                f"(IQR **{q25:.1f}–{q75:.1f}**)."
            )

    st.markdown("---")

    # ---- Timeline ----
    if "Timestamp" in df.columns:
        st.subheader("📈 Trends Over Time")

        df_ts = df.copy()
        df_ts["Timestamp"] = pd.to_datetime(df_ts["Timestamp"], errors="coerce")
        df_ts = df_ts.dropna(subset=["Timestamp"])

        if not df_ts.empty:
            df_ts["day"] = df_ts["Timestamp"].dt.date
            metrics = [c for c in ["resume_score", "matching_score_num", "actual_skills_count"] if c in df_ts.columns]

            daily = df_ts.groupby("day")[metrics].mean().reset_index()
            fig = px.line(daily, x="day", y=metrics, markers=True)
            st.plotly_chart(fig, use_container_width=True)

            if len(daily) >= 2 and all(m in daily.columns for m in metrics):
                rs_diff = daily["resume_score"].iloc[-1] - daily["resume_score"].iloc[0] if "resume_score" in daily else np.nan
                ms_diff = daily["matching_score_num"].iloc[-1] - daily["matching_score_num"].iloc[0] if "matching_score_num" in daily else np.nan
                sc_diff = daily["actual_skills_count"].iloc[-1] - daily["actual_skills_count"].iloc[0] if "actual_skills_count" in daily else np.nan

                parts = []
                if not np.isnan(rs_diff):
                    parts.append(f"`resume_score` Δ **{rs_diff:.1f}**")
                if not np.isnan(ms_diff):
                    parts.append(f"`matching_score_num` Δ **{ms_diff:.1f}**")
                if not np.isnan(sc_diff):
                    parts.append(f"`actual_skills_count` Δ **{sc_diff:.1f}**")
                st.markdown("**Data-driven conclusion:** " + ", ".join(parts) + " (first → last day).")

    st.markdown("---")

    # ---- Skill count distribution ----
    if "actual_skills_count" in df.columns:
        st.subheader("🧠 Distribution of Extracted Skill Counts")
        fig = px.violin(df, y="actual_skills_count", box=True, points="all")
        st.plotly_chart(fig, use_container_width=True)

        q1, q3 = df["actual_skills_count"].quantile([0.25, 0.75])
        st.markdown(
            f"**Data-driven conclusion:** Most resumes have `actual_skills_count` between **{q1:.0f}–{q3:.0f}** "
            f"(tails indicate very few or very many extracted skills)."
        )

    st.markdown("---")

    # ---- Role distribution ----
    if "Predicted_Field" in df.columns:
        st.subheader("📌 Role Distribution")
        role_counts = df["Predicted_Field"].value_counts().reset_index()
        role_counts.columns = ["Role", "Count"]

        fig = px.bar(role_counts, x="Role", y="Count")
        st.plotly_chart(fig, use_container_width=True)

        top = role_counts.iloc[0]
        pct = 100 * top["Count"] / role_counts["Count"].sum()
        st.markdown(
            f"**Data-driven conclusion:** Most users align with `Predicted_Field = {top['Role']}` "
            f"(~**{pct:.1f}%** of records)."
        )

    st.markdown("---")

    # ---- Experience vs role ----
    if {"User_level", "Predicted_Field"}.issubset(df.columns):
        st.subheader("📊 Experience Level vs Role")

        cross = df.groupby(["Predicted_Field", "User_level"]).size().reset_index(name="Count")
        fig = px.bar(cross, x="Predicted_Field", y="Count", color="User_level", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

        dominant = df["User_level"].value_counts().idxmax()
        st.markdown(
            f"**Data-driven conclusion:** `User_level = {dominant}` is the most common experience group in the dataset."
        )

    st.markdown("---")

    # ---- Correlation heatmap ----
    st.subheader("🧬 Correlation Heatmap")
    corr_df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

    if corr_df.shape[1] >= 2:
        corr = corr_df.corr()
        fig = px.imshow(corr, text_auto=True)
        st.plotly_chart(fig, use_container_width=True)

        if "resume_score" in corr.columns and "matching_score_num" in corr.columns:
            val = corr.loc["resume_score", "matching_score_num"]
            st.markdown(
                f"**Data-driven conclusion:** `resume_score` and `matching_score_num` correlation ≈ **{val:.2f}**."
            )


# -----------------------------
# ML models section
# -----------------------------
@st.cache_resource
def fit_models_cached(X_train, y_train, feature_cols_tuple, use_gridsearch: bool):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    return scaler, lr, rf


@st.cache_resource
def run_gridsearch_rf(X_train, y_train):
    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    grid = GridSearchCV(
        base_rf,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring="accuracy",
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def train_and_evaluate_models(
    df_ml: pd.DataFrame, feature_cols: list[str], target_col: str
):
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

    if y.dtype == "object":
        y = y.astype("category")
        class_names = list(y.cat.categories)
        y_encoded = y.cat.codes
    else:
        y_encoded = y
        class_names = sorted(list(np.unique(y_encoded)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)

    y_pred_lr = log_reg.predict(X_test_scaled)
    acc_lr = (y_pred_lr == y_test).mean()

    # Random Forest (GridSearch)
    st.markdown("#### 🌲 Random Forest (with hyperparameter tuning)")
    rf, best_params = run_gridsearch_rf(X_train, y_train)
    st.write("Best RF params:", best_params)

    y_pred_rf = rf.predict(X_test)
    acc_rf = (y_pred_rf == y_test).mean()

    # Accuracy comparison
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

    present_classes = np.unique(y_test)
    present_names = [str(class_names[i]) for i in present_classes] if class_names else [str(c) for c in present_classes]

    # Logistic report
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

    # RF report
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

    # Confusion Matrix (best model)
    better_model_name = "Random Forest" if acc_rf >= acc_lr else "Logistic Regression"
    st.markdown(f"#### 🔀 Confusion Matrix – Best Model ({better_model_name})")

    cm = confusion_matrix(y_test, y_pred_rf if better_model_name == "Random Forest" else y_pred_lr, labels=present_classes)
    cm_df = pd.DataFrame(cm, index=present_names, columns=present_names)

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
        - The best model correctly classifies about **{acc_cm:.2%}** of examples.
        """
    )

    # Feature importances
    st.markdown("#### 🧩 Feature Importance (Random Forest)")
    importances = rf.feature_importances_
    imp_df = pd.DataFrame({"Feature": feature_cols, "Importance": importances}).sort_values("Importance", ascending=False)

    fig_imp = px.bar(imp_df, x="Feature", y="Importance", labels={"Importance": "Relative Importance"})
    st.plotly_chart(fig_imp, use_container_width=True)

    top_feats = imp_df.head(3)
    top_feats = imp_df.head(3)

    feat_summary = ", ".join(
        [
            f"`{row['Feature']}` (≈ {row['Importance']:.2f})"
            for _, row in top_feats.iterrows()
        ]
    )

    st.markdown(
        f"**Data-driven conclusion (Feature Importance):** Most influential features are {feat_summary}."
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

    possible_targets: list[str] = []
    if "User_level" in df_ml.columns:
        possible_targets.append("User_level")
    if "Predicted_Field" in df_ml.columns:
        possible_targets.append("Predicted_Field")

    if not possible_targets:
        st.error("No suitable target columns (User_level / Predicted_Field) found.")
        return

    target_col = st.selectbox("Target variable", options=possible_targets)

    st.markdown("**Features used (X):**")
    st.write(feature_cols)

    if st.button("🚀 Train & Evaluate Models"):
        train_and_evaluate_models(df_ml, feature_cols, target_col)


def render_ml_eda_page():
    st.title("📊 ML & EDA – Learning from user_data.csv")

    df_raw = load_user_data()
    if df_raw is None:
        return

    df_ml, feature_cols = engineer_features(df_raw)

    tab_eda, tab_ml = st.tabs(["📊 EDA", "🤖 ML Models"])

    with tab_eda:
        render_eda_section(df_ml)

    with tab_ml:
        render_ml_section(df_ml, feature_cols)
