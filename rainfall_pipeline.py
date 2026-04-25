"""
Rainfall prediction pipeline.

This module contains the full workflow for a binary classification task:
predict whether rainfall will occur (1) or not (0) using weather features.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Use a non-interactive backend so plots can be saved without opening windows.
matplotlib.use("Agg")
sns.set_theme(style="whitegrid")


def normalize_target(series: pd.Series) -> pd.Series:
    """
    Convert the target column into binary labels.

    Why this exists:
    Datasets may store rainfall labels in different forms:
    - numeric: 0 / 1
    - text: Yes/No, True/False, Rain/No Rain

    This function standardizes them so all models train on a consistent target.
    """
    # If already numeric, keep it numeric and cast to int for consistency.
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)

    # For string labels, normalize formatting before mapping.
    lowered = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "rain": 1,
        "no rain": 0,
        "1": 1,
        "0": 0,
    }
    return lowered.map(mapping)


def save_plot(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    """
    Save a matplotlib figure to disk and close it.

    Closing figures prevents memory growth when many plots are generated.
    """
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    """
    Build preprocessing steps for mixed-type data.

    - Numeric columns are standardized (mean=0, std=1).
    - Categorical columns are one-hot encoded.

    These transformations are applied inside each model pipeline.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def get_models() -> dict[str, object]:
    """
    Return all classification algorithms we want to compare.

    Keeping this in one function makes experimentation easier:
    we can add/remove models without touching the training loop.
    """
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM (RBF)": SVC(probability=True, random_state=42),
    }


def run_pipeline(data_path: str = "weather_based_rain_prediction_dataset.csv", output_dir: str = "outputs") -> pd.DataFrame:
    """
    Run the full machine learning experiment end-to-end.

    Workflow:
    1) Load and inspect data
    2) Clean target and missing values
    3) Split train/test data
    4) Train multiple models
    5) Evaluate metrics
    6) Save comparison results and visualizations

    Returns:
    A DataFrame sorted by ROC-AUC, containing model performance metrics.
    """
    data_path_obj = Path(data_path)
    output_dir_obj = Path(output_dir)
    output_dir_obj.mkdir(exist_ok=True)

    # -----------------------------
    # Step 1: Load and inspect data
    # -----------------------------
    df = pd.read_csv(data_path_obj)
    print("Data shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nMissing values:\n", df.isna().sum())

    # Identify the target automatically from common rainfall target names.
    possible_targets = ["RainTomorrow", "Rain", "RainToday", "rain", "target"]
    target_col = next((col for col in possible_targets if col in df.columns), None)
    if target_col is None:
        raise ValueError("Target column not found. Please set target_col manually.")

    print(f"\nTarget column selected: {target_col}")

    # -----------------------------------
    # Step 2: Clean and prepare the target
    # -----------------------------------
    # Convert mixed target formats into 0/1 and keep only valid binary rows.
    df[target_col] = normalize_target(df[target_col])
    df = df[df[target_col].isin([0, 1])].copy()

    # Separate predictors by data type, because each type needs different preprocessing.
    numeric_cols = [c for c in df.columns if c != target_col and pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in df.columns if c != target_col and c not in numeric_cols]

    # Fill missing values:
    # - numeric -> median (robust to outliers)
    # - categorical -> most frequent value (mode)
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        mode_val = df[col].mode(dropna=True)
        df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown")

    print("\nTarget distribution:")
    print(df[target_col].value_counts(normalize=True).rename("ratio"))

    # X = input features, y = target label.
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    # -----------------------------------
    # Step 3: Train/test split
    # -----------------------------------
    # Stratify keeps class proportions similar in train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build reusable preprocessing and model list.
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    models = get_models()
    results = []
    roc_data = []

    # -----------------------------------
    # Step 4: Exploratory/summary plots
    # -----------------------------------
    # 1) Target distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x=target_col, hue=target_col, ax=ax, palette="Set2", legend=False)
    ax.set_title("Rain Occurrence Distribution")
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    save_plot(fig, output_dir_obj, "01_target_distribution.png")

    # 2) Correlation heatmap
    corr_cols = numeric_cols + [target_col]
    if len(corr_cols) > 1:
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(df[corr_cols].corr(), cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        save_plot(fig, output_dir_obj, "02_correlation_heatmap.png")

    # 3) Numeric distributions
    if numeric_cols:
        ncols = 3
        nrows = (len(numeric_cols) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4 * nrows))
        axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
        for i, col in enumerate(numeric_cols):
            sns.histplot(df[col], kde=True, ax=axes[i], color="#2C7FB8")
            axes[i].set_title(f"{col} Distribution")
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")
        save_plot(fig, output_dir_obj, "03_numeric_distributions.png")

    # 4) Boxplots by class
    if numeric_cols:
        melted = df.melt(id_vars=target_col, value_vars=numeric_cols, var_name="Feature", value_name="Value")
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=melted, x="Feature", y="Value", hue=target_col, ax=ax)
        ax.set_title("Feature Distribution by Rain Class")
        ax.tick_params(axis="x", rotation=45)
        save_plot(fig, output_dir_obj, "04_boxplots_by_class.png")

    # -----------------------------------
    # Step 5: Train and evaluate each model
    # -----------------------------------
    for name, estimator in models.items():
        # Pipeline guarantees preprocessing is learned from training data only.
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)

        # Class predictions and positive-class probabilities.
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        # Core evaluation metrics.
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)

        print(f"\n{name}")
        print("=" * 50)
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "ROC_AUC": auc,
            }
        )

        # Confusion matrix helps explain error types (FP/FN) per model.
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(f"{name} - Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_plot(fig, output_dir_obj, f"cm_{safe_name}.png")

        # Store ROC points so we can compare all models on one chart.
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data.append((name, fpr, tpr, auc))

    # -----------------------------------
    # Step 6: Save final tabular results
    # -----------------------------------
    results_df = pd.DataFrame(results).sort_values(by="ROC_AUC", ascending=False)
    results_path = output_dir_obj / "model_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\nFinal Results Table:")
    print(results_df)
    print(f"\nSaved results to: {results_path}")

    # -----------------------------------
    # Step 7: Final comparison visualizations
    # -----------------------------------
    # 5) Model comparison chart
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df = results_df.melt(
        id_vars="Model",
        value_vars=["Accuracy", "F1", "ROC_AUC"],
        var_name="Metric",
        value_name="Score",
    )
    sns.barplot(data=plot_df, x="Model", y="Score", hue="Metric", ax=ax)
    ax.set_title("Model Performance Comparison")
    ax.tick_params(axis="x", rotation=25)
    save_plot(fig, output_dir_obj, "05_model_comparison.png")

    # 6) ROC curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, fpr, tpr, auc in roc_data:
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    save_plot(fig, output_dir_obj, "06_roc_curves.png")

    # 7) Random Forest feature importances
    # This explains which transformed features contributed most to predictions.
    rf_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=300, random_state=42)),
        ]
    )
    rf_pipeline.fit(X_train, y_train)
    rf_model = rf_pipeline.named_steps["model"]
    feature_names = rf_pipeline.named_steps["preprocessor"].get_feature_names_out()
    feat_imp = (
        pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    feat_imp.sort_values().plot(kind="barh", ax=ax, color="#1B9E77")
    ax.set_title("Top 15 Feature Importances (Random Forest)")
    ax.set_xlabel("Importance")
    save_plot(fig, output_dir_obj, "07_feature_importance_rf.png")

    print("\nAll visualizations saved in 'outputs/' directory.")
    return results_df
