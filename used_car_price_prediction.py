"""Used car price prediction workflow for the CarDekho Vehicle Dataset."""

from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")  # Save figures without requiring a desktop display or Tk installation.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "dataset" / "Car details v3.csv"
OUTPUT_DIR = ROOT / "outputs"
REFERENCE_YEAR = 2026
RANDOM_STATE = 42
TARGET = "selling_price"


def brand_from_name(value):
    """Use the first name token as a practical brand proxy for this dataset."""
    if pd.isna(value) or not str(value).strip():
        return np.nan
    return str(value).strip().split()[0].title()


def numeric_from_text(series):
    """Extract the first numeric value from specification strings."""
    return pd.to_numeric(series.astype("string").str.extract(r"([-+]?[0-9]*\.?[0-9]+)", expand=False), errors="coerce")


def save_inspection(df):
    with (OUTPUT_DIR / "dataset_inspection.txt").open("w", encoding="utf-8") as f:
        f.write(f"Source: {DATA_PATH.relative_to(ROOT)}\nShape: {df.shape}\n\nColumns and dtypes:\n{df.dtypes.to_string()}\n\n")
        f.write("Missing values and percentages:\n")
        f.write(pd.DataFrame({"count": df.isna().sum(), "percent": df.isna().mean() * 100}).to_string())
        f.write(f"\n\nExact duplicate rows: {df.duplicated().sum()}\n\nFirst five rows:\n{df.head().to_string(index=False)}")
        f.write(f"\n\nLast five rows:\n{df.tail().to_string(index=False)}\n\nDescriptive statistics:\n{df.describe(include='all').transpose().to_string()}")


def clean_and_engineer(raw):
    df = raw.copy()
    # Standardize headings and categorical text before validation.
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    for col in ["name", "fuel", "seller_type", "transmission", "owner"]:
        if col in df:
            df[col] = df[col].astype("string").str.strip().replace({"": pd.NA, "-": pd.NA, "nan": pd.NA})
    # Normalize known category spellings/casing while preserving meaningful values.
    for col in ["fuel", "seller_type", "transmission", "owner"]:
        if col in df:
            df[col] = df[col].str.title()
    for col in ["year", "selling_price", "km_driven", "seats"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Specification values are often stored as strings with units; retain numeric magnitudes.
    for col in ["mileage", "engine", "max_power"]:
        if col in df:
            df[col] = numeric_from_text(df[col])
    before_dupes = len(df)
    df = df.drop_duplicates().copy()
    removed_duplicates = before_dupes - len(df)
    # These values cannot represent a sale record or a usable nonnegative distance.
    invalid_target = df[TARGET].isna() | (df[TARGET] <= 0)
    invalid_year = df["year"].isna() | (df["year"] < 1900) | (df["year"] > REFERENCE_YEAR)
    invalid_distance = df["km_driven"].isna() | (df["km_driven"] < 0)
    invalid_count = int((invalid_target | invalid_year | invalid_distance).sum())
    df = df.loc[~(invalid_target | invalid_year | invalid_distance)].copy()
    df["brand"] = df["name"].map(brand_from_name)
    df["vehicle_age"] = REFERENCE_YEAR - df["year"]
    return df, {"duplicates_removed": removed_duplicates, "invalid_rows_removed": invalid_count}


def save_eda(df):
    sns.set_theme(style="whitegrid")
    # Price distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[TARGET], bins=40, kde=True, ax=ax)
    ax.set(title="Distribution of Used Car Selling Prices", xlabel="Selling price (dataset currency units)", ylabel="Number of cars")
    fig.tight_layout(); fig.savefig(OUTPUT_DIR / "price_distribution.png", dpi=150); plt.close(fig)
    # Price vs year and distance
    for x, title, xlabel, filename in [("year", "Selling Price vs Manufacturing Year", "Manufacturing year", "price_vs_year.png"), ("km_driven", "Selling Price vs Kilometers Driven", "Kilometers driven", "price_vs_km_driven.png")]:
        fig, ax = plt.subplots(figsize=(8, 5)); sns.scatterplot(data=df, x=x, y=TARGET, alpha=.45, ax=ax)
        ax.set(title=title, xlabel=xlabel, ylabel="Selling price (dataset currency units)"); fig.tight_layout(); fig.savefig(OUTPUT_DIR / filename, dpi=150); plt.close(fig)
    # Categorical comparisons, limited to readable groups.
    for col, agg in [("fuel", "median"), ("transmission", "median")]:
        grouped = df.groupby(col, dropna=False)[TARGET].median().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(x=grouped.index.astype(str), y=grouped.values, ax=ax)
        ax.set(title=f"Median Selling Price by {col.replace('_', ' ').title()}", xlabel=col.replace("_", " ").title(), ylabel="Median selling price"); ax.tick_params(axis="x", rotation=25)
        fig.tight_layout(); fig.savefig(OUTPUT_DIR / f"price_by_{col}.png", dpi=150); plt.close(fig)
    brand_counts = df["brand"].value_counts(); common = brand_counts[brand_counts >= 10].index
    brand_medians = df[df["brand"].isin(common)].groupby("brand")[TARGET].median().sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(10, 6)); sns.barplot(x=brand_medians.values, y=brand_medians.index, ax=ax)
    ax.set(title="Median Selling Price by Brand (brands with at least 10 records; top 15)", xlabel="Median selling price", ylabel="Brand"); fig.tight_layout(); fig.savefig(OUTPUT_DIR / "price_by_brand.png", dpi=150); plt.close(fig)
    # IQR outlier flags are reported and visualized, not automatically removed.
    numeric = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(10, 5)); sns.boxplot(data=df[[TARGET, "km_driven", "vehicle_age"]], orient="h", ax=ax)
    ax.set(title="IQR Outlier Review for Price, Distance, and Vehicle Age", xlabel="Value (features retain their own units)"); fig.tight_layout(); fig.savefig(OUTPUT_DIR / "outlier_boxplots.png", dpi=150); plt.close(fig)
    corr_cols = [c for c in [TARGET, "year", "km_driven", "mileage", "engine", "max_power", "seats", "vehicle_age"] if c in numeric]
    fig, ax = plt.subplots(figsize=(9, 7)); sns.heatmap(df[corr_cols].corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    ax.set(title="Correlation Heatmap of Numeric Features"); fig.tight_layout(); fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=150); plt.close(fig)
    # Export IQR bounds/counts for transparent outlier review.
    rows = []
    for col in [c for c in [TARGET, "km_driven", "vehicle_age", "engine", "max_power", "mileage"] if c in numeric]:
        q1, q3 = df[col].quantile([.25, .75]); iqr = q3 - q1
        rows.append({"feature": col, "q1": q1, "q3": q3, "lower_bound": q1 - 1.5 * iqr, "upper_bound": q3 + 1.5 * iqr,
                     "outlier_count": int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "iqr_outlier_summary.csv", index=False)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Expected source CSV was not found: {DATA_PATH}")
    raw = pd.read_csv(DATA_PATH)
    required = {"name", "year", "selling_price", "km_driven", "fuel", "seller_type", "transmission", "owner"}
    if not required.issubset({c.strip().lower().replace(" ", "_") for c in raw.columns}):
        raise ValueError(f"CSV is missing required CarDekho columns: {sorted(required)}")
    print(f"Loaded {DATA_PATH.name}; shape={raw.shape}")
    print("Columns:", list(raw.columns))
    print("First rows:\n", raw.head(), "\nLast rows:\n", raw.tail())
    print("Dtypes:\n", raw.dtypes)
    save_inspection(raw)
    cleaned, cleaning_notes = clean_and_engineer(raw)
    cleaned.to_csv(OUTPUT_DIR / "cleaned_car_data.csv", index=False)
    (OUTPUT_DIR / "cleaning_notes.txt").write_text(
        "Cleaning decisions: exact duplicate rows removed; rows with missing/nonpositive selling price, missing/out-of-range year, or missing/negative km_driven removed because they cannot support the target or valid vehicle age. Specification unit strings converted to numeric magnitudes. Legitimate IQR outliers are flagged and visualized but retained.\n"
        + f"Duplicates removed: {cleaning_notes['duplicates_removed']}\nInvalid rows removed: {cleaning_notes['invalid_rows_removed']}\nRows retained: {len(cleaned)}\n",
        encoding="utf-8")
    save_eda(cleaned)

    # Rows without a target are excluded above. Remaining feature gaps are imputed inside CV-safe pipelines.
    features = [c for c in ["brand", "year", "vehicle_age", "km_driven", "fuel", "seller_type", "transmission", "owner", "mileage", "engine", "max_power", "seats"] if c in cleaned]
    X, y = cleaned[features], cleaned[TARGET]
    categorical = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numerical = [c for c in features if c not in categorical]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    preprocessor = ColumnTransformer([("num", numeric_pipe, numerical), ("cat", categorical_pipe, categorical)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=RANDOM_STATE)
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE, min_samples_leaf=2),
        "Random Forest": RandomForestRegressor(n_estimators=250, random_state=RANDOM_STATE, n_jobs=-1, min_samples_leaf=2),
    }
    results, fitted = [], {}
    for name, estimator in models.items():
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results.append({"model": name, "MAE": mean_absolute_error(y_test, pred), "RMSE": np.sqrt(mean_squared_error(y_test, pred)), "R2": r2_score(y_test, pred)})
        fitted[name] = (pipe, pred)
    scores = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    scores.to_csv(OUTPUT_DIR / "model_evaluation_results.csv", index=False)
    best_name = scores.loc[0, "model"]
    best_pipe, best_pred = fitted[best_name]
    joblib.dump(best_pipe, OUTPUT_DIR / "best_model_pipeline.joblib")
    pd.DataFrame({"actual_price": y_test.to_numpy(), "predicted_price": best_pred, "residual": y_test.to_numpy() - best_pred}).to_csv(OUTPUT_DIR / "actual_vs_predicted.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5)); sns.barplot(data=scores.melt(id_vars="model", value_vars=["MAE", "RMSE"], var_name="metric", value_name="error"), x="model", y="error", hue="metric", ax=ax)
    ax.set(title="Test Set Error Comparison (lower is better)", xlabel="Model", ylabel="Error in selling-price units"); ax.tick_params(axis="x", rotation=15); fig.tight_layout(); fig.savefig(OUTPUT_DIR / "model_comparison.png", dpi=150); plt.close(fig)
    # Feature importance is meaningful for the tree ensemble; names include encoded categories.
    rf_pipe = fitted["Random Forest"][0]
    names = rf_pipe.named_steps["preprocessor"].get_feature_names_out()
    importances = rf_pipe.named_steps["model"].feature_importances_
    pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False).to_csv(OUTPUT_DIR / "random_forest_feature_importance.csv", index=False)
    top_features = pd.DataFrame({"feature": names, "importance": importances}).sort_values("importance", ascending=False).head(5)
    missing = raw.isna().sum()
    nonzero_missing = missing[missing > 0]
    missing_text = "No raw missing values were found." if nonzero_missing.empty else "; ".join(
        f"{col}: {int(count)} ({raw[col].isna().mean():.1%})" for col, count in nonzero_missing.items())
    summary = [
        "PROJECT FINDINGS",
        f"Source dataset: {DATA_PATH.name} ({len(raw):,} rows, {raw.shape[1]} columns).",
        f"Raw missing values: {missing_text}",
        f"Exact duplicate rows: {int(raw.duplicated().sum()):,}; removed as repeated records.",
        "Raw specification fields mileage, engine, max_power, and torque include unit-bearing text; numeric specifications were extracted for mileage, engine, and max_power. Torque remains available in the cleaned file but is not used as a model feature because its formats and units are inconsistent.",
        "Potentially extreme numeric values were flagged by the IQR rule and retained because an IQR flag alone does not establish that a vehicle record is invalid.",
        f"After cleaning: {len(cleaned):,} rows. Brand is the first name token; vehicle_age is 2026 minus manufacturing year.",
        "Median price charts compare groups without letting a small number of very expensive vehicles dominate the summary. Correlations and group differences are descriptive and do not establish causation.",
        "Test metrics (same held-out 20% split; lower MAE/RMSE and higher R-squared are preferable):",
        scores.to_string(index=False),
        f"Selected model by lowest test RMSE: {best_name}.",
        "Top Random Forest features (impurity-based importance; descriptive rather than causal):",
        top_features.to_string(index=False),
    ]
    (OUTPUT_DIR / "project_findings.txt").write_text("\n\n".join(summary), encoding="utf-8")
    print("\nModel results:\n", scores.to_string(index=False))
    print(f"Best model by test RMSE: {best_name}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
