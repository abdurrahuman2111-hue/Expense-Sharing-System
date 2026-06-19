import os
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")


REQUIRED_COLUMNS = {
    "START_DATE*",
    "MILES*",
    "PURPOSE*",
    "CATEGORY*",
    "START*",
    "STOP*",
}


def _load_uber_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Uber dataset file not found at: {path}. "
            "Update DATA_PATH in this script to point to your CSV."
        )

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{sorted(missing)}. Available columns: {list(df.columns)[:30]}..."
        )

    # Drop last row if the common UBER notebook does that (safe: only if last row is fully empty)
    if df.tail(1).isna().all(axis=1).iloc[0]:
        df = df.iloc[:-1].copy()

    # Parse timestamps
    df = df.copy()
    df["START_DATE*"] = pd.to_datetime(df["START_DATE*"], errors="coerce")
    df["END_DATE*"] = pd.to_datetime(df.get("END_DATE*"), errors="coerce") if "END_DATE*" in df.columns else pd.NaT

    # Target candidates
    # Prefer miles as demand proxy; if you also want ride_duration, you can extend.
    df["target_miles"] = pd.to_numeric(df["MILES*"], errors="coerce")
    df = df.dropna(subset=["START_DATE*", "target_miles"])

    # Feature engineering
    df["hour"] = df["START_DATE*"].dt.hour
    df["weekday"] = df["START_DATE*"].dt.day_name()
    df["date"] = df["START_DATE*"].dt.date

    # Simple day-part bins (matches notebook idea)
    df["day_part"] = pd.cut(
        df["hour"],
        bins=[-1, 10, 15, 19, 23],
        labels=["Morning", "Afternoon", "Evening", "Night"],
    )

    # Clean object columns
    obj_cols = ["PURPOSE*", "CATEGORY*", "START*", "STOP*", "weekday", "day_part"]
    for c in obj_cols:
        if c in df.columns:
            df[c] = df[c].astype("object").fillna("unknown")

    # More numeric signals
    df["month"] = df["START_DATE*"].dt.month

    return df


def build_and_train_model(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    feature_cols = ["hour", "month", "weekday", "day_part", "PURPOSE*", "CATEGORY*", "START*", "STOP*"]
    target_col = "target_miles"

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Identify categorical + numeric
    numeric_features = ["hour", "month"]
    categorical_features = [c for c in feature_cols if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = RandomForestRegressor(
        n_estimators=400,
        random_state=random_state,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=2,
    )

    from sklearn.pipeline import Pipeline

    pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)

    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
        "R2": float(r2_score(y_test, pred)),
    }

    return pipe, metrics


def predict_top_peak_periods(model, df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    # Forecasting-like: compare average predicted demand across hour/day_part
    grid = df[["hour", "month", "weekday", "day_part", "PURPOSE*", "CATEGORY*", "START*", "STOP*"]].copy()

    # Build a reduced grid: focus on time and day-part by averaging over other categories.
    # Approach: sample representative values for categorical columns.
    sample_cat = {
        "PURPOSE*": df["PURPOSE*"].mode().iloc[0],
        "CATEGORY*": df["CATEGORY*"].mode().iloc[0],
        "START*": df["START*"].mode().iloc[0],
        "STOP*": df["STOP*"].mode().iloc[0],
        "weekday": df["weekday"].mode().iloc[0],
    }

    # Use most common month for stability
    month_mode = int(df["month"].mode().iloc[0])

    hours = sorted(df["hour"].unique())
    rows = []
    for h in hours:
        part = pd.cut([h], bins=[-1, 10, 15, 19, 23], labels=["Morning", "Afternoon", "Evening", "Night"])[0]
        rows.append(
            {
                "hour": int(h),
                "month": month_mode,
                "weekday": sample_cat["weekday"],
                "day_part": str(part),
                "PURPOSE*": sample_cat["PURPOSE*"],
                "CATEGORY*": sample_cat["CATEGORY*"],
                "START*": sample_cat["START*"],
                "STOP*": sample_cat["STOP*"],
            }
        )

    grid_df = pd.DataFrame(rows)
    preds = model.predict(grid_df)
    grid_df["predicted_miles"] = preds

    return grid_df.sort_values("predicted_miles", ascending=False).head(n)


def main():
    # Change this to where your Uber CSV exists.
    # In your notebook it reads: C:\\Users\\abudr\\uberdrive.csv
    DATA_PATH = r"C:\Users\abudr\uberdrive.csv"

    df = _load_uber_csv(DATA_PATH)
    df = _preprocess(df)

    model, metrics = build_and_train_model(df)

    print("\n[Ride Demand Forecast Model - RandomForestRegressor]")
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    top = predict_top_peak_periods(model, df, n=8)
    print("\nTop peak time windows (based on predicted miles proxy):")
    print(top[["hour", "day_part", "predicted_miles"]].to_string(index=False))

    # Save model for reuse
    try:
        import joblib

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/ride_demand_model.joblib")
        print("\nSaved model to models/ride_demand_model.joblib")
    except Exception:
        print("\nJoblib not available or could not save model; continuing without saving.")


if __name__ == "__main__":
    main()

