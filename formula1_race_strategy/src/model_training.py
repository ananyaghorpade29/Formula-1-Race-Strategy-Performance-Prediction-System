
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "processed" / "f1_ml_ready.csv"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports" / "model_evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. FEATURES AND TARGET
# ==========================================

TARGET = "lap_duration"

# Maximum lap time allowed for race-pace modelling.
MAX_LAP_TIME = 180

# Features available before or during the lap.
# Same-lap sector times and speeds are excluded
# to prevent target leakage.

NUMERIC_FEATURES = [
    "lap_number",
    "tyre_age",
    "stint_number",
    "is_pit_out_lap"
]

CATEGORICAL_FEATURES = [
    "driver_number",
    "compound",
    "GrandPrix",
    "Season"
]


# ==========================================
# 3. LOAD AND PREPARE DATA
# ==========================================

def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Run data_preprocessing.py first."
        )

    df = pd.read_csv(DATA_PATH)

    original_rows = len(df)

    print("\n" + "=" * 55)
    print("DATASET LOADED SUCCESSFULLY")
    print("=" * 55)

    print("Original shape:", df.shape)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' is missing."
        )

    if "GrandPrix" not in df.columns:
        raise ValueError(
            "GrandPrix column is required."
        )

    # --------------------------------------
    # Convert lap duration to numeric seconds
    # --------------------------------------

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    # --------------------------------------
    # DATA CLEANING
    # --------------------------------------

    # Keep only positive, finite lap times
    # that do not exceed 180 seconds.

    valid_target = (
        df[TARGET].notna()
        & np.isfinite(df[TARGET])
        & (df[TARGET] > 0)
        & (df[TARGET] <= MAX_LAP_TIME)
    )

    df = df.loc[valid_target].copy()

    excluded_rows = original_rows - len(df)

    print("\nDATA CLEANING RESULTS")
    print("Original records:", original_rows)
    print("Excluded records:", excluded_rows)
    print("Retained records:", len(df))
    print("Maximum permitted lap time:", MAX_LAP_TIME)

    print(
        "Cleaned median lap time:",
        round(df[TARGET].median(), 3),
        "seconds"
    )

    # --------------------------------------
    # CREATE RACE GROUPS
    # --------------------------------------

    if "Season" in df.columns:
        df["_RaceGroup"] = (
            df["Season"].astype(str)
            + "_"
            + df["GrandPrix"].astype(str)
        )
    else:
        df["_RaceGroup"] = df["GrandPrix"].astype(str)

    df = df[df["_RaceGroup"].notna()].copy()

    # --------------------------------------
    # SELECT AVAILABLE FEATURES
    # --------------------------------------

    numeric_features = [
        col for col in NUMERIC_FEATURES
        if col in df.columns and df[col].notna().any()
    ]

    categorical_features = [
        col for col in CATEGORICAL_FEATURES
        if col in df.columns and df[col].notna().any()
    ]

    if not numeric_features and not categorical_features:
        raise ValueError("No usable model features found.")

    # --------------------------------------
    # CONVERT NUMERIC FEATURES
    # --------------------------------------

    for col in numeric_features:
        if col != "is_pit_out_lap":
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    # Convert pit-out values to numeric 0/1.
    if "is_pit_out_lap" in numeric_features:

        df["is_pit_out_lap"] = (
            df["is_pit_out_lap"]
            .astype(str)
            .str.lower()
            .map({
                "true": 1,
                "false": 0,
                "1": 1,
                "0": 0
            })
        )

    # --------------------------------------
    # PREPARE CATEGORICAL FEATURES
    # --------------------------------------

    for col in categorical_features:

        df[col] = df[col].astype("object")

        df[col] = df[col].where(
            df[col].notna(),
            np.nan
        )

    # --------------------------------------
    # CREATE X, Y AND GROUPS
    # --------------------------------------

    feature_columns = (
        numeric_features + categorical_features
    )

    X = df[feature_columns].copy()

    y = df[TARGET].astype(float).copy()

    groups = df["_RaceGroup"].copy()

    print("\nDATA PREPARATION COMPLETE")

    print("Rows available:", len(df))
    print("Race groups:", groups.nunique())
    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)
    print("Prediction target:", TARGET)

    if groups.nunique() < 5:
        raise ValueError(
            "At least five race groups are required "
            "for 5-fold GroupKFold validation."
        )

    return (
        df,
        X,
        y,
        groups,
        numeric_features,
        categorical_features,
        excluded_rows
    )


# ==========================================
# 4. BUILD MACHINE LEARNING PIPELINE
# ==========================================

def build_pipeline(
    model,
    numeric_features,
    categorical_features
):

    transformers = []

    # --------------------------------------
    # NUMERIC DATA PIPELINE
    # --------------------------------------

    if numeric_features:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                )
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    # --------------------------------------
    # CATEGORICAL DATA PIPELINE
    # --------------------------------------

    if categorical_features:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value="UNKNOWN"
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    # --------------------------------------
    # COMBINE TRANSFORMERS
    # --------------------------------------

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", model)
        ]
    )

    return pipeline


# ==========================================
# 5. EVALUATION METRICS
# ==========================================

def calculate_metrics(actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = (
        r2_score(actual, predicted)
        if len(actual) >= 2
        else np.nan
    )

    return mae, rmse, r2


# ==========================================
# 6. RACE-BASED CROSS-VALIDATION
# ==========================================

def evaluate_models(
    X,
    y,
    groups,
    numeric_features,
    categorical_features
):

    cv = GroupKFold(n_splits=5)

    models = {

        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    }

    all_metrics = []
    all_predictions = []

    for model_name, estimator in models.items():

        print(f"\nEvaluating {model_name}...")

        fold_number = 1

        for train_index, test_index in cv.split(
            X,
            y,
            groups=groups
        ):

            X_train = X.iloc[train_index]
            X_test = X.iloc[test_index]

            y_train = y.iloc[train_index]
            y_test = y.iloc[test_index]

            # Build a fresh model for each fold.
            pipeline = build_pipeline(
                clone(estimator),
                numeric_features,
                categorical_features
            )

            pipeline.fit(
                X_train,
                y_train
            )

            predictions = pipeline.predict(X_test)

            mae, rmse, r2 = calculate_metrics(
                y_test,
                predictions
            )

            all_metrics.append({

                "Model": model_name,
                "Fold": fold_number,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,

                "TestRaceGroups": groups.iloc[
                    test_index
                ].nunique()

            })

            fold_predictions = pd.DataFrame({

                "Model": model_name,

                "ActualLapTime": y_test.to_numpy(),

                "PredictedLapTime": predictions,

                "RaceGroup": groups.iloc[
                    test_index
                ].to_numpy()

            })

            all_predictions.append(fold_predictions)

            print(
                f"Fold {fold_number}: "
                f"MAE={mae:.3f}s, "
                f"RMSE={rmse:.3f}s, "
                f"R2={r2:.3f}"
            )

            fold_number += 1

    # --------------------------------------
    # SAVE CROSS-VALIDATION RESULTS
    # --------------------------------------

    metrics_df = pd.DataFrame(all_metrics)

    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True
    )

    metrics_df.to_csv(
        REPORT_DIR / "cross_validation_metrics.csv",
        index=False
    )

    predictions_df.to_csv(
        REPORT_DIR / "cross_validation_predictions.csv",
        index=False
    )

    # --------------------------------------
    # MODEL COMPARISON
    # --------------------------------------

    summary = (
        metrics_df.groupby("Model")
        .agg(
            Mean_MAE=("MAE", "mean"),
            Std_MAE=("MAE", "std"),
            Mean_RMSE=("RMSE", "mean"),
            Mean_R2=("R2", "mean")
        )
        .reset_index()
        .sort_values("Mean_MAE")
    )

    summary.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    print("\n" + "=" * 55)
    print("MODEL COMPARISON")
    print("=" * 55)

    print(summary.to_string(index=False))

    return summary, models


# ==========================================
# 7. TRAIN AND SAVE FINAL MODELS
# ==========================================

def train_final_models(
    X,
    y,
    models,
    numeric_features,
    categorical_features,
    summary,
    excluded_rows
):

    saved_pipelines = {}

    for model_name, estimator in models.items():

        pipeline = build_pipeline(
            clone(estimator),
            numeric_features,
            categorical_features
        )

        # Train using all cleaned training records.
        pipeline.fit(X, y)

        filename = (
            "linear_regression_model.pkl"
            if model_name == "Linear Regression"
            else "random_forest_model.pkl"
        )

        joblib.dump(
            pipeline,
            MODEL_DIR / filename
        )

        saved_pipelines[model_name] = pipeline

        print(f"Saved {model_name}: {filename}")

    # Select model with the lowest mean CV MAE.
    best_model_name = summary.iloc[0]["Model"]

    best_pipeline = saved_pipelines[best_model_name]

    joblib.dump(
        best_pipeline,
        MODEL_DIR / "lap_time_model.pkl"
    )

    # --------------------------------------
    # SAVE MODEL METADATA
    # --------------------------------------

    metadata = {

        "project": (
            "Formula 1 Race Strategy "
            "and Performance Prediction System"
        ),

        "target": TARGET,

        "best_model": best_model_name,

        "features": list(X.columns),

        "numeric_features": numeric_features,

        "categorical_features": categorical_features,

        "maximum_lap_time_seconds": MAX_LAP_TIME,

        "excluded_training_records": excluded_rows,

        "selection_metric": (
            "Lowest mean race-based cross-validation MAE"
        ),

        "cross_validation": "5-fold GroupKFold",

        "training_records": len(X)

    }

    with open(
        MODEL_DIR / "model_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print("\nBest model:", best_model_name)

    print(
        "Saved selected model: "
        "models/lap_time_model.pkl"
    )

    return saved_pipelines


# ==========================================
# 8. RANDOM FOREST FEATURE IMPORTANCE
# ==========================================

def save_feature_importance(random_forest_pipeline):

    preprocessor = random_forest_pipeline.named_steps[
        "preprocessor"
    ]

    regressor = random_forest_pipeline.named_steps[
        "regressor"
    ]

    feature_names = preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": regressor.feature_importances_

    })

    importance_df = importance_df.sort_values(
        "Importance",
        ascending=False
    )

    importance_df.to_csv(
        REPORT_DIR / "random_forest_feature_importance.csv",
        index=False
    )

    print("\n" + "=" * 55)
    print("TOP 10 IMPORTANT FEATURES")
    print("=" * 55)

    print(
        importance_df.head(10).to_string(index=False)
    )


# ==========================================
# 9. MAIN PROGRAM
# ==========================================

def main():

    (
        df,
        X,
        y,
        groups,
        numeric_features,
        categorical_features,
        excluded_rows
    ) = load_data()

    print("\n" + "=" * 55)
    print("FORMULA 1 LAP TIME PREDICTION")
    print("=" * 55)

    # Evaluate both machine learning models.
    summary, models = evaluate_models(
        X,
        y,
        groups,
        numeric_features,
        categorical_features
    )

    # Train and save final models.
    saved_pipelines = train_final_models(
        X,
        y,
        models,
        numeric_features,
        categorical_features,
        summary,
        excluded_rows
    )

    # Save Random Forest feature importance.
    save_feature_importance(
        saved_pipelines["Random Forest"]
    )

    print("\n" + "=" * 55)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 55)

    print("\nTraining records:", len(X))
    print("Race groups:", groups.nunique())
    print("Excluded records:", excluded_rows)

    print("\nModels saved in:", MODEL_DIR)

    print("Evaluation reports saved in:", REPORT_DIR)


if __name__ == "__main__":
    main()
