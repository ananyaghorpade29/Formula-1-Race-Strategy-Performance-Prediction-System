
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT_DIR / "data" / "processed" / "f1_ml_ready.csv"

OUTPUT_DIR = ROOT_DIR / "reports" / "tire_degradation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------



def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Please complete Phase 2 preprocessing first."
        )

    df = pd.read_csv(DATA_PATH)

    # Convert OpenF1 column names to analysis column names.
    column_mapping = {
        "lap_duration": "LapTime",
        "tyre_age": "TyreLife",
        "compound": "Compound",
        "GrandPrix": "GrandPrix",
        "driver_number": "Driver",
        "stint_number": "Stint",
    }

    df = df.rename(columns=column_mapping)

    required_columns = [
        "LapTime",
        "TyreLife",
        "Compound",
        "GrandPrix",
        "Driver",
        "Stint",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Convert lap time, tire age and stint to numeric.
    df["LapTime"] = pd.to_numeric(
        df["LapTime"], errors="coerce"
    )

    df["TyreLife"] = pd.to_numeric(
        df["TyreLife"], errors="coerce"
    )

    df["Stint"] = pd.to_numeric(
        df["Stint"], errors="coerce"
    )

    # Keep valid observations.
    df = df[
        df["LapTime"].notna()
        & np.isfinite(df["LapTime"])
        & (df["LapTime"] > 0)
        & (df["LapTime"] <= 180)
        & df["TyreLife"].notna()
        & (df["TyreLife"] > 0)
        & df["Compound"].notna()
        & df["GrandPrix"].notna()
        & df["Driver"].notna()
        & df["Stint"].notna()
    ].copy()

    # Standardize tire compound names.
    df["Compound"] = (
        df["Compound"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df = df[
        ~df["Compound"].isin(
            ["UNKNOWN", "NAN", "NONE", ""]
        )
    ].copy()

    # Create a unique label for each race event.
    # Example: 2023 - Bahrain, 2024 - Bahrain.
    if "Season" in df.columns:
        df["RaceLabel"] = (
            df["Season"].astype(str)
            + " - "
            + df["GrandPrix"].astype(str)
        )
    else:
        df["RaceLabel"] = df["GrandPrix"].astype(str)

    print("Dataset loaded successfully.")
    print("Valid lap observations:", len(df))
    print("Unique race events:", df["RaceLabel"].nunique())
    print("Compounds:", sorted(df["Compound"].unique()))
    print("Drivers:", df["Driver"].nunique())
    print("Stints:", df["Stint"].nunique())

    return df

# --------------------------------------------------
# 3. AGGREGATE LAP TIMES BY TIRE AGE
# --------------------------------------------------

def create_compound_summary(df):

    group_columns = [
        "GrandPrix",
        "Compound",
        "TyreLife",
    ]

    if "Season" in df.columns:
        group_columns.insert(0, "Season")

    summary = (
        df.groupby(group_columns, observed=True)
        .agg(
            MedianLapTime=("LapTime", "median"),
            MeanLapTime=("LapTime", "mean"),
            LapCount=("LapTime", "size"),
            DriverCount=("Driver", "nunique"),
        )
        .reset_index()
    )

    # Avoid drawing conclusions from very small groups.
    summary = summary[
        summary["LapCount"] >= 3
    ].copy()

    summary.to_csv(
        OUTPUT_DIR / "tire_degradation_summary.csv",
        index=False,
    )

    return summary


# --------------------------------------------------
# 4. COMPOUND DEGRADATION CHART
# --------------------------------------------------

def plot_compound_degradation(summary):

    if summary.empty:
        print(
            "Not enough observations for the compound chart."
        )
        return

    facet_column = (
        "Season" if "Season" in summary.columns
        else "GrandPrix"
    )

    fig = px.line(
        summary,
        x="TyreLife",
        y="MedianLapTime",
        color="Compound",
        facet_col=facet_column,
        markers=True,
        hover_data=[
            "LapCount",
            "DriverCount",
            "MeanLapTime",
        ],
        title=(
            "Tire Degradation: Median Lap Time "
            "Against Tire Age"
        ),
        labels={
            "TyreLife": "Tire Age (Laps)",
            "MedianLapTime": "Median Lap Time (Seconds)",
            "Compound": "Tire Compound",
        },
    )

    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
    )

    fig.write_html(
        OUTPUT_DIR / "tire_degradation_by_compound.html"
    )

    print("Saved compound degradation chart.")


# --------------------------------------------------
# 5. CALCULATE DEGRADATION FOR EACH STINT
# --------------------------------------------------

def calculate_stint_degradation(df):

    if "Stint" not in df.columns:
        raise ValueError(
            "Stint column is required for stint-level analysis."
        )

    group_columns = [
        "GrandPrix",
        "Driver",
        "Stint",
        "Compound",
    ]

    if "Season" in df.columns:
        group_columns.insert(0, "Season")

    results = []

    for group_values, stint_df in df.groupby(
        group_columns,
        observed=True,
        dropna=True,
    ):

        stint_df = stint_df.dropna(
            subset=["TyreLife", "LapTime"]
        ).copy()

        # Require enough observations and tire ages.
        if len(stint_df) < 5:
            continue

        if stint_df["TyreLife"].nunique() < 3:
            continue

        X = stint_df[["TyreLife"]].to_numpy()
        y = stint_df["LapTime"].to_numpy()

        model = LinearRegression()
        model.fit(X, y)

        predicted = model.predict(X)

        # Unpack the group identity.
        if "Season" in df.columns:
            season, grand_prix, driver, stint, compound = (
                group_values
            )
        else:
            grand_prix, driver, stint, compound = group_values
            season = None

        results.append({
            "Season": season,
            "GrandPrix": grand_prix,
            "Driver": driver,
            "Stint": stint,
            "Compound": compound,
            "LapCount": len(stint_df),
            "MinimumTyreLife": stint_df["TyreLife"].min(),
            "MaximumTyreLife": stint_df["TyreLife"].max(),
            "DegradationSecondsPerLap": float(
                model.coef_[0]
            ),
            "Intercept": float(model.intercept_),
            "R2": float(model.score(X, y)),
            "MeanAbsoluteFitError": float(
                np.mean(np.abs(y - predicted))
            ),
        })

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        OUTPUT_DIR / "tire_stint_regression_details.csv",
        index=False,
    )

    print(
        "Stints with enough data for regression:",
        len(result_df),
    )

    return result_df


# --------------------------------------------------
# 6. PLOT STINT-LEVEL DEGRADATION
# --------------------------------------------------

def plot_stint_degradation(stint_results):

    if stint_results.empty:
        print(
            "Not enough complete stints for regression chart."
        )
        return

    # Median slope across stints for each race and compound.
    group_columns = ["GrandPrix", "Compound"]

    if "Season" in stint_results.columns:
        group_columns.insert(0, "Season")

    compound_rates = (
        stint_results.groupby(
            group_columns,
            observed=True,
        )
        .agg(
            MedianDegradation=(
                "DegradationSecondsPerLap", "median"
            ),
            StintCount=("DegradationSecondsPerLap", "size"),
        )
        .reset_index()
    )

    compound_rates.to_csv(
        OUTPUT_DIR / "degradation_rates_by_compound.csv",
        index=False,
    )

    facet_column = (
        "Season" if "Season" in compound_rates.columns
        else "GrandPrix"
    )

    fig = px.bar(
        compound_rates,
        x="Compound",
        y="MedianDegradation",
        color="Compound",
        facet_col=facet_column,
        hover_data=["StintCount"],
        title=(
            "Estimated Tire Degradation Rate "
            "by Compound and Race"
        ),
        labels={
            "Compound": "Tire Compound",
            "MedianDegradation": (
                "Estimated Lap-Time Change (Seconds/Lap)"
            ),
        },
    )

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
    )

    fig.update_layout(
        template="plotly_white",
        showlegend=False,
    )

    fig.write_html(
        OUTPUT_DIR / "tire_degradation_by_stint.html"
    )

    print("Saved stint degradation chart.")


# --------------------------------------------------
# 7. MAIN PROGRAM
# --------------------------------------------------

def main():

    print("\n" + "=" * 55)
    print("PHASE 5: TIRE DEGRADATION ANALYSIS")
    print("=" * 55)

    df = load_data()

    compound_summary = create_compound_summary(df)

    plot_compound_degradation(compound_summary)

    stint_results = calculate_stint_degradation(df)

    plot_stint_degradation(stint_results)

    print("\n" + "=" * 55)
    print("TIRE DEGRADATION ANALYSIS COMPLETED")
    print("=" * 55)

    print("Reports saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
