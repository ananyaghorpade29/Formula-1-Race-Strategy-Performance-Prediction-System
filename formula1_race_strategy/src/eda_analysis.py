
import os
import pandas as pd
import numpy as np
import plotly.express as px


# ==========================================
# 1. PROJECT PATHS
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "f1_ml_ready.csv"
)

EDA_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "eda"
)

os.makedirs(EDA_DIR, exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_data():

    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            "ML-ready dataset not found. "
            "Run data_preprocessing.py first."
        )

    df = pd.read_csv(DATA_FILE)

    print("\nDATA LOADED SUCCESSFULLY")
    print("Dataset shape:", df.shape)

    required_columns = [
        "lap_duration",
        "lap_number",
        "driver_number",
        "GrandPrix"
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns missing: {missing}"
        )

    numeric_columns = [
        "lap_duration",
        "lap_number",
        "driver_number",
        "tyre_age",
        "duration_sector_1",
        "duration_sector_2",
        "duration_sector_3",
        "i1_speed",
        "i2_speed",
        "st_speed",
        "average_sector_time",
        "average_speed"
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    # Keep valid lap times
    df = df[
        df["lap_duration"].notna()
        & np.isfinite(df["lap_duration"])
        & (df["lap_duration"] > 0)
    ].copy()

    print("Valid lap records:", len(df))
    print("\nAvailable tyre compounds:")
    print(df["compound"].value_counts(dropna=False))

    return df


# ==========================================
# 3. SAVE CHART
# ==========================================

def save_chart(fig, filename):

    filepath = os.path.join(
        EDA_DIR,
        filename
    )

    fig.write_html(
    filepath,
    include_plotlyjs="cdn"
)

    print("Chart saved:", filepath)


# ==========================================
# 4. LAP TIME DISTRIBUTION
# ==========================================

def lap_time_analysis(df):

    print("\n--- LAP TIME ANALYSIS ---")

    print(df["lap_duration"].describe())

    fig = px.histogram(
        df,
        x="lap_duration",
        nbins=50,
        title="Formula 1 Lap Time Distribution",
        labels={
            "lap_duration": "Lap Time (seconds)"
        },
        marginal="box"
    )

    fig.update_layout(
        xaxis_title="Lap Time (seconds)",
        yaxis_title="Number of Laps"
    )

    save_chart(
        fig,
        "lap_time_distribution.html"
    )


# ==========================================
# 5. DRIVER PERFORMANCE
# ==========================================

def driver_analysis(df):

    print("\n--- DRIVER PERFORMANCE ---")

    stats = (
        df.groupby(
            ["GrandPrix", "driver_number"]
        )["lap_duration"]
        .agg(
            AverageLapTime="mean",
            MedianLapTime="median",
            FastestLap="min",
            LapTimeStd="std",
            TotalLaps="count"
        )
        .reset_index()
    )

    stats.to_csv(
        os.path.join(
            EDA_DIR,
            "driver_statistics.csv"
        ),
        index=False
    )

    print(stats.to_string(index=False))

    fig = px.bar(
        stats,
        x="driver_number",
        y="AverageLapTime",
        color="GrandPrix",
        barmode="group",
        hover_data=[
            "MedianLapTime",
            "FastestLap",
            "TotalLaps"
        ],
        title="Driver Performance by Grand Prix",
        labels={
            "driver_number": "Driver Number",
            "AverageLapTime": "Average Lap Time (seconds)"
        }
    )

    save_chart(
        fig,
        "driver_comparison.html"
    )


# ==========================================
# 6. TYRE COMPOUND ANALYSIS
# ==========================================

def compound_analysis(df):

    if "compound" not in df.columns:
        print("\nTyre compound data unavailable.")
        return

    tire_df = df.dropna(
        subset=["compound", "lap_duration"]
    ).copy()

    if tire_df.empty:
        print("\nNo valid tyre compound records.")
        return

    print("\n--- TYRE COMPOUND ANALYSIS ---")

    compound_stats = (
        tire_df.groupby("compound")["lap_duration"]
        .agg(
            AverageLapTime="mean",
            MedianLapTime="median",
            FastestLap="min",
            TotalLaps="count"
        )
        .reset_index()
    )

    print(compound_stats.to_string(index=False))

    fig = px.box(
        tire_df,
        x="compound",
        y="lap_duration",
        color="compound",
        facet_col="GrandPrix",
        title="Tyre Compound Comparison by Grand Prix",
        points="outliers"
    )

    fig.update_layout(
        xaxis_title="Tyre Compound",
        yaxis_title="Lap Time (seconds)"
    )

    save_chart(
        fig,
        "tire_compound_comparison.html"
    )


# ==========================================
# 7. TYRE DEGRADATION ANALYSIS
# ==========================================

def tire_degradation_analysis(df):

    required = [
        "tyre_age",
        "compound",
        "lap_duration"
    ]

    if not all(col in df.columns for col in required):
        print("\nTyre degradation columns unavailable.")
        return

    tire_df = df.dropna(
        subset=required
    ).copy()

    tire_df = tire_df[
        tire_df["tyre_age"] > 0
    ]

    if tire_df.empty:
        print("\nNo valid tyre age records.")
        return

    degradation = (
        tire_df.groupby(
            ["GrandPrix", "compound", "tyre_age"]
        )["lap_duration"]
        .agg(
            MedianLapTime="median",
            LapCount="count"
        )
        .reset_index()
    )

    # Keep groups with at least three observations
    degradation = degradation[
        degradation["LapCount"] >= 3
    ]

    if degradation.empty:
        print("\nInsufficient repeated observations for degradation chart.")
        return

    fig = px.line(
        degradation,
        x="tyre_age",
        y="MedianLapTime",
        color="compound",
        facet_col="GrandPrix",
        markers=True,
        title="Tyre Degradation by Grand Prix",
        hover_data=["LapCount"],
        labels={
            "tyre_age": "Tyre Age (laps)",
            "MedianLapTime": "Median Lap Time (seconds)"
        }
    )

    save_chart(
        fig,
        "tire_degradation.html"
    )


# ==========================================
# 8. SECTOR PERFORMANCE
# ==========================================

def sector_analysis(df):

    sector_columns = [
        "duration_sector_1",
        "duration_sector_2",
        "duration_sector_3"
    ]

    available = [
        col for col in sector_columns
        if col in df.columns
    ]

    if not available:
        print("\nSector timing data unavailable.")
        return

    print("\n--- SECTOR PERFORMANCE ---")

    sector_stats = df[available].median()

    print("\nMedian sector times:")
    print(sector_stats)

    sector_df = df[available].melt(
        var_name="Sector",
        value_name="SectorTime"
    ).dropna()

    fig = px.box(
        sector_df,
        x="Sector",
        y="SectorTime",
        color="Sector",
        title="Sector Time Distribution"
    )

    fig.update_layout(
        xaxis_title="Race Sector",
        yaxis_title="Sector Time (seconds)"
    )

    save_chart(
        fig,
        "sector_analysis.html"
    )


# ==========================================
# 9. LAP PROGRESSION
# ==========================================

def lap_progression_analysis(df):

    progression = (
        df.groupby(
            ["GrandPrix", "lap_number"]
        )["lap_duration"]
        .median()
        .reset_index()
    )

    fig = px.line(
        progression,
        x="lap_number",
        y="lap_duration",
        color="GrandPrix",
        title="Median Lap Time Across Race Laps",
        markers=False,
        labels={
            "lap_number": "Lap Number",
            "lap_duration": "Median Lap Time (seconds)"
        }
    )

    save_chart(
        fig,
        "lap_progression.html"
    )


# ==========================================
# 10. SPEED ANALYSIS
# ==========================================

def speed_analysis(df):

    if "average_speed" not in df.columns:
        print("\nAverage speed data unavailable.")
        return

    speed_df = df.dropna(
        subset=["average_speed", "lap_duration"]
    )

    if len(speed_df) < 3:
        print("\nInsufficient speed records.")
        return

    fig = px.scatter(
        speed_df,
        x="average_speed",
        y="lap_duration",
        color="GrandPrix",
        hover_data=["driver_number", "lap_number"],
        title="Average Speed vs Lap Time",
        labels={
            "average_speed": "Average Speed",
            "lap_duration": "Lap Time (seconds)"
        }
    )

    save_chart(
        fig,
        "speed_analysis.html"
    )


# ==========================================
# 11. MAIN EDA FUNCTION
# ==========================================

def main():

    df = load_data()

    print("\n================================")
    print("FORMULA 1 EXPLORATORY ANALYSIS")
    print("================================")

    lap_time_analysis(df)

    driver_analysis(df)

    compound_analysis(df)

    tire_degradation_analysis(df)

    sector_analysis(df)

    lap_progression_analysis(df)

    speed_analysis(df)

    print("\n================================")
    print("EDA COMPLETED SUCCESSFULLY")
    print("================================")

    print("\nAll charts saved in:")
    print(EDA_DIR)


# ==========================================
# 12. EXECUTION
# ==========================================

if __name__ == "__main__":
    main()
