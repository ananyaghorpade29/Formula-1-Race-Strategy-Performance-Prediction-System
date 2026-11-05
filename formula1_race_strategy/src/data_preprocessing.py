import os
from pathlib import Path

import pandas as pd


# ==========================================
# 1. PROJECT DIRECTORIES
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CLEANED_FILE = PROCESSED_DIR / "f1_cleaned_laps.csv"
ML_FILE = PROCESSED_DIR / "f1_ml_ready.csv"
REPORT_FILE = PROCESSED_DIR / "data_quality_report.txt"

SUPPORTED_SEASONS = {"2023", "2024"}


# ==========================================
# 2. DISCOVER AND LOAD RACE LAP FILES
# ==========================================

print("\n========== LOADING LAP DATA ==========")

lap_files = sorted(
    file for file in RAW_DIR.glob("*_laps.csv")
    if file.name[:4] in SUPPORTED_SEASONS
)

if not lap_files:
    raise FileNotFoundError(
        "No individual 2023 or 2024 lap files found in data/raw."
    )

lap_data = []

for filepath in lap_files:

    race_df = pd.read_csv(filepath)

    # Check essential columns
    required_lap_columns = [
        "session_key",
        "driver_number",
        "lap_number",
        "lap_duration"
    ]

    missing_columns = [
        column for column in required_lap_columns
        if column not in race_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{filepath.name} is missing columns: {missing_columns}"
        )

    # Add season and Grand Prix from filename if unavailable
    filename_parts = filepath.stem.split("_", 1)

    if "Season" not in race_df.columns:
        race_df["Season"] = int(filename_parts[0])

    if "GrandPrix" not in race_df.columns:
        race_df["GrandPrix"] = filename_parts[1]

    race_df["source_file"] = filepath.name

    lap_data.append(race_df)

    print(f"Loaded {filepath.name}: {race_df.shape}")

laps = pd.concat(lap_data, ignore_index=True)

print("\nCombined lap dataset:", laps.shape)
print("Race files loaded:", len(lap_files))
print("Seasons:", sorted(laps["Season"].dropna().unique()))


# ==========================================
# 3. BASIC DATA CLEANING
# ==========================================

print("\n========== DATA CLEANING ==========")

numeric_columns = [
    "session_key",
    "driver_number",
    "lap_number",
    "lap_duration",
    "duration_sector_1",
    "duration_sector_2",
    "duration_sector_3",
    "i1_speed",
    "i2_speed",
    "st_speed"
]

for column in numeric_columns:
    if column in laps.columns:
        laps[column] = pd.to_numeric(
            laps[column],
            errors="coerce"
        )

# Remove duplicate lap records
key_columns = [
    "session_key",
    "driver_number",
    "lap_number"
]

before_duplicates = len(laps)

laps = laps.drop_duplicates(
    subset=key_columns,
    keep="first"
)

duplicates_removed = before_duplicates - len(laps)

# Remove invalid non-positive lap times.
# Missing lap times are retained for data-quality reporting.
laps = laps[
    laps["lap_duration"].isna()
    | (laps["lap_duration"] > 0)
].copy()

print("Duplicate records removed:", duplicates_removed)
print("Cleaned lap dataset:", laps.shape)

# Save cleaned lap data
laps.to_csv(CLEANED_FILE, index=False)

print("Cleaned data saved:", CLEANED_FILE)


# ==========================================
# 4. DISCOVER AND LOAD TYRE STINT FILES
# ==========================================

print("\n========== LOADING TYRE STINT DATA ==========")

stint_data = []
missing_stint_files = []

for lap_file in lap_files:

    # Example:
    # 2023_Bahrain_laps.csv
    # becomes:
    # 2023_Bahrain_stints.csv

    stint_filename = lap_file.name.replace(
        "_laps.csv",
        "_stints.csv"
    )

    stint_filepath = RAW_DIR / stint_filename

    if not stint_filepath.exists():
        missing_stint_files.append(stint_filename)
        print("WARNING: Stint file not found:", stint_filename)
        continue

    stint_df = pd.read_csv(stint_filepath)

    stint_df["source_file"] = stint_filename

    print(f"Loaded {stint_filename}: {stint_df.shape}")

    stint_data.append(stint_df)

if not stint_data:
    raise FileNotFoundError(
        "No matching tyre stint files were found."
    )

stints = pd.concat(stint_data, ignore_index=True)

print("\nCombined stint dataset:", stints.shape)
print("Stint files loaded:", len(stint_data))


# ==========================================
# 5. VALIDATE AND EXPAND STINT DATA
# ==========================================

print("\n========== PREPARING TYRE INFORMATION ==========")

required_stint_columns = [
    "session_key",
    "driver_number",
    "stint_number",
    "lap_start",
    "lap_end",
    "compound",
    "tyre_age_at_start"
]

for column in required_stint_columns:
    if column not in stints.columns:
        raise ValueError(
            f"Missing required stint column: {column}"
        )

numeric_stint_columns = [
    "session_key",
    "driver_number",
    "stint_number",
    "lap_start",
    "lap_end",
    "tyre_age_at_start"
]

for column in numeric_stint_columns:
    stints[column] = pd.to_numeric(
        stints[column],
        errors="coerce"
    )

if stints[required_stint_columns].isna().any().any():
    raise ValueError(
        "Missing or invalid values found in required stint columns."
    )

# Identify invalid stint lap ranges
invalid_stints = (
    (stints["lap_start"] < 1)
    | (stints["lap_end"] < stints["lap_start"])
)

invalid_stint_count = int(invalid_stints.sum())

print("Invalid stint records found:", invalid_stint_count)

# Exclude invalid stint records
stints = stints.loc[~invalid_stints].copy()

print("Valid stint records remaining:", len(stints))
print("Invalid stint records excluded:", invalid_stint_count)

stint_laps = []

for _, stint in stints.iterrows():

    start_lap = int(stint["lap_start"])
    end_lap = int(stint["lap_end"])

    for lap_number in range(start_lap, end_lap + 1):

        tyre_age = (
            stint["tyre_age_at_start"]
            + lap_number
            - start_lap
        )

        stint_laps.append({
            "session_key": stint["session_key"],
            "driver_number": stint["driver_number"],
            "lap_number": lap_number,
            "stint_number": stint["stint_number"],
            "compound": stint["compound"],
            "tyre_age_at_start": stint["tyre_age_at_start"],
            "tyre_age": tyre_age
        })

stint_laps_df = pd.DataFrame(stint_laps)

print("Expanded tyre lap records:", stint_laps_df.shape)


# ==========================================
# 6. CHECK FOR OVERLAPPING STINT MATCHES
# ==========================================

print("\n========== VALIDATING STINT DATA ==========")

join_keys = [
    "session_key",
    "driver_number",
    "lap_number"
]

overlaps = stint_laps_df.duplicated(
    subset=join_keys,
    keep=False
)

overlap_count = int(overlaps.sum())

print("Overlapping stint rows:", overlap_count)

if overlap_count > 0:
    raise ValueError(
        "Overlapping stint ranges detected. "
        "Please inspect the original stint files."
    )


# ==========================================
# 7. MERGE LAP AND TYRE DATA
# ==========================================

print("\n========== MERGING DATASETS ==========")

ml_data = laps.merge(
    stint_laps_df,
    on=join_keys,
    how="left",
    validate="one_to_one"
)

matched = int(ml_data["compound"].notna().sum())
unmatched = int(ml_data["compound"].isna().sum())

print("Total lap records:", len(ml_data))
print("Matched lap records:", matched)
print("Unmatched lap records:", unmatched)

match_percentage = (
    matched / len(ml_data) * 100
    if len(ml_data) > 0 else 0
)

print(f"Tyre matching percentage: {match_percentage:.2f}%")


# ==========================================
# 8. FEATURE ENGINEERING
# ==========================================

print("\n========== FEATURE ENGINEERING ==========")

# Average sector time
sector_columns = [
    "duration_sector_1",
    "duration_sector_2",
    "duration_sector_3"
]

available_sectors = [
    column for column in sector_columns
    if column in ml_data.columns
]

if available_sectors:
    ml_data["average_sector_time"] = (
        ml_data[available_sectors].mean(axis=1)
    )

# Average speed from available speed traps
speed_columns = [
    "i1_speed",
    "i2_speed",
    "st_speed"
]

available_speeds = [
    column for column in speed_columns
    if column in ml_data.columns
]

if available_speeds:
    ml_data["average_speed"] = (
        ml_data[available_speeds].mean(axis=1)
    )

# Keep compound as a categorical value
if "compound" in ml_data.columns:
    ml_data["compound"] = ml_data["compound"].astype("category")

print("Feature engineering completed.")


# ==========================================
# 9. SAVE ML-READY DATASET
# ==========================================

ml_data.to_csv(ML_FILE, index=False)

print("\nML-ready dataset saved:", ML_FILE)


# ==========================================
# 10. GENERATE DATA QUALITY REPORT
# ==========================================

print("\n========== DATA QUALITY REPORT ==========")

with open(REPORT_FILE, "w", encoding="utf-8") as report:

    report.write("FORMULA 1 DATA QUALITY REPORT\n")
    report.write("==============================\n\n")

    report.write(f"Lap files loaded: {len(lap_files)}\n")
    report.write(f"Stint files loaded: {len(stint_data)}\n")
    report.write(f"Missing stint files: {len(missing_stint_files)}\n")

    report.write(
        f"Total records: {len(ml_data)}\n"
    )
    report.write(
        f"Total columns: {len(ml_data.columns)}\n"
    )
    report.write(
        f"Duplicate records removed: {duplicates_removed}\n"
    )
    report.write(
        f"Matched tyre records: {matched}\n"
    )
    report.write(
        f"Unmatched tyre records: {unmatched}\n"
    )
    report.write(
        f"Tyre matching percentage: {match_percentage:.2f}%\n"
    )
    report.write(
        f"Overlapping stint rows: {overlap_count}\n\n"
    )

    report.write("Missing stint files:\n")
    report.write(
        "\n".join(missing_stint_files)
        if missing_stint_files else "None"
    )

    report.write("\n\nRecords by season:\n")
    report.write(
        ml_data["Season"].value_counts().sort_index().to_string()
    )

    report.write("\n\nRecords by Grand Prix:\n")
    report.write(
        ml_data.groupby(
            ["Season", "GrandPrix"],
            observed=True
        ).size().to_string()
    )

    report.write("\n\nMissing values by column:\n")
    report.write(f"Invalid stint records excluded: {invalid_stint_count}\n")
    report.write(
        ml_data.isnull().sum().to_string()
    )

    report.write("\n\nLap time statistics:\n")
    report.write(
        ml_data["lap_duration"].describe().to_string()
    )

print("Data quality report saved:", REPORT_FILE)


# ==========================================
# 11. FINAL SUMMARY
# ==========================================

print("\n==========================================")
print("MULTI-SEASON PREPROCESSING COMPLETED")
print("==========================================")

print("Seasons included:", sorted(ml_data["Season"].dropna().unique()))
print("Race files loaded:", len(lap_files))
print("Stint files loaded:", len(stint_data))
print("Final dataset dimensions:", ml_data.shape)
print("Valid lap times:", ml_data["lap_duration"].notna().sum())

print("\nRecords by season:")
print(ml_data["Season"].value_counts().sort_index())

print("\nTyre compound distribution:")
print(ml_data["compound"].value_counts(dropna=False))

print("\nFinal output files:")
print("1.", CLEANED_FILE)
print("2.", ML_FILE)
print("3.", REPORT_FILE)