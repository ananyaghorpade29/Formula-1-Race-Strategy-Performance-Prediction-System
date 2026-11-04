
import os
import pandas as pd

# ==========================================
# 1. PROJECT DIRECTORIES
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

INPUT_FILE = os.path.join(RAW_DIR, "f1_combined_laps.csv")
CLEANED_FILE = os.path.join(PROCESSED_DIR, "f1_cleaned_laps.csv")
ML_FILE = os.path.join(PROCESSED_DIR, "f1_ml_ready.csv")
REPORT_FILE = os.path.join(PROCESSED_DIR, "data_quality_report.txt")


# ==========================================
# 2. LOAD LAP DATA
# ==========================================

print("\n========== LOADING LAP DATA ==========")

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Combined lap dataset not found: {INPUT_FILE}"
    )

laps = pd.read_csv(INPUT_FILE)

print("Lap dataset loaded:", laps.shape)


# ==========================================
# 3. BASIC DATA CLEANING
# ==========================================

print("\n========== DATA CLEANING ==========")

# Remove duplicate lap records
key_columns = [
    "session_key",
    "driver_number",
    "lap_number"
]

if all(column in laps.columns for column in key_columns):
    laps = laps.drop_duplicates(
        subset=key_columns,
        keep="first"
    )
else:
    laps = laps.drop_duplicates()

# Convert numeric columns
numeric_columns = [
    "lap_duration",
    "duration_sector_1",
    "duration_sector_2",
    "duration_sector_3",
    "lap_number",
    "driver_number",
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

# Remove invalid lap durations, but retain missing values
if "lap_duration" in laps.columns:
    laps = laps[
        laps["lap_duration"].isna()
        | (laps["lap_duration"] > 0)
    ]

print("Cleaned lap dataset:", laps.shape)

# Save cleaned lap data
laps.to_csv(CLEANED_FILE, index=False)

print("Cleaned data saved:", CLEANED_FILE)


# ==========================================
# 4. LOAD TYRE STINT DATA
# ==========================================

print("\n========== LOADING TYRE STINT DATA ==========")

stint_files = [
    "2023_Bahrain_stints.csv",
    "2023_Spain_stints.csv"
]

stint_data = []

for filename in stint_files:

    filepath = os.path.join(RAW_DIR, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Stint file not found: {filepath}"
        )

    stint_df = pd.read_csv(filepath)

    print(filename, ":", stint_df.shape)

    stint_data.append(stint_df)

stints = pd.concat(
    stint_data,
    ignore_index=True
)

print("Combined stint dataset:", stints.shape)


# ==========================================
# 5. EXPAND STINTS INTO INDIVIDUAL LAPS
# ==========================================

print("\n========== PREPARING TYRE INFORMATION ==========")

required_columns = [
    "session_key",
    "driver_number",
    "stint_number",
    "lap_start",
    "lap_end",
    "compound",
    "tyre_age_at_start"
]

for column in required_columns:
    if column not in stints.columns:
        raise ValueError(
            f"Missing required stint column: {column}"
        )

for column in [
    "session_key",
    "driver_number",
    "stint_number",
    "lap_start",
    "lap_end",
    "tyre_age_at_start"
]:
    stints[column] = pd.to_numeric(
        stints[column],
        errors="coerce"
    )

if stints[required_columns].isna().any().any():
    raise ValueError(
        "Missing or invalid values found in required stint columns."
    )

stint_laps = []

for _, stint in stints.iterrows():

    start_lap = int(stint["lap_start"])
    end_lap = int(stint["lap_end"])

    if end_lap < start_lap:
        raise ValueError("Invalid stint lap range found.")

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
        "Please inspect the original stint data."
    )


# ==========================================
# 7. MERGE LAP AND TYRE DATA
# ==========================================

print("\n========== MERGING DATASETS ==========")

laps["lap_number"] = pd.to_numeric(
    laps["lap_number"],
    errors="coerce"
)

laps["session_key"] = pd.to_numeric(
    laps["session_key"],
    errors="coerce"
)

laps["driver_number"] = pd.to_numeric(
    laps["driver_number"],
    errors="coerce"
)

ml_data = laps.merge(
    stint_laps_df,
    on=join_keys,
    how="left",
    validate="one_to_one"
)

matched = ml_data["compound"].notna().sum()
unmatched = ml_data["compound"].isna().sum()

print("Total lap records:", len(ml_data))
print("Matched lap records:", matched)
print("Unmatched lap records:", unmatched)


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

# Total speed indicator from available speed traps
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

# Encode tyre compound as a categorical feature
if "compound" in ml_data.columns:
    ml_data["compound"] = (
        ml_data["compound"].astype("category")
    )

print("New features created where source data was available.")


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

    report.write(f"Total records: {len(ml_data)}\n")
    report.write(f"Total columns: {len(ml_data.columns)}\n")
    report.write(f"Matched tyre records: {matched}\n")
    report.write(f"Unmatched tyre records: {unmatched}\n")
    report.write(f"Overlapping stint rows: {overlap_count}\n\n")

    report.write("Missing values by column:\n")
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
print("PHASE 3 PREPROCESSING COMPLETED")
print("==========================================")

print("Final dataset dimensions:", ml_data.shape)
print("Valid lap times:", ml_data["lap_duration"].notna().sum())

print("\nTyre compound distribution:")
print(ml_data["compound"].value_counts(dropna=False))

print("\nFinal output files:")
print("1.", CLEANED_FILE)
print("2.", ML_FILE)
print("3.", REPORT_FILE)
