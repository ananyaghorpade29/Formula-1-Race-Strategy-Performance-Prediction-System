
import os
import pandas as pd

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FILE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "f1_cleaned_laps.csv"
)

df = pd.read_csv(FILE_PATH)

print("\n========== DATASET VERIFICATION ==========")

print("\n1. Dataset dimensions:")
print(df.shape)

print("\n2. First five records:")
print(df.head())

print("\n3. Available columns:")
print(df.columns.tolist())

print("\n4. Missing values:")
print(df.isnull().sum())

print("\n5. Driver count:")
print(df["driver_number"].nunique())

print("\n6. Race count:")
print(df["GrandPrix"].nunique())

print("\n7. Tire compounds:")
print("Tyre compound information is not available in the current lap dataset.")
print("We will add tyre compound data from the stint dataset in the next stage.")
#print(df["Compound"].value_counts())

print("\n8. Lap time statistics:")
print(df["lap_duration"].describe())

print("\n9. Pit-out laps:")
print(df["is_pit_out_lap"].value_counts(dropna=False))

print("\n10. Available lap records:")
print(df["lap_duration"].notna().sum())

print("\n========== VERIFICATION COMPLETE ==========")
