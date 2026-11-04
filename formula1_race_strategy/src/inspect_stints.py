
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

files = [
    "2023_Bahrain_stints.csv",
    "2023_Spain_stints.csv"
]

for filename in files:
    filepath = os.path.join(RAW_DIR, filename)

    print("\n====================================")
    print("FILE:", filename)
    print("====================================")

    df = pd.read_csv(filepath)

    print("\nDataset shape:", df.shape)

    print("\nAvailable columns:")
    print(df.columns.tolist())

    print("\nFirst five records:")
    print(df.head().to_string(index=False))

    print("\nMissing values:")
    print(df.isnull().sum())

