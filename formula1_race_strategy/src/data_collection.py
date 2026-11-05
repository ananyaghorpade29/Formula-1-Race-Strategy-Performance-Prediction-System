
import os
import time
import requests
import pandas as pd

# Project directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
SESSION_FILE = os.path.join(
    RAW_DIR, "f1_race_sessions_2023_2024.csv"
)

os.makedirs(RAW_DIR, exist_ok=True)

BASE_URL = "https://api.openf1.org/v1"

# Existing filenames use country names rather than circuit names.
# Recognise them so existing files are not downloaded again.
EXISTING_FILE_ALIASES = {
    (2023, "Sakhir"): "2023_Bahrain",
    (2023, "Catalunya"): "2023_Spain",
}

ENDPOINTS = {
    "laps": "laps",
    "weather": "weather",
    "stints": "stints",
    "pitstops": "pit",
}


def fetch_data(endpoint, session_key):
    """Fetch one type of race data from OpenF1."""

    url = f"{BASE_URL}/{endpoint}"
    params = {"session_key": int(session_key)}

    response = requests.get(url, params=params, timeout=60)

    # Some sessions do not have data for every endpoint.
    if response.status_code == 404:
        return []

    response.raise_for_status()
    return response.json()


def collect_endpoint(session, data_type, endpoint, filename_base):
    """Download and save one endpoint, skipping existing files."""

    filename = f"{filename_base}_{data_type}.csv"
    filepath = os.path.join(RAW_DIR, filename)

    if os.path.exists(filepath):
        print(f"  SKIPPED (already exists): {filename}")
        return "skipped"

    try:
        records = fetch_data(endpoint, session["session_key"])

        if not records:
            print(f"  UNAVAILABLE: {data_type}")
            return "unavailable"

        df = pd.DataFrame(records)

        # Add season and Grand Prix information to lap records.
        if data_type == "laps":
            df["Season"] = int(session["year"])
            df["GrandPrix"] = session["country_name"]

        df.to_csv(filepath, index=False)

        print(f"  SAVED: {filename} ({len(df)} records)")
        return "downloaded"

    except requests.RequestException as error:
        print(f"  API ERROR ({data_type}): {error}")
        return "error"

    except (ValueError, KeyError) as error:
        print(f"  DATA ERROR ({data_type}): {error}")
        return "error"


def collect_race_data(session):
    """Collect available data for one race session."""

    year = int(session["year"])
    circuit = session["circuit_short_name"]
    country = session["country_name"]
    session_key = session["session_key"]

    filename_base = EXISTING_FILE_ALIASES.get(
        (year, circuit),
        f"{year}_{circuit}"
    )

    print("\n" + "=" * 55)
    print(f"RACE: {country} | {circuit} | {year}")
    print(f"Session key: {session_key}")
    print("=" * 55)

    results = {}

    for data_type, endpoint in ENDPOINTS.items():
        results[data_type] = collect_endpoint(
            session,
            data_type,
            endpoint,
            filename_base
        )

        # Small pause between API requests.
        time.sleep(0.2)

    return results


def main():
    if not os.path.exists(SESSION_FILE):
        print("ERROR: Race session index not found.")
        print("Expected:", SESSION_FILE)
        return

    sessions_df = pd.read_csv(SESSION_FILE)

    sessions_df = sessions_df[
        sessions_df["is_cancelled"].astype(str).str.lower() == "false"
    ]

    sessions_df = sessions_df.sort_values(
        ["year", "date_start"]
    )

    totals = {
        "downloaded": 0,
        "skipped": 0,
        "unavailable": 0,
        "error": 0
    }

    print("FORMULA 1 DATA COLLECTION")
    print(f"Race sessions to process: {len(sessions_df)}")

    for _, session in sessions_df.iterrows():
        try:
            results = collect_race_data(session)

            for status in results.values():
                totals[status] += 1

        except Exception as error:
            print(f"RACE PROCESSING ERROR: {error}")
            totals["error"] += 1

    print("\n" + "=" * 55)
    print("DATA COLLECTION SUMMARY")
    print("=" * 55)
    print("Race sessions processed:", len(sessions_df))
    print("Files downloaded:", totals["downloaded"])
    print("Files skipped:", totals["skipped"])
    print("Files unavailable:", totals["unavailable"])
    print("Errors:", totals["error"])
    print("Output directory:", RAW_DIR)


if __name__ == "__main__":
    main()
