
import os
import requests
import pandas as pd

# Project directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")

os.makedirs(RAW_DIR, exist_ok=True)

BASE_URL = "https://api.openf1.org/v1"

RACES = [
    (2023, "Bahrain"),
    (2023, "Spain"),
    (2023, "Great Britain")
]


def fetch_data(endpoint, params):
    """Fetch historical F1 data from OpenF1."""

    url = f"{BASE_URL}/{endpoint}"

    response = requests.get(
        url,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    return response.json()


def collect_race_data(year, country):
    print(f"\nCollecting {country} Grand Prix ({year})")

    # Find the race session
    sessions = fetch_data(
        "sessions",
        {
            "year": year,
            "country_name": country,
            "session_name": "Race"
        }
    )

    if not sessions:
        print(f"No race session found for {country}")
        return None

    session = sessions[0]
    session_key = session["session_key"]

    print("Session found:", session_key)

    # Collect lap data
    laps = fetch_data(
        "laps",
        {"session_key": session_key}
    )

    if laps:
        laps_df = pd.DataFrame(laps)
        laps_df["Season"] = year
        laps_df["GrandPrix"] = country

        filename = f"{year}_{country}_laps.csv"
        path = os.path.join(RAW_DIR, filename)

        laps_df.to_csv(path, index=False)

        print("Lap data saved:", path)
        print("Total lap records:", len(laps_df))
    else:
        print("No lap data available.")

    # Collect weather data
    weather = fetch_data(
        "weather",
        {"session_key": session_key}
    )

    if weather:
        weather_df = pd.DataFrame(weather)

        filename = f"{year}_{country}_weather.csv"
        path = os.path.join(RAW_DIR, filename)

        weather_df.to_csv(path, index=False)

        print("Weather data saved:", path)

    # Collect tyre stint data
    stints = fetch_data(
        "stints",
        {"session_key": session_key}
    )

    if stints:
        stints_df = pd.DataFrame(stints)

        filename = f"{year}_{country}_stints.csv"
        path = os.path.join(RAW_DIR, filename)

        stints_df.to_csv(path, index=False)

        print("Tyre stint data saved:", path)

    # Collect pit stop data
    pits = fetch_data(
        "pit",
        {"session_key": session_key}
    )

    if pits:
        pits_df = pd.DataFrame(pits)

        filename = f"{year}_{country}_pitstops.csv"
        path = os.path.join(RAW_DIR, filename)

        pits_df.to_csv(path, index=False)

        print("Pit stop data saved:", path)

    return True


def main():
    successful = 0

    for year, country in RACES:
        try:
            result = collect_race_data(year, country)

            if result:
                successful += 1

        except requests.RequestException as error:
            print(f"Network/API error for {country}: {error}")

        except Exception as error:
            print(f"Error collecting {country}: {error}")

    print("\n================================")
    print("DATA COLLECTION SUMMARY")
    print("================================")
    print(f"Successful races: {successful}/{len(RACES)}")
    print("Output directory:", RAW_DIR)


if __name__ == "__main__":
    main()
