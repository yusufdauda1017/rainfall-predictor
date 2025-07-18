import pandas as pd
import requests
from tqdm import tqdm
import time
from datetime import datetime

def get_nasa_data(lat, lon, start_date, end_date):
    base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "RH2M,CLRSKY_SRF_ALB,GWETTOP",
        "latitude": lat,
        "longitude": lon,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "community": "AG",
        "format": "JSON"
    }

    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()["properties"]["parameter"]
        df = pd.DataFrame({
            "time": list(data["RH2M"].keys()),
            "relative_humidity_2m": list(data["RH2M"].values()),
            "cloud_cover_proxy": list(data["CLRSKY_SRF_ALB"].values()),
            "soil_moisture_0_to_10cm": list(data["GWETTOP"].values())
        })
        df["time"] = pd.to_datetime(df["time"], format="%Y%m%d")
        return df
    else:
        print(f"❌ NASA API Error: {response.status_code}")
        return pd.DataFrame()

def enhance_city_data(city_name, lat, lon):
    print(f"\n⚡ Enhancing {city_name} data...")
    
    try:
        basic_df = pd.read_csv(f"{city_name}_basic.csv")
        basic_df["time"] = pd.to_datetime(basic_df["time"])
    except FileNotFoundError:
        print(f"⚠️ No basic data found for {city_name}")
        return

    years = basic_df["time"].dt.year.unique()
    all_enhanced = []

    for year in tqdm(years, desc=f"Processing {city_name}"):
        start = datetime(year, 1, 1)
        end = datetime(year, 12, 31)

        nasa_df = get_nasa_data(lat, lon, start, end)
        if not nasa_df.empty:
            year_df = basic_df[basic_df["time"].dt.year == year]
            merged_df = year_df.merge(nasa_df, on="time", how="left")
            all_enhanced.append(merged_df)
        
        time.sleep(1.5)  # Be kind to API

    if all_enhanced:
        final_df = pd.concat(all_enhanced)
        final_df.to_csv(f"{city_name}_enhanced.csv", index=False)
        print(f"💾 Saved enhanced data for {city_name}")
    else:
        print(f"⚠️ No enhanced data for {city_name}")

# Define cities and their coordinates
cities = {
    "Gombe": (10.29, 11.17),
    "Bauchi": (10.31, 9.84),
    "Potiskum": (11.71, 11.08),
    "Yola": (9.21, 12.48)
}

# Run enhancement
for city, (lat, lon) in cities.items():
    enhance_city_data(city, lat, lon)

print("\n✨ All NASA enhancements complete!")
