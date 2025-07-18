import requests
import pandas as pd
from tqdm import tqdm
import time 

# Basic reliable parameters
BASIC_PARAMS = "temperature_2m_max,temperature_2m_min,precipitation_sum"

# Fetch data for one city at a time
def fetch_city_data(city_name, lat, lon):
    print(f"\n📡 Fetching BASIC data for {city_name}...")
    all_years = []
    
    # Get 3 years at a time (safe chunk size)
    for year_chunk in [(2007,2009), (2010,2012), (2013,2015), (2016,2018), (2019,2021), (2022,2024)]:
        start = f"{year_chunk[0]}-01-01"
        end = f"{year_chunk[1]}-12-31"
        
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}&daily={BASIC_PARAMS}"
        
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                data = response.json().get('daily', {})
                if data:
                    df = pd.DataFrame(data)
                    df['city'] = city_name
                    all_years.append(df)
                    print(f"✅ {year_chunk[0]}-{year_chunk[1]} success")
                else:
                    print(f"⚠️ No data for {year_chunk[0]}-{year_chunk[1]}")
            else:
                print(f"❌ Failed {year_chunk[0]}-{year_chunk[1]} (HTTP {response.status_code})")
        except Exception as e:
            print(f"⚠️ Error in {year_chunk[0]}-{year_chunk[1]}: {str(e)}")
        
        time.sleep(2)  # Be kind to the API
    
    if all_years:
        pd.concat(all_years).to_csv(f"{city_name}_basic.csv", index=False)
        return True
    return False

# Run for all cities
cities = {
    "Gombe": (10.29, 11.17),
    "Bauchi": (10.31, 9.84),
    "Potiskum": (11.71, 11.08),
    "Yola": (9.21, 12.48)
}

for city, (lat, lon) in cities.items():
    fetch_city_data(city, lat, lon)

print("\n🎉 BASIC data collection complete!")