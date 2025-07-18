import pandas as pd
import glob

# Combine all enhanced city files
all_files = glob.glob("*_enhanced.csv")
combined = pd.concat([pd.read_csv(f) for f in all_files])

# Clean and save final dataset
combined['time'] = pd.to_datetime(combined['time'])
combined.sort_values(['city', 'time'], inplace=True)
combined.to_csv("nigeria_weather_final.csv", index=False)

print(f"🚀 Combined {len(all_files)} cities into nigeria_weather_final.csv")
print(f"📊 Total rows: {len(combined):,}")
print("Columns available:", list(combined.columns))