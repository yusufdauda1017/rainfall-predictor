import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def load_and_clean():
    # Load raw data
    df = pd.read_csv("nigeria_weather_final.csv")
    
    # Convert and extract dates
    df['time'] = pd.to_datetime(df['time'])
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month
    df['day_of_year'] = df['time'].dt.dayofyear

    # Advanced missing value imputation
    imputer = IterativeImputer(random_state=42)
    cols_to_impute = ['temperature_2m_max', 'relative_humidity_2m', 'soil_moisture_0_to_10cm']
    df[cols_to_impute] = imputer.fit_transform(df[cols_to_impute])

    # Lag features
    df['precipitation_lag1'] = df.groupby('city')['precipitation_sum'].shift(1)
    df['precipitation_lag2'] = df.groupby('city')['precipitation_sum'].shift(2)

    # Seasonal features
    df['is_monsoon'] = df['month'].isin([6, 7, 8, 9]).astype(int)
    df['season'] = pd.cut(df['month'], 
                         bins=[0, 3, 6, 9, 12],
                         labels=['Winter', 'Spring', 'Summer', 'Autumn'])

    # Rolling averages
    df['temp_rolling_7'] = df.groupby('city')['temperature_2m_max'].rolling(7).mean().reset_index(level=0, drop=True)

    # Save processed data
    df.to_csv("enhanced_weather.csv", index=False)
    return df

def visualize_trends(df):
    # Rainfall by season
    plt.figure(figsize=(12, 6))
    df.groupby(['city', 'season'])['precipitation_sum'].mean().unstack().plot(kind='bar')
    plt.title("Seasonal Rainfall Patterns by City")
    plt.ylabel("Mean Rainfall (mm)")
    plt.tight_layout()
    plt.savefig("seasonal_rainfall.png")
    plt.close()

if __name__ == "__main__":
    df = load_and_clean()
    visualize_trends(df)
    print("✅ Processing complete. Saved to enhanced_weather.csv")
