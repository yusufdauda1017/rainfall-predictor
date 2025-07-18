import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("enhanced_weather.csv")
df['time'] = pd.to_datetime(df['time'])
df['month'] = df['time'].dt.month

# 📊 1. Max Temperature Over Time
plt.figure(figsize=(14, 6))
for city in df['city'].unique():
    plt.plot(df[df['city'] == city]['time'], df[df['city'] == city]['temperature_2m_max'], label=city)
plt.title("🌡️ Daily Max Temperature Over Time (2007–2024)")
plt.xlabel("Date")
plt.ylabel("Max Temperature (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("temperature_trend.png")
plt.close()

# 💧 2. Relative Humidity Over Time
plt.figure(figsize=(14, 6))
for city in df['city'].unique():
    plt.plot(df[df['city'] == city]['time'], df[df['city'] == city]['relative_humidity_2m'], label=city)
plt.title("💧 Daily Relative Humidity Over Time (2007–2024)")
plt.xlabel("Date")
plt.ylabel("Relative Humidity (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("humidity_trend.png")
plt.close()

# ☁️ 3. Average Monthly Cloud Cover (Bar Chart)
monthly_cloud = df.groupby(['city', 'month'])['cloud_cover_proxy'].mean().unstack()
monthly_cloud.T.plot(kind='bar', figsize=(14, 6))
plt.title("☁️ Average Monthly Cloud Cover by City")
plt.xlabel("Month")
plt.ylabel("Cloud Cover Proxy")
plt.legend(title="City")
plt.tight_layout()
plt.savefig("monthly_cloud_cover.png")
plt.close()

print("✅ Visualizations saved:")
print("- temperature_trend.png")
print("- humidity_trend.png")
print("- monthly_cloud_cover.png")
