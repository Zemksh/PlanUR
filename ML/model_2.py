#FAILED

# 🌍 **1. Import Libraries**
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
import geopandas as gpd
from shapely.geometry import Point

# ✅ **2. Generate Simulated Urban Sustainability Dataset**
np.random.seed(42)

cities = ["New York", "Los Angeles", "Berlin", "Tokyo", "Mumbai", "Sydney"]
data = {
    "City": np.random.choice(cities, 200),
    "Temperature": np.random.uniform(10, 35, 200),
    "Rainfall": np.random.uniform(500, 2000, 200),
    "AQI": np.random.randint(30, 200, 200),
    "Urban_Heat": np.random.uniform(0.5, 5, 200),
    "Transit_Usage": np.random.uniform(20, 90, 200),
    "Walkability": np.random.uniform(30, 100, 200),
    "Green_Space": np.random.uniform(10, 60, 200),
    "Biodiversity_Index": np.random.randint(100, 1000, 200),
    "Energy_Consumption": np.random.uniform(5000, 15000, 200),
    "Building_Efficiency": np.random.uniform(50, 100, 200),
    "Water_Efficiency": np.random.uniform(10, 70, 200),
    "Recycling_Rate": np.random.uniform(20, 90, 200)
}

# Create DataFrame
df = pd.DataFrame(data)
df.to_csv("sustainability_data.csv", index=False)

# ✅ **3. Preprocessing**
scaler = StandardScaler()
X = df.drop("City", axis=1)
X_scaled = scaler.fit_transform(X)

# ✅ **4. Sustainability Score Calculation**
def calculate_sustainability_score(row):
    score = (
        row["Green_Space"] * 0.2 +
        row["Biodiversity_Index"] * 0.15 +
        (100 - row["AQI"]) * 0.15 +
        row["Walkability"] * 0.1 +
        row["Transit_Usage"] * 0.1 +
        row["Building_Efficiency"] * 0.1 +
        row["Water_Efficiency"] * 0.1 +
        row["Recycling_Rate"] * 0.1
    )
    return min(max(score, 0), 100)

# Add Sustainability Score
df["Sustainability_Score"] = df.apply(calculate_sustainability_score, axis=1)

# ✅ **5. AI Model Training**
X = df.drop(["City", "Sustainability_Score"], axis=1)
y = df["Sustainability_Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Model Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# ✅ **6. Generate Recommendations**
def generate_recommendations(row):
    rec = []
    if row["Green_Space"] < 30:
        rec.append("Increase green spaces and promote rooftop gardens.")
    if row["AQI"] > 100:
        rec.append("Introduce stricter emissions regulations and expand electric transit.")
    if row["Energy_Consumption"] > 10000:
        rec.append("Implement energy efficiency programs in public infrastructure.")
    if row["Water_Efficiency"] < 30:
        rec.append("Promote water recycling and conservation practices.")
    if row["Recycling_Rate"] < 50:
        rec.append("Enhance waste segregation and recycling campaigns.")
    return rec if rec else ["City is already highly sustainable."]

# Add recommendations
df["Recommendations"] = df.apply(generate_recommendations, axis=1)

# ✅ **7. Load and Process GeoTIFF for NDVI Calculation**
def load_bands(geotiff_path):
    with open(geotiff_path) as src:
        red = src.read(1).astype('float32')
        nir = src.read(2).astype('float32')
    return red, nir

def calculate_ndvi(red, nir):
    np.seterr(divide='ignore', invalid='ignore')
    ndvi = (nir - red) / (nir + red)
    ndvi = np.nan_to_num(ndvi)
    return ndvi

# Example: Replace with your GeoTIFF file path
geotiff_path = "path/to/vegetation_image.tif"
red_band, nir_band = load_bands(geotiff_path)
ndvi = calculate_ndvi(red_band, nir_band)

# ✅ **8. Green Cover Percentage Calculation**
def calculate_green_cover_percentage(ndvi, threshold=0.2):
    green_pixels = np.sum(ndvi > threshold)
    total_pixels = ndvi.size
    green_cover_percentage = (green_pixels / total_pixels) * 100
    return green_cover_percentage

green_cover = calculate_green_cover_percentage(ndvi)
print(f"Green Cover Percentage: {green_cover:.2f}%")

# ✅ **9. Vegetation Classification with K-means**
ndvi_reshaped = ndvi.reshape(-1, 1)

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(ndvi_reshaped)
classified = kmeans.labels_.reshape(ndvi.shape)

# ✅ **10. Export Green Metrics as GeoJSON**
green_metrics = {
    "City": ["Sample City"],
    "Green_Cover_Percentage": [green_cover],
    "Tree_Canopy_Coverage": [35.4],
    "Per_Capita_Green_Space": [22.1],
}

gdf = gpd.GeoDataFrame(green_metrics, geometry=[Point(-74.006, 40.7128)])
gdf.to_file("green_cover_analysis.geojson", driver="GeoJSON")
print("Green cover metrics exported as GeoJSON.")

# ✅ **11. Visualizations**

# Enhanced NDVI Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(ndvi, cmap='YlGn', cbar=True, vmin=-1, vmax=1)
plt.title("NDVI Heatmap - Green Cover Distribution")
plt.show()

# Enhanced Sustainability Boxplot
def color_based_on_score(score):
    if score >= 80:
        return "green"
    elif score >= 50:
        return "orange"
    else:
        return "red"

city_colors = df.groupby('City')['Sustainability_Score'].median().apply(color_based_on_score)

fig, ax = plt.subplots(figsize=(14, 7))
sns.boxplot(x="City", y="Sustainability_Score", data=df, palette=city_colors.to_dict())

plt.axhline(y=70, color='blue', linestyle='--', label='Good Sustainability Threshold (70)')

medians = df.groupby("City")["Sustainability_Score"].median()
for i, median in enumerate(medians):
    ax.text(i, median + 1, f"{median:.1f}", color='black', ha="center", fontweight='bold')

plt.title("Urban Sustainability Scores by City", fontsize=16)
plt.xlabel("City", fontsize=12)
plt.ylabel("Sustainability Score", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()