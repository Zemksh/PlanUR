import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 🌍 **1. Generate Simulated Dataset**
np.random.seed(42)

cities = ["New York", "Los Angeles", "Berlin", "Tokyo", "Mumbai", "Sydney"]
data = {
    "City": np.random.choice(cities, 200),
    "Temperature": np.random.uniform(10, 35, 200),  # °C
    "Rainfall": np.random.uniform(500, 2000, 200),  # mm
    "AQI": np.random.randint(30, 200, 200),  # Air quality index
    "Urban_Heat": np.random.uniform(0.5, 5, 200),  # Heat island effect
    "Transit_Usage": np.random.uniform(20, 90, 200),  # Public transit usage %
    "Walkability": np.random.uniform(30, 100, 200),  # Walkability score
    "Green_Space": np.random.uniform(10, 60, 200),  # Green space %
    "Biodiversity_Index": np.random.randint(100, 1000, 200),  # Biodiversity score
    "Energy_Consumption": np.random.uniform(5000, 15000, 200),  # kWh/person/year
    "Building_Efficiency": np.random.uniform(50, 100, 200),  # % compliance
    "Water_Efficiency": np.random.uniform(10, 70, 200),  # Water conservation %
    "Recycling_Rate": np.random.uniform(20, 90, 200)  # Waste recycling %
}

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv("sustainability_data.csv", index=False)

# ✅ **2. Preprocessing: Scale Features**
scaler = StandardScaler()
X = df.drop("City", axis=1)  # Features only
X_scaled = scaler.fit_transform(X)

# ✅ **3. Sustainability Score Calculation**
def calculate_sustainability_score(row):
    score = (
        row["Green_Space"] * 0.2 +
        row["Biodiversity_Index"] * 0.15 +
        (100 - row["AQI"]) * 0.15 +  # Lower AQI is better
        row["Walkability"] * 0.1 +
        row["Transit_Usage"] * 0.1 +
        row["Building_Efficiency"] * 0.1 +
        row["Water_Efficiency"] * 0.1 +
        row["Recycling_Rate"] * 0.1
    )
    return min(max(score, 0), 200)  # Clamp between 0-100

# Add sustainability scores to the dataset
df["Sustainability_Score"] = df.apply(calculate_sustainability_score, axis=1)

# ✅ **4. AI Model Training**
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

# ✅ **5. Generate Recommendations**
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

# Add recommendations to dataset
df["Recommendations"] = df.apply(generate_recommendations, axis=1)

# ✅ **6. Improved Visualization**

# Assign colors based on median sustainability score
def color_based_on_score(score):
    if score >= 80:
        return "green"   # High sustainability
    elif score >= 50:
        return "orange"  # Moderate sustainability
    else:
        return "red"     # Low sustainability

# Calculate median scores and apply colors
city_colors = df.groupby('City')['Sustainability_Score'].median().apply(color_based_on_score)

# Plot with enhanced visualization
fig, ax = plt.subplots(figsize=(14, 7))
sns.boxplot(x="City", y="Sustainability_Score", data=df, palette=city_colors.to_dict())

# Add a reference line at a "good sustainability threshold" (e.g., 70)
plt.axhline(y=70, color='blue', linestyle='--', label='Good Sustainability Threshold (70)')

# Annotate median scores
medians = df.groupby("City")["Sustainability_Score"].median()
for i, median in enumerate(medians):
    ax.text(i, median + 1, f"{median:.1f}", color='black', ha="center", fontweight='bold')

# Styling and formatting
plt.title("Urban Sustainability Scores by City", fontsize=16)
plt.xlabel("City", fontsize=12)
plt.ylabel("Sustainability Score", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()
plt.tight_layout()

plt.show()
