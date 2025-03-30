# 🌍 **1. Import Libraries**
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import cv2
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt

# ✅ **2. Google Maps API Key**
API_KEY = "AIzaSyC5P_A7ZmDK22JxOpYfyH9adOE6piR4y6M"  # Replace with your actual API key

# ✅ **3. Fetch Satellite Image from Google Maps**
def get_satellite_image(lat, lng, zoom=18, size="640x640"):
    """
    Fetches a satellite image from Google Maps Static API.
    """
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lng}&zoom={zoom}&size={size}&maptype=satellite&key={API_KEY}"
    )

    response = requests.get(url)

    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        image.save("satellite_image.png")
        print("✅ Satellite image saved successfully.")
    else:
        print(f"❌ Failed to fetch image: {response.status_code}")

# ✅ **4. Green Cover Extraction**
def extract_green_cover(image_path, lower_green = (25, 40, 20), upper_green = (90, 255, 255)):
    """
    Extracts green cover percentage from the satellite image using color thresholding.
    """
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create a mask for green vegetation
    mask = cv2.inRange(hsv, np.array(lower_green), np.array(upper_green))

    # Calculate green area percentage
    green_pixels = np.sum(mask > 0)
    total_pixels = image.shape[0] * image.shape[1]
    green_cover_percentage = (green_pixels / total_pixels) * 100

    # Display results
    print(f"🌿 Green Cover Percentage: {green_cover_percentage:.2f}%")

    # Show the original image and green mask
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Satellite Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title("Green Cover Mask")
    plt.imshow(mask, cmap="Greens")

    plt.show()

    return green_cover_percentage

# ✅ **5. Export Green Cover Data to GeoJSON**
def export_geojson(lat, lng, green_cover_percentage, filename="green_cover_metrics.geojson"):
    """
    Exports green cover metrics as GeoJSON.
    """
    metrics = {
        "City": ["Location"],
        "Green_Cover_Percentage": [green_cover_percentage],
        "Latitude": [lat],
        "Longitude": [lng]
    }

    gdf = gpd.GeoDataFrame(
        metrics,
        geometry=[Point(lng, lat) for _ in range(len(metrics["City"]))]
    )

    gdf.to_file(filename, driver="GeoJSON")
    print(f"✅ Exported green cover metrics to {filename}.")

# ✅ **6. Main Execution**
def main():
    # 🌍 Example coordinates (Central Park, NY)
    lat, lng = 10.529459704568222, 76.22184885536457  # Replace with your desired location

    print("📥 Fetching satellite image from Google Maps...")
    get_satellite_image(lat, lng)

    print("\n🌿 Extracting green cover percentage...")
    green_cover = extract_green_cover("satellite_image.png")

    print("\n📊 Exporting green cover data to GeoJSON...")
    export_geojson(lat, lng, green_cover)

    print("\n✅ Analysis Complete!")

# ✅ **7. Run Program**
if __name__ == "__main__":
    main()