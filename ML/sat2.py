# ✅ **1. Import Libraries**
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import cv2
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt
import os


# ✅ **2. Google Maps API Key**
API_KEY = "AIzaSyC5P_A7ZmDK22JxOpYfyH9adOE6piR4y6M"  # Replace with your valid API key


# ✅ **3. Fetch Satellite and Terrain Images**
def get_google_maps_images(lat, lng, zoom=18, size="640x640"):
    """
    Fetches satellite and terrain images from Google Maps Static API.
    """
    sat_url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lng}&zoom={zoom}&size={size}&maptype=satellite&key={API_KEY}"
    )

    ter_url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lng}&zoom={zoom}&size={size}&maptype=terrain&key={API_KEY}"
    )

    try:
        sat_response = requests.get(sat_url)
        ter_response = requests.get(ter_url)

        if sat_response.status_code == 200 and ter_response.status_code == 200:
            sat_img = Image.open(BytesIO(sat_response.content))
            ter_img = Image.open(BytesIO(ter_response.content))

            sat_img.save("satellite_image.png")
            ter_img.save("terrain_image.png")
            print("✅ Satellite and Terrain images saved successfully.")
        else:
            print(f"❌ Failed to fetch images. Satellite: {sat_response.status_code}, Terrain: {ter_response.status_code}")

    except Exception as e:
        print(f"❌ Error fetching images: {e}")


# ✅ **4. Green Cover Extraction (Generalized Function)**
def extract_green_mask(image_path, lower_green=(25, 40, 20), upper_green=(90, 255, 255)):
    """
    Extracts green vegetation mask from an image using color thresholding.
    """
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return None

    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Green mask extraction
    mask = cv2.inRange(hsv, np.array(lower_green), np.array(upper_green))

    return mask, image


# ✅ **5. Overlay and Refined Green Cover Calculation**
def refined_green_cover(sat_mask, terrain_mask):
    """
    Refines green cover calculation by overlaying terrain green mask over satellite green mask.
    """
    if sat_mask is None or terrain_mask is None:
        print("❌ One or both masks are missing.")
        return 0

    # Combine masks (only green areas in both masks are considered)
    combined_mask = cv2.bitwise_and(sat_mask, terrain_mask)

    # Green area calculation
    green_pixels = np.sum(combined_mask > 0)
    total_pixels = sat_mask.shape[0] * sat_mask.shape[1]
    refined_green_cover_percentage = (green_pixels / total_pixels) * 100

    print(f"🌿 Refined Green Cover Percentage (overlayed): {refined_green_cover_percentage:.2f}%")

    # Display the results
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.title("Satellite Green Cover")
    plt.imshow(sat_mask, cmap="Greens")

    plt.subplot(1, 3, 2)
    plt.title("Terrain Green Cover")
    plt.imshow(terrain_mask, cmap="Greens")

    plt.subplot(1, 3, 3)
    plt.title("Refined Green Cover (Overlay)")
    plt.imshow(combined_mask, cmap="Greens")

    plt.show()

    return refined_green_cover_percentage


# ✅ **6. Export Green Cover Data to GeoJSON**
def export_geojson(lat, lng, green_cover_percentage, filename="green_cover_overlay.geojson"):
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


# ✅ **7. Main Execution**
def main():
    # 🌍 Example coordinates (Bengaluru)
    lat, lng = 13.102137528529248, 77.57940080843787  # Replace with your desired location

    print("📥 Fetching satellite and terrain images from Google Maps...")
    get_google_maps_images(lat, lng)

    print("\n🌿 Extracting green cover from satellite image...")
    sat_mask, sat_image = extract_green_mask("satellite_image.png")

    print("\n🌿 Extracting green cover from terrain image...")
    terrain_mask, terrain_image = extract_green_mask("terrain_image.png")

    print("\n🔍 Overlaying terrain and satellite green cover masks...")
    refined_cover = refined_green_cover(sat_mask, terrain_mask)

    print("\n📊 Exporting refined green cover data to GeoJSON...")
    export_geojson(lat, lng, refined_cover)

    print("\n✅ Analysis Complete!")


# ✅ **8. Run Program**
if __name__ == "__main__":
    main()
