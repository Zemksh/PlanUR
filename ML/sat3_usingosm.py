# 1. Import Libraries
import requests
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from shapely.geometry import Point
import geopandas as gpd
import matplotlib.pyplot as plt

# 2. Fetch OSM Standard Layer Image with Headers
def get_osm_standard_image(lat, lng, zoom=15, size=640):
    """
    Fetches OSM standard layer image with proper headers.
    """
    x = int((lng + 180) / 360 * (2**zoom))
    y = int((1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi) / 2 * (2**zoom))
    
    url = f"https://a.tile.openstreetmap.org/{zoom}/{x}/{y}.png"

    # Add User-Agent header to avoid 403 error
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python-Requests/2.28.1"
    }
    
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img.save("osm_standard_image.png")
        print("OSM standard layer image saved successfully.")
        return "osm_standard_image.png"
    else:
        print(f"Failed to fetch OSM standard image. Status code: {response.status_code}")
        return None

# 3. Green Cover Extraction
def extract_green_cover(image_path, lower_green=(40, 40, 20), upper_green=(90, 255, 255)):
    """
    Extracts green cover percentage from OSM map image using color thresholding.
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
    print(f"Green Cover Percentage: {green_cover_percentage:.2f}%")

    # Show the original image and green mask
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original OSM Map Image (Standard Layer)")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title("Green Cover Mask")
    plt.imshow(mask, cmap="Greens")

    plt.show()

    return green_cover_percentage

# 4. Export Green Cover Data to GeoJSON
def export_geojson(lat, lng, green_cover_percentage, filename="green_cover_osm.geojson"):
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
    print(f"Exported green cover metrics to {filename}.")

# 5. Main Execution
def main():
    # Example coordinates (Bengaluru, India)
    lat, lng = 13.101002816234251, 77.6027369800865  # Replace with your desired location

    print("Fetching OSM standard layer image...")
    image_path = get_osm_standard_image(lat, lng)

    if image_path:
        print("\nExtracting green cover percentage...")
        green_cover = extract_green_cover(image_path)

        print("\nExporting green cover data to GeoJSON...")
        export_geojson(lat, lng, green_cover)

        print("\nAnalysis Complete!")
    else:
        print("OSM image could not be retrieved. Analysis aborted.")

# 6. Run Program
if __name__ == "__main__":
    main()

