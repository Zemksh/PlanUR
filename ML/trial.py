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

# 3. Green Cover Extraction Excluding Water Bodies
def extract_green_cover_exclude_water(image_path, lower_green=(40, 40, 20), upper_green=(90, 255, 255), 
                                      lower_water=(90, 50, 50), upper_water=(140, 255, 255)):
    """
    Extracts green cover percentage excluding water bodies.
    """
    image = cv2.imread(image_path)

    if image is None:
        print("Failed to load image.")
        return 0.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Mask for green vegetation
    green_mask = cv2.inRange(hsv, np.array(lower_green), np.array(upper_green))

    # Mask for water bodies
    water_mask = cv2.inRange(hsv, np.array(lower_water), np.array(upper_water))

    # Combine masks: exclude water
    valid_land_mask = cv2.bitwise_not(water_mask)  # Land area (excluding water)
    green_on_land = cv2.bitwise_and(green_mask, green_mask, mask=valid_land_mask)

    # Calculate green cover percentage excluding water
    green_pixels = np.sum(green_on_land > 0)
    land_pixels = np.sum(valid_land_mask > 0)

    # Handle case where there is no land area
    green_cover_percentage = (green_pixels / land_pixels) * 100 if land_pixels > 0 else 0.0

    print(f"Green Cover (excluding water): {green_cover_percentage:.2f}%")

    # Display only original map and green mask
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.title("Original OSM Map Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Green mask excluding water
    plt.subplot(1, 2, 2)
    plt.title("Green Cover Mask (Excluding Water)")
    plt.imshow(green_on_land, cmap="Greens")

    plt.show()

    return green_cover_percentage

# 4. Export Green Cover Data to GeoJSON
def export_geojson(lat, lng, green_cover_percentage, filename="green_cover_excluding_water.geojson"):
    """
    Exports green cover metrics to GeoJSON.
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
        print("\nExtracting green cover percentage excluding water...")
        green_cover = extract_green_cover_exclude_water(image_path)

        print("\nExporting green cover data to GeoJSON...")
        export_geojson(lat, lng, green_cover)

        print("\nAnalysis Complete!")
    else:
        print("OSM image could not be retrieved. Analysis aborted.")

# 6. Run Program
if __name__ == "__main__":
    main()
