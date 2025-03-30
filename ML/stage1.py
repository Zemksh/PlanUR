# 1. Import Libraries
import requests
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from geopandas import GeoDataFrame, points_from_xy
import matplotlib.pyplot as plt


# 2. Fetch OSM Standard Layer Image with Headers
def get_osm_standard_image(lat, lng, zoom=15, size=640):
    """
    Fetches OSM standard layer image with proper headers.
    """
    x = int((lng + 180) / 360 * (2**zoom))
    y = int((1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi) / 2 * (2**zoom))
    
    url = f"https://a.tile.openstreetmap.org/{zoom}/{x}/{y}.png"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python-Requests/2.28.1"
    }
    
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img.save("osm_standard_image.png")
        print("✅ OSM standard layer image saved successfully.")
        return "osm_standard_image.png"
    else:
        print(f"❌ Failed to fetch OSM standard image. Status code: {response.status_code}")
        return None


# 3. Fetch Google Maps Traffic Layer Image (Using Dynamic Tiles API)
def get_google_traffic_image(lat, lng, zoom=15, size="640x640", api_key="YOUR_GOOGLE_MAPS_API_KEY"):
    """
    Fetches a Google Maps traffic image using Dynamic Tiles API.
    """
    # Tile calculations
    x = int((lng + 180) / 360 * (2 ** zoom))
    y = int((1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi) / 2 * (2 ** zoom))
    
    url = (
        f"https://mt1.google.com/vt/lyrs=m@221097413,traffic&x={x}&y={y}&z={zoom}&scale=1&key={api_key}"
    )

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Python-Requests/2.28.1"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img.save("google_traffic_image.png")
        print("✅ Google Maps traffic image saved successfully.")
        return "google_traffic_image.png"
    else:
        print(f"❌ Failed to fetch Google Maps traffic image. Status code: {response.status_code}")
        return None


# 4. Green Cover Extraction with Water Exclusion
def extract_green_cover_exclude_water(image_path, lower_green=(40, 40, 20), upper_green=(90, 255, 255),
                                      lower_water=(90, 50, 50), upper_water=(140, 255, 255)):
    """
    Extracts green cover percentage from the image excluding water bodies.
    """
    image = cv2.imread(image_path)

    if image is None:
        print("❌ Failed to load image.")
        return None, 0.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Create masks
    green_mask = cv2.inRange(hsv, np.array(lower_green), np.array(upper_green))
    water_mask = cv2.inRange(hsv, np.array(lower_water), np.array(upper_water))

    # Exclude water area
    land_mask = cv2.bitwise_not(water_mask)  # Invert water mask → land area
    green_on_land = cv2.bitwise_and(green_mask, green_mask, mask=land_mask)

    # Calculate green cover excluding water
    green_pixels = np.sum(green_on_land > 0)
    land_pixels = np.sum(land_mask > 0)

    # Handle division by zero in case there are no land pixels
    green_cover_percentage = (green_pixels / land_pixels) * 100 if land_pixels > 0 else 0.0

    print(f"🌿 Green Cover (excluding water): {green_cover_percentage:.2f}%")

    # Return resized mask and percentage
    return cv2.resize(green_on_land, (640, 640)), green_cover_percentage


# 5. Overlay Green Mask on Traffic Image with Green Color
def overlay_green_on_traffic(traffic_image_path, green_mask):
    """
    Overlays green cover mask (excluding water) on top of the traffic image with actual green color.
    """
    traffic_image = cv2.imread(traffic_image_path)

    # Ensure both images are the same size
    green_mask_resized = cv2.resize(green_mask, (traffic_image.shape[1], traffic_image.shape[0]))

    # Convert mask to true green overlay
    green_overlay = np.zeros_like(traffic_image)
    green_overlay[:, :, 1] = green_mask_resized  # Green channel only

    # Blend the green mask with the traffic image
    overlay = cv2.addWeighted(traffic_image, 0.7, green_overlay, 0.5, 0)

    # Save and display the overlay
    cv2.imwrite("traffic_green_overlay_excl_water.png", overlay)
    print("✅ Green overlay (excluding water) image saved successfully.")

    # Display the result
    plt.figure(figsize=(12, 6))
    plt.title("Green Cover Overlay on Traffic Layer (Excluding Water)")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# 6. Export Green Cover Data to GeoJSON
def export_geojson(lat, lng, green_cover_percentage, filename="green_cover_excl_water.geojson"):
    """
    Exports green cover metrics excluding water to GeoJSON.
    """
    metrics = {
        "City": ["Location"],
        "Green_Cover_Percentage": [green_cover_percentage],
        "Latitude": [lat],
        "Longitude": [lng]
    }

    gdf = GeoDataFrame(
        metrics,
        geometry=points_from_xy([lng], [lat])
    )

    gdf.to_file(filename, driver="GeoJSON")
    print(f"✅ Exported green cover metrics to {filename}.")


# 7. Main Execution
def main():
    """
    Main program execution.
    """
    # Example coordinates (Bengaluru, India)
    lat, lng = 13.101002816234251, 77.6027369800865
    google_maps_api_key = "YOUR_GOOGLE_MAPS_API_KEY"  # Replace with your actual API key

    print("🚀 Fetching OSM standard layer image...")
    osm_image_path = get_osm_standard_image(lat, lng)

    print("\n🚦 Fetching Google Maps traffic image...")
    traffic_image_path = get_google_traffic_image(lat, lng, api_key=google_maps_api_key)

    if osm_image_path and traffic_image_path:
        print("\n🌿 Extracting green cover percentage excluding water...")
        green_mask, green_cover = extract_green_cover_exclude_water(osm_image_path)

        if green_mask is not None:
            print("\n🛣️ Overlaying green cover on traffic image...")
            overlay_green_on_traffic(traffic_image_path, green_mask)

            print("\n🌍 Exporting green cover data to GeoJSON...")
            export_geojson(lat, lng, green_cover)

            print("\n✅ Analysis Complete!")
        else:
            print("❌ Failed to generate green cover mask.")
    else:
        print("❌ Failed to retrieve one or both images. Analysis aborted.")


# 8. Run Program
if __name__ == "__main__":
    main()
