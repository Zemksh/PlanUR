# 1. Import Libraries
import requests
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
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


# 3. Fetch Google Maps Traffic Layer Image
def get_google_traffic_image(lat, lng, zoom=15, size="640x640", api_key="YOUR_GOOGLE_MAPS_API_KEY"):
    """
    Fetches Google Maps traffic layer image.
    """
    url = (
        f"https://maps.googleapis.com/maps/api/staticmap?"
        f"center={lat},{lng}&zoom={zoom}&size={size}&maptype=roadmap"
        f"&style=feature:road|element:geometry|visibility:on"
        f"&style=feature:poi|visibility:off"
        f"&style=feature:water|visibility:off"
        f"&style=feature:landscape|visibility:off"
        f"&key={api_key}&traffic=true"
    )

    response = requests.get(url)

    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        img.save("google_traffic_image.png")
        print("Google Maps traffic image saved successfully.")
        return "google_traffic_image.png"
    else:
        print(f"Failed to fetch Google Maps traffic image. Status code: {response.status_code}")
        return None


# 4. Green Cover Extraction Excluding Water Bodies
def extract_green_cover_exclude_water(image_path, lower_green=(40, 40, 20), upper_green=(90, 255, 255),
                                      lower_water=(90, 50, 50), upper_water=(140, 255, 255)):
    """
    Extracts green cover percentage excluding water bodies.
    """
    image = cv2.imread(image_path)

    if image is None:
        print("Failed to load image.")
        return None, 0.0

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Mask for green vegetation
    green_mask = cv2.inRange(hsv, np.array(lower_green), np.array(upper_green))

    # Mask for water bodies
    water_mask = cv2.inRange(hsv, np.array(lower_water), np.array(upper_water))

    # Exclude water bodies from the green mask
    valid_land_mask = cv2.bitwise_not(water_mask)
    green_on_land = cv2.bitwise_and(green_mask, green_mask, mask=valid_land_mask)

    # Calculate green cover percentage excluding water
    green_pixels = np.sum(green_on_land > 0)
    land_pixels = np.sum(valid_land_mask > 0)

    green_cover_percentage = (green_pixels / land_pixels) * 100 if land_pixels > 0 else 0.0

    print(f"Green Cover (excluding water): {green_cover_percentage:.2f}%")

    return green_on_land, green_cover_percentage


# 5. Overlay Green Cover on Traffic Layer
def overlay_green_on_traffic(traffic_image_path, green_mask):
    """
    Overlays the green cover mask on the traffic layer image.
    """
    traffic_image = cv2.imread(traffic_image_path)

    if traffic_image is None:
        print("Failed to load traffic image.")
        return

    # Resize green mask to match traffic image dimensions
    green_mask_resized = cv2.resize(green_mask, (traffic_image.shape[1], traffic_image.shape[0]))

    # Convert green mask to 3 channels
    green_overlay = cv2.cvtColor(green_mask_resized, cv2.COLOR_GRAY2BGR)

    # Apply green mask on traffic image
    overlay = cv2.addWeighted(traffic_image, 0.7, green_overlay, 0.3, 0)

    # Save and display the overlay
    cv2.imwrite("traffic_green_overlay.png", overlay)
    print("Overlay image saved successfully.")

    # Display the result
    plt.figure(figsize=(12, 6))
    plt.title("Green Cover Overlay on Traffic Layer")
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# 6. Export Green Cover Data to GeoJSON
def export_geojson(lat, lng, green_cover_percentage, filename="green_cover_traffic_overlay.geojson"):
    """
    Exports green cover metrics to GeoJSON.
    """
    metrics = {
        "City": ["Location"],
        "Green_Cover_Percentage": [green_cover_percentage],
        "Latitude": [lat],
        "Longitude": [lng]
    }

    # Use geopandas points_from_xy to avoid shapely
    gdf = gpd.GeoDataFrame(
        metrics,
        geometry=gpd.points_from_xy([lng], [lat])
    )

    gdf.to_file(filename, driver="GeoJSON")
    print(f"Exported green cover metrics to {filename}.")


# 7. Main Execution
def main():
    # Example coordinates (Bengaluru, India)
    lat, lng = 13.101002816234251, 77.6027369800865  # Replace with your desired location
    google_maps_api_key = "YOUR_GOOGLE_MAPS_API_KEY"  # Replace with your actual API key

    print("Fetching OSM standard layer image...")
    osm_image_path = get_osm_standard_image(lat, lng)

    print("\nFetching Google Maps traffic image...")
    traffic_image_path = get_google_traffic_image(lat, lng, api_key=google_maps_api_key)

    if osm_image_path and traffic_image_path:
        print("\nExtracting green cover percentage excluding water...")
        green_mask, green_cover = extract_green_cover_exclude_water(osm_image_path)

        if green_mask is not None:
            print("\nOverlaying green cover on traffic image...")
            overlay_green_on_traffic(traffic_image_path, green_mask)

            print("\nExporting green cover data to GeoJSON...")
            export_geojson(lat, lng, green_cover)

            print("\nAnalysis Complete!")
        else:
            print("Failed to extract green cover. Analysis aborted.")
    else:
        print("Failed to retrieve one or both images. Analysis aborted.")


# 8. Run Program
if __name__ == "__main__":
    main()
