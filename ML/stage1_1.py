
import cv2
import numpy as np
import matplotlib.pyplot as plt
import io
import requests
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, Response, url_for
app = Flask(__name__)
def get_google_traffic_image(lat, lng, zoom=15, size="640x640", api_key="AIzaSyC5P_A7ZmDK22JxOpYfyH9adOE6piR4y6M"):
    """
    Fetches a Google Maps traffic image using Dynamic Tiles API.
    """
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
        return "google_traffic_image.png"
    else:
        print(f"❌ Failed to fetch Google Maps traffic image. Status code: {response.status_code}")
        return None

def main(lat,lng):
    # Hardcoded coordinates for Delhi
    latitude = lat
    longitude = lng
    google_maps_api_key = "AIzaSyC5P_A7ZmDK22JxOpYfyH9adOE6piR4y6M"  # Replace with your actual API key
    
    # Fetch Google Maps traffic image dynamically
    overlay_image_path = get_google_traffic_image(latitude, longitude, api_key=google_maps_api_key)
    if overlay_image_path is None:
        print("❌ Could not retrieve traffic image.")
        return None
    
    overlay_image = cv2.imread(overlay_image_path)
    if overlay_image is None:
        print("❌ Failed to load the overlayed image.")
        return None
    
    hsv = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2HSV)
    
    lower_dark_green = np.array([35, 50, 50])
    upper_dark_green = np.array([90, 255, 255])
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([150, 255, 255])
    lower_gray1 = np.array([0, 0, 30])
    upper_gray1 = np.array([180, 30, 200])
    
    gray_hex_bgr = np.uint8([[[162, 157, 151]]])
    gray_hex_hsv = cv2.cvtColor(gray_hex_bgr, cv2.COLOR_BGR2HSV)[0][0]
    lower_gray_hex = np.array([gray_hex_hsv[0] - 15, gray_hex_hsv[1] - 50, gray_hex_hsv[2] - 50])
    upper_gray_hex = np.array([gray_hex_hsv[0] + 15, gray_hex_hsv[1] + 50, gray_hex_hsv[2] + 50])
    
    green_mask = cv2.inRange(hsv, lower_dark_green, upper_dark_green)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    gray_mask1 = cv2.inRange(hsv, lower_gray1, upper_gray1)
    gray_hex_mask = cv2.inRange(hsv, lower_gray_hex, upper_gray_hex)
    
    excluded_mask = cv2.bitwise_or(green_mask, blue_mask)
    excluded_mask = cv2.bitwise_or(excluded_mask, gray_mask1)
    excluded_mask = cv2.bitwise_or(excluded_mask, gray_hex_mask)
    
    fillable_mask = cv2.bitwise_not(excluded_mask)
    
    kernel = np.ones((3, 3), np.uint8)
    fillable_mask = cv2.erode(fillable_mask, kernel, iterations=2)
    
    green_filled = overlay_image.copy()
    green_filled[np.where(fillable_mask > 0)] = (0, 255, 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    axes[0].imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Overlay")
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(green_filled, cv2.COLOR_BGR2RGB))
    axes[1].set_title("New Potential Green Spaces")
    axes[1].axis('off')
    
    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)
    
    return img