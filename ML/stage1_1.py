import cv2
import numpy as np
import matplotlib.pyplot as plt
import io

def produce_image():
    # 1️⃣ Load the overlayed image
    overlay_image = cv2.imread("C:/Users/Korou Kshetrimayum/Desktop/code/PLANUR/PlanUR/ML/terrain_image.png")
    # Ensure the image is loaded correctly
    if overlay_image is None:
        print("❌ Failed to load the overlayed image.")
        exit()
    # 2️⃣ Convert the image to HSV for color filtering
    hsv = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2HSV)
    
    # 3️⃣ Define color ranges for areas to EXCLUDE from green filling
    # Dark green (existing vegetation) - making range wider
    lower_dark_green = np.array([35, 50, 50])
    upper_dark_green = np.array([90, 255, 255])
    
    # Blue (water) - wider range
    lower_blue = np.array([90, 40, 40])
    upper_blue = np.array([150, 255, 255])
    
    # Roads and grays (multiple ranges)
    lower_gray1 = np.array([0, 0, 30])
    upper_gray1 = np.array([180, 30, 200])
    
    # Specific gray with hex #979da2
    gray_hex_bgr = np.uint8([[[162, 157, 151]]]) # BGR value of #979da2
    gray_hex_hsv = cv2.cvtColor(gray_hex_bgr, cv2.COLOR_BGR2HSV)[0][0]
    lower_gray_hex = np.array([gray_hex_hsv[0] - 15, gray_hex_hsv[1] - 50, gray_hex_hsv[2] - 50])
    upper_gray_hex = np.array([gray_hex_hsv[0] + 15, gray_hex_hsv[1] + 50, gray_hex_hsv[2] + 50])
    
    # 4️⃣ Create masks for all elements to exclude
    green_mask = cv2.inRange(hsv, lower_dark_green, upper_dark_green)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    gray_mask1 = cv2.inRange(hsv, lower_gray1, upper_gray1)
    gray_hex_mask = cv2.inRange(hsv, lower_gray_hex, upper_gray_hex)
    
    # 5️⃣ Combine all exclusion masks
    excluded_mask = cv2.bitwise_or(green_mask, blue_mask)
    excluded_mask = cv2.bitwise_or(excluded_mask, gray_mask1)
    excluded_mask = cv2.bitwise_or(excluded_mask, gray_hex_mask)
    
    # 6️⃣ Invert the mask to target only the fillable areas
    fillable_mask = cv2.bitwise_not(excluded_mask)
    
    # 7️⃣ Apply erosion to ensure excluded areas are fully covered
    kernel = np.ones((3, 3), np.uint8)
    fillable_mask = cv2.erode(fillable_mask, kernel, iterations=2)  # Increased iterations
    
    # 8️⃣ Apply the mask to fill only non-excluded areas with green
    green_filled = overlay_image.copy()
    green_filled[np.where(fillable_mask > 0)] = (0, 255, 0)
    
    # 9️⃣ Display the side-by-side visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Original Image
    axes[0].imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Overlay")
    axes[0].axis('off')
    
    # Processed Image
    axes[1].imshow(cv2.cvtColor(green_filled, cv2.COLOR_BGR2RGB))
    axes[1].set_title("New Potential Green Spaces")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # ✅ REMOVE plt.show()
    
    img = io.BytesIO()
    plt.savefig(img, format='png') # ✅ Save to memory
    img.seek(0) # ✅ Go back to the start of the buffer
    plt.close(fig) # ✅ Close the figure
    
    return img # ✅ Return the image buffer
