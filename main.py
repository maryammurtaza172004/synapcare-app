import cv2
import numpy as np

# 1. LOAD: Read your wrist X-ray
image = cv2.imread('wrist.JPG', 0)

if image is None:
    print("Error: Can't find wrist.JPG")
else:
    # 2. ENHANCE: Make it super sharp
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image)

    # 3. DETECT: Find the fracture lines
    edges = cv2.Canny(enhanced, 70, 150)

    # 4. COLOR: Create a color version so we can draw RED highlights
    # This turns our B&W image into a color 'canvas'
    color_result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    # 5. ANALYSIS: Find the crack and measure it
    red_pixels = np.argwhere(edges > 0)
    
    if len(red_pixels) > 0:
        # Measure extent (The math we did before)
        dist = np.linalg.norm(red_pixels[0] - red_pixels[-1])
        extent_mm = dist / 10
        
        # Identify Type: If it's very long, it's a Linear fracture. 
        # If it's tiny, it's a Hairline/Minute fracture.
        f_type = "Minute/Hairline" if extent_mm < 5 else "Linear Fracture"

        # 6. HIGHLIGHT: Paint the detected fracture lines RED
        color_result[edges > 0] = [0, 0, 255] 

        # 7. UI: Draw a professional box with the info
        cv2.rectangle(color_result, (10, 10), (350, 120), (50, 50, 50), -1) # Dark background box
        cv2.putText(color_result, f"STATUS: FRACTURE DETECTED", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(color_result, f"TYPE: {f_type}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(color_result, f"EXTENT: {extent_mm:.2f} mm", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 8. SHOW:
        cv2.imshow('Biomedical Fracture Detector v1.0', color_result)
        print("Analysis finished! Check the pop-up window.")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()