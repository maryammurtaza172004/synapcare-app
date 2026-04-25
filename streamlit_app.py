import streamlit as st
import cv2
import numpy as np
import os

# Set up the look of the website
st.set_page_config(page_title="SynapCare Fracture Detector", layout="wide")

st.title("🦴 SynapCare: Minute Fracture Detection")
st.write("Biomedical Engineering Prototype for Home-Based Care")

# --- SIDEBAR: Where we pick our images ---
st.sidebar.header("Select Image Source")
source = st.sidebar.radio("Go to:", ["Sample Dataset", "Upload New X-ray"])

image_to_process = None

# Logic to load images from your folder
if source == "Sample Dataset":
    folder = "sample_data"
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.JPG', '.jpeg'))]
        if files:
            selected_file = st.sidebar.selectbox("Choose a sample wrist:", files)
            image_path = os.path.join(folder, selected_file)
            image_to_process = cv2.imread(image_path, 0)
        else:
            st.sidebar.warning("No images found in 'sample_data' folder.")
    else:
        st.sidebar.error("Folder 'sample_data' not found!")

else:
    uploaded_file = st.sidebar.file_uploader("Upload an X-ray", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_to_process = cv2.imdecode(file_bytes, 0)

# --- MAIN ANALYSIS AREA ---
if image_to_process is not None:
    # 1. Processing Logic (THE CLEAN ENGINE)
    # Step A: Boost contrast so minute cracks show up
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(image_to_process)
    
    # Step B: Slight blur to ignore tiny 'noise' and smooth healthy bone edges
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # Step C: Edge detection (Focusing on sharp breaks)
    edges = cv2.Canny(blurred, 100, 200)

    # Step D: The 'Top-Hat' Clean up (Removes long lines, keeps small cracks)
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_TOPHAT, kernel) 

    # 2. Measurement Math
    red_pixels = np.argwhere(edges > 0)
    if len(red_pixels) > 0:
        dist = np.linalg.norm(red_pixels[0] - red_pixels[-1])
        extent_mm = dist / 10 # Assuming 10px = 1mm for the prototype
    else:
        extent_mm = 0

    # 3. Display Results in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original X-ray")
        st.image(image_to_process, use_container_width=True)
        
    with col2:
        st.subheader("AI Analysis (Detection Mode)")
        # Convert to color so we can show RED highlights
        color_result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        color_result[edges > 0] = [255, 0, 0] # Red highlights for edges
        st.image(color_result, use_container_width=True)

    # 4. The Final Report
    st.divider()
    if extent_mm > 0.5: # Only show if we found a significant line
        st.error(f"**ALERT:** Potential Fracture/Crack Detected.")
        st.info(f"**Measurement:** Estimated Extent is {extent_mm:.2f} mm")
    else:
        st.success("No significant minute fractures detected in this scan.")
else:
    st.info("Select an image from the sidebar to begin.")