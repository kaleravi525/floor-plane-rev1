import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Floor Plan Symbol Auto-Counter")
st.write("Upload a floor plan and a symbol. The app will find and count all matches.")

# 1. Upload files directly on the web page
plan_file = st.file_uploader("Upload Floor Plan Image", type=["png", "jpg", "jpeg"])
symbol_file = st.file_uploader("Upload the Symbol to Find", type=["png", "jpg", "jpeg"])

if plan_file and symbol_file:
    # Convert uploaded files into readable image data
    plan_img = np.array(Image.open(plan_file).convert('L')) 
    symbol_img = np.array(Image.open(symbol_file).convert('L'))
    
    # Get the width and height of the symbol
    w, h = symbol_img.shape[::-1]
    
    # 2. Add a slider to the web page to adjust search strictness
    threshold = st.slider("Match Accuracy Threshold (0.8 = 80% match)", 0.5, 1.0, 0.8)
    
    # 3. The OpenCV engine scans the floor plan for the symbol
    res = cv2.matchTemplate(plan_img, symbol_img, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    
    # 4. Draw red boxes around everything it finds
    plan_color = cv2.cvtColor(plan_img, cv2.COLOR_GRAY2BGR)
    count = 0
    
    # We use a set to avoid counting the exact same pixel twice
    found_points = []
    for pt in zip(*loc[::-1]):
        # Check if we already found a match very close to this spot
        if not any(abs(pt[0] - fp[0]) < 10 and abs(pt[1] - fp[1]) < 10 for fp in found_points):
            cv2.rectangle(plan_color, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 3)
            found_points.append(pt)
            count += 1
            
    # 5. Display the final counted results on the web page!
    st.success(f"System found {count} matching symbols!")
    st.image(plan_color, caption="Automatically Highlighted Floor Plan", use_container_width=True)