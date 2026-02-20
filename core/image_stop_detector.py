import cv2
import numpy as np
import pandas as pd


def detect_stops_from_image(uploaded_file, base_lat=23.8512, base_lon=72.1266):
    """
    Detect circular markers from map image.
    Returns pseudo coordinates usable by routing engine.
    """

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 1.5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=50,
        param2=30, # Increased slightly to reduce false positives
        minRadius=5,
        maxRadius=60, # Increased to catch larger markers
    )

    if circles is None:
        raise ValueError("No stops detected in image. Try an image with clearer circular markers.")

    circles = np.round(circles[0, :]).astype("int")

    # normalize to pseudo geo space
    h, w = gray.shape
    
    # Scale factor: 0.05 degrees is roughly 5km
    scale_lat = 0.05
    scale_lon = 0.05

    data = []
    for i, (x, y, r) in enumerate(circles):
        # Invert Y for latitude (up is positive)
        # Normalize x,y to -0.5 to 0.5 range, then multiply by scale
        norm_x = (x / w) - 0.5
        norm_y = ((h - y) / h) - 0.5 # Inverted Y
        
        lat = base_lat + (norm_y * scale_lat)
        lon = base_lon + (norm_x * scale_lon)
        
        data.append([f"Stop_{i+1}", lat, lon])

    df = pd.DataFrame(data, columns=["stop_name", "lat", "lon"])
    return df
