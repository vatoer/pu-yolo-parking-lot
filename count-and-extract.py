import cv2
import pandas as pd
import numpy as np
import json
import os
from ultralytics import YOLO
import time

# Initialize the YOLO model
model = YOLO('yolov8s.pt')

# Parking lot areas
parking_areas = {
    "space_1": [(52, 364), (30, 417), (73, 412), (88, 369)],
    "space_2": [(105, 353), (86, 428), (137, 427), (146, 358)],
    "space_3": [(159, 354), (150, 427), (204, 425), (203, 353)],
    "space_4": [(217, 352), (219, 422), (273, 418), (261, 347)],
    "space_5": [(274, 345), (286, 417), (338, 415), (321, 345)],
    "space_6": [(336, 343), (357, 410), (409, 408), (382, 340)],
    "space_7": [(396, 338), (426, 404), (479, 399), (439, 334)],
    "space_8": [(458, 333), (494, 397), (543, 390), (495, 330)],
    "space_9": [(511, 327), (557, 388), (603, 383), (549, 324)],
    "space_10": [(564, 323), (615, 381), (654, 372), (596, 315)],
    "space_11": [(616, 316), (666, 369), (703, 363), (642, 312)],
    "space_12": [(674, 311), (730, 360), (764, 355), (707, 308)]
}

# Load class names for YOLO
with open("coco.txt", "r") as f:
    class_list = f.read().split("\n")

# Create output directories
output_images = "extracted_images"
output_labels = "ground_truth_labels"
os.makedirs(output_images, exist_ok=True)
os.makedirs(output_labels, exist_ok=True)

# Ground truth storage
ground_truth_data = []

# Video input
cap = cv2.VideoCapture('parking1.mp4')
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_counter += 1
    frame = cv2.resize(frame, (1020, 500))

    # YOLO prediction
    results = model.predict(frame, verbose=False)
    detections = results[0].boxes.data
    detections_df = pd.DataFrame(detections).astype("float")

    # Initialize occupancy tracker for each parking space
    space_occupancy = {key: 0 for key in parking_areas.keys()}

    # Process detections
    for _, row in detections_df.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row[:6])
        class_name = class_list[class_id]

        # Check only 'car' detections
        if 'car' in class_name:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of bounding box

            # Check if the car belongs to a parking space
            for space, area in parking_areas.items():
                result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
                if result >= 0:  # Point is inside the polygon
                    space_occupancy[space] = 1
                    # Draw car bounding box and center
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    break  # Skip other spaces once matched

    # Draw parking space polygons with color based on occupancy
    for space, area in parking_areas.items():
        color = (0, 0, 255) if space_occupancy[space] == 1 else (0, 255, 0)
        cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
        cv2.putText(frame, space.split("_")[1], area[0], cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

    # Save the frame as an image
    image_path = os.path.join(output_images, f"frame_{frame_counter}.jpg")
    cv2.imwrite(image_path, frame)

    # Append ground truth labels for the frame
    ground_truth_data.append({
        "frame": frame_counter,
        "labels": space_occupancy
    })

    # Display the frame
    cv2.imshow("Parking Lot Detection", frame)

    # Exit on pressing 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Save ground truth labels to a JSON file
with open(os.path.join(output_labels, "ground_truth.json"), "w") as json_file:
    json.dump(ground_truth_data, json_file, indent=4)

cap.release()
cv2.destroyAllWindows()
