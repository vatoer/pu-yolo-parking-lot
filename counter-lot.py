import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

# Load the YOLO model and class labels
model = YOLO('yolov8s.pt')
with open("coco.txt", "r") as file:
    class_list = file.read().split("\n")

# Parking areas defined as a list of polygons
parking_areas = [
    [(52, 364), (30, 417), (73, 412), (88, 369)],    # Area 1
    [(105, 353), (86, 428), (137, 427), (146, 358)], # Area 2
    [(159, 354), (150, 427), (204, 425), (203, 353)], # Area 3
    [(217, 352), (219, 422), (273, 418), (261, 347)], # Area 4
    [(274, 345), (286, 417), (338, 415), (321, 345)], # Area 5
    [(336, 343), (357, 410), (409, 408), (382, 340)], # Area 6
    [(396, 338), (426, 404), (479, 399), (439, 334)], # Area 7
    [(458, 333), (494, 397), (543, 390), (495, 330)], # Area 8
    [(511, 327), (557, 388), (603, 383), (549, 324)], # Area 9
    [(564, 323), (615, 381), (654, 372), (596, 315)], # Area 10
    [(616, 316), (666, 369), (703, 363), (642, 312)], # Area 11
    [(674, 311), (730, 360), (764, 355), (707, 308)]  # Area 12
]

# Function for mouse movement (optional)
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print([x, y])

cv2.namedWindow('Parking Lot')
cv2.setMouseCallback('Parking Lot', RGB)

# Video capture setup
cap = cv2.VideoCapture('parking1.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    detections = pd.DataFrame(results[0].boxes.data).astype("float")

    # Initialize counts for each parking area
    occupied = [0] * len(parking_areas)

    # Process each detection
    for _, row in detections.iterrows():
        x1, y1, x2, y2, _, class_id = row
        class_name = class_list[int(class_id)]

        if 'car' in class_name:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Check if car is in any parking area
            for i, area in enumerate(parking_areas):
                if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                    occupied[i] = 1  # Mark the area as occupied
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    cv2.putText(frame, class_name, (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                    break

    # Draw parking areas and display availability
    total_spaces = len(parking_areas)
    free_spaces = total_spaces - sum(occupied)

    for i, area in enumerate(parking_areas):
        color = (0, 0, 255) if occupied[i] else (0, 255, 0)  # Red if occupied, Green if free
        cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
        cv2.putText(frame, str(i + 1), area[0], cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

    # Display free space count
    cv2.putText(frame, f"Free Spaces: {free_spaces}", (23, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

    # Show the result
    cv2.imshow("Parking Lot", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Exit on 'ESC'
        break

cap.release()
cv2.destroyAllWindows()
