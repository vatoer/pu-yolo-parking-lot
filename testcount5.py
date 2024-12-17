import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import os

# Initialize YOLO model
model = YOLO('yolov8s.pt')

# Load class names
with open("coco.txt", "r") as f:
    class_list = f.read().strip().split("\n")

# Define parking areas
parking_areas = [
    [(52, 364), (30, 417), (73, 412), (88, 369)],
    [(105, 353), (86, 428), (137, 427), (146, 358)],
    [(159, 354), (150, 427), (204, 425), (203, 353)],
    [(217, 352), (219, 422), (273, 418), (261, 347)],
    [(274, 345), (286, 417), (338, 415), (321, 345)],
    [(336, 343), (357, 410), (409, 408), (382, 340)],
    [(396, 338), (426, 404), (479, 399), (439, 334)],
    [(458, 333), (494, 397), (543, 390), (495, 330)],
    [(511, 327), (557, 388), (603, 383), (549, 324)],
    [(564, 323), (615, 381), (654, 372), (596, 315)],
    [(616, 316), (666, 369), (703, 363), (642, 312)],
    [(674, 311), (730, 360), (764, 355), (707, 308)]
]

def draw_parking_area(frame, area, occupied, index):
    color = (0, 0, 255) if occupied else (0, 255, 0)
    cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
    position = tuple(np.mean(area, axis=0).astype(int))
    cv2.putText(frame, str(index + 1), position, cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

def save_frame_and_annotations(frame, annotations, output_dir, frame_number):
    os.makedirs(output_dir, exist_ok=True)
    frame_path = os.path.join(output_dir, f"frame_{frame_number}.jpg")
    annotation_path = os.path.join(output_dir, f"frame_{frame_number}.txt")
    
    cv2.imwrite(frame_path, frame)
    with open(annotation_path, "w") as f:
        for annotation in annotations:
            f.write(" ".join(map(str, annotation)) + "\n")

def process_video(video_path, output_dir, extraction_rate=1):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every nth frame (based on extraction_rate)
        if frame_number % extraction_rate == 0:
            frame = cv2.resize(frame, (1020, 500))
            results = model.predict(frame, verbose=False)

            # Extract predictions
            detections = results[0].boxes.data.cpu().numpy() if results[0].boxes else []
            annotations = []

            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                class_name = class_list[int(cls)]
                if class_name == 'car':
                    cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
                    for i, area in enumerate(parking_areas):
                        if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                            annotations.append([class_name, x1, y1, x2, y2, i + 1])
                            draw_parking_area(frame, area, True, i)
                            break

            # Save frame and annotations
            save_frame_and_annotations(frame, annotations, output_dir, frame_number)

        frame_number += 1

        # Show frame
        cv2.imshow("Parking Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the video processing
process_video("parking1.mp4", "output", extraction_rate=30)
