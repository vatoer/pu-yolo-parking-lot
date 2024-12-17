import cv2
import json
import numpy as np
from ultralytics import YOLO
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import csv

# Paths
ground_truth_file = "ground_truth_labels/ground_truth.json"
images_folder = "extracted_images"
model_path = "yolov8s.pt"
report_file = "detailed_parking_report.csv"  # Output report file

# Load YOLO model
model = YOLO(model_path)

# Parking lot areas (must match the original)
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

# Load ground truth data
with open(ground_truth_file, "r") as f:
    ground_truth_data = json.load(f)

# Initialize lists for evaluation and detailed report
y_true = []  # Ground truth labels
y_pred = []  # Predicted labels
detailed_report = []

# Process each frame/image
for frame_data in ground_truth_data:
    frame_id = frame_data["frame"]
    ground_truth_labels = frame_data["labels"]

    # Load the corresponding image
    image_path = os.path.join(images_folder, f"frame_{frame_id}.jpg")
    if not os.path.exists(image_path):
        continue

    frame = cv2.imread(image_path)
    results = model.predict(frame, verbose=False)
    detections = results[0].boxes.data

    # Initialize predictions for this frame
    space_occupancy_pred = {key: 0 for key in parking_areas.keys()}

    # Process YOLO detections
    for *box, _, class_id in detections:
        class_id = int(class_id)
        if class_id == 2:  # Class 'car' in COCO dataset
            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of bounding box

            # Check parking spaces
            for space, area in parking_areas.items():
                result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
                if result >= 0:
                    space_occupancy_pred[space] = 1
                    break

    # Append ground truth and predictions for all spaces
    y_true.extend([ground_truth_labels[space] for space in parking_areas.keys()])
    y_pred.extend([space_occupancy_pred[space] for space in parking_areas.keys()])

    # Generate detailed report for this frame
    for space, actual in ground_truth_labels.items():
        prediction = space_occupancy_pred[space]
        result = (
            "True Positive" if actual == 1 and prediction == 1 else
            "True Negative" if actual == 0 and prediction == 0 else
            "False Positive" if actual == 0 and prediction == 1 else
            "False Negative"
        )
        detailed_report.append({
            "Frame": frame_id,
            "Parking Space": space,
            "Ground Truth": actual,
            "Prediction": prediction,
            "Result": result
        })

# Calculate performance metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Display results
print("Performance Metrics:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Write detailed report to CSV
with open(report_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=["Frame", "Parking Space", "Ground Truth", "Prediction", "Result"])
    writer.writeheader()
    writer.writerows(detailed_report)

print(f"Detailed parking analysis report saved to {report_file}")
