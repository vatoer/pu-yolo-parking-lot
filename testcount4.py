import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time
import os
import json

# Define class list (assuming it's a separate file)
class_list = ["car", "bus", "motorcycle", ...]  # Replace with actual classes

# Define parking area coordinates as a dictionary with empty lists for each area
parking_areas = {
    "area1": [(52, 364), (30, 417), (73, 412), (88, 369)],
    "area2": [(105, 353), (86, 428), (137, 427), (146, 358)],
    "area3":[(159,354),(150,427),(204,425),(203,353)],
    "area4":[(217,352),(219,422),(273,418),(261,347)],
    "area5":[(274,345),(286,417),(338,415),(321,345)],
    "area6":[(336,343),(357,410),(409,408),(382,340)],
    "area7":[(396,338),(426,404),(479,399),(439,334)],
    "area8":[(458,333),(494,397),(543,390),(495,330)],
    "area9":[(511,327),(557,388),(603,383),(549,324)],
    "area10":[(564,323),(615,381),(654,372),(596,315)],
    "area11":[(616,316),(666,369),(703,363),(642,312)],
    "area12":[(674,311),(730,360),(764,355),(707,308)],
    # ...
}

model = YOLO('yolov8s.pt')

extraction_interval = 1  # Extract frames every second


def count_cars_in_area(frame, area_name):
    """
    Counts cars within a specific parking area.

    Args:
        frame (np.ndarray): The video frame image.
        area_name (str): The name of the parking area (key in parking_areas dict).

    Returns:
        int: The number of cars detected in the area.
    """
    area_points = np.array(parking_areas[area_name], np.int32)
    results = model.predict(frame)
    cars = 0

    # Iterate over the detected objects
    for result in results:
        for box in result.boxes:
            # Check if the detected object is within the parking area
            x1, y1, x2, y2 = box.xyxy[0]
            center_x = float((x1 + x2) / 2)
            center_y = float((y1 + y2) / 2)
            if cv2.pointPolygonTest(area_points, (center_x, center_y), False) >= 0:
                cars += 1

    return cars


def display_parking_info(frame, car_counts):
    """
    Displays the number of cars in each parking area.

    Args:
        frame (np.ndarray): The video frame image.
        car_counts (dict): A dictionary containing car counts for each area.
    """
    font = cv2.FONT_HERSHEY_COMPLEX
    font_scale = 0.5
    font_thickness = 1
    color_white = (255, 255, 255)
    color_black = (0, 0, 0)

    # Display total free spaces
    total_spaces = len(parking_areas)
    free_spaces = total_spaces - sum(car_counts.values())
    cv2.putText(frame, f"Free Spaces: {free_spaces}", (20, 30), font, font_scale, color_white, font_thickness)

    # Display car count for each area
    for area_name, count in car_counts.items():
        area_points = np.array(parking_areas[area_name], np.int32)
        text_color = color_black if count == 0 else color_white
        if count == 1:
            cv2.polylines(frame, [area_points], True, text_color, 2)
            cv2.putText(frame, str(count), get_text_center(area_points), font, font_scale, text_color, font_thickness)
        else:
            cv2.polylines(frame, [area_points], True, color_white, 2)
            cv2.putText(frame, str(count), get_text_center(area_points), font, font_scale, color_black, font_thickness)


def get_text_center(points):
    """
    Calculates the center point for displaying text within an area.

    Args:
        points (np.ndarray): An array of points defining the area.

    Returns:
        tuple: The (x, y) coordinates of the center point.
    """
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    return (x_min + x_max) // 2, (y_min + y_max) // 2

def generate_ground_truth(frame, predictions, frame_count, timestamp):
    ground_truth = {
        "frame_id": frame_count,
        "timestamp": timestamp,
        "parking_areas": {}
    }

    car_class_id = 2  # Assuming car class is the first class (adjust as needed)

    for prediction in predictions:
        x1, y1, x2, y2, confidence, class_id = prediction

        if class_id == car_class_id and confidence > 0.5:  # Adjust confidence threshold as needed
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            for area_name, area_points in parking_areas.items():
                if cv2.pointPolygonTest(np.array(area_points), (cx, cy), False) >= 0:
                    ground_truth["parking_areas"][area_name].append([x1, y1, x2, y2])
                    break

    return ground_truth

# Main loop
cap = cv2.VideoCapture('parking1.mp4')  # Replace with your video file path

frame_count = 0
start_time = time.time()

output_dir = "annotated_frames"
os.makedirs(output_dir, exist_ok=True)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame if needed
    frame = cv2.resize(frame, (1020, 500))

    # Count cars in each area
    car_counts = {}
    for area_name in parking_areas:
        car_counts[area_name] = count_cars_in_area(frame, area_name)

    # Display parking information on the frame
    display_parking_info(frame, car_counts)

    cv2.imshow('Parking', frame)

    elapsed_time = time.time() - start_time
    if elapsed_time >= extraction_interval:
        frame_count += 1
        timestamp = int(elapsed_time)

        # Run model prediction for current frame
        results = model.predict(frame)

        # Generate ground truth using model predictions
        ground_truth = generate_ground_truth(frame, results.pandas().xyxy[0].values.tolist(), frame_count, timestamp)

        # Save the frame and ground truth
        frame_filename = f"frame_{frame_count}.jpg"
        ground_truth_filename = f"ground_truth_{frame_count}.json"
        cv2.imwrite(os.path.join(output_dir, frame_filename), frame)
        with open(os.path.join(output_dir, ground_truth_filename), 'w') as f:
            json.dump(ground_truth, f)

        # Reset the timer
        start_time = time.time()

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()