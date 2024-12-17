import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

# Define class list (assuming it's a separate file)
class_list = ["car", "bus", "motorcycle", ...]  # Replace with actual classes

# Define parking area coordinates as a dictionary for easier access
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
    # ... define all other areas
}

model = YOLO('yolov8s.pt')


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
    for index, row in results.pandas().xyxy[0].iterrows():
        x1, y1, x2, y2, d = row.values
        class_name = class_list[d]
        if class_name == "car":
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if cv2.pointPolygonTest(area_points, (cx, cy), False) >= 0:
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

def calculate_iou(box1, box2):
  """Calculates the Intersection over Union (IoU) of two bounding boxes.

  Args:
    box1: A tuple or list of four coordinates (x1, y1, x2, y2) representing the top-left and bottom-right corners of the first box.
    box2: A tuple or list of four coordinates (x1, y1, x2, y2) representing the top-left and bottom-right corners of the second box.

  Returns:
    The IoU value between the two boxes.
  """

  x1, y1, x2, y2 = box1
  x1_, y1_, x2_, y2_ = box2

  # Determine the coordinates of the intersection rectangle
  x_intersection = max(x1, x1_)
  y_intersection = max(y1, y1_)
  x_union = min(x2, x2_)
  y_union = min(y2, y2_)

  # Compute the area of intersection rectangle
  intersection_area = max(0, x_union - x_intersection + 1) * max(0, y_union - y_intersection + 1)

  # Compute the area of both the prediction and ground-truth boxes
  box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
  box2_area = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)

  # Compute the union area by adding the individual areas and subtracting the intersection area
  union_area = box1_area + box2_area - intersection_area

  # Compute the IoU
  iou = intersection_area / union_area

  return iou

def evaluate_model(ground_truth, predictions):
    """
    Evaluates the model's performance based on ground truth and predicted bounding boxes.

    Args:
        ground_truth: A dictionary containing ground truth annotations for each frame.
        predictions: A dictionary containing predicted bounding boxes for each frame.

    Returns:
        A tuple of precision, recall, and F1-score.
    """

    total_tp, total_fp, total_fn = 0, 0, 0

    for frame_id, gt_boxes in ground_truth.items():
        pred_boxes = predictions.get(frame_id, [])

        for gt_box in gt_boxes:
            matched = False
            for pred_box in pred_boxes:
                # Calculate IoU between ground truth and predicted boxes
                iou = calculate_iou(gt_box, pred_box)
                if iou > 0.5:  # Adjust IoU threshold as needed
                    matched = True
                    total_tp += 1
                    break
            if not matched:
                total_fn += 1

        for pred_box in pred_boxes:
            if not any(iou > 0.5 for gt_box in gt_boxes):
                total_fp += 1

    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1_score

# Main loop
cap = cv2.VideoCapture('parking1.mp4')  # Replace with your video file path

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

    if cv2.waitKey(1) & 0xFF == 27:
        break

# After processing the video
precision, recall, f1_score = evaluate_model(ground_truth, predictions)
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}")

cap.release()
cv2.destroyAllWindows()