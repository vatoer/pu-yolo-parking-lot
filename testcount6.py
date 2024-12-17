import csv

def save_predictions(predictions, output_file):
    """Save predictions to a CSV file."""
    fieldnames = ['frame_number', 'parking_space_id', 'occupied']
    file_exists = os.path.isfile(output_file)

    with open(output_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Write header if file does not exist
        if not file_exists:
            writer.writeheader()

        # Write predictions
        for pred in predictions:
            writer.writerow(pred)

def process_video_with_predictions(video_path, output_dir, prediction_file, extraction_rate=1):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % extraction_rate == 0:
            frame = cv2.resize(frame, (1020, 500))
            results = model.predict(frame, verbose=False)

            # Extract predictions
            detections = results[0].boxes.data.cpu().numpy() if results[0].boxes else []
            predictions = []

            for i, area in enumerate(parking_areas):
                occupied = False
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    class_name = class_list[int(cls)]
                    if class_name == 'car':
                        cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
                        if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                            occupied = True
                            break
                predictions.append({
                    'frame_number': frame_number,
                    'parking_space_id': i + 1,
                    'occupied': int(occupied)
                })
                draw_parking_area(frame, area, occupied, i)

            # Save predictions
            save_predictions(predictions, prediction_file)

        frame_number += 1

        # Show frame
        cv2.imshow("Parking Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the video processing with predictions
process_video_with_predictions(
    "parking1.mp4", 
    "output", 
    "predictions.csv", 
    extraction_rate=30
)
