
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

model = YOLO('yolov8s.pt')

# Evaluate on a validation dataset
results = model.val()

# Print the evaluation results
print(results)