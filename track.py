import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
try:
    model = YOLO("yolov8n.pt")
except Exception as e:
    print("Error loading YOLOv8 model:", e)
    exit()

# Open the video file
video_path = "Test.mp4"
cap = cv2.VideoCapture(video_path)

# Check if video file is opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Get the width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the field of view (FOV) for different perspectives
fov_degrees = {
    "front": 90,  # Front perspective
    "back": 90,   # Back perspective
    "left": 90,   # Left perspective
    "right": 90   # Right perspective
}

# Define function to transform frame based on selected perspective
def transform_perspective(frame, perspective):
    # Define the region of interest (ROI) for the selected perspective
    if perspective == "front":
        fov_center = {"x": frame_width // 2, "y": frame_height // 2}
    elif perspective == "back":
        fov_center = {"x": frame_width // 2, "y": frame_height // 2}
    elif perspective == "left":
        fov_center = {"x": frame_width // 4, "y": frame_height // 2}
    elif perspective == "right":
        fov_center = {"x": 3 * frame_width // 4, "y": frame_height // 2}
    else:
        raise ValueError("Invalid perspective")

    # Calculate the width and height of the FOV based on the FOV degrees
    fov_width = int(frame_width * fov_degrees[perspective] / 360)
    fov_height = int(frame_height * fov_degrees[perspective] / 360)

    # Apply perspective projection to the frame
    perspective_frame = frame[
        fov_center["y"] - fov_height // 2 : fov_center["y"] + fov_height // 2,
        fov_center["x"] - fov_width // 2 : fov_center["x"] + fov_width // 2,
    ]

    return perspective_frame

# Initialize perspective as 'front'
current_perspective = "front"

# Track a specific object (e.g., bike)
tracked_object_label = "bike"

# Loop through the video frames
while True:
    # Read a frame from the video
    success, frame = cap.read()

    # Check if frame is retrieved successfully
    if not success:
        print("Error: Couldn't read frame.")
        break

    # Run YOLOv8 detection on the frame
    try:
        results = model(frame)
    except Exception as e:
        print("Error during YOLOv8 detection:", e)
        continue

    # Filter results to find the specified object
    tracked_object = None
    for label, conf, bbox in zip(results.xyxy[0].names, results.xyxy[0].scores, results.xyxy[0].xyxy):
        if label == tracked_object_label:
            tracked_object = bbox
            break

    # If the specified object is found, adjust the field of view (FOV) to focus on it
    if tracked_object is not None:
        # Calculate the centroid of the tracked object
        centroid_x = int((tracked_object[0] + tracked_object[2]) / 2)
        centroid_y = int((tracked_object[1] + tracked_object[3]) / 2)

        # Determine the new perspective based on the centroid position
        if centroid_x < frame_width // 2:
            current_perspective = "left"
        else:
            current_perspective = "right"

    # Transform the frame based on the current perspective
    transformed_frame = transform_perspective(frame, current_perspective)

    # Visualize the transformed frame
    cv2.imshow("Transformed Frame", transformed_frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
