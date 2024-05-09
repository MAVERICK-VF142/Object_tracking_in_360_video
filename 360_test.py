import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# Get the width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the field of view (FOV) for different perspectives
fov_degrees = {
    'front': 90,   # Front perspective
    'back': 90,    # Back perspective
    'left': 90,    # Left perspective
    'right': 90    # Right perspective
}

# Define function to transform frame based on selected perspective
def transform_perspective(frame, perspective):
    # Define the region of interest (ROI) for the selected perspective
    if perspective == 'front':
        fov_center = {'x': frame_width // 2, 'y': frame_height // 2}
    elif perspective == 'back':
        fov_center = {'x': frame_width // 2, 'y': frame_height // 2}
    elif perspective == 'left':
        fov_center = {'x': frame_width // 4, 'y': frame_height // 2}
    elif perspective == 'right':
        fov_center = {'x': 3 * frame_width // 4, 'y': frame_height // 2}
    else:
        raise ValueError("Invalid perspective")

    # Calculate the width and height of the FOV based on the FOV degrees
    fov_width = int(frame_width * fov_degrees[perspective] / 360)
    fov_height = int(frame_height * fov_degrees[perspective] / 360)
    
    # Apply perspective projection to the frame
    perspective_frame = frame[fov_center['y'] - fov_height // 2: fov_center['y'] + fov_height // 2,
                              fov_center['x'] - fov_width // 2: fov_center['x'] + fov_width // 2]

    return perspective_frame

# Create a display window
cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)

# Initialize perspective as 'front'
current_perspective = 'front'

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Transform the frame based on the current perspective
        transformed_frame = transform_perspective(frame, current_perspective)
        
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(transformed_frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Resize the frame to fit within the window
        resized_frame = cv2.resize(annotated_frame, (frame_width, frame_height))

        # Display the resized frame
        cv2.imshow("YOLOv8 Tracking", resized_frame)

        # Check for key press to switch perspective
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("f"):
            current_perspective = 'front'
        elif key == ord("b"):
            current_perspective = 'back'
        elif key == ord("l"):
            current_perspective = 'left'
        elif key == ord("r"):
            current_perspective = 'right'
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
