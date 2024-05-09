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

# Create a display window
cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)

# Set the desired field of view
fov_degrees = 90  # Adjust as needed

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Convert the equi-rectangular frame to perspective projection with FOV in the center
        # Define the region of interest (ROI) for the center FOV
        fov_center = {
            'x': frame_width // 2,
            'y': frame_height // 2,
            'w': int(frame_width * fov_degrees / 360),
            'h': int(frame_height * fov_degrees / 360)
        }
        
        # Apply perspective projection to the frame
        perspective_frame = frame[fov_center['y'] - fov_center['h'] // 2: fov_center['y'] + fov_center['h'] // 2,
                                  fov_center['x'] - fov_center['w'] // 2: fov_center['x'] + fov_center['w'] // 2]

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(perspective_frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Resize the frame to fit within the window
        resized_frame = cv2.resize(annotated_frame, (frame_width, frame_height))

        # Display the resized frame
        cv2.imshow("YOLOv8 Tracking", resized_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
