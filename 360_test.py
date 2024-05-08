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

# Define output frame dimensions for normal FOV projection
output_width = 2024  # Adjust as needed
output_height = 2880  # Adjust as needed

# Create a display window
cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Placeholder for determining the current viewing angle or position within the 360 video
        # For demonstration purposes, let's assume a fixed viewing angle
        viewing_angle = 360  # Modify this value based on actual viewing angle or position determination

        # Placeholder for computing remap parameters based on the current viewing angle or position
        # For demonstration purposes, let's assume we have fixed remap parameters here
        map_x, map_y = cv2.initUndistortRectifyMap(np.eye(3), None, None, np.eye(3),
                                                    (output_width, output_height), cv2.CV_32FC1)

        # Remap the frame from equirectangular to normal FOV
        normal_fov_frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR)

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(normal_fov_frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Resize the frame to fit within the window
        resized_frame = cv2.resize(annotated_frame, (output_width, output_height))

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
