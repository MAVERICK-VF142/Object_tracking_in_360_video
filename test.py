import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov9e.pt')

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

# Define the threshold for motion detection
threshold = 130  # Adjust as needed

# Define function to transform frame based on selected perspective
def transform_perspective(frame, perspective):
    if perspective == 'front':
        fov_center = {'x': frame_width // 2, 'y': frame_height // 2}
    elif perspective == 'back':
        fov_center = {"x": 3 * frame_width // 4, "y":frame_height // 2}
    elif perspective == 'left':
        fov_center = {'x': frame_width // 4, 'y': frame_height // 2}
    elif perspective == 'right':
        fov_center = {'x': 3 * frame_width // 4, 'y': frame_height // 2}
    else:
        raise ValueError("Invalid perspective")

    # Calculate the width and height of the FOV based on the FOV degrees
    fov_width = int(frame_width * fov_degrees[perspective] / 360)
    fov_height = int(frame_height * fov_degrees[perspective] / 360)
    
    # Adjust the FOV height for the back perspective
    if perspective == 'back':
        fov_height = min(frame_height, int(fov_width * frame_height / frame_width))
    
    # Apply perspective projection to the frame
    perspective_frame = frame[fov_center['y'] - fov_height // 2: fov_center['y'] + fov_height // 2,
                              fov_center['x'] - fov_width // 2: fov_center['x'] + fov_width // 2]

    return perspective_frame

# Function for motion-based saliency detection
def detect_saliency(frame, previous_frame):
    # Check if previous_frame is not None and has non-zero size
    if previous_frame is None or previous_frame.size == 0:
        return 0

    # Resize the previous frame to match the size of the current frame
    previous_frame_resized = cv2.resize(previous_frame, (frame.shape[1], frame.shape[0]))
    # Compute the saliency based on the difference between the frames
    saliency = np.mean(np.abs(frame - previous_frame_resized))
    return saliency

# Initialize previous frames for motion-based saliency for each perspective
previous_frames = {'front': None, 'back': None, 'left': None, 'right': None}

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Initialize dictionary to store detected motion for each perspective
        motion_detected = {'front': False, 'back': False, 'left': False, 'right': False}
        
        # Loop through perspectives
        for perspective, previous_frame in previous_frames.items():
            # Transform the frame based on the current perspective
            transformed_frame = transform_perspective(frame, perspective)
            
            # Detect motion-based saliency for the transformed frame
            saliency = detect_saliency(transformed_frame, previous_frame)
            
            # If motion is detected, set the flag for the perspective
            if saliency > threshold:
                motion_detected[perspective] = True
                
                # Run YOLOv8 object detection and tracking
                results = model.track(transformed_frame, persist=True)
                annotated_frame = results[0].plot()

                # Display the annotated frame in the perspective window
                cv2.imshow(f"{perspective.capitalize()} Perspective", annotated_frame)

            # Update the previous frame for the next iteration
            previous_frames[perspective] = frame.copy()

        # Check for key press to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
