import cv2
import numpy as np
from ultralytics import YOLO
import random

# Load the YOLOv8 model
model = YOLO("yolov9e.pt")

# Open the video file
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# Get the width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the field of view (FOV) for different perspectives
fov_degrees = {
    "front": 90,  # Front perspective
    "left": 90,  # Left perspective
    "right": 90,  # Right perspective
    "leftmost": 90,  # Leftmost perspective
    "rightmost": 90,  # Rightmost perspective
}

threshold = random.randint(120, 140)  # Random threshold ranging from 120 to 140
print("Threshold for motion detection:", threshold)


# Define function to transform frame based on selected perspective
def transform_perspective(frame, perspective):
    if perspective == "front":
        fov_center = {"x": frame_width // 2, "y": frame_height // 2}
    elif perspective == "left":
        fov_center = {"x": frame_width // 4, "y": frame_height // 2}
    elif perspective == "right":
        fov_center = {"x": 3 * frame_width // 4, "y": frame_height // 2}
    elif perspective == "leftmost":
        fov_center = {"x": frame_width // 8, "y": frame_height // 2}
    elif perspective == "rightmost":
        fov_center = {"x": 7 * frame_width // 8, "y": frame_height // 2}
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


# Function for motion-based saliency detection
def detect_saliency(frame, previous_frame):
    # Check if previous_frame is not None and has non-zero size
    if previous_frame is None or previous_frame.size == 0:
        return 0

    # Resize the previous frame to match the size of the current frame
    previous_frame_resized = cv2.resize(
        previous_frame, (frame.shape[1], frame.shape[0])
    )
    # Compute the saliency based on the difference between the frames
    saliency = np.mean(np.abs(frame - previous_frame_resized))
    return saliency


# Initialize previous frames for motion-based saliency for each perspective
previous_frames = {
    "front": None,
    "left": None,
    "right": None,
    "leftmost": None,
    "rightmost": None,
}

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Initialize dictionary to store detected motion for each perspective
        motion_detected = {
            "front": False,
            "left": False,
            "right": False,
            "leftmost": False,
            "rightmost": False,
        }

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
                # Check if results is a list (adjust the index based on your needs)
                if isinstance(results, list) and len(results) > 0:
                    # Get the detected objects' coordinates from the first element (if results is a list)
                    coordinates = results[0].boxes

                    # Loop through the detected objects
                    for coord in coordinates:
                        x1, y1, x2, y2 = coord.xyxy[0]  # Extract coordinates
                        conf = coord.conf.item()  # Extract confidence
                        cls = coord.cls.item()  # Extract class
                        label = model.names[cls]  # Get the label of the detected object

                        # Calculate the center of the object
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

                        # Map the (x, y) coordinates to 360-degree video coordinates
                        theta = np.degrees(
                            (center_x / frame_width) * 360
                        )  # Convert x to theta in degrees
                        phi = np.degrees(
                            (center_y / frame_height) * 180
                        )  # Convert y to phi in degrees

                        # Draw bounding box and label with formatted coordinates
                        cv2.putText(
                            annotated_frame,
                            f"[x: {theta:.2f}, y: {phi:.2f}]",
                            (int(x1), int(y1) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2,
                        )

                # Display the annotated frame in the perspective window
                if perspective == "rightmost":
                    # Define the region of interest to display only the right portion of the frame
                    roi_width = annotated_frame.shape[1] // 2
                    overlap = 50  # Adjust this value for the desired overlap
                    roi = annotated_frame[:, roi_width - overlap :, :]

                    # Display the ROI in the perspective window
                    cv2.imshow(f"{perspective.capitalize()} Perspective", roi)
                else:
                    cv2.imshow(
                        f"{perspective.capitalize()} Perspective", annotated_frame
                    )

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
