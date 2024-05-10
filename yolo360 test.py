import cv2
from yolo360 import YOLO360

# Load the YOLO360 model
model = YOLO360()

# Open the 360-degree video file
video_path = "360_video.mp4"
cap = cv2.VideoCapture(video_path)

# Get the original frame width and height
original_frame_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
original_frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Define the scale factor for resizing
scale_factor = 0.5  # Adjust as needed

# Calculate the new frame width and height
new_frame_width = int(original_frame_width * scale_factor)
new_frame_height = int(original_frame_height * scale_factor)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if ret:
        # Resize the frame
        resized_frame = cv2.resize(frame, (new_frame_width, new_frame_height))

        # Detect objects in the resized frame
        detections = model.detect(resized_frame)

        # Visualize the detections on the resized frame
        for detection in detections:
            label = detection['label']
            confidence = detection['confidence']
            bbox = detection['bbox']
            # Scale the bounding box coordinates back to original size
            scaled_bbox = [int(coord / scale_factor) for coord in bbox]
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (scaled_bbox[0], scaled_bbox[1]), (scaled_bbox[2], scaled_bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (scaled_bbox[0], scaled_bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow('Object Detection', frame)

        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()