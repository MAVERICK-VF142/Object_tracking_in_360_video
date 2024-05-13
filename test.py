import cv2
import numpy as np

# Open the video file
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# Get the width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a display window
cv2.namedWindow("Motion-Based Saliency", cv2.WINDOW_NORMAL)

# Function to perform motion-based saliency detection
def detect_motion_saliency(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur the grayscale frame to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Compute absolute difference between consecutive frames
    frame_diff = cv2.absdiff(prev_gray, blurred)

    # Threshold the difference image to identify motion regions
    _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Set non-motion regions to white (255, 255, 255)
    motion_highlight = np.zeros_like(frame)
    motion_highlight[motion_mask == 255] = [255, 0, 0]  # Blue color for motion regions

    return motion_highlight

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Unable to read the video file")
    exit()

# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if ret:
        # Perform motion-based saliency detection
        motion_highlight = detect_motion_saliency(frame)

        # Display the motion-based saliency map
        cv2.imshow("Motion-Based Saliency", motion_highlight)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
