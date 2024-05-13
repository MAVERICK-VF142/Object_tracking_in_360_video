import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov9e.pt')

# Open the video file
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# Get the width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the reduced size for zoomed frame
zoomed_width = frame_width // 10
zoomed_height = frame_height // 5

# Create a display window
cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)

# Variables for storing selected object coordinates
selected_object_coords = None

# Function to handle mouse events for selecting objects
def select_object(event, x, y, flags, param):
    global selected_object_coords

    if event == cv2.EVENT_LBUTTONDOWN:
        print("Mouse Clicked at:", x, y)
        # Store the coordinates of the clicked point
        selected_object_coords = (x, y)

# Set mouse event handler
cv2.setMouseCallback("YOLOv8 Tracking", select_object)

# Read the first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Unable to read the video file")
    exit()

# Convert the first frame to grayscale for motion-based saliency detection
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Convert frame to grayscale for motion-based saliency detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference between consecutive frames for motion-based saliency
        frame_diff = cv2.absdiff(prev_gray, gray_frame)

        # Threshold the difference image to identify motion regions
        _, motion_mask = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

        # Run YOLOv8 detection on the frame
        results = model(frame)

        # Check if an object is selected
        if selected_object_coords is not None:
            for result in results:
                for detection in result.boxes.data:
                    print("Detection:", detection)  # Print out the detection to inspect its structure
                    x1, y1, x2, y2, conf, cls = detection.tolist()  # Unpack the detection
                    # Check if the click point falls within the bounding box
                    if x1 <= selected_object_coords[0] <= x2 and y1 <= selected_object_coords[1] <= y2:
                        print("Selected Object:", detection)

                        # Draw a rectangle around the selected object
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                        # Zoom in on the selected object
                        zoomed_frame = frame[int(y1):int(y2), int(x1):int(x2)]

                        # Resize the zoomed frame to the reduced size
                        zoomed_frame = cv2.resize(zoomed_frame, (zoomed_width, zoomed_height))

                        # Draw coordinates text on the zoomed frame with smaller font size
                        text = f"Coordinates: ({x1:.3f}, {y1:.3f}) - ({x2:.3f}, {y2:.3f})"
                        cv2.putText(zoomed_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                        # Display the zoomed frame
                        cv2.imshow("Zoomed Frame", zoomed_frame)

                        # Break the loop to select only one object
                        break

        # Visualize the results on the frame
        for result in results:
            annotated_frame = result.plot()

            # Overlay motion-based saliency regions on the frame
            motion_highlight = cv2.bitwise_and(frame, frame, mask=motion_mask)
            annotated_frame = cv2.addWeighted(annotated_frame, 0.7, motion_highlight, 0.3, 0)

            # Resize the frame to fit within the window
            resized_frame = cv2.resize(annotated_frame, (frame_width, frame_height))

            # Display the resized frame
            cv2.imshow("YOLOv8 Tracking", resized_frame)

        # Update the previous grayscale frame for motion-based saliency detection
        prev_gray = gray_frame.copy()

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display windows
cap.release()
cv2.destroyAllWindows()
