import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov9c.pt')

# Open the video file
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# Get the width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
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
                        zoom_factor = 1.5  # Reduced zoom factor
                        zoomed_width = int((x2 - x1) * zoom_factor)
                        zoomed_height = int((y2 - y1) * zoom_factor)
                        zoomed_frame = cv2.resize(frame[int(y1):int(y2), int(x1):int(x2)], (zoomed_width, zoomed_height))

                        # Display the zoomed frame
                        cv2.imshow("Zoomed Frame", zoomed_frame)

                        # Break the loop to select only one object
                        break

        # Visualize the results on the frame
        for result in results:
            annotated_frame = result.plot()

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

# Release the video capture object and close the display windows
cap.release()
cv2.destroyAllWindows()
