import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the video file
video_path = "Test.mp4"
cap = cv2.VideoCapture(video_path)

# Get the width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a display window
cv2.namedWindow("YOLOv8 Tracking", cv2.WINDOW_NORMAL)

# Initialize selected object and initial FOV
selected_object = None
current_fov = 90  # Example initial FOV: 90 degrees

# Define FOV adjustment parameters
fov_increment = 5  # Degrees to adjust FOV by

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Perform object selection
        if selected_object is None:
            # For simplicity, select the first detected object as the selected object
            selected_object = results[0].xyxy[0]  # (x1, y1, x2, y2) coordinates of the first detected object

        # Adjust FOV based on selected object
        current_fov = adjust_fov(selected_object, current_fov, frame_width, frame_height)

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

# Function to adjust FOV based on selected object position
def adjust_fov(selected_object, current_fov, frame_width, frame_height):
    # Calculate the center coordinates of the selected object
    obj_center_x = (selected_object[0] + selected_object[2]) / 2
    obj_center_y = (selected_object[1] + selected_object[3]) / 2

    # Calculate the horizontal and vertical FOV adjustment angles based on the object position
    horizontal_adjustment = (obj_center_x - frame_width / 2) / frame_width * current_fov
    vertical_adjustment = (obj_center_y - frame_height / 2) / frame_height * current_fov

    # Adjust the current FOV by the maximum of the horizontal and vertical adjustment angles
    max_adjustment = max(abs(horizontal_adjustment), abs(vertical_adjustment))
    new_fov = current_fov + max_adjustment

    return new_fov
