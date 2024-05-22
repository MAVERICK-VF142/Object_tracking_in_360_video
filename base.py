import cv2
from ultralytics import YOLO
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

# Initialize YOLOv9 model
model = YOLO("yolov9e.pt")

bbox_colors = [(255, 0, 255)]  # Magenta

# Initialize DeepSORT tracker
deep_sort_weights = "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
tracker = DeepSort(model_path=deep_sort_weights, max_age=10)

# Open the video file
video_path = "Test.mp4"
cap = cv2.VideoCapture(video_path)

cv2.namedWindow("Object Tracking", cv2.WINDOW_NORMAL)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Perform object detection
        results = model(frame, classes=0)

        if isinstance(results, list) and len(results) > 0:
            # Get the detected objects' coordinates from the first element (if results is a list)
            coordinates = results[0].boxes

            # Extract confidence and bbox coordinates
            conf = results[0].boxes.conf.cpu().numpy()
            xyxy = results[0].boxes.xywh.cpu().numpy()

            # Update tracker with detection results
            tracks = tracker.update(xyxy, conf, frame, frame)

            # Loop through the tracks from the tracker
            for track in tracker.tracker.tracks:
                track_id = track.track_id
                hits = track.hits
                x1, y1, x2, y2 = (
                    track.to_tlbr()
                )  # Get bounding box coordinates in (x1, y1, x2, y2) format
                centroid_x = int((x1 + x2) / 2)  # Calculate centroid x-coordinate
                centroid_y = int(y2)  # Calculate centroid y-coordinate

                # Draw bounding box
                cv2.rectangle(
                    frame, (int(x1), int(y1)), (int(x2), int(y2)), bbox_colors[0], 2
                )

                # Draw track ID and coordinates
                cv2.putText(
                    frame,
                    f"ID: {track_id} | Coords: ({centroid_x}, {centroid_y})",
                    (int(x1) + 10, int(y1) - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

        # Display the frame
        cv2.imshow("Object Tracking", frame)

        # Check for key press to exit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
