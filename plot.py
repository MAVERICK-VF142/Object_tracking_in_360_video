import cv2
from ultralytics import YOLO
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import matplotlib.pyplot as plt

# Initialize YOLOv9 model
model = YOLO("yolov9e.pt")

# Initialize DeepSORT tracker
deep_sort_weights = "deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7"
tracker = DeepSort(model_path=deep_sort_weights, max_age=10)

# Open the video file
video_path = "Test.mp4"
cap = cv2.VideoCapture(video_path)

# Initialize dictionaries to store centroid coordinates for each track ID
centroid_coordinates = {}

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Perform object detection
        results = model(frame, classes=0)

        if isinstance(results, list) and len(results) > 0:
            # Extract confidence and bbox coordinates
            conf = results[0].boxes.conf.cpu().numpy()
            xyxy = results[0].boxes.xywh.cpu().numpy()

            # Update tracker with detection results
            tracks = tracker.update(xyxy, conf, frame, frame)

            # Loop through the tracks from the tracker
            for track in tracker.tracker.tracks:
                track_id = track.track_id
                x1, y1, x2, y2 = (
                    track.to_tlbr()
                )  # Get bounding box coordinates in (x1, y1, x2, y2) format
                centroid_x = int((x1 + x2) / 2)  # Calculate centroid x-coordinate
                centroid_y = int((y1 + y2) / 2)  # Calculate centroid y-coordinate

                # Store centroid coordinates for each track ID
                if track_id not in centroid_coordinates:
                    centroid_coordinates[track_id] = {"x": [], "y": []}
                centroid_coordinates[track_id]["x"].append(centroid_x)
                centroid_coordinates[track_id]["y"].append(centroid_y)

    else:
        break

# Release the video capture object
cap.release()

# Plot centroid coordinates for each track ID
plt.figure(figsize=(8, 6))
for track_id, coordinates in centroid_coordinates.items():
    plt.plot(
        coordinates["x"], coordinates["y"], marker="o", label=f"Track ID: {track_id}"
    )
plt.xlabel("Centroid X")
plt.ylabel("Centroid Y")
plt.title("Centroid Coordinates for Each Track ID")
plt.legend()
plt.grid(True)
plt.show()
