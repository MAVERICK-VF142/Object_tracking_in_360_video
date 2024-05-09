import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict

track_history = defaultdict(lambda: [])
model = YOLO("yolov8n.pt")
names = model.model.names

cap = cv2.VideoCapture(0)  # Open the default camera (usually 0)

assert cap.isOpened(), "Error opening video stream"

while True:
    success, frame = cap.read()
    if not success:
        print("Error: Couldn't read frame.")
        break

    results = model(frame)

    if results.pred[0] is not None:
        # Extract prediction results
        pred = results.pred[0].cpu().numpy()
        boxes = pred[:, :4]
        clss = pred[:, 5].astype(int)
        confs = pred[:, 4]

        # Annotator Init
        annotator = Annotator(frame, line_width=2)

        for box, cls, conf in zip(boxes, clss, confs):
            xmin, ymin, xmax, ymax = map(int, box)
            label = names[cls]
            color = colors(cls, True)

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            # Add label
            text = f"{label}: {conf:.2f}"
            cv2.putText(frame, text, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Store tracking history
            track = track_history[label]
            track.append(((xmin + xmax) // 2, (ymin + ymax) // 2))
            if len(track) > 30:
                track.pop(0)

            # Plot tracks
            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            cv2.circle(frame, (track[-1][0], track[-1][1]), 7, color, -1)
            cv2.polylines(frame, [points], isClosed=False, color=color, thickness=2)

    cv2.imshow('Object Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
