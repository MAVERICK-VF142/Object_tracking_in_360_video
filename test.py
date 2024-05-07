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
    boxes = results[0].boxes.xyxy.cpu()

    if results[0].boxes.id is not None:
        # Extract prediction results
        clss = results[0].boxes.cls.cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confs = results[0].boxes.conf.float().cpu().tolist()

        # Annotator Init
        annotator = Annotator(frame, line_width=2)

        for box, cls, track_id in zip(boxes, clss, track_ids):
            annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

            # Store tracking history
            track = track_history[track_id]
            track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
            if len(track) > 30:
                track.pop(0)

            # Plot tracks
            points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
            cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
            cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

    cv2.imshow('Object Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
