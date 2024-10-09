import cv2
import numpy as np
import torch
import time
import pandas as pd
import yaml
import os

# Load configuration from a YAML file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Class to track objects and generate reports
class ObjectTracker:
    def __init__(self, model_path, video_source):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        self.video_source = video_source
        self.object_data = {}
        self.time_spent = {}
        self.object_count = {}
        self.tracking_paths = {}

    def track_objects(self):
        cap = cv2.VideoCapture(self.video_source)
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run object detection
            results = self.model(frame)
            detections = results.xyxy[0].numpy()
            
            for *box, conf, cls in detections:
                x1, y1, x2, y2 = map(int, box)
                object_id = int(cls)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                # Initialize tracking data
                if object_id not in self.object_data:
                    self.object_data[object_id] = {
                        "count": 0,
                        "coordinates": [],
                        "time_spent": 0
                    }
                self.object_data[object_id]["count"] += 1
                self.object_data[object_id]["coordinates"].append(center)
                
                # Update the time spent in certain areas
                if (x1 >= 100 and x2 <= 300) and (y1 >= 100 and y2 <= 300):  # Area of interest
                    self.object_data[object_id]["time_spent"] += 1  # Count each frame in the area

                # Path tracking
                if object_id not in self.tracking_paths:
                    self.tracking_paths[object_id] = []
                self.tracking_paths[object_id].append(center)

                # Draw bounding box and label
                label = f'ID: {object_id}, Conf: {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Show the processed frame
            cv2.imshow('Tracking', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.generate_report()

    def generate_report(self):
        # Create a DataFrame to hold the report data
        report_data = []

        for object_id, data in self.object_data.items():
            report_data.append({
                "Object ID": object_id,
                "Frequency of Appearance": data["count"],
                "Time Spent in Area": data["time_spent"],
                "Path Coordinates": data["coordinates"]
            })

        report_df = pd.DataFrame(report_data)
        report_file = "video_analysis_report.csv"
        report_df.to_csv(report_file, index=False)
        print(f"Report generated: {report_file}")

if __name__ == "__main__":
    config_path = 'config.yaml'  # Path to the config file
    config = load_config(config_path)
    
    video_source = config['video_source']  # Update according to your config structure
    model_path = 'yolov10e.pt'  # Path to the YOLOv10 weights

    tracker = ObjectTracker(model_path, video_source)
    tracker.track_objects()
