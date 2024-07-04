import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import defaultdict
from Dependencies import download_files
import random 

# Call the function to download the required files
download_files.download_dependencies()

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    print("GPU is available. Running on GPU.")
else:
    print("GPU is not available. Running on CPU.")

# Load the YOLO model and move it to the appropriate device
model = YOLO("yolov9e.pt").to(device)

deep_sort_weights = 'deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7'

# Define the field of view (FOV) for different perspectives
fov_degrees = 60  # Each FOV is 60 degrees

threshold = random.randint(120, 140)
print("Threshold for motion detection:", threshold)

# Initialize previous frames for motion-based saliency for each perspective
previous_frames = {
    "1": None,
    "2": None,
    "3": None,
    "4": None,
    "5": None,
    "6": None,
}

# Initialize dictionary to store centroid history for each track and each FOV
centroid_history = defaultdict(lambda: defaultdict(list))


def transform_perspective(frame, perspective):
    frame_height, frame_width = frame.shape[:2]
    
    # Assuming each FOV covers 60 degrees
    fov_degrees = 60  
    
    if perspective == "1":
        fov_center = {"x": frame_width // 12, "y": frame_height // 2}
    elif perspective == "2":
        fov_center = {"x": 3 * frame_width // 12, "y": frame_height // 2}
    elif perspective == "3":
        fov_center = {"x": 5 * frame_width // 12, "y": frame_height // 2}
    elif perspective == "4":
        fov_center = {"x": 7 * frame_width // 12, "y": frame_height // 2}
    elif perspective == "5":
        fov_center = {"x": 9 * frame_width // 12, "y": frame_height // 2}
    elif perspective == "6":
        fov_center = {"x": 11 * frame_width // 12, "y": frame_height // 2}
        # Extend the FOV more to the left for the sixth perspective
        fov_width = int(frame_width * fov_degrees / 360)  # 120 degrees FOV width
        fov_height = int(frame_height * fov_degrees / 360)  # 60 degrees FOV height
        perspective_frame = frame[
            fov_center["y"] - fov_height // 2 : fov_center["y"] + fov_height // 2,
            fov_center["x"] - fov_width //2 : fov_center["x"] + fov_width // 2,
        ]
        return perspective_frame
    else:
        raise ValueError("Invalid perspective")

    # Normal FOV extraction for perspectives 1 to 5
    fov_width = int(frame_width * fov_degrees / 360)  # 60 degrees FOV width
    fov_height = int(frame_height * fov_degrees / 360)  # 60 degrees FOV height
    perspective_frame = frame[
        fov_center["y"] - fov_height // 2 : fov_center["y"] + fov_height // 2,
        fov_center["x"] - fov_width // 2 : fov_center["x"] + fov_width // 2,
    ]

    return perspective_frame


def detect_saliency(frame, previous_frame):
    if previous_frame is None or previous_frame.size == 0:
        return 0
    previous_frame_resized = cv2.resize(previous_frame, (frame.shape[1], frame.shape[0]))
    saliency = np.mean(np.abs(frame - previous_frame_resized))
    return saliency

# Initialize dictionary to store trackers for each perspective
trackers = {perspective: DeepSort(model_path=deep_sort_weights, max_age=5) for perspective in previous_frames.keys()}

# Define colors for paths and bounding boxes
path_colors = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255), (0, 0, 255)]  # Yellow, Cyan, Green, Magenta, Red
bbox_colors = [(255, 0, 255)]  # Magenta

motion_detected = {"1": False, "2": False, "3": False, "4": False, "5": False, "6": False}

# Open the video file
video_path = "Dependencies/Test.mp4"
cap = cv2.VideoCapture(video_path)

# Get the width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize a global track ID counter
global_track_id = 0

# Dictionary to map perspective track IDs to global track IDs
global_id_map = defaultdict(dict)

# Initialize dictionary to store re-id features for each track and each perspective
reid_features = defaultdict(lambda: defaultdict(list))

# Placeholder for the re-id model
class ReIDModel(torch.nn.Module):
    def __init__(self):
        super(ReIDModel, self).__init__()
        # Load a pre-trained model (example)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        self.model.fc = torch.nn.Identity()  # Remove the classification layer

    def forward(self, img):
        return self.model(img)

# Initialize the re-id model
reid_model = ReIDModel()
reid_model.to(device).eval()  # Move the model to the appropriate device and set it to evaluation mode

def match_features(feature1, feature2, threshold=0.5):
    distance = np.linalg.norm(feature1 - feature2)
    return distance < threshold

while cap.isOpened():
    success, frame = cap.read()

    if success:
        for perspective, previous_frame in previous_frames.items():
            transformed_frame = transform_perspective(frame, perspective)
            saliency = detect_saliency(transformed_frame, previous_frame)

            if saliency > threshold:
                motion_detected[perspective] = True
                print(f"Saliency detected in {perspective}")

                tracker = trackers[perspective]

                results = model(transformed_frame, classes=[0, 2, 3], device=device)

                if isinstance(results, list) and len(results) > 0:
                    coordinates = results[0].boxes.cpu().numpy()

                    for coord in coordinates:
                        x1, y1, x2, y2 = coord.xyxy[0]
                        conf = coord.conf.item()
                        cls = coord.cls.item()
                        label = model.names[cls]
                        
                        # Extract appearance feature
                        obj_img = transformed_frame[int(y1):int(y2), int(x1):int(x2)]
                        obj_img = cv2.resize(obj_img, (128, 256))  # Resize to the input size expected by the re-id model
                        obj_img = torch.from_numpy(obj_img).permute(2, 0, 1).float().unsqueeze(0).to(device)  # Convert to tensor
                        feature = reid_model(obj_img).detach().cpu().numpy().flatten()  # Extract feature

                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2

                        object_id = f"{perspective}_{cls}_{conf:.2f}"

                        theta = np.degrees((center_x / frame_width) * 360)
                        phi = np.degrees((center_y / frame_height) * 180)

                        cv2.putText(
                            transformed_frame,
                            f"[x: {theta:.2f}, y: {phi:.2f}]",
                            (int(x1), int(y1) - 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2,
                        )

                conf = results[0].boxes.conf.detach().cpu().numpy()
                xywh = results[0].boxes.xywh.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()
                
                tracks = tracker.update(xywh, conf, clss, transformed_frame)

                for track in tracker.tracker.tracks:
                    track_id = track.track_id
                    if track_id not in global_id_map[perspective]:
                        global_id_map[perspective][track_id] = global_track_id
                        global_track_id += 1
                    global_id = global_id_map[perspective][track_id]

                    hits = track.hits
                    x1, y1, x2, y2 = track.to_tlbr()
                    centroid_x = int((x1 + x2) / 2)
                    centroid_y = int((y1 + y2) / 2)

                    centroid_history[perspective][global_id].append((centroid_x, centroid_y))

                    for point1, point2 in zip(centroid_history[perspective][global_id], centroid_history[perspective][global_id][1:]):
                        cv2.line(transformed_frame, point1, point2, path_colors[global_id % len(path_colors)], 2)

                    cv2.rectangle(transformed_frame, (int(x1), int(y1)), (int(x2), int(y2)), bbox_colors[0], 2)

                    cv2.putText(transformed_frame, f"ID: {global_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)

                    # Store the re-id feature for the global ID
                    reid_features[perspective][global_id].append(feature)

                person_count = len(centroid_history[perspective])

                cv2.imshow(f"{perspective.capitalize()} Perspective", transformed_frame)

            previous_frames[perspective] = frame.copy()

        # Re-id matching across perspectives
        for perspective1, features1 in reid_features.items():
            for perspective2, features2 in reid_features.items():
                if perspective1 != perspective2:
                    for global_id1, feats1 in features1.items():
                        for global_id2, feats2 in features2.items():
                            for feat1 in feats1:
                                for feat2 in feats2:
                                    if match_features(feat1, feat2):
                                        if global_id1 in global_id_map[perspective1] and global_id2 in global_id_map[perspective2]:
                                            # Merge the IDs across perspectives
                                            merged_global_id = global_id_map[perspective1][global_id1]
                                            old_global_id = global_id_map[perspective2].get(global_id2)

                                            if old_global_id is not None and old_global_id != merged_global_id:
                                                # Update all mappings to the new merged global ID
                                                for p, ids in global_id_map.items():
                                                    for track_id, gid in ids.items():
                                                        if gid == old_global_id:
                                                            global_id_map[p][track_id] = merged_global_id

                                            global_id_map[perspective2][global_id2] = merged_global_id

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
