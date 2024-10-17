import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from deep_sort_pytorch.deep_sort import DeepSort
from collections import defaultdict
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cdist
from skimage.feature import local_binary_pattern
from torchreid import models
from torchreid.reid.data import transforms as reid_transforms
import random
from Dependencies import download_files
import logging

# Set up logging configuration
logging.basicConfig(
    filename="boundary_tracking.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


# Call the function to download the required files
download_files.download_dependencies()

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("GPU is available. Running on GPU.")
else:
    print("GPU is not available. Running on CPU.")

# Load the YOLO model and move it to the appropriate device
model = YOLO("yolov10x.pt").to(device)

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

# Initialize a single tracker for the entire original frame
tracker = DeepSort(
    model_path="deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7", max_age=15
)

# Define colors for paths and bounding boxes
path_colors = [
    (0, 255, 255),
    (255, 255, 0),
    (0, 255, 0),
    (255, 0, 255),
    (0, 0, 255),
]  # Yellow, Cyan, Green, Magenta, Red
bbox_colors = [(255, 0, 255)]  # Magenta

motion_detected = {
    "1": False,
    "2": False,
    "3": False,
    "4": False,
    "5": False,
    "6": False,
}

# Open the video file
video_path = "Dependencies/Test.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    exit()

# Get the width and height of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize a global track ID counter
global_track_id = 0

# Dictionary to map perspective track IDs to global track IDs
global_id_map = defaultdict(dict)

# Initialize dictionary to store re-id features for each track and each perspective
reid_features = defaultdict(lambda: defaultdict(list))

# Load the OSNet model using torchreid
osnet_model = models.build_model(name="osnet_x1_0", num_classes=1, pretrained=True).to(
    device
)
osnet_model.eval()

# Define the transform for the image
transform = reid_transforms.Compose(
    [
        reid_transforms.Resize((256, 128)),  # Resize to (height, width)
        reid_transforms.ToTensor(),
        reid_transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def match_features(feature1, feature2, threshold=0.5):
    feature1 = normalize([feature1])[0]
    feature2 = normalize([feature2])[0]
    distance = cdist([feature1], [feature2], metric="euclidean")[0][0]
    return distance < threshold


def extract_color_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def extract_lbp_features(image, num_points=24, radius=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, num_points, radius, method="uniform")
    (hist, _) = np.histogram(
        lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2)
    )
    hist = hist.astype("float")
    hist /= hist.sum() + 1e-6
    return hist


def extract_hog_features(image):
    winSize = (64, 128)
    blockSize = (16, 16)
    blockStride = (8, 8)
    cellSize = (8, 8)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    return hog.compute(image).flatten()


def extract_features(frame, bbox):
    x1, y1, x2, y2 = bbox
    cropped_img = frame[int(y1) : int(y2), int(x1) : int(x2)]
    if cropped_img.size == 0:
        return None
    resized_img = cv2.resize(cropped_img, (128, 256))

    # Convert numpy array to PIL image
    pil_img = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        features_osnet = osnet_model(img_tensor).cpu().numpy().flatten()

    # Extract additional visual features
    color_hist = extract_color_histogram(resized_img)
    lbp_features = extract_lbp_features(resized_img)
    hog_features = extract_hog_features(resized_img)

    # Concatenate all features
    combined_features = np.hstack(
        [features_osnet, color_hist, lbp_features, hog_features]
    )

    return combined_features


def detect_saliency(frame, previous_frame):
    if previous_frame is None:
        return 0
    previous_frame_resized = cv2.resize(
        previous_frame, (frame.shape[1], frame.shape[0])
    )
    saliency = np.mean(np.abs(frame - previous_frame_resized))
    return saliency


object_fov_map = defaultdict(set)  # Import defaultdict from collections


def transform_perspective(frame, perspective, fov_height_factor=2.0):
    frame_height, frame_width = frame.shape[:2]
    fov_degrees = 60
    fov_width = int(frame_width * fov_degrees / 360)
    fov_height = int(frame_height * fov_degrees / 360 * fov_height_factor)

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
    else:
        raise ValueError("Invalid perspective")

    # Ensure coordinates are within frame bounds
    x1 = max(fov_center["x"] - fov_width // 2, 0)
    y1 = max(fov_center["y"] - fov_height // 2, 0)
    x2 = min(fov_center["x"] + fov_width // 2, frame_width)
    y2 = min(fov_center["y"] + fov_height // 2, frame_height)

    perspective_frame = frame[y1:y2, x1:x2]

    return perspective_frame, fov_center, fov_width, fov_height


# Define the virtual boundaries based on FOVs
virtual_boundaries = [0, 960, 2880, 3840, 4880, 5760]
boundary_threshold = 500  # Set a threshold to determine proximity to a boundary


def handle_boundary_crossing(global_id, x1, x2, frame_width, buffer_zone):
    """Handle cases where the object crosses boundaries or is near the boundary."""
    for boundary in virtual_boundaries:
        if is_near_boundary(x1, x2, boundary, buffer_zone):
            # Adjust bounding box if it is partially outside the frame
            if x1 < 0:
                x1 = 0
            if x2 > frame_width:
                x2 = frame_width

            # Check if the bounding box is near the boundary
            if x1 <= boundary + buffer_zone and x2 >= boundary - buffer_zone:
                return global_id
    return None


def merge_bounding_boxes_across_boundaries(bboxes, frame_width):
    merged_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        for boundary in virtual_boundaries:
            if is_near_boundary(x1, x2, boundary, boundary_threshold):
                # If bounding box crosses a boundary, adjust its coordinates
                if x1 < boundary and x2 > boundary:
                    # Extend the bounding box across the boundary
                    x2 = min(frame_width, boundary + (x2 - boundary))
                elif x2 > frame_width:
                    # Adjust bounding box if it goes beyond the frame width
                    x2 = frame_width
                merged_bboxes.append((x1, y1, x2, y2))
    return merged_bboxes


# Initialize dictionary to track objects crossing boundaries
crossing_objects = defaultdict(lambda: {"left": None, "right": None})

# Initialize dictionary to track objects crossing boundaries
crossing_objects = defaultdict(lambda: None)


# Update the function that checks and assigns IDs
def update_id_for_crossing_objects(
    global_id, track_id, x1, x2, frame_width, buffer_zone
):
    """Update the ID for objects crossing boundaries based on their coordinates."""
    for boundary in virtual_boundaries:
        if is_near_boundary(x1, x2, boundary, buffer_zone):
            # If crossing the boundary, assign the same ID if previously assigned
            if crossing_objects[global_id] is not None:
                track_id = crossing_objects[global_id]
            else:
                crossing_objects[global_id] = track_id
            return track_id
    return None


buffer_zone = 20

# Initialize dictionary to track objects crossing boundaries with their history
object_history = defaultdict(lambda: {"bbox": None, "frames": 0})


def is_near_boundary(x1, x2, boundary, buffer_zone):
    """Check if the bounding box is near the given boundary within the buffer zone."""
    return x1 <= boundary + buffer_zone and x2 >= boundary - buffer_zone


from collections import defaultdict

# Initialize dictionary to track objects near boundaries and their last known IDs
near_boundary_objects = defaultdict(lambda: {"last_seen": None, "last_id": None})


def update_near_boundary_objects(track_id, x1, x2, frame_width, buffer_zone):
    """Update objects near boundaries and their last known IDs."""
    for boundary in virtual_boundaries:
        if is_near_boundary(x1, x2, boundary, buffer_zone):
            near_boundary_objects[track_id] = {
                "last_seen": boundary,
                "last_id": global_id_map["all"].get(track_id),
            }
            return
    # If the object is not near a boundary, remove it from the dictionary
    if track_id in near_boundary_objects:
        del near_boundary_objects[track_id]


def check_reappeared_object(track_id, x1, x2, frame_width, buffer_zone):
    """Check if an object reappears near a boundary and reassign the same ID if necessary."""
    for boundary in virtual_boundaries:
        if is_near_boundary(x1, x2, boundary, buffer_zone):
            if track_id in near_boundary_objects:
                return near_boundary_objects[track_id]["last_id"]
    return None


# Inside your main loop where you process tracks:
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fov_detections = defaultdict(list)

    # Perform detection and transformation for each FOV
    for perspective in previous_frames.keys():
        perspective_frame, fov_center, fov_width, fov_height = transform_perspective(
            frame, perspective, fov_height_factor=2.0
        )

        # Perform detection on each perspective
        results = model(perspective_frame, classes=[0, 2, 3], device=device)[0]

        for result in results.boxes:
            conf = result.conf.item()
            cls = result.cls.item()
            if conf > 0.3:
                bbox_tensor = result.xyxy[0]
                if bbox_tensor.size() == torch.Size([4]):
                    orig_x1, orig_y1, orig_x2, orig_y2 = bbox_tensor.tolist()
                    orig_x1 = int(orig_x1) + fov_center["x"] - fov_width // 2
                    orig_y1 = int(orig_y1) + fov_center["y"] - fov_height // 2
                    orig_x2 = int(orig_x2) + fov_center["x"] - fov_width // 2
                    orig_y2 = int(orig_y2) + fov_center["y"] - fov_height // 2

                    # Check if bounding box is within the frame bounds
                    if (
                        orig_x1 < 0
                        or orig_x1 >= frame_width
                        or orig_y1 < 0
                        or orig_y1 >= frame_height
                        or orig_x2 < 0
                        or orig_x2 >= frame_width
                        or orig_y2 < 0
                        or orig_y2 >= frame_height
                    ):
                        continue

                    fov_detections[perspective].append(
                        {
                            "bbox": (orig_x1, orig_y1, orig_x2, orig_y2),
                            "conf": conf,
                            "cls": cls,
                        }
                    )

    bbox_xywh = []
    confs = []
    cls_ids = []

    # Process detections from all FOVs
    for perspective, detections in fov_detections.items():
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            conf = detection["conf"]
            cls = detection["cls"]
            bbox_xywh.append([int((x1 + x2) / 2), int((y1 + y2) / 2), x2 - x1, y2 - y1])
            confs.append(conf)
            cls_ids.append(cls)

    bbox_xywh = np.array(bbox_xywh)
    confs = np.array(confs)
    cls_ids = np.array(cls_ids)

    # Update tracks
    tracks = tracker.update(bbox_xywh, confs, cls_ids, frame)

    # Loop over tracks and process each track
    for track in tracker.tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue

        track_id = track.track_id
        if track_id not in global_id_map["all"]:
            global_id_map["all"][track_id] = global_track_id
            global_track_id += 1
        global_id = global_id_map["all"][track_id]

        # Extract track's bounding box
        x1, y1, x2, y2 = track.to_tlbr()

        # Update objects near boundaries
        update_near_boundary_objects(track_id, x1, x2, frame_width, buffer_zone)

        # Check if the object has reappeared near a boundary
        updated_id = check_reappeared_object(track_id, x1, x2, frame_width, buffer_zone)
        if updated_id is not None:
            global_id = updated_id

        # Draw bounding box and ID in each FOV
        for perspective in previous_frames.keys():
            perspective_frame, fov_center, fov_width, fov_height = (
                transform_perspective(frame, perspective, fov_height_factor=2.0)
            )

            # Adjust the coordinates to map onto the perspective frame
            perspective_x1 = x1 - (fov_center["x"] - fov_width // 2)
            perspective_y1 = y1 - (fov_center["y"] - fov_height // 2)
            perspective_x2 = x2 - (fov_center["x"] - fov_width // 2)
            perspective_y2 = y2 - (fov_center["y"] - fov_height // 2)

            # Check if the bounding box falls within this FOV
            if (0 <= perspective_x1 < perspective_frame.shape[1]) and (
                0 <= perspective_y1 < perspective_frame.shape[0]
            ):
                cv2.rectangle(
                    perspective_frame,
                    (int(perspective_x1), int(perspective_y1)),
                    (int(perspective_x2), int(perspective_y2)),
                    bbox_colors[0],
                    2,
                )
                cv2.putText(
                    perspective_frame,
                    f"ID: {global_id}",
                    (int(perspective_x1) + 10, int(perspective_y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                object_fov_map[global_id].add(perspective)

            # Display the FOV frame
            cv2.imshow(f"FOV {perspective}", perspective_frame)
            frame_resized = cv2.resize(frame, (1500, 780))
            cv2.imshow(f"obj", frame_resized)

    # Quit the loop if 'q' is pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
