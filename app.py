import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO
from deep_sort_pytorch.utils.parser import get_config
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
logging.basicConfig(filename='boundary_tracking.log',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Call the function to download the required files
download_files.download_dependencies()

# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logging.info("GPU is available. Running on GPU.")
else:
    logging.info("GPU is not available. Running on CPU.")

# Load the YOLO model and move it to the appropriate device
try:
    model = YOLO("yolov10x.pt").to(device)
except Exception as e:
    logging.error(f"Error loading YOLO model: {e}")
    exit()

# Define the field of view (FOV) for different perspectives
fov_degrees = 60  # Each FOV is 60 degrees
threshold = random.randint(120, 140)
logging.info(f"Threshold for motion detection: {threshold}")

# Initialize previous frames for motion-based saliency for each perspective
previous_frames = {str(i): None for i in range(1, 7)}

# Initialize dictionary to store centroid history for each track and each FOV
centroid_history = defaultdict(lambda: defaultdict(list))

# Initialize a single tracker for the entire original frame
tracker = DeepSort(model_path='deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7', max_age=15)

# Define colors for paths and bounding boxes
path_colors = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 255), (0, 0, 255)]  # Yellow, Cyan, Green, Magenta, Red
bbox_colors = [(255, 0, 255)]  # Magenta

motion_detected = {str(i): False for i in range(1, 7)}

# Open the video file
video_path = "Dependencies/Test.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    logging.error(f"Error: Unable to open video file {video_path}")
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
try:
    osnet_model = models.build_model(name='osnet_x1_0', num_classes=1, pretrained=True).to(device)
    osnet_model.eval()
except Exception as e:
    logging.error(f"Error loading OSNet model: {e}")
    exit()

# Define the transform for the image
transform = reid_transforms.Compose([
    reid_transforms.Resize((256, 128)),  # Resize to (height, width)
    reid_transforms.ToTensor(),
    reid_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def match_features(feature1, feature2, threshold=0.5):
    feature1 = normalize([feature1])[0]
    feature2 = normalize([feature2])[0]
    distance = cdist([feature1], [feature2], metric='euclidean')[0][0]
    return distance < threshold

def extract_color_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_lbp_features(image, num_points=24, radius=8):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, num_points, radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
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
    cropped_img = frame[int(y1):int(y2), int(x1):int(x2)]
    if cropped_img.size == 0:
        logging.warning(f"Empty cropped image for bbox: {bbox}")
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
    combined_features = np.hstack([features_osnet, color_hist, lbp_features, hog_features])

    return combined_features

def detect_saliency(frame, previous_frame):
    if previous_frame is None:
        return 0
    previous_frame_resized = cv2.resize(previous_frame, (frame.shape[1], frame.shape[0]))
    saliency = np.mean(np.abs(frame - previous_frame_resized))
    return saliency

object_fov_map = defaultdict(set)

def transform_perspective(frame, perspective, fov_height_factor=2.0):
    frame_height, frame_width = frame.shape[:2]
    fov_width = int(frame_width * fov_degrees / 360)
    fov_height = int(frame_height * fov_degrees / 360 * fov_height_factor)

    fov_centers = {
        "1": {"x": frame_width // 12, "y": frame_height // 2},
        "2": {"x": 3 * frame_width // 12, "y": frame_height // 2},
        "3": {"x": 5 * frame_width // 12, "y": frame_height // 2},
        "4": {"x": 7 * frame_width // 12, "y": frame_height // 2},
        "5": {"x": 9 * frame_width // 12, "y": frame_height // 2},
        "6": {"x": 11 * frame_width // 12, "y": frame_height // 2},
    }

    if perspective not in fov_centers:
        raise ValueError("Invalid perspective")

    fov_center = fov_centers[perspective]

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

def is_near_boundary(x1, x2, boundary, buffer_zone):
    """Check if the bounding box is near the given boundary within the buffer zone."""
    return (x1 <= boundary + buffer_zone and x2 >= boundary - buffer_zone)

buffer_zone = 20

# Initialize dictionary to track objects crossing boundaries
crossing_objects = defaultdict(lambda: None)

def update_near_boundary_objects(track_id, x1, x2, frame_width, buffer_zone):
