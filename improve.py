 import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import math
import argparse

# Constants
REFERENCE_BILLBOARD_WIDTH = 6.0  # Reference billboard width in meters
REFERENCE_BILLBOARD_HEIGHT = 3.0  # Reference billboard height in meters
REFERENCE_DISTANCE = 50.0  # Reference distance in meters for focal length calibration
REFERENCE_FOCAL_LENGTH = 1000.0  # Reference focal length in pixels

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

def detect_lanes(image):
    """Detect lanes using edge detection and Hough Transform."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=200)
    return [line[0] for line in lines] if lines is not None else []

def estimate_road_width(image, lanes):
    """Estimate road width if lanes are missing by detecting road edges."""
    if lanes:
        return None  # Lanes detected, no need to estimate width

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    left_edge = min(contours, key=lambda c: cv2.boundingRect(c)[0])
    right_edge = max(contours, key=lambda c: cv2.boundingRect(c)[0])
    x1, _, w1, _ = cv2.boundingRect(left_edge)
    x2, _, w2, _ = cv2.boundingRect(right_edge)
    return (x2 + w2) - x1  # Estimated road width

def detect_corners(image):
    """Detect road corners using Shi-Tomasi."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    return np.int0(corners) if corners is not None else []

def detect_billboards_yolo(image, conf_threshold=0.5):
    """Detect billboards using YOLOv8 with a confidence threshold."""
    results = model(image)
    billboards = []
    
    for r in results:
        for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if conf >= conf_threshold:
                x_min, y_min, x_max, y_max = box
                billboards.append((int(x_min), int(y_min), int(x_max), int(y_max)))
    return billboards

def classify_billboard_position(bbox, image_width):
    """Classify billboard as LHS, RHS, or Overhead."""
    x_min, _, x_max, _ = bbox
    center_x = (x_min + x_max) / 2
    road_center_x = image_width // 2
    return "Overhead" if abs(center_x - road_center_x) < 50 else ("LHS" if center_x < road_center_x else "RHS")

def detect_occlusion(billboards, non_billboard_objects):
    """Detect occlusion by checking overlapping bounding boxes."""
    occlusion_rates = []
    for i, bbox1 in enumerate(billboards):
        x1_min, y1_min, x1_max, y1_max = bbox1
        occluded_area = sum(
            max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) * max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            for j, bbox2 in enumerate(billboards + non_billboard_objects) if i != j
            for x2_min, y2_min, x2_max, y2_max in [bbox2]
        )
        billboard_area = (x1_max - x1_min) * (y1_max - y1_min)
        occlusion_rates.append((occluded_area / billboard_area) * 100 if billboard_area > 0 else 0)
    return occlusion_rates

def detect_cluttered_billboards(billboards):
    """Detect billboards that are too close together using clustering."""
    if len(billboards) < 2:
        return []

    centers = np.array([((x_min + x_max) / 2, (y_min + y_max) / 2) for x_min, y_min, x_max, y_max in billboards])
    clustering = DBSCAN(eps=50, min_samples=2).fit(centers)
    cluttered = [billboards[i] for i in range(len(billboards)) if clustering.labels_[i] != -1]
    return cluttered

def calculate_focal_length(perceived_width, distance):
    """Calculate focal length dynamically based on perceived width and distance."""
    return (perceived_width * distance) / REFERENCE_BILLBOARD_WIDTH

def estimate_billboard_distance(bbox, focal_length):
    """Estimate billboard distance using reference width and focal length."""
    x_min, _, x_max, _ = bbox
    perceived_width = x_max - x_min
    
    if perceived_width <= 0:
        return None  # Avoid division by zero

    # Distance formula: (Actual Width * Focal Length) / Perceived Width
    return (REFERENCE_BILLBOARD_WIDTH * focal_length) / perceived_width

def estimate_billboard_real_world_dimensions(bbox, distance, focal_length):
    """Estimate real-world width and height of the billboard using perspective transformation."""
    x_min, y_min, x_max, y_max = bbox
    perceived_width = x_max - x_min
    perceived_height = y_max - y_min

    # Real-world width and height using perspective scaling
    real_width = (perceived_width * distance) / focal_length
    real_height = (perceived_height * distance) / focal_length
    return real_width, real_height

def estimate_billboard_angle(bbox, road_width, image_width):
    """Estimate billboard angle using perspective transform."""
    x_min, _, x_max, _ = bbox
    center_x = (x_min + x_max) / 2
    road_center_x = image_width // 2

    if road_width is None:
        return None  # Cannot estimate without road reference
    
    relative_position = (center_x - road_center_x) / road_width
    return math.degrees(math.atan(relative_position))  # Angle in degrees

def estimate_billboard_elevation(bbox, image_height):
    """Estimate billboard elevation using reference height and road width."""
    _, y_min, _, y_max = bbox
    perceived_height = y_max - y_min
    
    if perceived_height <= 0:
        return None  # Avoid division by zero
    
    # Elevation estimation using reference billboard height
    elevation_ratio = perceived_height / image_height
    return REFERENCE_BILLBOARD_HEIGHT * elevation_ratio

def calculate_setback_distance(bbox, corners):
    """Calculate the setback distance of the nearest billboard edge from the nearest road corner."""
    x_min, y_min, x_max, y_max = bbox
    billboard_edges = [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]
    
    min_distance = float('inf')
    for corner in corners:
        corner_x, corner_y = corner.ravel()
        for edge_x, edge_y in billboard_edges:
            distance = math.sqrt((edge_x - corner_x)**2 + (edge_y - corner_y)**2)
            if distance < min_distance:
                min_distance = distance
    return min_distance

def detect_non_billboard_objects(image):
    """Detect non-billboard objects like buildings using YOLOv8."""
    results = model(image)
    non_billboard_objects = []
    
    for r in results:
        for box, conf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            if conf >= 0.5 and r.names[int(box[-1])] != 'billboard':
                x_min, y_min, x_max, y_max = box
                non_billboard_objects.append((int(x_min), int(y_min), int(x_max), int(y_max)))
    return non_billboard_objects

def main(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    lanes = detect_lanes(image)
    road_width = estimate_road_width(image, lanes)
    corners = detect_corners(image)
    billboards = detect_billboards_yolo(image)
    non_billboard_objects = detect_non_billboard_objects(image)
    occlusion_rates = detect_occlusion(billboards, non_billboard_objects)
    cluttered_billboards = detect_cluttered_billboards(billboards)
    
    # Estimate focal length dynamically
    if billboards:
        first_bbox = billboards[0]
        x_min, _, x_max, _ = first_bbox
        perceived_width = x_max - x_min
        focal_length = calculate_focal_length(perceived_width, REFERENCE_DISTANCE)
    else:
        focal_length = REFERENCE_FOCAL_LENGTH  # Default focal length

    distances = [estimate_billboard_distance(bbox, focal_length) for bbox in billboards]
    real_world_dimensions = [estimate_billboard_real_world_dimensions(bbox, distance, focal_length) 
                            for bbox, distance in zip(billboards, distances)]
    elevations = [estimate_billboard_elevation(bbox, image.shape[0]) for bbox in billboards]
    angles = [estimate_billboard_angle(bbox, road_width, image.shape[1]) for bbox in billboards]
    setback_distances = [calculate_setback_distance(bbox, corners) for bbox in billboards]

    print(f"\n[INFO] Detected {len(lanes)} lanes.")
    print(f"[INFO] Estimated Road Width: {road_width} pixels" if road_width else "[INFO] Lanes detected, no width estimation needed.")
    print(f"[INFO] Detected {len(corners)} road corners.")
    print(f"[INFO] Detected {len(billboards)} billboards.")
    print(f"[INFO] Detected {len(cluttered_billboards)} cluttered billboards.\n")

    for i, bbox in enumerate(billboards):
        position = classify_billboard_position(bbox, image.shape[1])
        occlusion = occlusion_rates[i]
        distance = distances[i]
        real_width, real_height = real_world_dimensions[i]
        elevation = elevations[i]
        angle = angles[i] if angles[i] is not None else 0  # Default value if None
        setback_distance = setback_distances[i]

        print(f"Billboard {i+1}: Position: {position}, "
              f"Occlusion: {occlusion:.2f}%, "
              f"Distance: {distance:.2f} meters, "
              f"Real-World Dimensions: {real_width:.2f}m x {real_height:.2f}m, "
              f"Elevation: {elevation:.2f} pixels, "
              f"Angle: {angle:.2f}°, "
              f"Setback Distance: {setback_distance:.2f} pixels")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect billboards and lanes from an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()
    
    main(args.image_path)
