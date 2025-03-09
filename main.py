import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import math
import argparse

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

def detect_occlusion(billboards):
    """Detect occlusion by checking overlapping bounding boxes."""
    occlusion_rates = []
    for i, bbox1 in enumerate(billboards):
        x1_min, y1_min, x1_max, y1_max = bbox1
        occluded_area = sum(
            max(0, min(x1_max, x2_max) - max(x1_min, x2_min)) * max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            for j, bbox2 in enumerate(billboards) if i != j
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

def estimate_billboard_distance(bbox, known_width=3, focal_length=700):
    """Estimate billboard distance using focal length and known size."""
    x_min, _, x_max, _ = bbox
    perceived_width = x_max - x_min
    return (known_width * focal_length) / perceived_width if perceived_width > 0 else None

def estimate_billboard_angle(bbox, road_width, image_width):
    """Estimate billboard angle using perspective transform."""
    x_min, _, x_max, _ = bbox
    center_x = (x_min + x_max) / 2
    road_center_x = image_width // 2

    if road_width is None:
        return None  # Cannot estimate without road reference
    
    relative_position = (center_x - road_center_x) / road_width
    return math.degrees(math.atan(relative_position))  # Angle in degrees

def estimate_billboard_elevation(bbox, road_width, image_height):
    """Estimate billboard elevation using road width as reference."""
    _, y_min, _, y_max = bbox
    perceived_height = y_max - y_min
    
    if road_width is None:
        return None  # Cannot estimate without road width
    
    elevation_ratio = perceived_height / image_height
    return road_width * elevation_ratio  # Estimated elevation in pixels

def main(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    lanes = detect_lanes(image)
    road_width = estimate_road_width(image, lanes)
    corners = detect_corners(image)
    billboards = detect_billboards_yolo(image)
    occlusion_rates = detect_occlusion(billboards)
    cluttered_billboards = detect_cluttered_billboards(billboards)
    
    distances = [estimate_billboard_distance(bbox) for bbox in billboards]
    elevations = [estimate_billboard_elevation(bbox, road_width, image.shape[0]) for bbox in billboards]
    angles = [estimate_billboard_angle(bbox, road_width, image.shape[1]) for bbox in billboards]

    print(f"\n[INFO] Detected {len(lanes)} lanes.")
    print(f"[INFO] Estimated Road Width: {road_width} pixels" if road_width else "[INFO] Lanes detected, no width estimation needed.")
    print(f"[INFO] Detected {len(corners)} road corners.")
    print(f"[INFO] Detected {len(billboards)} billboards.")
    print(f"[INFO] Detected {len(cluttered_billboards)} cluttered billboards.\n")

    for i, bbox in enumerate(billboards):
        position = classify_billboard_position(bbox, image.shape[1])
        occlusion = occlusion_rates[i]
        distance = distances[i]
        elevation = elevations[i]
        angle = angles[i] if angles[i] is not None else 0  # Default value if None

        print(f"Billboard {i+1}: Position: {position}, "
              f"Occlusion: {occlusion:.2f}%, "
              f"Distance: {distance:.2f} units, "
              f"Elevation: {elevation:.2f} pixels, "
              f"Angle: {angle:.2f}°")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect billboards and lanes from an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()
    
    main(args.image_path)
