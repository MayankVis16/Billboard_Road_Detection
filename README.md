# Billboard Detection and Road Analysis using YOLOv8

## Overview

This project aims to detect and analyze billboards and road features using YOLOv8. The model is trained on a custom dataset annotated using Roboflow. The main objectives include:

- Detecting lanes and estimating road width
- Identifying road corners
- Detecting billboards and their attributes (size, position, occlusion, distance, etc.)
- Estimating billboard clutter and angle

## Problem Statement

From a 2D image, the system extracts the following information:

1. Number of lanes and road width estimation (if lanes are unmarked).
2. Road corner detection.
3. Billboard detection.
4. Billboard position classification (LHS, RHS, Overhead).
5. Billboard size estimation (Width, Height, Elevation, Setback Distance, Angle).
6. Billboard occlusion percentage estimation.
7. Billboard clutter detection (detecting closely placed billboards causing distractions).
8. Billboard distance estimation considering zoom effects.

## Dataset and Annotation

- **Annotation Tool**: Roboflow
- **Annotation Format**: YOLO format
- **Dataset Composition**: Public datasets (MS COCO, Mapillary Vistas) + custom annotated dataset

## Dependencies

Ensure the following packages are installed before running the code:

```bash
pip install ultralytics opencv-python numpy matplotlib scikit-learn
```

## Training the YOLOv8 Model

The training script uses a pre-trained YOLOv8 model and fine-tunes it on the custom dataset.

```python
from ultralytics import YOLO

data_yaml_path = "C:\\Users\\Mayank Sharma\\OneDrive\\Desktop\\Data Detect\\dataset\\data.yaml"
model = YOLO("yolov8m.pt")  # Load pre-trained model
model.train(data=data_yaml_path, epochs=20)  # Train on dataset
```

## Running Inference

Once trained, the model can detect billboards and analyze road features.

```python
model = YOLO("runs/detect/train16/weights/best.pt")
results = model("C:\\Users\\Mayank Sharma\\OneDrive\\Desktop\\Data Detect\\img52.jpg")
results[0].save("C:\\Users\\Mayank Sharma\\OneDrive\\Desktop\\Data Detect\\output.jpg")
```

## Solution Code

The main solution script includes:

1. **Lane Detection**: Using edge detection and Hough Transform.
2. **Road Width Estimation**: Based on lane/road edge detection.
3. **Road Corner Detection**: Using Shi-Tomasi corner detection.
4. **Billboard Detection**: YOLOv8-based detection.
5. **Billboard Position Classification**: LHS, RHS, Overhead.
6. **Occlusion Detection**: Calculating overlapping billboard areas.
7. **Clutter Detection**: Using DBSCAN clustering to detect closely packed billboards.
8. **Billboard Distance Estimation**: Using focal length and known size.
9. **Billboard Angle and Elevation Estimation**: Using image perspective and road width reference.

## Running the Main Script

```bash
python main.py --image_path path/to/image.jpg
```

## Output

The script prints:

- Number of lanes detected
- Estimated road width
- Number of detected billboards and their attributes
- Occlusion percentage and clutter detection results

## Future Enhancements

- **Depth-based road width estimation**
- **Improved YOLO dataset for billboard detection**
- **Advanced occlusion detection methods**
- **Better angle estimation techniques**

## Author

**Mayank Sharma** (Intern)

 
