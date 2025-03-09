from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Define paths
data_yaml_path = "C:\\Users\\Mayank Sharma\\OneDrive\\Desktop\\Data Detect\\dataset\\data.yaml"
trained_model_path = "runs/detect/train16/weights/best.pt"
input_image_path = "C:\\Users\\Mayank Sharma\\OneDrive\\Desktop\\Data Detect\\img52.jpg"
output_image_path = "C:\\Users\\Mayank Sharma\\OneDrive\\Desktop\\Data Detect\\output.jpg"

def train_model():
    """ Train the YOLOv8 model """
    model = YOLO("yolov8m.pt")  # Load pre-trained YOLOv8 model
    model.train(data=data_yaml_path, epochs=20)  # Train the model
    print("Training completed. Model saved in 'runs/detect/train16/weights/'")

def run_inference():
    """ Run inference on an input image using trained model """
    model = YOLO(trained_model_path)  # Load trained model
    results = model(input_image_path)  # Run inference

    # Save the output image
    for r in results:
        r.save(output_image_path)  # Saves annotated image
    print(f"Inference completed. Output saved as {output_image_path}")

    # Display the output image
    output_image = cv2.imread(output_image_path)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

    plt.imshow(output_image)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    train_model()  # Train the model
    run_inference()  # Run inference on img52.jpg
