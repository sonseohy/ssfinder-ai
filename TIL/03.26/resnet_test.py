import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from ultralytics import YOLO
from matplotlib import font_manager, rc
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# Path settings
image_path = "blueumb.jpg"  # Test image path
output_dir = "comparison_results"  # Output directory
resnet_model_path = "fasterrcnn_resnet50_fpn.pth"  # ResNet model path (user-defined)

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load YOLO model
# Set path for fine-tuned model
finetuned_model_path = "best.pt"  # Fine-tuned model path
finetuned_model = YOLO(finetuned_model_path)

# Load YOLOv8m-oiv7 model
yolov8m_oiv7_model = YOLO("yolov8m-oiv7.pt")

# Load ResNet model
# Load pre-trained ResNet model or custom model
if os.path.exists(resnet_model_path):
    # Load custom model
    resnet_model = fasterrcnn_resnet50_fpn(pretrained=False)
    resnet_model.load_state_dict(torch.load(resnet_model_path))
else:
    # Load pre-trained model
    resnet_model = fasterrcnn_resnet50_fpn(pretrained=True)

resnet_model.eval()
if torch.cuda.is_available():
    resnet_model = resnet_model.cuda()

# COCO dataset class names (for ResNet model)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Load image
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Could not load image: {image_path}")

# Convert BGR to RGB (OpenCV uses BGR, matplotlib uses RGB)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Run YOLO model inference
# Inference with fine-tuned model
finetuned_results = finetuned_model(image_rgb)
# Inference with YOLOv8m-oiv7 model
yolov8m_oiv7_results = yolov8m_oiv7_model(image_rgb)

# Run ResNet model inference
def get_resnet_prediction(image_rgb, model, threshold=0.5):
    # Preprocess image
    image_tensor = F.to_tensor(image_rgb)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    
    # Inference
    with torch.no_grad():
        prediction = model([image_tensor])
    
    # Filter results
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    
    # Keep predictions above threshold
    keep = scores > threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    return boxes, scores, labels

# Inference with ResNet model
resnet_boxes, resnet_scores, resnet_labels = get_resnet_prediction(image_rgb, resnet_model)

# Function to visualize YOLO results
def visualize_yolo_results(image, results, model_name):
    # Copy result image
    img_copy = image.copy()
    
    # Number of detected objects
    num_detections = len(results[0].boxes)
    
    # Extract bounding boxes, classes, and confidence from results
    for i in range(num_detections):
        box = results[0].boxes[i].xyxy[0].cpu().numpy()
        conf = float(results[0].boxes[i].conf[0].cpu().numpy())
        cls = int(results[0].boxes[i].cls[0].cpu().numpy())
        cls_name = results[0].names[cls]
        
        # Draw bounding box
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display class name and confidence
        label = f"{cls_name}: {conf:.2f}"
        cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add title
    title = f"Model: {model_name}, Detected Objects: {num_detections}"
    
    return img_copy, title, num_detections

# Function to visualize ResNet results
def visualize_resnet_results(image, boxes, scores, labels, model_name):
    # Copy result image
    img_copy = image.copy()
    
    # Number of detected objects
    num_detections = len(boxes)
    
    # Get detected class distribution
    class_counts = {}
    for label in labels:
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    
    # Extract bounding boxes, classes, and confidence from results
    for i in range(num_detections):
        box = boxes[i]
        conf = scores[i]
        cls = labels[i]
        cls_name = COCO_INSTANCE_CATEGORY_NAMES[cls]
        
        # Draw bounding box
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Display class name and confidence
        label = f"{cls_name}: {conf:.2f}"
        cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Add title
    title = f"Model: {model_name}, Detected Objects: {num_detections}"
    
    # Add ResNet model analysis to the image
    y_pos = 30
    cv2.putText(img_copy, "ResNet Model Analysis:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)
    y_pos += 30
    cv2.putText(img_copy, f"- Total detected objects: {num_detections}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)
    y_pos += 25
    
    # Add class distribution
    if num_detections > 0:
        cv2.putText(img_copy, "- Class distribution:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)
        y_pos += 25
        
        for cls_name, count in class_counts.items():
            if cls_name != "N/A":  # Skip N/A classes
                text = f"  * {cls_name}: {count}"
                cv2.putText(img_copy, text, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)
                y_pos += 25
                
                # Limit the number of classes displayed to prevent overflow
                if (y_pos > image.shape[0] - 30) or (len(class_counts) > 5 and list(class_counts.keys()).index(cls_name) >= 4):
                    cv2.putText(img_copy, "    ... and more", (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)
                    break
    
    return img_copy, title, num_detections

# Updated YOLO visualization function with summary
def visualize_yolo_results_with_summary(image, results, model_name):
    # Call original visualization function
    img_copy, title, num_detections = visualize_yolo_results(image, results, model_name)
    
    # Count detected classes
    class_counts = {}
    
    # Aggregate class counts
    for i in range(num_detections):
        cls = int(results[0].boxes[i].cls[0].cpu().numpy())
        cls_name = results[0].names[cls]
        
        if cls_name in class_counts:
            class_counts[cls_name] += 1
        else:
            class_counts[cls_name] = 1
    
    # Add summary information to the image
    if num_detections > 0:
        y_pos = 30
        cv2.putText(img_copy, "Detected Objects:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)
        y_pos += 30
        
        for cls_name, count in class_counts.items():
            text = f"- {cls_name}: {count}"
            cv2.putText(img_copy, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)
            y_pos += 25
            
            # Limit the number of displayed classes
            if (y_pos > image.shape[0] - 30) or (len(class_counts) > 8 and list(class_counts.keys()).index(cls_name) >= 7):
                cv2.putText(img_copy, "... and more", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)
                break
    
    return img_copy, title, num_detections

# Visualize and save results
finetuned_img, finetuned_title, finetuned_detections = visualize_yolo_results_with_summary(image_rgb, finetuned_results, "Fine-tuned YOLO")
yolov8m_img, yolov8m_title, yolov8m_detections = visualize_yolo_results_with_summary(image_rgb, yolov8m_oiv7_results, "YOLOv8m-OIV7")
resnet_img, resnet_title, resnet_detections = visualize_resnet_results(image_rgb, resnet_boxes, resnet_scores, resnet_labels, "ResNet50-FPN")

# Save results
cv2.imwrite(os.path.join(output_dir, "finetuned_result.jpg"), cv2.cvtColor(finetuned_img, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "yolov8m_result.jpg"), cv2.cvtColor(yolov8m_img, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "resnet_result.jpg"), cv2.cvtColor(resnet_img, cv2.COLOR_RGB2BGR))

# Visualize and save using matplotlib
plt.figure(figsize=(18, 6))

# First image (fine-tuned model)
plt.subplot(1, 3, 1)
plt.imshow(finetuned_img)
plt.title(finetuned_title)
plt.axis('off')

# Second image (YOLOv8m-OIV7 model)
plt.subplot(1, 3, 2)
plt.imshow(yolov8m_img)
plt.title(yolov8m_title)
plt.axis('off')

# Third image (ResNet model)
plt.subplot(1, 3, 3)
plt.imshow(resnet_img)
plt.title(resnet_title)
plt.axis('off')

# Set overall title
plt.suptitle("Object Detection Model Performance Comparison", fontsize=16)
plt.tight_layout()

# Save results
plt.savefig(os.path.join(output_dir, "three_model_comparison.png"), dpi=300, bbox_inches='tight')
print(f"Results saved to {output_dir} directory.")

# Output detection results
print(f"Fine-tuned YOLO model detections: {finetuned_detections}")
print(f"YOLOv8m-OIV7 model detections: {yolov8m_detections}")
print(f"ResNet50-FPN model detections: {resnet_detections}")

# Create model performance comparison table
performance_fig = plt.figure(figsize=(10, 5))
ax = performance_fig.add_subplot(111)

# Model names and detection counts
models = ["Fine-tuned YOLO", "YOLOv8m-OIV7", "ResNet50-FPN"]
detections = [finetuned_detections, yolov8m_detections, resnet_detections]

# Create bar chart
bars = ax.bar(models, detections, color=['skyblue', 'lightgreen', 'salmon'])
ax.set_ylabel('Number of Detections')
ax.set_title('Object Detection Model Performance Comparison')

# Display numbers above each bar
for i, (bar, count) in enumerate(zip(bars, detections)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            str(count), ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "three_model_detection_comparison.png"), dpi=300, bbox_inches='tight')

# Additional performance comparison - measure processing time
import time

def measure_inference_time(model, image, n_iterations=10):
    # Warm-up
    if isinstance(model, YOLO):
        _ = model(image)
    else:  # ResNet model
        image_tensor = F.to_tensor(image)
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        with torch.no_grad():
            _ = model([image_tensor])
    
    # Start time measurement
    start_time = time.time()
    
    # Repeat multiple times for average measurement
    for _ in range(n_iterations):
        if isinstance(model, YOLO):
            _ = model(image)
        else:  # ResNet model
            with torch.no_grad():
                _ = model([image_tensor])
    
    # End time measurement
    end_time = time.time()
    
    # Calculate average inference time (in milliseconds)
    avg_time = (end_time - start_time) * 1000 / n_iterations
    
    return avg_time

# Measure inference time for each model
print("Measuring inference time for each model...")
finetuned_time = measure_inference_time(finetuned_model, image_rgb)
yolov8m_time = measure_inference_time(yolov8m_oiv7_model, image_rgb)
resnet_time = measure_inference_time(resnet_model, image_rgb)

print(f"Fine-tuned YOLO model average inference time: {finetuned_time:.2f} ms")
print(f"YOLOv8m-OIV7 model average inference time: {yolov8m_time:.2f} ms")
print(f"ResNet50-FPN model average inference time: {resnet_time:.2f} ms")

# Inference time comparison graph
time_fig = plt.figure(figsize=(10, 5))
ax = time_fig.add_subplot(111)

# Model names and inference times
times = [finetuned_time, yolov8m_time, resnet_time]

# Create bar chart
bars = ax.bar(models, times, color=['skyblue', 'lightgreen', 'salmon'])
ax.set_ylabel('Average Inference Time (ms)')
ax.set_title('Object Detection Model Inference Time Comparison')

# Display numbers above each bar
for i, (bar, t) in enumerate(zip(bars, times)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f"{t:.2f} ms", ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "inference_time_comparison.png"), dpi=300, bbox_inches='tight')

# Comprehensive performance evaluation - detections vs. inference time
efficiency_fig = plt.figure(figsize=(12, 6))
ax1 = efficiency_fig.add_subplot(111)

# Set up two y-axes
ax2 = ax1.twinx()

# First axis shows detection counts
bars1 = ax1.bar(np.arange(len(models)) - 0.2, detections, width=0.4, color='skyblue', label='Number of Detections')
ax1.set_ylabel('Number of Detections', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Second axis shows inference time
bars2 = ax2.bar(np.arange(len(models)) + 0.2, times, width=0.4, color='salmon', label='Inference Time (ms)')
ax2.set_ylabel('Inference Time (ms)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Set x-axis labels
ax1.set_xticks(np.arange(len(models)))
ax1.set_xticklabels(models)

# Set title
plt.title('Object Detection Model Efficiency Comparison (Detections vs Time)')

# Display legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "model_efficiency_comparison.png"), dpi=300, bbox_inches='tight')

print("Comparative analysis of the three models completed.")

# ResNet model analysis and results
print("=" * 50)
print("ResNet Model Analysis:")
print(f"- Detected objects: {resnet_detections}")
print(f"- Average inference time: {resnet_time:.2f} ms")

# Check class distribution
if resnet_detections > 0:
    class_counts = {}
    for label in resnet_labels:
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    
    print("- Detected object class distribution:")
    for cls, count in class_counts.items():
        print(f"  * {cls}: {count}")

print("=" * 50)