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

# 경로 설정
image_path = "blueumb.jpg"  # 테스트할 이미지 경로
output_dir = "comparison_results"  # 결과 저장 디렉토리
resnet_model_path = "fasterrcnn_resnet50_fpn.pth"  # ResNet 모델 경로 (사용자 지정)

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# YOLO 모델 로드
# 파인튜닝된 모델 경로 설정
finetuned_model_path = "best.pt"  # 파인튜닝된 모델 경로
finetuned_model = YOLO(finetuned_model_path)

# YOLOv8m-oiv7 모델 로드
yolov8m_oiv7_model = YOLO("yolov8m-oiv7.pt")

# ResNet 모델 로드
# 사전 학습된 ResNet 모델 또는 사용자 지정 모델 로드
if os.path.exists(resnet_model_path):
    # 사용자 지정 모델 로드
    resnet_model = fasterrcnn_resnet50_fpn(pretrained=False)
    resnet_model.load_state_dict(torch.load(resnet_model_path))
else:
    # 사전 학습된 모델 로드
    resnet_model = fasterrcnn_resnet50_fpn(pretrained=True)

resnet_model.eval()
if torch.cuda.is_available():
    resnet_model = resnet_model.cuda()

# COCO 데이터셋 클래스 이름 (ResNet 모델용)
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

# 이미지 로드
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

# BGR을 RGB로 변환 (OpenCV는 BGR, matplotlib는 RGB 사용)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# YOLO 모델 추론 실행
# 파인튜닝된 모델로 추론
finetuned_results = finetuned_model(image_rgb)
# YOLOv8m-oiv7 모델로 추론
yolov8m_oiv7_results = yolov8m_oiv7_model(image_rgb)

# ResNet 모델 추론 실행
def get_resnet_prediction(image_rgb, model, threshold=0.5):
    # 이미지 전처리
    image_tensor = F.to_tensor(image_rgb)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    
    # 추론
    with torch.no_grad():
        prediction = model([image_tensor])
    
    # 결과 필터링
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    
    # 임계값 이상의 예측만 유지
    keep = scores > threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    
    return boxes, scores, labels

# ResNet 모델로 추론
resnet_boxes, resnet_scores, resnet_labels = get_resnet_prediction(image_rgb, resnet_model)

# YOLO 결과 시각화 함수
def visualize_yolo_results(image, results, model_name):
    # 결과 이미지 복사
    img_copy = image.copy()
    
    # 발견된 객체 수
    num_detections = len(results[0].boxes)
    
    # 결과에서 바운딩 박스, 클래스, 신뢰도 추출
    for i in range(num_detections):
        box = results[0].boxes[i].xyxy[0].cpu().numpy()
        conf = float(results[0].boxes[i].conf[0].cpu().numpy())
        cls = int(results[0].boxes[i].cls[0].cpu().numpy())
        cls_name = results[0].names[cls]
        
        # 바운딩 박스 그리기
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 클래스 이름과 신뢰도 표시
        label = f"{cls_name}: {conf:.2f}"
        cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 제목 추가
    title = f"모델: {model_name}, 탐지 객체 수: {num_detections}"
    
    return img_copy, title, num_detections

# ResNet 결과 시각화 함수
def visualize_resnet_results(image, boxes, scores, labels, model_name):
    # 결과 이미지 복사
    img_copy = image.copy()
    
    # 발견된 객체 수
    num_detections = len(boxes)
    
    # 결과에서 바운딩 박스, 클래스, 신뢰도 추출
    for i in range(num_detections):
        box = boxes[i]
        conf = scores[i]
        cls = labels[i]
        cls_name = COCO_INSTANCE_CATEGORY_NAMES[cls]
        
        # 바운딩 박스 그리기
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 클래스 이름과 신뢰도 표시
        label = f"{cls_name}: {conf:.2f}"
        cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 제목 추가
    title = f"모델: {model_name}, 탐지 객체 수: {num_detections}"
    
    return img_copy, title, num_detections

# YOLO 결과 시각화 함수도 업데이트 - 탐지 결과 요약 추가
def visualize_yolo_results_with_summary(image, results, model_name):
    # 기존 시각화 함수 호출
    img_copy, title, num_detections = visualize_yolo_results(image, results, model_name)
    
    # 탐지된 클래스 카운트
    class_counts = {}
    
    # 클래스 카운트 집계
    for i in range(num_detections):
        cls = int(results[0].boxes[i].cls[0].cpu().numpy())
        cls_name = results[0].names[cls]
        
        if cls_name in class_counts:
            class_counts[cls_name] += 1
        else:
            class_counts[cls_name] = 1
    
    # 탐지된 객체 요약 정보를 이미지에 추가
    if num_detections > 0:
        y_pos = 30
        cv2.putText(img_copy, "Detected Objects:", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2)
        y_pos += 30
        
        for cls_name, count in class_counts.items():
            text = f"- {cls_name}: {count}"
            cv2.putText(img_copy, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)
            y_pos += 25
            
            # 너무 많은 클래스가 있으면 이미지 높이를 벗어날 수 있어 최대 8개까지만 표시
            if (y_pos > image.shape[0] - 30) or (len(class_counts) > 8 and list(class_counts.keys()).index(cls_name) >= 7):
                cv2.putText(img_copy, "... and more", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)
                break
    
    return img_copy, title, num_detections

# 결과 시각화 및 저장
finetuned_img, finetuned_title, finetuned_detections = visualize_yolo_results_with_summary(image_rgb, finetuned_results, "Fine-tuned YOLO")
yolov8m_img, yolov8m_title, yolov8m_detections = visualize_yolo_results_with_summary(image_rgb, yolov8m_oiv7_results, "YOLOv8m-OIV7")
resnet_img, resnet_title, resnet_detections = visualize_resnet_results(image_rgb, resnet_boxes, resnet_scores, resnet_labels, "ResNet50-FPN")

# 결과 저장
cv2.imwrite(os.path.join(output_dir, "finetuned_result.jpg"), cv2.cvtColor(finetuned_img, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "yolov8m_result.jpg"), cv2.cvtColor(yolov8m_img, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "resnet_result.jpg"), cv2.cvtColor(resnet_img, cv2.COLOR_RGB2BGR))

# matplotlib을 사용한 시각화 및 저장
plt.figure(figsize=(18, 6))

# 첫 번째 이미지 (파인튜닝된 모델)
plt.subplot(1, 3, 1)
plt.imshow(finetuned_img)
plt.title(finetuned_title)
plt.axis('off')

# 두 번째 이미지 (YOLOv8m-OIV7 모델)
plt.subplot(1, 3, 2)
plt.imshow(yolov8m_img)
plt.title(yolov8m_title)
plt.axis('off')

# 세 번째 이미지 (ResNet 모델)
plt.subplot(1, 3, 3)
plt.imshow(resnet_img)
plt.title(resnet_title)
plt.axis('off')

# 전체 제목 설정
plt.suptitle("Object Detection Model Performance Comparison", fontsize=16)
plt.tight_layout()

# 결과 저장
plt.savefig(os.path.join(output_dir, "three_model_comparison.png"), dpi=300, bbox_inches='tight')
print(f"Results saved to {output_dir} directory.")

# 탐지 결과 출력
print(f"Fine-tuned YOLO model detections: {finetuned_detections}")
print(f"YOLOv8m-OIV7 model detections: {yolov8m_detections}")
print(f"ResNet50-FPN model detections: {resnet_detections}")

# 모델 성능 비교 테이블 생성
performance_fig = plt.figure(figsize=(10, 5))
ax = performance_fig.add_subplot(111)

# 모델 이름과 탐지 수
models = ["Fine-tuned YOLO", "YOLOv8m-OIV7", "ResNet50-FPN"]
detections = [finetuned_detections, yolov8m_detections, resnet_detections]

# 막대 그래프 생성
bars = ax.bar(models, detections, color=['skyblue', 'lightgreen', 'salmon'])
ax.set_ylabel('Number of Detections')
ax.set_title('Object Detection Model Performance Comparison')

# 각 막대 위에 숫자 표시
for i, (bar, count) in enumerate(zip(bars, detections)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            str(count), ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "three_model_detection_comparison.png"), dpi=300, bbox_inches='tight')

# 추가 성능 비교 - 처리 시간 측정
import time

def measure_inference_time(model, image, n_iterations=10):
    # 워밍업
    if isinstance(model, YOLO):
        _ = model(image)
    else:  # ResNet 모델
        image_tensor = F.to_tensor(image)
        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()
        with torch.no_grad():
            _ = model([image_tensor])
    
    # 시간 측정 시작
    start_time = time.time()
    
    # 여러 번 반복하여 평균 측정
    for _ in range(n_iterations):
        if isinstance(model, YOLO):
            _ = model(image)
        else:  # ResNet 모델
            with torch.no_grad():
                _ = model([image_tensor])
    
    # 시간 측정 종료
    end_time = time.time()
    
    # 평균 추론 시간 계산 (밀리초 단위)
    avg_time = (end_time - start_time) * 1000 / n_iterations
    
    return avg_time

# 각 모델의 추론 시간 측정
print("Measuring inference time for each model...")
finetuned_time = measure_inference_time(finetuned_model, image_rgb)
yolov8m_time = measure_inference_time(yolov8m_oiv7_model, image_rgb)
resnet_time = measure_inference_time(resnet_model, image_rgb)

print(f"Fine-tuned YOLO model average inference time: {finetuned_time:.2f} ms")
print(f"YOLOv8m-OIV7 model average inference time: {yolov8m_time:.2f} ms")
print(f"ResNet50-FPN model average inference time: {resnet_time:.2f} ms")

# 추론 시간 비교 그래프
time_fig = plt.figure(figsize=(10, 5))
ax = time_fig.add_subplot(111)

# 모델 이름과 추론 시간
times = [finetuned_time, yolov8m_time, resnet_time]

# 막대 그래프 생성
bars = ax.bar(models, times, color=['skyblue', 'lightgreen', 'salmon'])
ax.set_ylabel('Average Inference Time (ms)')
ax.set_title('Object Detection Model Inference Time Comparison')

# 각 막대 위에 숫자 표시
for i, (bar, t) in enumerate(zip(bars, times)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
            f"{t:.2f} ms", ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "inference_time_comparison.png"), dpi=300, bbox_inches='tight')

# 종합 성능 평가 - 탐지 수 대비 추론 시간
efficiency_fig = plt.figure(figsize=(12, 6))
ax1 = efficiency_fig.add_subplot(111)

# 두 개의 y축 설정
ax2 = ax1.twinx()

# 첫 번째 축에 탐지 수 표시
bars1 = ax1.bar(np.arange(len(models)) - 0.2, detections, width=0.4, color='skyblue', label='Number of Detections')
ax1.set_ylabel('Number of Detections', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# 두 번째 축에 추론 시간 표시
bars2 = ax2.bar(np.arange(len(models)) + 0.2, times, width=0.4, color='salmon', label='Inference Time (ms)')
ax2.set_ylabel('Inference Time (ms)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# x축 레이블 설정
ax1.set_xticks(np.arange(len(models)))
ax1.set_xticklabels(models)

# 제목 설정
plt.title('Object Detection Model Efficiency Comparison (Detections vs Time)')

# 범례 표시
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "model_efficiency_comparison.png"), dpi=300, bbox_inches='tight')

print("Comparative analysis of the three models completed.")

# ResNet 모델의 특징 및 결과 분석
print("=" * 50)
print("ResNet 모델 분석:")
print(f"- 탐지된 객체 수: {resnet_detections}")
print(f"- 평균 추론 시간: {resnet_time:.2f} ms")

# 탐지된 클래스 분포 확인
if resnet_detections > 0:
    class_counts = {}
    for label in resnet_labels:
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
        if class_name in class_counts:
            class_counts[class_name] += 1
        else:
            class_counts[class_name] = 1
    
    print("- 탐지된 객체 클래스 분포:")
    for cls, count in class_counts.items():
        print(f"  * {cls}: {count}개")

print("=" * 50)