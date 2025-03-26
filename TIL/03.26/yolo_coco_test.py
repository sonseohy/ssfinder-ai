import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO
from matplotlib import font_manager, rc
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

# 경로 설정
image_path = "carkey.jpg"  # 테스트할 이미지 경로
output_dir = "comparison_results"  # 결과 저장 디렉토리

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# 모델 로드
# 파인튜닝된 모델 경로 설정
finetuned_model_path = "best.pt"  # 파인튜닝된 모델 경로 수정 필요
finetuned_model = YOLO(finetuned_model_path)

# YOLOv8m-oiv7 모델 로드
yolov8m_oiv7_model = YOLO("yolov8m-oiv7.pt")

# 이미지 로드
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"이미지를 불러올 수 없습니다: {image_path}")

# BGR을 RGB로 변환 (OpenCV는 BGR, matplotlib는 RGB 사용)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 모델 추론 실행
# 파인튜닝된 모델로 추론
finetuned_results = finetuned_model(image_rgb)
# YOLOv8m-oiv7 모델로 추론
yolov8m_oiv7_results = yolov8m_oiv7_model(image_rgb)

# 결과 시각화 함수
def visualize_results(image, results, model_name):
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
    
    # 제목 추가 (영어로 표시)
    title = f"Model: {model_name}, Detections: {num_detections}"
    
    return img_copy, title, num_detections

# 결과 시각화 및 저장
finetuned_img, finetuned_title, finetuned_detections = visualize_results(image_rgb, finetuned_results, "Fine-tuned YOLO")
yolov8m_img, yolov8m_title, yolov8m_detections = visualize_results(image_rgb, yolov8m_oiv7_results, "YOLOv8m-OIV7")

# 결과 저장
cv2.imwrite(os.path.join(output_dir, "finetuned_result.jpg"), cv2.cvtColor(finetuned_img, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(output_dir, "yolov8m_result.jpg"), cv2.cvtColor(yolov8m_img, cv2.COLOR_RGB2BGR))

# matplotlib을 사용한 시각화 및 저장
plt.figure(figsize=(12, 6))

# 첫 번째 이미지 (파인튜닝된 모델)
plt.subplot(1, 2, 1)
plt.imshow(finetuned_img)
plt.title(finetuned_title)
plt.axis('off')

# 두 번째 이미지 (YOLOv8m-OIV7 모델)
plt.subplot(1, 2, 2)
plt.imshow(yolov8m_img)
plt.title(yolov8m_title)
plt.axis('off')

# 전체 제목 설정
plt.suptitle("YOLO Model Comparison Test", fontsize=16)
plt.tight_layout()

# 결과 저장
plt.savefig(os.path.join(output_dir, "comparison_result.png"), dpi=300, bbox_inches='tight')
print(f"결과가 {output_dir} 디렉토리에 저장되었습니다.")

# 탐지 결과 출력
print(f"Fine-tuned YOLO 모델 탐지 객체 수: {finetuned_detections}")
print(f"YOLOv8m-OIV7 모델 탐지 객체 수: {yolov8m_detections}")

# 모델 성능 비교 테이블 생성
performance_fig = plt.figure(figsize=(8, 4))
ax = performance_fig.add_subplot(111)

# 모델 이름과 탐지 수
models = ["Fine-tuned YOLO", "YOLOv8m-OIV7"]
detections = [finetuned_detections, yolov8m_detections]

# 막대 그래프 생성
ax.bar(models, detections, color=['skyblue', 'lightgreen'])
ax.set_ylabel('Number of Detections')
ax.set_title('Object Detection Performance Comparison')

# 각 막대 위에 숫자 표시
for i, count in enumerate(detections):
    ax.text(i, count + 0.1, str(count), ha='center')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "detection_comparison.png"), dpi=300, bbox_inches='tight')

print("비교 분석이 완료되었습니다.")