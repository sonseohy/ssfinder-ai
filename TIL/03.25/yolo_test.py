import os
import time
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd

# # GPU 설정 (필요에 따라 변경)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 사용할 GPU 번호

# 이미지 경로 설정
image_path = "test_images/blueumb.jpg"  # 테스트할 이미지 경로
output_dir = "model_comparison_results"
os.makedirs(output_dir, exist_ok=True)

# 테스트할 모델 목록
models = {
    "YOLOv8n": "models/yolov8n-oiv7.pt",
    "YOLOv8s": "models/yolov8s-oiv7.pt",
    "YOLOv8m": "models/yolov8m-oiv7.pt",
    "YOLOv8l": "models/yolov8l-oiv7.pt",
    "YOLOv8x": "models/yolov8x-oiv7.pt",
}

# 결과를 저장할 데이터프레임 준비
results_data = []

# 각 모델 테스트
for model_name, model_path in models.items():
    try:
        print(f"\n테스트 중: {model_name} ({model_path})")
        
        # 모델이 존재하는지 확인
        if not os.path.exists(model_path):
            print(f"  - 모델 파일을 찾을 수 없습니다: {model_path}")
            print(f"  - 모델을 자동으로 다운로드합니다...")
        
        # 모델 로드 시간 측정
        load_start = time.time()
        model = YOLO(model_path)
        load_time = time.time() - load_start
        print(f"  - 모델 로드 시간: {load_time:.2f}초")
        
        # 추론 시간 측정
        infer_start = time.time()
        results = model.predict(source=image_path, 
                              save=True,
                              conf=0.25,  # 신뢰도 임계값
                              project=output_dir,
                              name=model_name)
        infer_time = time.time() - infer_start
        print(f"  - 추론 시간: {infer_time:.4f}초")
        
        # 결과 분석
        boxes = results[0].boxes
        detections = []
        
        if len(boxes) > 0:
            print(f"  - 감지된 객체: {len(boxes)}개")
            
            for j, box in enumerate(boxes):
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                
                # 클래스 이름 가져오기
                class_names = model.names
                cls_name = class_names[cls_id] if cls_id in class_names else f"클래스 {cls_id}"
                
                # 바운딩 박스 좌표
                bbox = box.xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2] 형식
                
                print(f"  - 객체 {j+1}: {cls_name}, 신뢰도: {conf:.4f}, 박스: {bbox}")
                detections.append({
                    "class": cls_name,
                    "confidence": conf,
                    "bbox": bbox
                })
        else:
            print("  - 감지된 객체 없음")
        
        # GPU 메모리 사용량 확인
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 3)  # GB 단위
            torch.cuda.reset_peak_memory_stats()
        else:
            gpu_memory = 0
        
        # 결과 저장
        results_data.append({
            "Model": model_name,
            "Load Time (s)": load_time,
            "Inference Time (s)": infer_time,
            "Detected Objects": len(boxes),
            "GPU Memory (GB)": gpu_memory,
            "Detections": detections
        })
        
        # GPU 캐시 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"  - 오류 발생: {e}")
        results_data.append({
            "Model": model_name,
            "Load Time (s)": None,
            "Inference Time (s)": None,
            "Detected Objects": 0,
            "GPU Memory (GB)": 0,
            "Detections": [],
            "Error": str(e)
        })

# 결과 요약
df = pd.DataFrame([{k: v for k, v in d.items() if k != "Detections"} for d in results_data])
print("\n결과 요약:")
print(df)

# CSV로 결과 저장
df.to_csv(os.path.join(output_dir, "model_comparison_results.csv"), index=False)
print(f"\n결과가 {output_dir} 디렉토리에 저장되었습니다.")

# 모델별 탐지 결과 시각화 (선택 사항)
try:
    plt.figure(figsize=(15, 10))
    
    # 원본 이미지 로드
    img = Image.open(image_path)
    
    for idx, model_result_path in enumerate(os.listdir(output_dir)):
        if model_result_path.endswith(".jpg") or model_result_path.endswith(".png"):
            result_img = Image.open(os.path.join(output_dir, model_result_path))
            plt.subplot(3, 2, idx+1)
            plt.imshow(np.array(result_img))
            plt.title(model_result_path.split("/")[-1])
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_models_comparison.jpg"))
    print(f"모델 비교 시각화 이미지가 저장되었습니다: {os.path.join(output_dir, 'all_models_comparison.jpg')}")
except Exception as e:
    print(f"시각화 중 오류 발생: {e}")