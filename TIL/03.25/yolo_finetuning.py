import os
from ultralytics import YOLO
import torch

# GPU 설정
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 분실물로 자주 발견되는 COCO 클래스 ID
# 24: 백팩, 25: 우산, 26: 핸드백, 28: 가방/슈트케이스, 63: 노트북, 67: 휴대폰, 73: 책, 74: 시계
lost_item_classes = [24, 25, 26, 28, 63, 67, 73, 74]

# YOLOv8m 모델 다운로드 및 로드
print("YOLOv8m 모델 다운로드 및 로드 중...")
model_path = "yolov8m-oiv7.pt"

# 모델 파일이 존재하는지 확인하고, 없으면 다운로드
if not os.path.exists(model_path):
    print(f"모델 파일 {model_path}이 없습니다. 다운로드를 시작합니다...")
    # 모델 다운로드 코드
    try:
        # Ultralytics API를 사용하여 모델 다운로드
        model = YOLO("yolov8m-oiv7.pt")  # 이 코드로 자동 다운로드
        print(f"모델 파일 다운로드 완료: {model_path}")
    except Exception as e:
        print(f"모델 다운로드 중 오류 발생: {e}")
        print("수동으로 모델을 다운로드하려면 다음 링크를 사용하세요:")
        print("https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-oiv7.pt")
        exit(1)
else:
    print(f"모델 파일 {model_path}이 이미 존재합니다.")
    model = YOLO(model_path)

# 현재 가용 GPU 메모리 출력
if torch.cuda.is_available():
    print(f"사용 가능한 GPU: {torch.cuda.get_device_name(0)}")
    print(f"현재 GPU 메모리 사용량: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    print(f"최대 GPU 메모리: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")

# 특정 클래스만 학습하도록 설정
training_args = {
    'data': 'coco.yaml',           # 기본 COCO 데이터셋 사용
    'epochs': 100,                 # 학습 에포크 수
    'imgsz': 640,                  # 입력 이미지 크기
    'batch': 16,                   # 배치 크기 (GPU 메모리에 따라 조정)
    'patience': 15,                # 조기종료 참을성
    'device': 2,                   # GPU 디바이스
    'workers': 8,                  # 데이터 로더 워커 수
    'project': 'lost_items',       # 프로젝트 이름
    'name': 'yolov8m_lost_items',  # 실험 이름 (m으로 변경)
    'exist_ok': True,              # 기존 실험 폴더 덮어쓰기
    'pretrained': True,            # 사전 학습된 가중치 사용
    'optimizer': 'Adam',           # 최적화 알고리즘
    'lr0': 0.001,                  # 초기 학습률
    'weight_decay': 0.0005,        # 가중치 감쇠
    'warmup_epochs': 3,            # 워밍업 에포크
    'cos_lr': True,                # 코사인 학습률 스케줄링
    'close_mosaic': 10,            # 모자이크 증강 종료 에포크
    'classes': lost_item_classes,  # 선택된 클래스만 학습
    'cache': 'disk',               # 이미지 캐싱으로 학습 속도 향상
    'amp': True,                   # 혼합 정밀도 학습 활성화
    'fraction': 1.0,               # 데이터셋 사용 비율 (1.0 = 전체)
}

print("분실물 관련 클래스만 선택하여 YOLOv8m 모델 학습을 시작합니다...")
print(f"선택된 클래스 ID: {lost_item_classes}")

# 메모리 문제가 발생하면 배치 크기를 줄이라는 안내
print("참고: GPU 메모리 부족 오류가 발생하면 'batch' 값을 줄여보세요 (예: 16 → 8 → 4)")

# 모델 학습 시작
try:
    results = model.train(**training_args)
    print("학습이 완료되었습니다!")
except Exception as e:
    print(f"학습 중 오류 발생: {e}")
    if "CUDA out of memory" in str(e):
        print("GPU 메모리 부족 오류가 발생했습니다. 배치 크기를 줄이고 다시 시도하세요.")
        training_args['batch'] = training_args['batch'] // 2
        print(f"배치 크기를 {training_args['batch']}로 줄이고 다시 시도합니다...")
        try:
            results = model.train(**training_args)
            print("줄어든 배치 크기로 학습이 완료되었습니다!")
        except Exception as e2:
            print(f"두 번째 시도 중 오류 발생: {e2}")
            print("학습을 중단합니다. 수동으로 배치 크기와 이미지 크기를 조정하세요.")
            exit(1)

# 학습된 모델 로드 (best.pt)
best_model_path = f"lost_items/yolov8m_lost_items/weights/best.pt"
if os.path.exists(best_model_path):
    model = YOLO(best_model_path)
    print(f"학습된 최상의 모델을 로드했습니다: {best_model_path}")
else:
    print(f"경고: 최상의 모델을 찾을 수 없습니다. 마지막으로 학습된 모델을 사용합니다.")
    model = YOLO(f"lost_items/yolov8m_lost_items/weights/last.pt")

# COCO 테스트 데이터셋으로 자동 평가 (별도 다운로드 없이)
print("COCO 검증 데이터셋으로 모델을 평가합니다...")
val_results = model.val(data='coco.yaml', 
                         classes=lost_item_classes,  # 분실물 클래스만 평가
                         batch=16,
                         device=2)
print(f"분실물 클래스에 대한 평가 결과: {val_results.box.map}")

# 몇 가지 예시 이미지에 대해 예측 수행
print("분실물 관련 테스트 이미지에 대해 예측을 수행합니다...")
# 테스트를 위한 예시 이미지 URL (백팩, 노트북, 휴대폰 등이 있는 이미지)
test_sources = [
    "https://images.unsplash.com/photo-1553062407-98eeb64c6a62?q=80&w=1000",  # 백팩
    "https://images.unsplash.com/photo-1496181133206-80ce9b88a853?q=80&w=1000",  # 노트북
    "https://images.unsplash.com/photo-1598327105666-5b89351aff97?q=80&w=1000",  # 핸드백
    "https://images.unsplash.com/photo-1472495010058-65576a9959e4?q=80&w=1000",  # 책
    "wallet1.jpg"  # 로컬 파일 (있는 경우)
]

# 예측 수행
for i, source in enumerate(test_sources):
    try:
        print(f"테스트 이미지 {i+1}/{len(test_sources)} 예측 중...")
        results = model.predict(source=source, 
                               save=True,
                               conf=0.25,
                               classes=lost_item_classes,
                               project='lost_items',
                               name=f'test_prediction_{i+1}')
        
        # 결과 요약
        boxes = results[0].boxes
        if len(boxes) > 0:
            print(f"  - 감지된 객체: {len(boxes)}개")
            for j, box in enumerate(boxes):
                cls_id = int(box.cls.item())
                conf = box.conf.item()
                
                # 클래스 이름 가져오기
                class_names = model.names
                cls_name = class_names[cls_id] if cls_id in class_names else f"클래스 {cls_id}"
                
                print(f"  - 객체 {j+1}: {cls_name}, 신뢰도: {conf:.2f}")
        else:
            print("  - 감지된 객체 없음")
            
    except Exception as e:
        print(f"  - 오류 발생: {e}")

print("\n테스트가 완료되었습니다. 결과는 'lost_items' 디렉토리에서 확인할 수 있습니다.")