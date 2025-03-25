# 서버 실행 명령:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
import cv2
import torch
from PIL import Image
import io
import os
import sys
from datetime import datetime
import colorsys
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("분실물_AI")

# 시작 로그
logger.info("=== 분실물 AI 서버 시작 ===")

# YOLOv8 로드
try:
    logger.info("YOLOv8 모델 로딩 시작...")
    from ultralytics import YOLO
    
    model_path = "models/yolov8m-oiv7.pt"
    if not os.path.exists(model_path):
        logger.warning(f"모델 파일이 존재하지 않습니다: {model_path}")
        os.makedirs("models", exist_ok=True)
        logger.info("기본 YOLOv8m 모델을 사용합니다.")
        model_path = "yolov8m.pt"
    
    logger.info(f"모델 로드 중: {model_path}")
    yolo_model = YOLO(model_path)
    logger.info("YOLOv8 모델 로딩 완료")
except ImportError as e:
    logger.error(f"YOLOv8 모듈을 가져올 수 없습니다: {e}")
    logger.error("pip install ultralytics 명령으로 설치하세요.")
    yolo_model = None
except Exception as e:
    logger.error(f"YOLOv8 모델 로드 중 오류 발생: {e}")
    yolo_model = None

# 전처리 변환
import torchvision.transforms as transforms
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 물품 및 카테고리 매핑
item_categories = {
    "cellphone": "휴대전화", "phone": "휴대전화", "mobile phone": "휴대전화",
    "wallet": "지갑", "purse": "지갑",
    "backpack": "백팩", "bag": "가방", "handbag": "핸드백",
    "laptop": "노트북", "computer": "노트북", 
    "umbrella": "우산",
    "book": "책",
    "glasses": "안경", "sunglasses": "선글라스"
}

app = FastAPI(title="분실물 AI 분석 서비스")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """서버 동작 확인 기본 엔드포인트"""
    return {"message": "분실물 AI 분석 서비스가 정상 동작 중입니다."}

# 색상 분석 함수
def analyze_colors(img_array):
    """이미지에서 주요 색상을 추출"""
    try:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (100, 100))
        pixels = img_resized.reshape(-1, 3).astype(np.float32)
        
        # K-means 클러스터링으로 주요 색상 추출
        K = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # 각 클러스터의 픽셀 수 계산
        counts = np.bincount(labels.flatten())
        
        # 색상 결과 생성
        colors = []
        for i in range(K):
            r, g, b = centers[i]
            h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
            
            # 색상 이름 결정
            if v < 0.2:
                color_name = "검정"
            elif s < 0.1 and v > 0.8:
                color_name = "흰색"
            elif s < 0.1:
                color_name = "회색"
            elif h < 0.05 or h > 0.95:
                color_name = "빨강"
            elif 0.05 <= h < 0.11:
                color_name = "주황"
            elif 0.11 <= h < 0.17:
                color_name = "노랑"
            elif 0.17 <= h < 0.45:
                color_name = "초록"
            elif 0.45 <= h < 0.65:
                color_name = "파랑"
            elif 0.65 <= h < 0.85:
                color_name = "보라"
            elif 0.85 <= h < 0.95:
                color_name = "분홍"
            else:
                color_name = "기타"
                
            hex_color = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
            percentage = (counts[i] / counts.sum()) * 100
            
            logger.info(f"색상 #{i+1}: {color_name} ({hex_color}), 비율: {percentage:.1f}%")
            
            colors.append({
                "name": color_name,
                "hex": hex_color,
                "percentage": round(percentage, 1)
            })
        
        # 비율 기준으로 정렬
        colors = sorted(colors, key=lambda x: x["percentage"], reverse=True)
        return colors
    except Exception as e:
        logger.error(f"색상 분석 중 오류: {e}")
        return [{"name": "회색", "hex": "#808080", "percentage": 100}]

# 재질 분석 함수
def analyze_material(img_array):
    """이미지의 재질을 분석"""
    try:
        # 질감 특성 기반 간단 분류
        edges = cv2.Canny(img_array, 100, 200)
        edge_density = np.mean(edges) / 255.0
        
        # HSV 변환으로 색상 특성 추출
        hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:,:,1]) / 255.0
        
        # 명도 계산
        brightness = np.mean(img_array) / 255.0
        
        # 특성 기반 간단 분류
        if edge_density > 0.1 and saturation < 0.4:
            return "금속"
        elif edge_density < 0.05 and brightness > 0.7:
            return "플라스틱"
        elif edge_density > 0.08 and 0.3 < saturation < 0.7:
            return "가죽"
        elif 0.05 < edge_density < 0.1 and saturation < 0.3:
            return "천"
        else:
            return "기타"
    except Exception as e:
        logger.error(f"재질 분석 중 오류: {e}")
        return "기타"

def analyze_image(image):
    """이미지 분석 파이프라인"""
    logger.info("이미지 분석 시작")
    results = {}
    
    # 기본값 설정
    results["item_type"] = "알 수 없음"
    results["confidence"] = 0.5
    results["colors"] = [{"name": "회색", "hex": "#808080", "percentage": 100}]
    results["material"] = "알 수 없음"
    results["description"] = "이 물품은 AI가 정확히 분석하지 못했습니다."
    results["auto_tags"] = ["미확인"]
    results["category_id"] = 7  # 기타 카테고리
    
    try:
        # 이미지를 NumPy 배열로 변환
        img_array = np.array(image)
        logger.info(f"이미지 크기: {img_array.shape}")
        
        # 크롭된 객체 이미지 초기화
        detected_object_image = None
        
        # 1. YOLO로 객체 감지
        detected_objects = []
        if yolo_model:
            try:
                yolo_results = yolo_model(img_array)
                
                # 결과 처리
                for result in yolo_results:
                    boxes = result.boxes
                    total_objects = len(boxes)
                    
                    # 로그에 객체 수 명확하게 출력
                    print(f"\n### YOLO 모델이 {total_objects}개의 객체를 감지했습니다 ###")
                    logger.info(f"### YOLO 모델이 {total_objects}개의 객체를 감지했습니다 ###")
                    
                    if total_objects == 0:
                        logger.info("감지된 객체가 없습니다.")
                    
                    for i, box in enumerate(boxes):
                        try:
                            conf = float(box.conf[0].item()) if hasattr(box, 'conf') and len(box.conf) > 0 else 0
                            cls = int(box.cls[0].item()) if hasattr(box, 'cls') and len(box.cls) > 0 else -1
                            
                            if conf > 0.3:  # 신뢰도 임계값
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                class_id = int(cls)
                                class_name = result.names[class_id] if class_id in result.names else "unknown"
                                
                                logger.info(f"감지된 객체 #{i+1}: 클래스={class_name}, 신뢰도={conf:.2f}, 위치=[{x1}, {y1}, {x2}, {y2}]")
                                
                                detected_objects.append({
                                    "class": class_name,
                                    "confidence": conf,
                                    "box": [x1, y1, x2, y2]
                                })
                            else:
                                logger.info(f"낮은 신뢰도로 무시된 객체 #{i+1}: 신뢰도={conf:.2f}")
                        except Exception as e:
                            logger.error(f"객체 #{i+1} 처리 중 오류: {str(e)}")
            except Exception as e:
                logger.error(f"YOLO 추론 중 오류: {str(e)}")
                
        # 객체 감지 결과 처리
        if detected_objects:
            # 가장 높은 신뢰도의 객체 선택
            main_object = max(detected_objects, key=lambda x: x["confidence"])
            results["item_type"] = main_object["class"]
            results["confidence"] = main_object["confidence"]
            logger.info(f"선택된 객체: {main_object['class']}(신뢰도: {main_object['confidence']:.2f})")
            
            # 감지된 객체 부분만 크롭
            x1, y1, x2, y2 = main_object["box"]
            
            # 이미지 범위 검사 및 조정
            h, w = img_array.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # 유효한 크롭 영역인지 확인
            if x1 < x2 and y1 < y2 and (x2-x1)*(y2-y1) > 100:  # 최소 크기 체크
                detected_object_image = img_array[y1:y2, x1:x2]
                logger.info(f"객체 이미지 크롭: 크기 {detected_object_image.shape}")
                print(f"객체 영역 크롭: ({x1},{y1},{x2},{y2}), 크기={detected_object_image.shape}")
            else:
                logger.warning(f"유효하지 않은 객체 영역: [{x1},{y1},{x2},{y2}]")
                detected_object_image = img_array  # 전체 이미지 사용
        
        # 객체가 감지되지 않았거나 크롭에 실패한 경우 전체 이미지 사용
        if detected_object_image is None:
            logger.info("감지된 객체가 없거나 크롭에 실패하여 전체 이미지를 사용합니다.")
            print("감지된 객체가 없어 전체 이미지로 분석합니다.")
            detected_object_image = img_array
        
        # 2. 색상 분석 - 객체 영역만 분석
        colors = analyze_colors(detected_object_image)
        results["colors"] = colors
        
        # 3. 재질 분석 - 객체 영역만 분석
        material = analyze_material(detected_object_image)
        results["material"] = material
        logger.info(f"감지된 재질: {material}")
        
        # 4. 카테고리 매핑
        item_type = results["item_type"].lower()
        korean_item = None
        
        # 한국어 물품명 매핑
        for eng, kor in item_categories.items():
            if eng in item_type:
                korean_item = kor
                break
        
        if korean_item:
            results["item_type"] = korean_item
        
        # 카테고리 ID 매핑
        category_map = {
            "지갑": 1,
            "휴대전화": 2,
            "가방": 3, "백팩": 3, "핸드백": 3,
            "노트북": 4,
            "의류": 5, "옷": 5, "셔츠": 5, "바지": 5,
            "액세서리": 6, "시계": 6, "안경": 6,
        }
        
        category_id = 7  # 기본값: 기타
        for key, value in category_map.items():
            if key in results["item_type"]:
                category_id = value
                break
        
        results["category_id"] = category_id
        
        # 5. 설명 생성
        try:
            item_type = results.get("item_type", "물품")
            main_color = colors[0]["name"] if colors else "알 수 없는 색상"
            
            description = f"이 {item_type}은(는) {main_color} 색상의 {material} 재질로 보입니다."
            results["description"] = description
            logger.info(f"생성된 설명: {description}")
        except Exception as e:
            logger.error(f"설명 생성 중 오류: {str(e)}")
        
        # 6. 자동 태그 생성
        main_color = colors[0]["name"] if colors else "알 수 없는 색상"
        auto_tags = [main_color, material, results["item_type"]]
        auto_tags = [tag for tag in auto_tags if tag not in ["알 수 없음", "unknown"]]
        auto_tags = list(set(auto_tags))  # 중복 제거
        results["auto_tags"] = auto_tags
        
    except Exception as e:
        logger.error(f"이미지 분석 중 오류: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("이미지 분석 완료")
    return results

@app.post("/analyze/")
async def analyze_lost_item(
    file: UploadFile = File(...),
    location: str = Form(None),
    lost_at: str = Form(None)
):
    """분실물 이미지를 분석하는 엔드포인트"""
    logger.info(f"이미지 분석 요청: 파일명={file.filename}, 위치={location}, 분실시간={lost_at}")
    
    # 파일 읽기
    try:
        contents = await file.read()
        logger.info(f"파일 크기: {len(contents)} 바이트")
        
        if not contents:
            logger.error("빈 파일이 업로드되었습니다")
            raise HTTPException(status_code=400, detail="빈 파일이 업로드되었습니다")
        
        # 이미지 처리
        image_bytes = io.BytesIO(contents)
        image = Image.open(image_bytes).convert("RGB")
        logger.info(f"이미지 크기: {image.width}x{image.height}")
        
        # 이미지 분석
        analysis_results = analyze_image(image)
        
        # 메모리 정리
        image_bytes.close()
        image.close()
        
        # 메타데이터 추가
        if location:
            analysis_results["location"] = location
        if lost_at:
            analysis_results["lost_at"] = lost_at
        
        analysis_results["analyzed_at"] = datetime.now().isoformat()
        
        return JSONResponse(content=analysis_results)
        
    except HTTPException as he:
        logger.error(f"HTTP 예외: {he.detail}")
        raise he
    except Exception as e:
        logger.error(f"이미지 분석 오류: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"이미지 분석 오류: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)