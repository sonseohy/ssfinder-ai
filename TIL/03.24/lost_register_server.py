# ai_model_server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
import tensorflow as tf
from ultralytics import YOLO
import numpy as np
import cv2
import io

app = FastAPI(title="분실물 AI 모델 서버")

# 모델 로드
yolo_model = YOLO('models/yolov8x.pt')
efficientnet_model = tf.keras.models.load_model('models/efficientnet.h5')
resnet_model = tf.keras.models.load_model('models/resnet.h5')

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    # 이미지 읽기
    image_bytes = await file.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="이미지를 읽을 수 없습니다")
    
    # YOLO로 물체 감지
    yolo_results = yolo_model(image)
    
    # 물체 추출 및 분석
    # (실제 구현에 맞게 코드 추가)
    
    # 분석 결과
    analysis_result = {
        "item_type": "지갑",
        "category": "액세서리",
        "confidence": 0.95,
        "colors": [
            {"name": "갈색", "hex": "#8B4513", "percentage": 75},
            {"name": "검정", "hex": "#000000", "percentage": 15}
        ],
        "material": "가죽",
        "material_confidence": 0.87,
        "pattern": "단색",
        "pattern_confidence": 0.92,
        "text_detected": [{"text": "GUCCI", "confidence": 0.92}],
        "brand": {"brand": "GUCCI", "confidence": 0.92},
        "description": "이 물건은 갈색 가죽 재질의 GUCCI 브랜드 지갑입니다.",
        "auto_tags": ["지갑", "가죽", "갈색", "GUCCI", "액세서리"]
    }
    
    return {"result": analysis_result}

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "AI 모델 서버"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)