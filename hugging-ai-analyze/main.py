import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from models.models import LostItemAnalyzer
from typing import Dict, Any
import base64
from PIL import Image
import io

# FastAPI 애플리케이션 생성
app = FastAPI(title="분실물 이미지 분석 API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한하세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 분석기 초기화
analyzer = LostItemAnalyzer()

# API 루트 경로 핸들러
@app.get("/")
async def root():
    return {"message": "분실물 이미지 분석 API가 실행 중입니다."}

# 파일 업로드를 통한 이미지 분석
@app.post("/api/analyze/upload")
async def analyze_image_upload(file: UploadFile = File(...)):
    # 파일 확장자 검증
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in valid_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"지원되지 않는 파일 형식입니다. 지원되는 형식: {', '.join(valid_extensions)}"
        )
    
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            temp_path = temp.name
            content = await file.read()
            temp.write(content)
        
        # 이미지 분석
        result = analyzer.analyze_lost_item(temp_path)
        
        # 임시 파일 삭제
        os.unlink(temp_path)
        
        if result["success"]:
            # 한국어 번역 결과만 반환
            ko_result = {
                "status": "success",
                "data": {
                    "title": result["data"]["translated"]["title"],
                    "category": result["data"]["translated"]["category"],
                    "color": result["data"]["translated"]["color"],
                    "material": result["data"]["translated"]["material"],
                    "brand": result["data"]["translated"]["brand"],
                    "description": result["data"]["translated"]["description"],
                    "distinctive_features": result["data"]["translated"]["distinctive_features"]
                }
            }
            return ko_result
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        # 예외 발생 시 임시 파일 삭제 시도
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"이미지 분석 중 오류 발생: {str(e)}")

# Base64 인코딩된 이미지 분석 (Java에서 사용할 엔드포인트)
# 메모리에서 직접 처리
@app.post("/api/analyze/base64")
async def analyze_image_base64(payload: Dict[str, Any] = Body(...)):
    try:
        if "image" not in payload:
            raise HTTPException(status_code=400, detail="요청에 'image' 필드가 필요합니다")
        
        base64_str = payload["image"]
        
        # Base64 문자열에서 헤더 제거 (있을 경우)
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        
        # Base64 디코딩
        image_data = base64.b64decode(base64_str)
        image = Image.open(io.BytesIO(image_data))
        
        # 메모리 내 이미지를 바이트 스트림으로 저장
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # 임시 파일 경로 동적 생성
        temp_path = f"/tmp/uploads/temp_{os.getpid()}_{id(image)}.jpg"
        
        # 이미지 저장
        with open(temp_path, 'wb') as f:
            f.write(img_byte_arr)
        
        # 이미지 분석
        result = analyzer.analyze_lost_item(temp_path)
        
        # 임시 파일 삭제
        os.unlink(temp_path)
        
        if result["success"]:
            # 한국어 번역 결과만 반환
            ko_result = {
                "status": "success",
                "data": {
                    "title": result["data"]["translated"]["title"],
                    "category": result["data"]["translated"]["category"],
                    "color": result["data"]["translated"]["color"],
                    "material": result["data"]["translated"]["material"],
                    "brand": result["data"]["translated"]["brand"],
                    "description": result["data"]["translated"]["description"],
                    "distinctive_features": result["data"]["translated"]["distinctive_features"]
                }
            }
            return ko_result
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 분석 중 오류 발생: {str(e)}")

# API 상태, 환경변수 확인
@app.get("/api/status")
async def status():
    return {
        "status": "ok",
        "papago_api": "active" if analyzer.translator.use_papago else "inactive",
        "models_loaded": True
    }

# 메인 실행 코드 (로컬 테스트용)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)