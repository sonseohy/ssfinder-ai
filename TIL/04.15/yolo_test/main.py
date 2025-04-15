import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from models.models import LostItemAnalyzer
from typing import Dict, Any, Optional
import base64
from PIL import Image
import io
import logging
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

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
logger.info("분석기 초기화 완료: use_papago=%s", analyzer.translator.use_papago)

# 라우터 임포트 및 등록
try:
    from api.routers.img_analyze_router import router
    app.include_router(router, tags=["이미지 분석"])
    logger.info("이미지 분석 라우터 등록 완료")
except ImportError as e:
    logger.warning(f"이미지 분석 라우터 임포트 실패: {e}")

# API 루트 경로 핸들러
@app.get("/")
async def root():
    logger.info("루트 경로 접속")
    return {"message": "분실물 이미지 분석 API가 실행 중입니다."}

# 파일 업로드를 통한 이미지 분석
@app.post("/api/analyze/upload")
async def analyze_image_upload(
    file: UploadFile = File(...),
    use_yolo_clip: Optional[bool] = Query(True, description="YOLO와 CLIP을 함께 사용하여 분석할지 여부")
):
    logger.info(f"파일 업로드 분석 요청: {file.filename}, use_yolo_clip={use_yolo_clip}")
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
        
        # 이미지 분석 시작 시간
        start_time = time.time()
        
        # 이미지 분석
        logger.info(f"이미지 분석 시작: {temp_path}, use_yolo_clip={use_yolo_clip}")
        result = analyzer.analyze_lost_item(temp_path, use_yolo_clip=use_yolo_clip)
        
        # 분석 소요 시간
        analysis_time = time.time() - start_time
        logger.info(f"이미지 분석 완료. 소요 시간: {analysis_time:.2f}초")
        
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
                },
                "timing": result["data"].get("timing", {}),
                "analysis_method": "yolo_clip" if use_yolo_clip else "blip_only"
            }
            logger.info(f"분석 결과: {ko_result['data']['title']}, 소요 시간: {analysis_time:.2f}초")
            return ko_result
        else:
            logger.error(f"분석 실패: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        # 예외 발생 시 임시 파일 삭제 시도
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass
        
        logger.exception(f"이미지 분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"이미지 분석 중 오류 발생: {str(e)}")

# Base64 인코딩된 이미지 분석 (Java에서 사용할 엔드포인트)
# 메모리에서 직접 처리
@app.post("/api/analyze/base64")
async def analyze_image_base64(
    payload: Dict[str, Any] = Body(...),
    use_yolo_clip: Optional[bool] = Query(True, description="YOLO와 CLIP을 함께 사용하여 분석할지 여부")
):
    logger.info(f"Base64 이미지 분석 요청, use_yolo_clip={use_yolo_clip}")
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
        
        # 임시 디렉토리 확인 및 생성
        temp_dir = "/tmp/uploads"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        
        # 임시 파일 경로 동적 생성
        temp_path = f"{temp_dir}/temp_{os.getpid()}_{id(image)}.jpg"
        
        # 이미지 저장
        with open(temp_path, 'wb') as f:
            f.write(img_byte_arr)
        
        # 이미지 분석 시작 시간
        start_time = time.time()
        
        # 이미지 분석
        logger.info(f"Base64 이미지 분석 시작: {temp_path}, use_yolo_clip={use_yolo_clip}")
        result = analyzer.analyze_lost_item(temp_path, use_yolo_clip=use_yolo_clip)
        
        # 분석 소요 시간
        analysis_time = time.time() - start_time
        logger.info(f"Base64 이미지 분석 완료. 소요 시간: {analysis_time:.2f}초")
        
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
                },
                "timing": result["data"].get("timing", {}),
                "analysis_method": "yolo_clip" if use_yolo_clip else "blip_only"
            }
            logger.info(f"Base64 분석 결과: {ko_result['data']['title']}, 소요 시간: {analysis_time:.2f}초")
            return ko_result
        else:
            logger.error(f"Base64 분석 실패: {result['error']}")
            raise HTTPException(status_code=500, detail=result["error"])
            
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Base64 이미지 분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"이미지 분석 중 오류 발생: {str(e)}")

# 비교 분석 API - 두 방식을 모두 실행하고 시간 비교
@app.post("/api/analyze/compare")
async def compare_analysis_methods(file: UploadFile = File(...)):
    logger.info(f"비교 분석 요청: {file.filename}")
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
        
        # YOLO+CLIP 방식 분석
        logger.info(f"YOLO+CLIP 방식 분석 시작: {temp_path}")
        yolo_clip_start = time.time()
        yolo_clip_result = analyzer.analyze_lost_item(temp_path, use_yolo_clip=True)
        yolo_clip_time = time.time() - yolo_clip_start
        
        # 기존 방식 분석
        logger.info(f"기존 방식 분석 시작: {temp_path}")
        blip_only_start = time.time()
        blip_only_result = analyzer.analyze_lost_item(temp_path, use_yolo_clip=False)
        blip_only_time = time.time() - blip_only_start
        
        # 임시 파일 삭제
        os.unlink(temp_path)
        
        # 결과 비교
        comparison_result = {
            "status": "success",
            "yolo_clip": {
                "time": yolo_clip_time,
                "success": yolo_clip_result["success"],
                "data": yolo_clip_result["data"]["translated"] if yolo_clip_result["success"] else None,
                "timing_details": yolo_clip_result["data"].get("timing", {}) if yolo_clip_result["success"] else None
            },
            "blip_only": {
                "time": blip_only_time,
                "success": blip_only_result["success"],
                "data": blip_only_result["data"]["translated"] if blip_only_result["success"] else None,
                "timing_details": blip_only_result["data"].get("timing", {}) if blip_only_result["success"] else None
            },
            "time_difference": {
                "absolute": abs(yolo_clip_time - blip_only_time),
                "percentage": abs(yolo_clip_time - blip_only_time) / blip_only_time * 100 if blip_only_time > 0 else 0,
                "faster_method": "yolo_clip" if yolo_clip_time < blip_only_time else "blip_only"
            }
        }
        
        logger.info(f"비교 분석 완료: YOLO+CLIP: {yolo_clip_time:.2f}초, 기존 방식: {blip_only_time:.2f}초")
        return comparison_result
            
    except Exception as e:
        # 예외 발생 시 임시 파일 삭제 시도
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass
        
        logger.exception(f"비교 분석 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"비교 분석 중 오류 발생: {str(e)}")

# API 상태, 환경변수 확인
@app.get("/api/status")
async def status():
    status_info = {
        "status": "ok",
        "papago_api": "active" if analyzer.translator.use_papago else "inactive",
        "models_loaded": True,
        "yolo_model": "yolov8m-oiv7",
        "clip_model": "ViT-B/32",
        "blip_models": {
            "caption": analyzer.image_analyzer.caption_model.__class__.__name__,
            "vqa": analyzer.image_analyzer.vqa_model.__class__.__name__
        }
    }
    logger.info(f"API 상태 확인: {status_info}")
    return status_info

# 메인 실행 코드 (로컬 테스트용)
if __name__ == "__main__":
    import uvicorn
    logger.info("서버 시작 중...")
    uvicorn.run(app, host="0.0.0.0", port=5001, log_level="info")