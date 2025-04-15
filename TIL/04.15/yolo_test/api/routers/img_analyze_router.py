import os
import tempfile
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query
from typing import Dict, Any, Optional
import time

# API 라우터 생성
router = APIRouter()

# 분석기 가져오기
def get_analyzer():
    from main import analyzer
    if not analyzer:
        raise HTTPException(status_code=500, detail="분석기가 초기화되지 않았습니다.")
    
    # 인스턴스 확인을 위한 로그 추가
    print(f"get_analyzer 호출됨: translator.use_papago={analyzer.translator.use_papago}")

    return analyzer

# API 루트 경로 핸들러
@router.get("/")
async def root():
    return {"message": "분실물 이미지 분석 API가 실행 중입니다."}

# 이미지 분석 후 정보 JSON으로 반환
@router.post("/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    use_yolo_clip: Optional[bool] = Query(True, description="YOLO와 CLIP을 함께 사용하여 분석할지 여부"),
    analyzer = Depends(get_analyzer)
):
    
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
        
        # 분석 시작 시간
        start_time = time.time()
        
        # 이미지 분석
        print(f"이미지 분석 시작: {temp_path}, use_yolo_clip={use_yolo_clip}")
        result = analyzer.analyze_lost_item(temp_path, use_yolo_clip=use_yolo_clip)
        
        # 소요 시간
        analysis_time = time.time() - start_time
        print(f"이미지 분석 완료. 소요 시간: {analysis_time:.2f}초")
        
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

# 비교 분석 API - 두 방식을 모두 실행하고 시간 비교
@router.post("/compare")
async def compare_analysis_methods(
    file: UploadFile = File(...),
    analyzer = Depends(get_analyzer)
):
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
        print(f"YOLO+CLIP 방식 분석 시작: {temp_path}")
        yolo_clip_start = time.time()
        yolo_clip_result = analyzer.analyze_lost_item(temp_path, use_yolo_clip=True)
        yolo_clip_time = time.time() - yolo_clip_start
        
        # 기존 방식 분석
        print(f"기존 방식 분석 시작: {temp_path}")
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
        
        print(f"비교 분석 완료: YOLO+CLIP: {yolo_clip_time:.2f}초, 기존 방식: {blip_only_time:.2f}초")
        return comparison_result
            
    except Exception as e:
        # 예외 발생 시 임시 파일 삭제 시도
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"비교 분석 중 오류 발생: {str(e)}")

# API 상태, 환경변수 확인
@router.get("/status")
async def status(analyzer = Depends(get_analyzer)):
    print(f"상태 확인: analyzer.translator.use_papago={analyzer.translator.use_papago}")
    return {
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