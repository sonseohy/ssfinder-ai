"""
습득물 매칭 관련 API 라우트
"""
import os
import sys
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import aiofiles
import uuid
from datetime import datetime

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 추가하여 모듈 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SIMILARITY_THRESHOLD
from utils.simple_db import fetch_found_items
from models.clip_model import KoreanCLIPModel
from utils.similarity import find_similar_items

# 라우터 생성
router = APIRouter(
    prefix="/api/matching",
    tags=["matching"],
    responses={404: {"description": "Not found"}},
)

# 모델 초기화 함수 (의존성 주입용)
clip_model = None

def get_clip_model():
    """
    한국어 CLIP 모델 인스턴스를 반환 (싱글톤 패턴)
    """
    global clip_model
    if clip_model is None:
        try:
            clip_model = KoreanCLIPModel()
        except Exception as e:
            logger.error(f"CLIP 모델 초기화 실패: {str(e)}")
            # 실패 시 None 반환 (텍스트 기반 매칭만 가능)
    return clip_model

# Pydantic 모델 정의
class LostItemPost(BaseModel):
    """사용자가 분실한 물품 게시글 모델"""
    category: str = Field(..., description="분실물 카테고리 (예: 지갑, 가방, 전자기기)")
    item_name: str = Field(..., description="물품명 (예: 검은색 가죽 지갑)")
    color: Optional[str] = Field(None, description="물품 색상")
    content: str = Field(..., description="게시글 내용")
    location: Optional[str] = Field(None, description="분실 장소")
    image_url: Optional[str] = Field(None, description="이미지 URL (있는 경우)")
    
    class Config:
        schema_extra = {
            "example": {
                "category": "지갑",
                "item_name": "검은색 가죽 지갑",
                "color": "검정색",
                "content": "지난주 토요일 강남역 근처에서 검정색 가죽 지갑을 잃어버렸습니다. 현금과 카드가 들어있어요.",
                "location": "강남역",
                "image_url": None
            }
        }

class MatchingResult(BaseModel):
    """매칭 결과 모델"""
    total_matches: int = Field(..., description="매칭된 항목 수")
    similarity_threshold: float = Field(..., description="유사도 임계값")
    matches: List[Dict[str, Any]] = Field(..., description="매칭된 항목 목록")

class MatchingResponse(BaseModel):
    """API 응답 모델"""
    success: bool = Field(..., description="요청 성공 여부")
    message: str = Field(..., description="응답 메시지")
    result: Optional[MatchingResult] = Field(None, description="매칭 결과")

# 이미지 업로드 처리 함수
async def save_upload_file(upload_file: UploadFile) -> str:
    """
    업로드된 파일을 저장하고 파일 경로를 반환
    
    Args:
        upload_file (UploadFile): 업로드된 파일
        
    Returns:
        str: 저장된 파일 경로
    """
    # 업로드 디렉토리 생성
    upload_dir = os.path.join("uploads", datetime.now().strftime("%Y-%m-%d"))
    os.makedirs(upload_dir, exist_ok=True)
    
    # 고유 파일명 생성
    file_extension = os.path.splitext(upload_file.filename)[1] if upload_file.filename else ".jpg"
    file_name = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join(upload_dir, file_name)
    
    # 파일 저장
    async with aiofiles.open(file_path, 'wb') as out_file:
        content = await upload_file.read()
        await out_file.write(content)
    
    return file_path

# API 엔드포인트 정의
@router.post("/find-similar", response_model=MatchingResponse)
async def find_similar_items_api(
    post: LostItemPost,
    threshold: float = Query(SIMILARITY_THRESHOLD, description="유사도 임계값 (0.0 ~ 1.0)"),
    limit: int = Query(10, description="반환할 최대 항목 수"),
    clip_model=Depends(get_clip_model)
):
    """
    사용자 게시글과 유사한 습득물을 찾는 API 엔드포인트
    """
    try:
        # 폼 데이터를 사전 형태로 변환
        user_post = post.dict()
        
        # MySQL에서 습득물 데이터 가져오기
        db_host = "j12c105.p.ssafy.io"
        db_port = 3306
        db_user = "ssafy"
        db_password = "tnatnavkdlsejssafyc!)%"
        db_name = "ssfinder"
        
        lost_items = fetch_found_items(
            host=db_host,
            port=db_port,
            user=db_user,
            password=db_password,
            database=db_name,
            limit=100
        )

        if not lost_items:
            raise HTTPException(status_code=500, detail="습득물 데이터를 가져오지 못했습니다.")
        
        # 유사한 항목 찾기
        similar_items = find_similar_items(user_post, lost_items, threshold, clip_model)
        
        # 결과 제한
        similar_items = similar_items[:limit]
        
        # 응답 구성
        result = MatchingResult(
            total_matches=len(similar_items),
            similarity_threshold=threshold,
            matches=[
                {
                    "item": item["item"],
                    "similarity": round(item["similarity"], 4),
                    "details": {
                        "text_similarity": round(item["details"]["text_similarity"], 4),
                        "image_similarity": round(item["details"]["image_similarity"], 4) if item["details"]["image_similarity"] else None,
                        "category_similarity": round(item["details"]["details"]["category"], 4),
                        "item_name_similarity": round(item["details"]["details"]["item_name"], 4),
                        "color_similarity": round(item["details"]["details"]["color"], 4),
                        "content_similarity": round(item["details"]["details"]["content"], 4)
                    }
                }
                for item in similar_items
            ]
        )
        
        return MatchingResponse(
            success=True,
            message=f"{len(similar_items)}개의 유사한 습득물을 찾았습니다.",
            result=result
        )
    
    except Exception as e:
        logger.error(f"API 호출 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"요청 처리 중 오류가 발생했습니다: {str(e)}")

@router.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    """
    이미지 업로드 API 엔드포인트
    
    Args:
        file (UploadFile): 업로드할 이미지 파일
        
    Returns:
        dict: 업로드 결과와 이미지 URL
    """
    try:
        # 이미지 파일 형식 검증
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")
        
        # 이미지 저장
        file_path = await save_upload_file(file)
        
        # 파일 URL (상대 경로)
        file_url = file_path.replace("\\", "/")
        
        return {
            "success": True,
            "message": "이미지가 업로드되었습니다.",
            "image_url": file_url
        }
    
    except Exception as e:
        logger.error(f"이미지 업로드 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"이미지 업로드 중 오류가 발생했습니다: {str(e)}")

@router.get("/test")
async def test_endpoint():
    """
    API 테스트용 엔드포인트
    
    Returns:
        dict: 테스트 응답
    """
    return {"message": "API가 정상적으로 작동 중입니다."}