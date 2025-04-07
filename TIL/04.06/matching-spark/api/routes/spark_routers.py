"""
스파크를 사용한 분실물 매칭 API 라우터
"""
import os
import sys
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 추가하여 모듈 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SIMILARITY_THRESHOLD
from hadoop.spark_similarity_matcher import match_lost_items

# 라우터 생성
router = APIRouter(
    prefix="/api/spark-matching",
    tags=["spark-matching"],
    responses={404: {"description": "Not found"}},
)

# Pydantic 모델 정의 (기존 LostItemPost와 동일)
class LostItemPost(BaseModel):
    """사용자가 분실한 물품 게시글 모델"""
    category: str = Field(..., description="분실물 카테고리 (예: 지갑, 가방, 전자기기)")
    item_name: str = Field(..., description="물품명 (예: 검은색 가죽 지갑)")
    color: Optional[str] = Field(None, description="물품 색상")
    content: str = Field(..., description="게시글 내용")
    location: Optional[str] = Field(None, description="분실 장소")
    image_url: Optional[str] = Field(None, description="이미지 URL (있는 경우)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "category": "지갑",
                "item_name": "검은색 가죽 지갑",
                "color": "검정색",
                "content": "지난주 토요일 강남역 근처에서 검정색 가죽 지갑을 잃어버렸습니다. 현금과 카드가 들어있어요.",
                "location": "강남역",
                "image_url": None
            }
        }

class SparkMatchingResponse(BaseModel):
    """스파크 매칭 API 응답 모델"""
    success: bool = Field(..., description="요청 성공 여부")
    message: str = Field(..., description="응답 메시지")
    threshold: Optional[float] = Field(None, description="유사도 임계값")
    total_matches: Optional[int] = Field(None, description="매칭된 항목 수")
    matches: Optional[List[Dict[str, Any]]] = Field(None, description="매칭된 항목 목록")

# API 엔드포인트 정의
@router.post("/find-similar", response_model=SparkMatchingResponse)
async def find_similar_items_spark_api(
    post: LostItemPost,
    threshold: float = Query(SIMILARITY_THRESHOLD, description="유사도 임계값 (0.0 ~ 1.0)"),
    limit: int = Query(10, description="반환할 최대 항목 수")
):
    """
    스파크를 사용하여 사용자 게시글과 유사한 습득물을 찾는 API 엔드포인트
    
    Args:
        post (LostItemPost): 사용자의 분실물 게시글
        threshold (float): 유사도 임계값
        limit (int): 반환할 최대 항목 수
        
    Returns:
        SparkMatchingResponse: 매칭 결과가 포함된 응답
    """
    try:
        # 폼 데이터를 사전 형태로 변환
        user_post = post.dict()
        
        # 스파크를 사용하여 유사한 분실물 찾기
        result = match_lost_items(user_post, threshold, limit)
        
        if not result["success"]:
            return SparkMatchingResponse(
                success=False,
                message=result["message"],
                matches=[]
            )
        
        return SparkMatchingResponse(
            success=True,
            message=result["message"],
            threshold=result["threshold"],
            total_matches=result["total_matches"],
            matches=result["matches"]
        )
    
    except Exception as e:
        logger.error(f"API 호출 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"요청 처리 중 오류가 발생했습니다: {str(e)}")

@router.get("/test")
async def test_endpoint():
    """
    스파크 API 테스트용 엔드포인트
    
    Returns:
        dict: 테스트 응답
    """
    return {"message": "스파크 매칭 API가 정상적으로 작동 중입니다."}