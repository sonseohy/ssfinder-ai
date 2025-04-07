"""
윈도우 환경에서 EC2 Spark API를 호출하는 FastAPI 라우터
"""
import os
import sys
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import traceback

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 추가하여 모듈 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import SIMILARITY_THRESHOLD
from spark_api_client import match_lost_items_via_api
from api.routes.matching_routers import save_upload_file  # 기존 파일 저장 함수 재사용

# 라우터 생성
router = APIRouter(
    prefix="/api/remote-matching",
    tags=["remote-matching"],
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

class SimilarItem(BaseModel):
    """유사한 분실물 아이템"""
    id: Optional[str] = Field(None, description="아이템 ID")
    category: Optional[str] = Field(None, description="카테고리")
    item_name: Optional[str] = Field(None, description="물품명")
    title: Optional[str] = Field(None, description="물품명 (원격 API용)")
    color: Optional[str] = Field(None, description="색상")
    content: Optional[str] = Field(None, description="내용")
    image_url: Optional[str] = Field(None, description="이미지 URL")
    similarity: Any = Field(None, description="유사도 정보")

class RemoteMatchingResponse(BaseModel):
    """원격 매칭 API 응답 모델"""
    success: bool = Field(..., description="요청 성공 여부")
    message: str = Field(..., description="응답 메시지")
    threshold: Optional[float] = Field(None, description="유사도 임계값")
    total_matches: Optional[int] = Field(None, description="매칭된 항목 수")
    matches: Optional[List[SimilarItem]] = Field(None, description="매칭된 항목 목록")

# API 엔드포인트 정의
@router.post("/find-similar", response_model=RemoteMatchingResponse)
async def find_similar_items_remote_api(
    post: LostItemPost,
    threshold: float = Query(SIMILARITY_THRESHOLD, description="유사도 임계값 (0.0 ~ 1.0)"),
    limit: int = Query(10, description="반환할 최대 항목 수")
):
    """
    EC2 Spark API를 호출하여 사용자 게시글과 유사한 습득물을 찾는 API 엔드포인트
    
    Args:
        post (LostItemPost): 사용자의 분실물 게시글
        threshold (float): 유사도 임계값
        limit (int): 반환할 최대 항목 수
        
    Returns:
        RemoteMatchingResponse: 매칭 결과가 포함된 응답
    """
    try:
        # 폼 데이터를 사전 형태로 변환
        user_post = post.dict()
        logger.info(f"사용자 게시글: {str(user_post)}")
        logger.info(f"임계값: {threshold}, 최대 결과 수: {limit}")
        
        # EC2 Spark API를 호출하여 유사한 분실물 찾기
        result = match_lost_items_via_api(user_post, threshold, limit)
        logger.info(f"API 호출 결과: {str(result)[:200]}...")  # 결과가 너무 길면 잘라서 로깅
        
        # 응답 유효성 확인
        if result is None:
            logger.error("API에서 None 응답을 받았습니다.")
            return RemoteMatchingResponse(
                success=False,
                message="API 서버에서 응답을 받지 못했습니다.",
                matches=[]
            )
        
        # 응답 데이터 처리 및 변환
        matches = result.get("matches", [])
        
        # title -> item_name 변환 (필요한 경우)
        for match in matches:
            if "title" in match and "item_name" not in match:
                match["item_name"] = match["title"]
        
        # 응답 데이터 구성
        response_data = {
            "success": result.get("success", True),
            "message": result.get("message", f"{len(matches)}개의 유사한 분실물을 찾았습니다."),
            "threshold": result.get("threshold", threshold),
            "total_matches": result.get("total_matches", len(matches)),
            "matches": matches
        }
        
        return RemoteMatchingResponse(**response_data)
    
    except Exception as e:
        logger.error(f"API 호출 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())  # 상세 오류 스택 트레이스 출력
        
        # 클라이언트에게 에러 응답 반환
        return RemoteMatchingResponse(
            success=False,
            message=f"요청 처리 중 오류가 발생했습니다: {str(e)}",
            matches=[]
        )

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
        logger.error(traceback.format_exc())  # 상세 오류 스택 트레이스 출력
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"이미지 업로드 중 오류가 발생했습니다: {str(e)}"
            }
        )

@router.get("/test")
async def test_endpoint():
    """
    원격 API 테스트용 엔드포인트
    
    Returns:
        dict: 테스트 응답
    """
    try:
        # API 클라이언트 테스트
        from spark_api_client import SparkAPIClient
        client = SparkAPIClient()
        
        # API 서버 상태 확인
        health = client.health_check()
        
        return {
            "message": "원격 Spark 매칭 API가 정상적으로 작동 중입니다.",
            "api_status": health,
            "endpoints": {
                "find_similar": "/api/remote-matching/find-similar",
                "upload_image": "/api/remote-matching/upload-image"
            }
        }
    except Exception as e:
        logger.error(f"API 테스트 중 오류 발생: {str(e)}")
        return {
            "message": f"API 테스트 중 오류 발생: {str(e)}",
            "status": "error"
        }