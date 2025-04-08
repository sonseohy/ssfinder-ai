"""
하둡 저장 관련 API 라우트
CLIP 임베딩을 하둡에 저장하고 검색하는 API 엔드포인트 제공
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
from config import HADOOP_HOST, HADOOP_PORT, HADOOP_USER
from models.clip_model_with_hadoop import KoreanCLIPModelWithHadoop

# 라우터 생성
router = APIRouter(
    prefix="/api/hadoop",
    tags=["hadoop"],
    responses={404: {"description": "Not found"}},
)

# 모델 초기화 함수 (의존성 주입용)
clip_hadoop_model = None

def get_clip_hadoop_model():
    """
    하둡 저장 기능이 있는 CLIP 모델 인스턴스를 반환 (싱글톤 패턴)
    """
    global clip_hadoop_model
    if clip_hadoop_model is None:
        try:
            clip_hadoop_model = KoreanCLIPModelWithHadoop(
                hdfs_host=HADOOP_HOST,
                hdfs_port=HADOOP_PORT,
                hdfs_user=HADOOP_USER
            )
        except Exception as e:
            logger.error(f"CLIP 하둡 모델 초기화 실패: {str(e)}")
            # 실패 시 None 반환
    return clip_hadoop_model

# Pydantic 모델 정의
class EmbeddingRequest(BaseModel):
    """임베딩 저장 요청 모델"""
    item_id: str = Field(..., description="항목 ID (저장 및 검색에 사용)")
    text: Optional[str] = Field(None, description="인코딩할 텍스트")
    image_url: Optional[str] = Field(None, description="인코딩할 이미지 URL 또는 경로")
    metadata: Optional[Dict[str, Any]] = Field(None, description="추가 메타데이터")
    
    @validator('item_id')
    def item_id_must_be_valid(cls, v):
        """항목 ID 유효성 검사"""
        if not v or not isinstance(v, str):
            raise ValueError('항목 ID는 필수이며 문자열이어야 합니다')
        # 특수 문자 제거 (안전한 파일명을 위해)
        return ''.join(c if c.isalnum() else '_' for c in v)
    
    @validator('text', 'image_url')
    def at_least_one_field_required(cls, v, values):
        """텍스트 또는 이미지가 하나 이상 필요"""
        if not values.get('text') and not values.get('image_url') and not v:
            raise ValueError('텍스트 또는 이미지 URL 중 하나는 필수입니다')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "item_id": "lost_item_001",
                "text": "검은색 가죽 지갑, 현금과 카드 포함",
                "image_url": "/uploads/2023-04-01/wallet.jpg",
                "metadata": {
                    "category": "지갑",
                    "color": "검정",
                    "location": "서울역",
                    "timestamp": "2023-04-01T12:30:00"
                }
            }
        }

class SearchRequest(BaseModel):
    """임베딩 검색 요청 모델"""
    query_text: Optional[str] = Field(None, description="검색할 텍스트 쿼리")
    query_image_url: Optional[str] = Field(None, description="검색할 이미지 URL 또는 경로")
    threshold: float = Field(0.5, description="유사도 임계값 (0.0 ~ 1.0)")
    limit: int = Field(10, description="반환할 최대 결과 수")
    
    @validator('threshold')
    def threshold_must_be_valid(cls, v):
        """임계값 유효성 검사"""
        if v < 0.0 or v > 1.0:
            raise ValueError('유사도 임계값은 0.0 ~ 1.0 사이여야 합니다')
        return v
    
    @validator('limit')
    def limit_must_be_valid(cls, v):
        """결과 수 제한 유효성 검사"""
        if v < 1 or v > 100:
            raise ValueError('결과 수는 1 ~ 100 사이여야 합니다')
        return v
    
    @validator('query_text', 'query_image_url')
    def at_least_one_query_required(cls, v, values):
        """텍스트 또는 이미지 쿼리가 하나 이상 필요"""
        if not values.get('query_text') and not values.get('query_image_url') and not v:
            raise ValueError('텍스트 쿼리 또는 이미지 URL 중 하나는 필수입니다')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "query_text": "검은색 가죽 지갑을 잃어버렸습니다",
                "query_image_url": None,
                "threshold": 0.5,
                "limit": 10
            }
        }

class DeleteRequest(BaseModel):
    """임베딩 삭제 요청 모델"""
    item_id: str = Field(..., description="삭제할 항목 ID")
    
    class Config:
        schema_extra = {
            "example": {
                "item_id": "lost_item_001"
            }
        }

# API 엔드포인트 정의
@router.post("/save-embedding", response_model=Dict[str, Any])
async def save_embedding(
    request: EmbeddingRequest,
    clip_model=Depends(get_clip_hadoop_model)
):
    """
    텍스트와/또는 이미지를 인코딩하여 임베딩을 하둡에 저장
    """
    if clip_model is None:
        raise HTTPException(status_code=500, detail="CLIP 모델 초기화에 실패했습니다")
    
    try:
        # 디버깅 정보 추가
        logger.info(f"하둡 스토리지 상태: 초기화됨={clip_model.hadoop_storage is not None}")
        if clip_model.hadoop_storage:
            logger.info(f"하둡 연결 정보: 호스트={clip_model.hadoop_storage.host}, 포트={clip_model.hadoop_storage.port}, 사용자={clip_model.hadoop_storage.user}")
        
        # 임베딩 저장
        file_path = clip_model.save_embedding_to_hadoop(
            item_id=request.item_id,
            text=request.text,
            image_source=request.image_url if request.image_url else None,
            metadata=request.metadata
        )
        
        return {
            "success": True,
            "message": "임베딩이 하둡에 저장되었습니다",
            "file_path": file_path,
            "item_id": request.item_id
        }
    
    except Exception as e:
        logger.error(f"임베딩 저장 중 오류 발생: {str(e)}")
        # 자세한 예외 스택 트레이스 출력
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"임베딩 저장 중 오류가 발생했습니다: {str(e)}")
    
@router.post("/search-similar", response_model=Dict[str, Any])
async def search_similar_items(
    request: SearchRequest,
    clip_model=Depends(get_clip_hadoop_model)
):
    """
    하둡에 저장된 임베딩 중 유사한 항목 검색
    
    Args:
        request (SearchRequest): 검색 요청 모델
        clip_model: CLIP-Hadoop 모델 인스턴스 (의존성 주입)
        
    Returns:
        dict: 검색 결과
    """
    if clip_model is None:
        raise HTTPException(status_code=500, detail="CLIP 모델 초기화에 실패했습니다")
    
    try:
        # 유사한 항목 검색
        similar_items = clip_model.search_similar_items_in_hadoop(
            query_text=request.query_text,
            query_image=request.query_image_url if request.query_image_url else None,
            threshold=request.threshold,
            limit=request.limit
        )
        
        return {
            "success": True,
            "message": f"{len(similar_items)}개의 유사한 항목을 찾았습니다",
            "items": similar_items
        }
    
    except Exception as e:
        logger.error(f"유사 항목 검색 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"유사 항목 검색 중 오류가 발생했습니다: {str(e)}")

@router.post("/load-embedding", response_model=Dict[str, Any])
async def load_embedding(
    item_id: str = Query(..., description="로드할 항목 ID"),
    clip_model=Depends(get_clip_hadoop_model)
):
    """
    하둡에서 임베딩 데이터 로드
    
    Args:
        item_id (str): 로드할 항목 ID
        clip_model: CLIP-Hadoop 모델 인스턴스 (의존성 주입)
        
    Returns:
        dict: 로드된 임베딩 데이터
    """
    if clip_model is None:
        raise HTTPException(status_code=500, detail="CLIP 모델 초기화에 실패했습니다")
    
    try:
        # 임베딩 로드
        embedding_data = clip_model.load_embedding_from_hadoop(item_id)
        
        # 임베딩 배열을 리스트로 변환 (JSON 직렬화 가능하도록)
        if "text_embedding" in embedding_data:
            embedding_data["text_embedding"] = embedding_data["text_embedding"].tolist()
        
        if "image_embedding" in embedding_data:
            embedding_data["image_embedding"] = embedding_data["image_embedding"].tolist()
        
        return {
            "success": True,
            "message": f"항목 '{item_id}'의 임베딩을 로드했습니다",
            "data": embedding_data
        }
    
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"항목 '{item_id}'의 임베딩을 찾을 수 없습니다")
    
    except Exception as e:
        logger.error(f"임베딩 로드 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"임베딩 로드 중 오류가 발생했습니다: {str(e)}")

@router.post("/delete-embedding", response_model=Dict[str, bool])
async def delete_embedding(
    request: DeleteRequest,
    clip_model=Depends(get_clip_hadoop_model)
):
    """
    하둡에서 임베딩 데이터 삭제
    
    Args:
        request (DeleteRequest): 삭제 요청 모델
        clip_model: CLIP-Hadoop 모델 인스턴스 (의존성 주입)
        
    Returns:
        dict: 삭제 결과
    """
    if clip_model is None:
        raise HTTPException(status_code=500, detail="CLIP 모델 초기화에 실패했습니다")
    
    try:
        # 임베딩 삭제
        if not hasattr(clip_model.hadoop_storage, 'delete_embedding'):
            raise HTTPException(status_code=500, detail="삭제 기능이 구현되지 않았습니다")
        
        success = clip_model.hadoop_storage.delete_embedding(request.item_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"항목 '{request.item_id}'의 임베딩을 찾을 수 없습니다")
        
        return {
            "success": True,
            "message": f"항목 '{request.item_id}'의 임베딩을 삭제했습니다"
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"임베딩 삭제 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"임베딩 삭제 중 오류가 발생했습니다: {str(e)}")

@router.get("/test")
async def test_hadoop_connection():
    """
    하둡 연결 테스트 엔드포인트
    
    Returns:
        dict: 테스트 결과
    """
    try:
        # 하둡 정보 표시
        info = {
            "hadoop_host": HADOOP_HOST,
            "hadoop_port": HADOOP_PORT,
            "hadoop_user": HADOOP_USER
        }
        
        return {
            "success": True,
            "message": "하둡 연결 정보 확인",
            "hadoop_info": info
        }
    
    except Exception as e:
        logger.error(f"하둡 테스트 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"하둡 테스트 중 오류가 발생했습니다: {str(e)}")