"""
게시글 임베딩 생성 관련 API 라우트
"""
import os
import sys
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import aiofiles
import uuid
from datetime import datetime
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 추가하여 모듈 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.clip_model import KoreanCLIPModel
from utils.embedding import generate_post_embedding

# 라우터 생성
router = APIRouter(
    prefix="/api/embedding",
    tags=["embedding"],
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
class Post(BaseModel):
    """게시글 모델"""
    category: str = Field(..., description="게시글 카테고리")
    title: str = Field(..., description="게시글 제목")
    content: str = Field(..., description="게시글 내용")
    image_url: Optional[str] = Field(None, description="이미지 URL (있는 경우)")
    
    class Config:
        schema_extra = {
            "example": {
                "category": "질문",
                "title": "CLIP 모델에 대해 질문합니다",
                "content": "CLIP 모델은 어떤 방식으로 텍스트와 이미지를 같은 임베딩 공간에 매핑하나요?",
                "image_url": None
            }
        }

class EmbeddingResponse(BaseModel):
    """API 응답 모델"""
    success: bool = Field(..., description="요청 성공 여부")
    message: str = Field(..., description="응답 메시지")
    embedding: Optional[Dict[str, Any]] = Field(None, description="생성된 임베딩 정보")
    
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
@router.post("/generate", response_model=EmbeddingResponse)
async def generate_embedding(
    post: Post,
    clip_model=Depends(get_clip_model)
):
    """
    게시글 임베딩 생성 API 엔드포인트
    
    Args:
        post (Post): 임베딩을 생성할 게시글 정보
        clip_model: CLIP 모델 인스턴스 (의존성 주입)
        
    Returns:
        EmbeddingResponse: 생성된 임베딩 정보가 포함된 응답
    """
    try:
        # 게시글 정보를 사전 형태로 변환
        post_data = post.dict()
        
        # 임베딩 생성
        embedding_result = generate_post_embedding(post_data, clip_model)
        
        # numpy 배열을 리스트로 변환 (JSON 직렬화 가능하도록)
        for key, value in embedding_result["embeddings"].items():
            if isinstance(value, np.ndarray):
                embedding_result["embeddings"][key] = value.tolist()
        
        return EmbeddingResponse(
            success=True,
            message="게시글 임베딩이 성공적으로 생성되었습니다.",
            embedding=embedding_result
        )
    
    except Exception as e:
        logger.error(f"임베딩 생성 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"임베딩 생성 중 오류가 발생했습니다: {str(e)}")

@router.post("/generate-with-image", response_model=EmbeddingResponse)
async def generate_embedding_with_image(
    category: str = Form(...),
    title: str = Form(...),
    content: str = Form(...),
    image: Optional[UploadFile] = File(None),
    clip_model=Depends(get_clip_model)
):
    """
    이미지가 포함된 게시글 임베딩 생성 API 엔드포인트
    
    Args:
        category (str): 게시글 카테고리
        title (str): 게시글 제목
        content (str): 게시글 내용
        image (UploadFile, optional): 업로드된 이미지 파일
        clip_model: CLIP 모델 인스턴스 (의존성 주입)
        
    Returns:
        EmbeddingResponse: 생성된 임베딩 정보가 포함된 응답
    """
    try:
        # 게시글 정보 구성
        post_data = {
            "category": category,
            "title": title,
            "content": content,
            "image_url": None
        }
        
        # 이미지가 업로드된 경우 저장
        if image:
            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")
            
            file_path = await save_upload_file(image)
            post_data["image_url"] = file_path
        
        # 임베딩 생성
        embedding_result = generate_post_embedding(post_data, clip_model)
        
        # numpy 배열을 리스트로 변환 (JSON 직렬화 가능하도록)
        for key, value in embedding_result["embeddings"].items():
            if isinstance(value, np.ndarray):
                embedding_result["embeddings"][key] = value.tolist()
        
        return EmbeddingResponse(
            success=True,
            message="게시글 임베딩이 성공적으로 생성되었습니다.",
            embedding=embedding_result
        )
    
    except Exception as e:
        logger.error(f"임베딩 생성 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"임베딩 생성 중 오류가 발생했습니다: {str(e)}")

@router.get("/test")
async def test_endpoint():
    """
    API 테스트용 엔드포인트
    
    Returns:
        dict: 테스트 응답
    """
    return {"message": "임베딩 생성 API가 정상적으로 작동 중입니다."}