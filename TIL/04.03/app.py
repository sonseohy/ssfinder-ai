"""
Spark와 FastAPI 통합 서비스
경량화된 CLIP 모델과 Spark를 연동하여 대규모 게시글 유사도 계산 처리
"""
import os
import sys
import logging
import time
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, Depends, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 추가하여 모듈 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SIMILARITY_THRESHOLD
from models.optimized_clip_model import OptimizedKoreanCLIPModel
from spark.similarity_processor import process_similarity
from db.mysql_connector import fetch_posts_from_db, save_post_to_db

# Pydantic 모델 정의
class PostData(BaseModel):
    """게시글 데이터 모델"""
    category: Optional[str] = Field(None, description="분실물 카테고리")
    item_name: Optional[str] = Field(None, description="물품명")
    color: Optional[str] = Field(None, description="물품 색상")
    content: str = Field(..., description="게시글 내용")
    location: Optional[str] = Field(None, description="위치 정보")
    image_url: Optional[str] = Field(None, description="이미지 URL")
    
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

class BatchSimilarityRequest(BaseModel):
    """배치 유사도 계산 요청 모델"""
    new_post: PostData
    candidates: List[Dict[str, Any]]

class SimilarityResponse(BaseModel):
    """유사도 계산 응답 모델"""
    success: bool = Field(..., description="요청 성공 여부")
    message: str = Field(..., description="응답 메시지")
    results: Optional[List[Dict[str, Any]]] = Field(None, description="유사도 계산 결과")

# 전역 모델 인스턴스
clip_model = None

def get_clip_model():
    """
    경량화된 CLIP 모델 인스턴스 가져오기 (싱글톤 패턴)
    """
    global clip_model
    if clip_model is None:
        try:
            clip_model = OptimizedKoreanCLIPModel()
        except Exception as e:
            logger.error(f"CLIP 모델 초기화 오류: {str(e)}")
            # 모델 초기화 실패 시 None 반환
    return clip_model

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="Spark 기반 게시글 유사도 계산 API",
    description="경량화된 CLIP 모델과 Spark를 이용한 대규모 게시글 유사도 계산 서비스",
    version="1.0.0"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """
    애플리케이션 시작 시 모델 초기화
    """
    global clip_model
    logger.info("CLIP 모델 초기화 중...")
    clip_model = get_clip_model()
    if clip_model:
        logger.info("CLIP 모델 초기화 완료")
    else:
        logger.warning("CLIP 모델 초기화 실패")

@app.get("/")
async def root():
    """
    루트 엔드포인트
    """
    return {
        "message": "Spark 기반 게시글 유사도 계산 API가 실행 중입니다.",
        "endpoints": {
            "find_similar": "/api/find-similar",
            "spark_similarity": "/api/spark/calculate-similarity",
            "batch_similarity": "/api/spark/batch-similarity"
        }
    }

@app.post("/api/find-similar", response_model=SimilarityResponse)
async def find_similar_posts(
    post_data: PostData,
    threshold: float = Query(SIMILARITY_THRESHOLD, description="유사도 임계값 (0.0 ~ 1.0)"),
    limit: int = Query(10, description="반환할 최대 항목 수")
):
    """
    Spark 기반 유사도 계산으로 비슷한 게시글 찾기
    """
    try:
        logger.info(f"새 게시글 유사도 계산 요청 수신: {post_data.item_name}")
        
        # 입력 데이터를 dict로 변환
        post_dict = post_data.dict()
        
        # Spark 처리 호출
        similar_posts = process_similarity(
            new_post=post_dict,
            threshold=threshold,
            limit=limit
        )
        
        # 응답 구성
        return SimilarityResponse(
            success=True,
            message=f"{len(similar_posts)}개의 유사한 게시글을 찾았습니다.",
            results=similar_posts
        )
        
    except Exception as e:
        logger.error(f"유사도 계산 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=f"처리 중 오류가 발생했습니다: {str(e)}")

@app.post("/api/spark/calculate-similarity")
async def calculate_similarity_api(
    post_data: Dict[str, Any] = Body(...),
    clip_model: OptimizedKoreanCLIPModel = Depends(get_clip_model)
):
    """
    Spark UDF에서 호출하는 CLIP 모델 기반 유사도 계산 API
    """
    try:
        start_time = time.time()
        
        if not clip_model:
            return JSONResponse(
                status_code=200,
                content={"similarity_scores": {}}
            )
        
        # 유사도 계산에 필요한 데이터 추출
        new_content = post_data.get('new_content', '')
        new_image_url = post_data.get('new_image_url', '')
        post_content = post_data.get('post_content', '')
        post_image_url = post_data.get('post_image_url', '')
        
        # 유사도 점수 초기화
        similarity_scores = {}
        
        # 텍스트 인코딩
        if new_content and post_content:
            new_text_embedding = clip_model.encode_text(new_content)
            post_text_embedding = clip_model.encode_text(post_content)
            text_similarity = clip_model.calculate_similarity(new_text_embedding, post_text_embedding)
            similarity_scores['text_similarity'] = float(text_similarity)
        
        # 이미지가 있는 경우
        if new_image_url and post_image_url:
            # 이미지 인코딩
            new_image_embedding = clip_model.encode_image(new_image_url)
            post_image_embedding = clip_model.encode_image(post_image_url)
            
            # 이미지-이미지 유사도
            image_similarity = clip_model.calculate_similarity(new_image_embedding, post_image_embedding)
            similarity_scores['image_similarity'] = float(image_similarity)
            
            # 교차 유사도 (텍스트-이미지, 이미지-텍스트)
            if new_content and post_image_url:
                text_to_image = clip_model.calculate_similarity(new_text_embedding, post_image_embedding)
                similarity_scores['text_to_image'] = float(text_to_image)
                
            if post_content and new_image_url:
                image_to_text = clip_model.calculate_similarity(post_text_embedding, new_image_embedding)
                similarity_scores['image_to_text'] = float(image_to_text)
            
            # 모든 유사도 점수의 평균
            scores = list(similarity_scores.values())
            similarity_scores['final_similarity'] = sum(scores) / len(scores)
        else:
            # 이미지가 없는 경우 텍스트 유사도만 사용
            if 'text_similarity' in similarity_scores:
                similarity_scores['final_similarity'] = similarity_scores['text_similarity']
            else:
                similarity_scores['final_similarity'] = 0.0
        
        process_time = time.time() - start_time
        logger.debug(f"유사도 계산 완료: {similarity_scores}, 처리 시간: {process_time:.4f}초")
        
        return {"similarity_scores": similarity_scores}
    
    except Exception as e:
        logger.error(f"CLIP 유사도 계산 중 오류 발생: {str(e)}")
        return JSONResponse(
            status_code=200,  # 오류가 있어도 200 반환 (Spark UDF 호출 시 오류 처리 용이성)
            content={"similarity_scores": {"error": str(e), "final_similarity": 0.0}}
        )

@app.post("/api/spark/batch-similarity")
async def batch_similarity_api(
    request: BatchSimilarityRequest,
    clip_model: OptimizedKoreanCLIPModel = Depends(get_clip_model)
):
    """
    여러 게시글에 대한 배치 유사도 계산 API
    """
    try:
        if not clip_model:
            return {"success": False, "message": "CLIP 모델 초기화 실패", "results": []}
        
        new_post = request.new_post.dict()
        candidates = request.candidates
        
        # 빈 배치인 경우
        if not candidates:
            return {"success": True, "message": "처리할 항목이 없습니다.", "results": []}
        
        start_time = time.time()
        logger.info(f"{len(candidates)}개 게시글에 대한 배치 유사도 계산 시작")
        
        # 새 게시글 임베딩 계산 (한 번만 수행)
        new_text_embedding = None
        new_image_embedding = None
        
        if new_post.get('content'):
            new_text_embedding = clip_model.encode_text(new_post['content'])
            
        if new_post.get('image_url'):
            new_image_embedding = clip_model.encode_image(new_post['image_url'])
        
        # 배치 처리 결과
        results = []
        
        for candidate in candidates:
            post_id = candidate.get('id')
            post_content = candidate.get('content', '')
            post_image_url = candidate.get('image_url', '')
            text_similarity = candidate.get('text_similarity', 0)
            
            result = {
                'id': post_id,
                'text_similarity': text_similarity,
                'final_similarity': text_similarity  # 기본값은 텍스트 유사도
            }
            
            try:
                clip_scores = {}
                
                # 텍스트-텍스트 유사도 계산
                if new_text_embedding is not None and post_content:
                    post_text_embedding = clip_model.encode_text(post_content)
                    text_clip_similarity = float(clip_model.calculate_similarity(new_text_embedding, post_text_embedding))
                    clip_scores['text_clip'] = text_clip_similarity
                
                # 이미지 유사도 계산 (양쪽 모두 이미지가 있는 경우)
                if new_image_embedding is not None and post_image_url:
                    post_image_embedding = clip_model.encode_image(post_image_url)
                    image_similarity = float(clip_model.calculate_similarity(new_image_embedding, post_image_embedding))
                    clip_scores['image'] = image_similarity
                    
                    # 텍스트-이미지 교차 유사도
                    if new_text_embedding is not None:
                        text_to_image = float(clip_model.calculate_similarity(new_text_embedding, post_image_embedding))
                        clip_scores['text_to_image'] = text_to_image
                    
                    if post_content:
                        post_text_embedding = clip_model.encode_text(post_content)
                        image_to_text = float(clip_model.calculate_similarity(post_text_embedding, new_image_embedding))
                        clip_scores['image_to_text'] = image_to_text
                
                # 최종 CLIP 유사도 계산
                if clip_scores:
                    clip_similarity = sum(clip_scores.values()) / len(clip_scores)
                    
                    # 텍스트 유사도와 CLIP 유사도를 결합
                    final_similarity = 0.7 * text_similarity + 0.3 * clip_similarity
                    
                    result['clip_similarity'] = clip_similarity
                    result['clip_scores'] = clip_scores
                    result['final_similarity'] = final_similarity
            
            except Exception as e:
                logger.warning(f"게시글 ID {post_id} 처리 중 오류: {str(e)}")
                # 오류 시 텍스트 유사도만 사용
            
            results.append(result)
        
        process_time = time.time() - start_time
        logger.info(f"배치 유사도 계산 완료: {len(results)}개 결과, 처리 시간: {process_time:.2f}초")
        
        return {
            "success": True, 
            "message": f"{len(results)}개 게시글 처리 완료", 
            "results": results
        }
        
    except Exception as e:
        logger.error(f"배치 처리 중 오류 발생: {str(e)}")
        return {
            "success": False, 
            "message": f"배치 처리 중 오류: {str(e)}", 
            "results": []
        }

# 서버 실행 함수
def start_server(host="0.0.0.0", port=8000, reload=True):
    """
    FastAPI 서버 시작 함수
    """
    import uvicorn
    uvicorn.run("app:app", host=host, port=port, reload=reload)

# 직접 실행 시 서버 시작
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Spark-FastAPI 통합 서비스')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='서버 호스트')
    parser.add_argument('--port', type=int, default=8000, help='서버 포트')
    parser.add_argument('--no-reload', action='store_true', help='자동 리로드 비활성화')
    
    args = parser.parse_args()
    
    logger.info(f"서버 시작: {args.host}:{args.port}")
    start_server(args.host, args.port, not args.no_reload)