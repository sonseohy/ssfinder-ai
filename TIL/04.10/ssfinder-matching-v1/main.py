"""
FastAPI 애플리케이션 메인 모듈
"""
import os
import sys
import logging
import tempfile
from fastapi import FastAPI, Request, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional, Union
import json
import base64
from io import BytesIO
from PIL import Image
import time
import traceback

# 캐시 디렉토리 설정 및 최적화
CACHE_DIRS = {
    'TRANSFORMERS_CACHE': '/tmp/transformers_cache',
    'HF_HOME': '/tmp/huggingface_cache',
    'TORCH_HOME': '/tmp/torch_hub_cache',
    'UPLOADS_DIR': '/tmp/uploads'
}

# 환경변수 설정
for key, path in CACHE_DIRS.items():
    os.environ[key] = path
    os.makedirs(path, exist_ok=True)

# 추가 환경변수 최적화
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# 로깅 설정 개선
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/app.log')
    ]
)
logger = logging.getLogger(__name__)

# 모델 클래스 정의 - Spring Boot와 호환되도록 수정
from pydantic import BaseModel, Field

class SpringMatchRequest(BaseModel):
    """Spring Boot에서 보내는 요청 구조에 맞춘 모델"""
    category: Optional[int] = None
    title: Optional[str] = None
    color: Optional[str] = None
    content: Optional[str] = None
    detail: Optional[str] = None  # Spring에서 detail이라는 필드명 사용
    location: Optional[str] = None
    image_url: Optional[str] = None
    threshold: Optional[float] = 0.7

class MatchingResult(BaseModel):
    total_matches: int
    similarity_threshold: float
    matches: List[Dict[str, Any]]

class MatchingResponse(BaseModel):
    success: bool
    message: str
    result: Optional[MatchingResult] = None

# 모델 초기화 (싱글톤으로 로드)
clip_model = None

def get_clip_model(force_reload=False):
    """
    한국어 CLIP 모델 인스턴스를 반환 (싱글톤 패턴)
    
    Args:
        force_reload (bool): 모델 강제 재로딩 여부
    """
    global clip_model
    
    # 모델 로딩 시작 시간 기록
    start_time = time.time()
    
    if clip_model is None or force_reload:
        try:
            # 로깅 및 성능 추적
            logger.info("🔄 CLIP 모델 초기화 시작...")
            
            # 메모리 사용량 기록 (가능한 경우)
            try:
                import psutil
                process = psutil.Process(os.getpid())
                logger.info(f"모델 로드 전 메모리 사용량: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            except ImportError:
                pass
            
            # 모델 로드
            from models.clip_model import KoreanCLIPModel
            clip_model = KoreanCLIPModel()
            
            # 로딩 시간 로깅
            load_time = time.time() - start_time
            logger.info(f"✅ CLIP 모델 로드 완료 (소요시간: {load_time:.2f}초)")
            
            # 메모리 사용량 기록 (가능한 경우)
            try:
                import psutil
                process = psutil.Process(os.getpid())
                logger.info(f"모델 로드 후 메모리 사용량: {process.memory_info().rss / 1024 / 1024:.2f} MB")
            except ImportError:
                pass
            
            return clip_model
        except Exception as e:
            # 상세한 에러 로깅
            logger.error(f"❌ CLIP 모델 초기화 실패: {str(e)}")
            logger.error(f"에러 상세: {traceback.format_exc()}")
            
            # 실패 시 None 반환
            return None
    return clip_model
    
# FastAPI 애플리케이션 생성
app = FastAPI(
    title="습득물 유사도 검색 API",
    description="한국어 CLIP 모델을 사용하여 사용자 게시글과 습득물 간의 유사도를 계산하는 API",
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

# 애플리케이션 시작 이벤트
@app.on_event("startup")
async def startup_event():
    """
    애플리케이션 시작 시 실행되는 이벤트 핸들러
    """
    logger.info("애플리케이션 시작 중...")
    try:
        # 모델 사전 다운로드 (비동기적으로)
        from models.clip_model import preload_clip_model
        preload_clip_model()
        logger.info("모델 사전 다운로드 완료")
    except Exception as e:
        logger.error(f"시작 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())

# 전역 예외 처리
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    전역 예외 처리기
    """
    logger.error(f"요청 처리 중 예외 발생: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": f"서버 오류가 발생했습니다: {str(exc)}"}
    )

# 유틸리티 모듈 임포트
from utils.similarity import calculate_similarity, find_similar_items, CATEGORY_WEIGHT, ITEM_NAME_WEIGHT, COLOR_WEIGHT, CONTENT_WEIGHT

# 내부적으로 습득물 목록을 가져오는 함수
async def fetch_found_items():
    """
    데이터베이스에서 습득물 목록을 가져오는 함수
    실제 구현에서는 DB에서 조회하거나 캐시에서 가져와야 함
    """
    # 예시 데이터 - 실제로는 DB에서 가져와야 함
    sample_found_items = [
        {
            "id": 1,
            "item_category_id": 1,
            "title": "검정 가죽 지갑",
            "color": "검정색",
            "content": "강남역 근처에서 검정색 가죽 지갑을 발견했습니다.",
            "location": "강남역",
            "image": None,
            "category": "지갑"
        },
        {
            "id": 2,
            "item_category_id": 1,
            "title": "갈색 가죽 지갑",
            "color": "갈색",
            "content": "서울대입구역 근처에서 갈색 가죽 지갑을 발견했습니다.",
            "location": "서울대입구역",
            "image": None,
            "category": "지갑"
        }
    ]
    return sample_found_items

# API 엔드포인트 정의 - Spring Boot에 맞게 수정
@app.post("/api/matching/find-similar", response_model=MatchingResponse)
async def find_similar_items_api(
    request: dict,
    threshold: float = Query(0.7, description="유사도 임계값 (0.0 ~ 1.0)"),
    limit: int = Query(10, description="반환할 최대 항목 수")
):
    """
    Spring Boot에서 보내는 요청 구조에 맞춰 사용자 게시글과 유사한 습득물을 찾는 API
    """
    try:
        logger.info(f"유사 습득물 검색 요청: threshold={threshold}, limit={limit}")
        logger.debug(f"요청 데이터: {request}")
        
        # 요청 데이터 변환
        user_post = {}
        
        # 중요: lostItemId 저장
        lostItemId = request.get('lostItemId')
        
        # Spring Boot에서 보내는 필드명 매핑
        if 'category' in request:
            user_post['category'] = request['category']
        elif 'itemCategoryId' in request:
            user_post['category'] = request['itemCategoryId']
            
        # 제목 필드
        if 'title' in request:
            user_post['item_name'] = request['title']
        
        # 색상 필드
        if 'color' in request:
            user_post['color'] = request['color']
        
        # 내용 필드 (Spring Boot에서는 detail로 보냄)
        if 'detail' in request:
            user_post['content'] = request['detail']
        elif 'content' in request:
            user_post['content'] = request['content']
        
        # 위치 필드
        if 'location' in request:
            user_post['location'] = request['location']
        
        # 이미지 URL 필드
        if 'image' in request and request['image']:
            user_post['image_url'] = request['image']
        elif 'image_url' in request and request['image_url']:
            user_post['image_url'] = request['image_url']
            
        # 요청에 들어온 threshold 값이 있으면 사용
        if 'threshold' in request and request['threshold']:
            threshold = float(request['threshold'])
        
        # 데이터베이스에서 습득물 목록 가져오기
        found_items = await fetch_found_items()
        
        logger.info(f"검색할 습득물 수: {len(found_items)}")
        
        # CLIP 모델 로드
        clip_model_instance = get_clip_model()
        
        if clip_model_instance is None:
            return MatchingResponse(
                success=False,
                message="CLIP 모델 로드에 실패했습니다. 잠시 후 다시 시도해주세요.",
                result=None
            )
        
        # 유사한 항목 찾기
        similar_items = find_similar_items(user_post, found_items, threshold, clip_model_instance)
        
        # 유사도 세부 정보 로깅
        logger.info("===== 유사도 세부 정보 =====")
        for idx, item in enumerate(similar_items):
            logger.info(f"항목 {idx+1}: {item['item']['title']}")
            logger.info(f"  최종 유사도: {item['similarity']:.4f}")
            
            details = item['details']
            logger.info(f"  텍스트 유사도: {details['text_similarity']:.4f}")
            if details['image_similarity'] is not None:
                logger.info(f"  이미지 유사도: {details['image_similarity']:.4f}")
            
            category_sim = details['details']['category']
            item_name_sim = details['details']['item_name']
            color_sim = details['details']['color']
            content_sim = details['details']['content']
            
            logger.info(f"  카테고리 유사도: {category_sim:.4f} (가중치: {CATEGORY_WEIGHT:.2f})")
            logger.info(f"  물품명 유사도: {item_name_sim:.4f} (가중치: {ITEM_NAME_WEIGHT:.2f})")
            logger.info(f"  색상 유사도: {color_sim:.4f} (가중치: {COLOR_WEIGHT:.2f})")
            logger.info(f"  내용 유사도: {content_sim:.4f} (가중치: {CONTENT_WEIGHT:.2f})")
        logger.info("==========================")
        
        # 결과 제한
        similar_items = similar_items[:limit]
        
        # Spring Boot 응답 형식에 맞게 결과 구성
        matches = []
        for item in similar_items:
            found_item = item['item']
            
            # 습득물 정보 구성 (추가 필드 포함)
            found_item_info = {
                "id": found_item["id"],
                "user_id": found_item.get("user_id", None),
                "item_category_id": found_item["item_category_id"],
                "title": found_item["title"],
                "color": found_item["color"],
                "lost_at": found_item.get("lost_at", None),
                "location": found_item["location"],
                "detail": found_item["content"],
                "image": found_item.get("image", None),
                "status": found_item.get("status", "ACTIVE"),
                "storedAt": found_item.get("storedAt", None),
                "majorCategory": found_item.get("majorCategory", None),  # 추가: 대분류
                "minorCategory": found_item.get("minorCategory", None),  # 추가: 소분류
                "management_id": found_item.get("management_id", None)    # 추가: 관리 번호
            }
            
            match_item = {
                "lostItemId": lostItemId,            # 요청 받은 lostItemId 사용
                "foundItemId": found_item["id"],
                "item": found_item_info,
                "similarity": round(item["similarity"], 4)
            }
            
            matches.append(match_item)
        
        # 응답 결과 구성 (Camel Case 필드명 사용)
        result = {
            "total_matches": len(matches),
            "similarity_threshold": threshold,
            "matches": matches
        }
        
        response_data = {
            "success": True,
            "message": f"{len(matches)}개의 유사한 습득물을 찾았습니다.",
            "result": result
        }

        # 응답 로깅 (디버깅용)
        logger.info(f"응답 데이터: {response_data}")
        
        return MatchingResponse(**response_data)
    
    except Exception as e:
        logger.error(f"API 호출 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 스택 트레이스 반환 (개발용)
        error_response = {
            "success": False,
            "message": f"요청 처리 중 오류가 발생했습니다: {str(e)}",
            "error_detail": traceback.format_exc()
        }
        
        return JSONResponse(status_code=500, content=error_response)

@app.get("/api/matching/test")
async def test_endpoint():
    """
    API 테스트용 엔드포인트
    
    Returns:
        dict: 테스트 응답
    """
    return {"message": "API가 정상적으로 작동 중입니다."}

@app.get("/api/status")
async def status():
    """
    API 상태 엔드포인트
    
    Returns:
        dict: API 상태 정보
    """
    # CLIP 모델 로드 시도
    model = get_clip_model()
    
    return {
        "status": "ok",
        "models_loaded": model is not None,
        "version": "1.0.0"
    }

# 루트 엔드포인트
@app.get("/")
async def root():
    """
    루트 엔드포인트 - API 정보 제공
    """
    return {
        "app_name": "습득물 유사도 검색 API",
        "version": "1.0.0",
        "description": "한국어 CLIP 모델을 사용하여 사용자 게시글과 습득물 간의 유사도를 계산합니다.",
        "api_endpoint": "/api/matching/find-similar",
        "test_endpoint": "/api/matching/test",
        "status_endpoint": "/api/status"
    }

# 애플리케이션 실행
if __name__ == "__main__":
    import uvicorn
    print("서버 실행 시도 중...")
    try:
        uvicorn.run(
            "main:app", 
            host="0.0.0.0", 
            port=7860,  # 허깅페이스 스페이스에서 사용할 기본 포트
            log_level="info",
            reload=True
        )
    except Exception as e:
        print(f"서버 실행 중 오류 발생: {e}")
        traceback.print_exc()