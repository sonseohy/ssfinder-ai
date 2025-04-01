from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from dotenv import load_dotenv

from app.routers import matching_router

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="분실물-습득물 매칭 API",
    description="CLIP과 BLIP 모델을 활용한 분실물-습득물 매칭 시스템 API",
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

# 라우터 포함
app.include_router(matching_router.router)

@app.get("/")
async def root():
    """API 기본 정보 반환"""
    return {"message": "분실물-습득물 매칭 API", "version": "1.0.0"}

@app.get("/api/health")
async def health_check():
    """서비스 헬스 체크"""
    return {"status": "healthy", "service": "lost-found-matching-api"}