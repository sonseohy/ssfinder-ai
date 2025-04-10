"""
FastAPI 애플리케이션 메인 모듈
"""
import os
import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from api import router as embedding_router
from config import UPLOAD_DIR

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 업로드 디렉토리 생성
os.makedirs(UPLOAD_DIR, exist_ok=True)

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="임베딩 생성 및 유사도 비교 API",
    description="한국어 CLIP 모델을 사용하여 텍스트와 이미지의 임베딩을 생성하고 유사도를 계산하는 API",
    version="1.0.0",
    docs_url=None,  # 기본 /docs 비활성화 (커스텀 설정 사용)
    redoc_url=None,  # 기본 /redoc 비활성화 (커스텀 설정 사용)
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 특정 도메인으로 제한해야 함
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(embedding_router)

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

# 정적 파일 서빙 설정 (업로드된 이미지 등)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# 커스텀 API 문서 설정
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    커스텀 Swagger UI
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - API 문서",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css",
    )

@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    """
    OpenAPI 스키마 반환
    """
    return get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

# 루트 엔드포인트
@app.get("/")
async def root():
    """
    루트 엔드포인트 - API 정보 제공
    """
    return {
        "app_name": "임베딩 생성 및 유사도 비교 API",
        "version": "1.0.0",
        "description": "한국어 CLIP 모델을 사용하여 텍스트와 이미지의 임베딩을 생성하고 유사도를 계산합니다.",
        "docs_url": "/docs",
        "api_prefix": "/api/embedding"
    }

# 서버 상태 체크 엔드포인트
@app.get("/health")
async def health_check():
    """
    서버 상태 체크
    """
    return {"status": "healthy"}

# 직접 실행 시 서버 시작
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)