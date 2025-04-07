"""
FastAPI 애플리케이션 메인 모듈
"""
import os
import sys
import logging
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from api.routes import matching_router, spark_router, remote_router, embedding_router

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 업로드 디렉토리 생성
os.makedirs("uploads", exist_ok=True)

# 상위 디렉토리 추가하여 모듈 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from api.routes import matching_router

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="습득물 유사도 검색 API",
    description="한국어 CLIP 모델을 사용하여 사용자 게시글과 습득물 간의 유사도를 계산하는 API",
    version="1.0.0",
    # docs_url과 redoc_url의 설정을 제거하거나 기본값으로 설정
    # docs_url="/docs",  # 기본값은 "/docs"이므로 명시적으로 설정하지 않아도 됨
    # redoc_url="/redoc",  # 기본값은 "/redoc"이므로 명시적으로 설정하지 않아도 됨
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
app.include_router(matching_router)
app.include_router(spark_router)
app.include_router(remote_router)
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
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# 커스텀 Swagger UI 관련 코드 제거
# @app.get("/docs", include_in_schema=False) 함수 제거
# @app.get("/openapi.json", include_in_schema=False) 함수 제거

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
        "docs_url": "/docs",
        "api_prefix": "/api"
    }

# 서버 실행 함수
def start_server(host="0.0.0.0", port=5000, reload=True):
    """
    FastAPI 서버 실행
    
    Args:
        host (str): 호스트 주소
        port (int): 포트 번호
        reload (bool): 자동 리로드 활성화 여부
    """
    uvicorn.run("main:app", host=host, port=port, reload=reload)

# 직접 실행 시 서버 시작
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='습득물 유사도 검색 API 서버')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='호스트 주소')
    parser.add_argument('--port', type=int, default=5000, help='포트 번호')
    parser.add_argument('--no-reload', action='store_true', help='자동 리로드 비활성화')
    
    args = parser.parse_args()
    
    logger.info(f"서버를 시작합니다. (host: {args.host}, port: {args.port})")
    start_server(args.host, args.port, not args.no_reload)