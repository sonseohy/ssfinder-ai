"""
API 라우트 패키지 초기화
"""
import logging

# 로깅 설정
logger = logging.getLogger(__name__)
logger.debug("API 라우트 패키지 초기화됨")

# 모든 라우터 노출
try:
    from api.routes.matching_routers import router as matching_router
    
    # 여기에 다른 라우터도 추가 가능
    __all__ = ['matching_router']
    
except ImportError as e:
    logger.warning(f"라우터 임포트 실패: {str(e)}")
    __all__ = []