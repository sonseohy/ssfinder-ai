"""
모델 패키지 초기화
"""
import logging

# 로깅 설정
logger = logging.getLogger(__name__)
logger.debug("모델 패키지 초기화됨")

# 모델 클래스 노출
try:
    from models.clip_model import KoreanCLIPModel
    
    __all__ = ['KoreanCLIPModel']
    
except ImportError as e:
    logger.warning(f"모델 임포트 실패: {str(e)}")
    __all__ = []