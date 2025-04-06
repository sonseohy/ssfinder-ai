"""
유틸리티 패키지 초기화
"""
import logging

# 로깅 설정
logger = logging.getLogger(__name__)
logger.debug("유틸리티 패키지 초기화됨")

# 유틸리티 함수 및 클래스 노출
try:
    from utils.similarity import (
        calculate_text_similarity, 
        calculate_category_similarity,
        calculate_similarity,
        find_similar_items
    )
    
    from utils.spark_processor import MySQLSparkProcessor
    
    __all__ = [
        'calculate_text_similarity', 
        'calculate_category_similarity',
        'calculate_similarity',
        'find_similar_items',
        'MySQLSparkProcessor'
    ]
    
except ImportError as e:
    logger.warning(f"유틸리티 임포트 실패: {str(e)}")
    __all__ = []