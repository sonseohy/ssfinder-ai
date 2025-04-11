"""
성능 측정 및 모니터링 유틸리티 모듈
"""
import time
import logging
import functools
import traceback
import psutil
import os
import platform
import gc
from typing import Callable, Any

# 로깅 설정
logger = logging.getLogger(__name__)

def performance_logger(func):
    """
    함수 실행 시간 및 메모리 사용량을 로깅하는 데코레이터
    
    Args:
        func (callable): 측정할 함수
        
    Returns:
        callable: 래핑된 함수
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # 시작 시간 및 메모리 사용량 측정
        start_time = time.time()
        start_memory = 0
        
        try:
            # 메모리 사용량 측정 (가능한 경우)
            if psutil:
                process = psutil.Process(os.getpid())
                start_memory = process.memory_info().rss / 1024 / 1024  # MB 단위
        except Exception:
            pass
        
        # 함수 호출
        try:
            result = await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"함수 {func.__name__} 실행 중 오류 발생: {str(e)}")
            logger.error(traceback.format_exc())
            raise
        
        # 종료 시간 및 메모리 사용량 측정
        end_time = time.time()
        end_memory = 0
        
        try:
            if psutil:
                process = psutil.Process(os.getpid())
                end_memory = process.memory_info().rss / 1024 / 1024  # MB 단위
        except Exception:
            pass
        
        # 실행 정보 로깅
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory if end_memory > 0 else 0
        
        logger.info(f"함수 {func.__name__} 실행 시간: {execution_time:.4f}초")
        
        if end_memory > 0:
            logger.info(f"함수 {func.__name__} 메모리 사용량 변화: {memory_usage:.2f} MB (시작: {start_memory:.2f} MB, 종료: {end_memory:.2f} MB)")
        
        return result
    
    return wrapper

def log_system_info():
    """시스템 정보 로깅"""
    try:
        logger.info("===== 시스템 정보 =====")
        logger.info(f"OS: {platform.system()} {platform.release()}")
        logger.info(f"Python 버전: {platform.python_version()}")
        
        if psutil:
            # CPU 정보
            cpu_count = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)
            logger.info(f"CPU: {cpu_count} 코어 ({cpu_count_logical} 논리 프로세서)")
            
            # 메모리 정보
            mem = psutil.virtual_memory()
            total_mem = mem.total / (1024 * 1024 * 1024)  # GB 단위
            available_mem = mem.available / (1024 * 1024 * 1024)  # GB 단위
            logger.info(f"메모리: 총 {total_mem:.2f} GB (사용 가능: {available_mem:.2f} GB)")
            
            # 디스크 정보
            disk = psutil.disk_usage('/')
            total_disk = disk.total / (1024 * 1024 * 1024)  # GB 단위
            free_disk = disk.free / (1024 * 1024 * 1024)  # GB 단위
            logger.info(f"디스크: 총 {total_disk:.2f} GB (여유 공간: {free_disk:.2f} GB)")
        
        logger.info("=======================")
    except Exception as e:
        logger.error(f"시스템 정보 로깅 중 오류 발생: {str(e)}")

def cleanup_memory():
    """
    메모리 정리 함수 - 긴 작업 후 메모리 정리에 사용
    """
    try:
        # 가비지 컬렉션 강제 실행
        collected = gc.collect()
        logger.debug(f"가비지 컬렉션 실행: {collected}개 객체 수집")
        
        # 메모리 사용량 로깅 (가능한 경우)
        if psutil:
            process = psutil.Process(os.getpid())
            memory_usage = process.memory_info().rss / 1024 / 1024  # MB 단위
            logger.debug(f"현재 메모리 사용량: {memory_usage:.2f} MB")
    except Exception as e:
        logger.error(f"메모리 정리 중 오류 발생: {str(e)}")