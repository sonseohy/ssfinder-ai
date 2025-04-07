"""
MySQL 데이터의 임베딩을 주기적으로 생성하고 하둡에 저장하는 스케줄러
"""
import os
import sys
import time
import logging
import schedule
from datetime import datetime
import argparse

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("embedding_scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 상위 디렉토리 추가하여 모듈 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from embedding_generator import generate_and_save_embeddings

def update_embeddings():
    """임베딩 업데이트 작업 실행"""
    try:
        logger.info("임베딩 업데이트 작업을 시작합니다...")
        
        # 임베딩 생성 및 하둡 저장 실행
        success = generate_and_save_embeddings()
        
        if success:
            logger.info("임베딩 업데이트 작업이 성공적으로 완료되었습니다.")
        else:
            logger.error("임베딩 업데이트 작업이 실패했습니다.")
        
        return success
    
    except Exception as e:
        logger.error(f"임베딩 업데이트 작업 중 오류 발생: {str(e)}")
        return False

def run_scheduler(interval_hours=24, run_immediately=True):
    """
    스케줄러 실행
    
    Args:
        interval_hours (int): 업데이트 주기 (시간)
        run_immediately (bool): 즉시 실행 여부
    """
    logger.info(f"임베딩 업데이트 스케줄러 시작 (주기: {interval_hours}시간)")
    
    # 스케줄 설정
    schedule.every(interval_hours).hours.do(update_embeddings)
    
    # 즉시 실행 옵션
    if run_immediately:
        logger.info("초기 임베딩 업데이트 작업을 실행합니다...")
        update_embeddings()
    
    # 스케줄러 무한 루프
    while True:
        try:
            schedule.run_pending()
            time.sleep(60)  # 1분마다 스케줄 확인
        except KeyboardInterrupt:
            logger.info("스케줄러가 사용자에 의해 중단되었습니다.")
            break
        except Exception as e:
            logger.error(f"스케줄러 실행 중 오류 발생: {str(e)}")
            time.sleep(300)  # 오류 발생 시 5분 대기 후 재시도

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="임베딩 업데이트 스케줄러")
    parser.add_argument('--interval', type=int, default=24, help='업데이트 주기 (시간)')
    parser.add_argument('--no-immediate', action='store_true', help='즉시 실행 비활성화')
    
    args = parser.parse_args()
    
    run_scheduler(interval_hours=args.interval, run_immediately=not args.no_immediate)