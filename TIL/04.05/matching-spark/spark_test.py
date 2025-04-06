"""
MySQL + Spark + CLIP 테스트 스크립트 (SparkRayTaskProcessor 없이 직접 구현)
"""
import os
import sys
import logging
import time
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 현재 디렉토리 기준으로 프로젝트 루트 설정 (유연하게 처리)
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(current_dir) == 'scripts':
    # scripts 디렉토리에 있는 경우
    project_root = os.path.dirname(current_dir)
else:
    # 프로젝트 루트에 있는 경우
    project_root = current_dir

sys.path.append(project_root)

# 환경 변수 로드
load_dotenv(os.path.join(project_root, '.env'))

# 모듈 임포트 (수정된 파일 사용)
try:
    # utils/spark_processor_fixed.py 사용
    from utils.spark_processor import MySQLSparkProcessor
    from config import (
        MYSQL_URL, MYSQL_USER, MYSQL_PASSWORD, 
        SPARK_MASTER, SPARK_APP_NAME, BATCH_SIZE
    )
except ImportError as e:
    logger.error(f"모듈 임포트 오류: {str(e)}")
    sys.exit(1)

def test_mysql_spark_clip():
    """
    MySQL + Spark + CLIP 테스트
    """
    logger.info("=== MySQL + Spark + CLIP 테스트 시작 ===")
    
    # 시작 시간 기록
    start_time = time.time()
    
    # MySQL-Spark 프로세서 초기화
    processor = MySQLSparkProcessor(
        app_name=SPARK_APP_NAME,
        master=SPARK_MASTER,
        mysql_url=MYSQL_URL,
        mysql_user=MYSQL_USER,
        mysql_password=MYSQL_PASSWORD
    )
    
    # 프로세서 시작
    if not processor.start():
        logger.error("프로세서 시작 실패")
        return
    
    try:
        # MySQL 테이블 이름과 조건 설정 (실제 환경에 맞게 수정)
        table_name = "posts"  # 처리할 테이블 이름
        where_clause = "1=1 LIMIT 100"  # 테스트용으로 100개만 가져옴
        
        logger.info(f"MySQL 테이블 처리 시작: {table_name}")
        
        # MySQL 테이블 처리
        result_df = processor.process_mysql_table(
            table_name=table_name,
            where_clause=where_clause,
            batch_size=BATCH_SIZE
        )
        
        # 결과 확인
        logger.info(f"처리된 행 수: {result_df.count()}")
        logger.info(f"결과 스키마: {result_df.schema}")
        
        # 샘플 데이터 출력
        logger.info("샘플 데이터 (처음 5행):")
        result_df.select("id", "content").show(5, truncate=True)
        
        # 임베딩 디버깅
        if "text_embedding" in result_df.columns:
            # 첫 번째 임베딩의 차원 확인
            first_embedding = result_df.select("text_embedding").first()[0]
            logger.info(f"임베딩 벡터 크기: {len(first_embedding)}")
            logger.info(f"임베딩 샘플 (처음 5개 값): {first_embedding[:5]}")
            
            # 첫번째 임베딩 저장 (디버깅용)
            import numpy as np
            np.save("first_embedding.npy", np.array(first_embedding))
            logger.info("첫 번째 임베딩 저장됨: first_embedding.npy")
        else:
            logger.warning("text_embedding 컬럼이 없습니다. 임베딩 생성에 실패했을 수 있습니다.")
        
        # 유사도 검색 테스트
        if "text_embedding" in result_df.columns:
            logger.info("=== 유사도 검색 테스트 ===")
            
            # 테스트 쿼리 텍스트
            query_text = "검은색 지갑을 잃어버렸습니다"
            logger.info(f"검색 쿼리: '{query_text}'")
            
            # 유사 항목 검색
            similar_items = processor.find_similar_items(
                query_text=query_text,
                result_df=result_df,
                top_n=5
            )
            
            # 결과 출력
            logger.info(f"검색 결과 ({len(similar_items)}개):")
            for i, item in enumerate(similar_items, 1):
                logger.info(f"{i}. ID: {item['id']}, 유사도: {item['similarity']:.4f}")
                # 주요 필드 출력 (id와 유사도 제외)
                for k, v in item['data'].items():
                    if k not in ['id', 'text_embedding', 'image_embedding']:
                        logger.info(f"   {k}: {v}")
        
        # 처리 시간 출력
        elapsed_time = time.time() - start_time
        logger.info(f"총 처리 시간: {elapsed_time:.2f}초")
        
        # Spark UI 정보 출력
        logger.info("Spark UI 확인: http://localhost:4040")
        
        # 결과 반환
        return result_df
    
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    finally:
        # 리소스 정리
        processor.stop()
        logger.info("=== 테스트 종료 ===")

if __name__ == "__main__":
    test_mysql_spark_clip()