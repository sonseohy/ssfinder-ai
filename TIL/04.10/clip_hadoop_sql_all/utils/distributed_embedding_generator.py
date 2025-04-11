#!/usr/bin/env python3
"""
스파크를 활용한 CLIP 임베딩 분산 생성기
MySQL에서 습득물 데이터를 가져와 로컬 API를 통해 CLIP 임베딩을 생성하고 하둡에 저장
"""
import os
import sys
import logging
import argparse
import time
import random
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, BooleanType, StructType, StructField

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("spark_embedding_generator.log")
    ]
)
logger = logging.getLogger(__name__)

def create_spark_session(app_name, master_url, executor_memory="4g", driver_memory="2g"):
    """
    스파크 세션 생성
    
    Args:
        app_name (str): 애플리케이션 이름
        master_url (str): 스파크 마스터 URL (예: spark://host:port)
        executor_memory (str): 실행기 메모리
        driver_memory (str): 드라이버 메모리
        
    Returns:
        SparkSession: 스파크 세션
    """
    logger.info(f"스파크 세션 생성 중 (master: {master_url})")
    
    # MySQL 커넥터 JAR 경로 - 다운로드: wget https://repo1.maven.org/maven2/mysql/mysql-connector-java/8.0.28/mysql-connector-java-8.0.28.jar
    mysql_jar = "mysql-connector-java-8.0.28.jar"
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(master_url) \
        .config("spark.executor.memory", executor_memory) \
        .config("spark.driver.memory", driver_memory) \
        .config("spark.python.worker.memory", "1g") \
        .config("spark.jars", mysql_jar) \
        .getOrCreate()
    
    logger.info("스파크 세션 생성 완료")
    return spark

def fetch_data_from_mysql(spark, db_config, batch_size=1000, offset=0):
    """
    MySQL에서 데이터를 배치로 가져오기
    
    Args:
        spark (SparkSession): 스파크 세션
        db_config (dict): 데이터베이스 설정
        batch_size (int): 배치 크기
        offset (int): 오프셋
        
    Returns:
        DataFrame: 스파크 데이터프레임
    """
    query = f"""
    (SELECT 
        id, detail, name, image, 
        location, status, found_at, management_id
    FROM 
        found_item
    ORDER BY 
        id DESC
    LIMIT {batch_size} OFFSET {offset}) as found_items
    """
    
    logger.info(f"MySQL에서 데이터 가져오기 (batch: {batch_size}, offset: {offset})")
    
    try:
        df = spark.read \
            .format("jdbc") \
            .option("url", f"jdbc:mysql://{db_config['host']}:{db_config['port']}/{db_config['database']}") \
            .option("driver", "com.mysql.cj.jdbc.Driver") \
            .option("dbtable", query) \
            .option("user", db_config['user']) \
            .option("password", db_config['password']) \
            .load()
        
        count = df.count()
        logger.info(f"{count}개 항목 가져옴")
        return df
    except Exception as e:
        logger.error(f"MySQL 데이터 가져오기 실패: {str(e)}")
        raise

def define_embedding_udf(api_urls):
    """
    임베딩 생성 및 하둡 저장을 위한 UDF 정의
    여러 API 서버에 라운드 로빈 방식으로 요청을 분산
    
    Args:
        api_urls (list): API 서버 URL 목록
        
    Returns:
        function: UDF 함수
    """
    # 결과 스키마 정의
    result_schema = StructType([
        StructField("success", BooleanType(), True),
        StructField("file_path", StringType(), True),
        StructField("message", StringType(), True)
    ])
    
    # 서버 인덱스 초기화 (라운드 로빈용)
    server_index = [0]  # 리스트로 감싸서 클로저 내부에서 수정 가능하게 함
    
    # UDF 정의
    @udf(result_schema)
    def generate_embedding_udf(item_id, name, detail, image, location, status, found_at):
        try:
            import requests
            import json
            from datetime import datetime
            
            # 라운드 로빈 방식으로 API 서버 선택
            current_index = server_index[0]
            server_url = api_urls[current_index]
            # 다음 서버로 인덱스 업데이트
            server_index[0] = (current_index + 1) % len(api_urls)
            
            # 날짜/시간 객체 처리
            if isinstance(found_at, datetime):
                found_at = found_at.isoformat()
            
            # 텍스트 데이터 구성
            text_parts = []
            if name:
                text_parts.append(f"이름: {name}")
            if detail:
                text_parts.append(f"상세: {detail}")
            if location:
                text_parts.append(f"위치: {location}")
                
            text = " ".join(text_parts)
            
            # 메타데이터 구성
            metadata = {
                "found_at": found_at,
                "status": status
            }
            
            # API 호출 (선택된 API 서버로)
            payload = {
                "item_id": str(item_id),
                "text": text,
                "image_url": image,
                "metadata": metadata
            }
            
            headers = {"Content-Type": "application/json"}
            
            # 재시도 로직 추가
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    # 지수 백오프를 적용한 재시도
                    if retry_count > 0:
                        backoff_time = 2 ** retry_count + random.uniform(0, 1)
                        time.sleep(min(backoff_time, 30))  # 최대 30초 대기
                    
                    endpoint = f"{server_url}/api/hadoop/save-embedding"
                    response = requests.post(
                        endpoint,
                        json=payload,
                        headers=headers,
                        timeout=60  # 60초 타임아웃
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        return (True, result.get('file_path'), result.get('message', '성공'))
                    elif response.status_code >= 500:  # 서버 오류는 재시도
                        retry_count += 1
                        if retry_count >= max_retries:
                            return (False, None, f"서버 오류 (코드: {response.status_code}): {response.text}")
                    else:  # 클라이언트 오류는 재시도하지 않음
                        return (False, None, f"API 오류 (코드: {response.status_code}): {response.text}")
                except requests.exceptions.RequestException as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        return (False, None, f"API 호출 실패: {str(e)}")
                        
        except Exception as e:
            return (False, None, f"처리 중 오류: {str(e)}")
    
    return generate_embedding_udf

def process_data_in_batches(spark, db_config, api_urls, batch_size=100, max_batches=None):
    """
    MySQL에서 데이터를 배치로 가져와 처리
    
    Args:
        spark (SparkSession): 스파크 세션
        db_config (dict): 데이터베이스 설정
        api_urls (list): API 서버 URL 목록
        batch_size (int): 배치 크기
        max_batches (int): 최대 배치 수 (None이면 모든 데이터 처리)
        
    Returns:
        dict: 처리 결과 통계
    """
    offset = 0
    batch_num = 0
    total_success = 0
    total_processed = 0
    
    # 임베딩 생성 UDF 정의
    generate_embedding = define_embedding_udf(api_urls)
    
    while True:
        batch_num += 1
        
        if max_batches and batch_num > max_batches:
            logger.info(f"최대 배치 수({max_batches})에 도달하여 종료")
            break
        
        # 배치 데이터 가져오기
        try:
            df = fetch_data_from_mysql(spark, db_config, batch_size, offset)
            
            if df.count() == 0:
                logger.info("더 이상 처리할 데이터가 없습니다")
                break
                
            # UDF 적용하여 임베딩 생성 및 하둡 저장
            logger.info(f"배치 {batch_num} 처리 중 (offset: {offset})")
            
            result_df = df.withColumn(
                "embedding_result",
                generate_embedding(
                    df["id"], df["name"], df["detail"], df["image"], 
                    df["location"], df["status"], df["found_at"]
                )
            )
            
            # 처리 결과 집계
            success_df = result_df.filter(col("embedding_result.success") == True)
            batch_success = success_df.count()
            batch_total = result_df.count()
            
            total_success += batch_success
            total_processed += batch_total
            
            logger.info(f"배치 {batch_num} 처리 완료: {batch_success}/{batch_total} 성공")
            
            # 실패한 항목 로깅
            failed_df = result_df.filter(col("embedding_result.success") == False)
            if failed_df.count() > 0:
                logger.warning("실패한 항목:")
                failed_items = failed_df.select("id", "embedding_result.message").collect()
                for item in failed_items:
                    logger.warning(f"  ID {item['id']}: {item['message']}")
            
            # 다음 배치로 이동
            offset += batch_size
            
            # 배치 크기보다 적은 데이터가 반환되면 종료
            if df.count() < batch_size:
                logger.info("모든 데이터 처리 완료")
                break
                
        except Exception as e:
            logger.error(f"배치 {batch_num} 처리 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 다음 배치로 진행
            offset += batch_size
    
    # 처리 결과 통계
    stats = {
        "total_batches": batch_num - 1,
        "total_processed": total_processed,
        "total_success": total_success,
        "success_rate": round(total_success / max(1, total_processed) * 100, 2)
    }
    
    logger.info(f"처리 완료: {stats['total_success']}/{stats['total_processed']} 항목 성공 ({stats['success_rate']}%)")
    
    return stats

def main():
    """
    메인 함수
    """
    parser = argparse.ArgumentParser(description="스파크를 활용한 CLIP 임베딩 분산 생성기")
    parser.add_argument("--master", required=True, help="스파크 마스터 URL (예: spark://host:port)")
    parser.add_argument("--api-urls", required=True, help="API 서버 URL 목록 (쉼표로 구분)")
    parser.add_argument("--db-host", default="j12c105.p.ssafy.io", help="MySQL 호스트")
    parser.add_argument("--db-port", type=int, default=3306, help="MySQL 포트")
    parser.add_argument("--db-user", default="ssafy", help="MySQL 사용자")
    parser.add_argument("--db-password", default="tnatnavkdlsejssafyc!)%", help="MySQL 비밀번호")
    parser.add_argument("--db-name", default="ssfinder", help="데이터베이스 이름")
    parser.add_argument("--batch-size", type=int, default=50, help="배치 크기")
    parser.add_argument("--max-batches", type=int, help="최대 배치 수 (기본: 무제한)")
    parser.add_argument("--executor-memory", default="4g", help="실행기 메모리")
    parser.add_argument("--driver-memory", default="2g", help="드라이버 메모리")
    
    args = parser.parse_args()
    
    # API URL 목록 파싱
    api_urls = args.api_urls.split(',')
    logger.info(f"API 서버 URL: {api_urls}")
    
    # 데이터베이스 설정
    db_config = {
        "host": args.db_host,
        "port": args.db_port,
        "user": args.db_user,
        "password": args.db_password,
        "database": args.db_name
    }
    
    try:
        # 스파크 세션 생성
        spark = create_spark_session(
            "CLIP Embedding Generator", 
            args.master,
            args.executor_memory,
            args.driver_memory
        )
        
        # 시작 시간 기록
        start_time = datetime.now()
        logger.info(f"처리 시작: {start_time}")
        
        # 데이터 처리
        stats = process_data_in_batches(
            spark, 
            db_config, 
            api_urls,
            args.batch_size,
            args.max_batches
        )
        
        # 종료 시간 기록 및 총 소요 시간 계산
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info(f"처리 종료: {end_time}")
        logger.info(f"총 소요 시간: {duration}")
        logger.info(f"처리 통계: {stats}")
        
        # 스파크 세션 종료
        spark.stop()
        
        return 0
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())