"""
MySQL 연결 및 데이터 로드 모듈
게시글 데이터를 MySQL에서 가져와 처리
"""
import os
import sys
import logging
import pymysql
import pandas as pd
from pymysql.cursors import DictCursor
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 추가하여 config.py 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, 
    BATCH_SIZE, MAX_POSTS_TO_FETCH, CACHE_EXPIRE_TIME
)

# 메모리 캐싱 설정
_cached_posts = None
_last_cache_time = 0

@contextmanager
def get_db_connection():
    """
    데이터베이스 연결을 생성하고 관리하는 컨텍스트 매니저
    
    Yields:
        pymysql.Connection: 데이터베이스 연결 객체
    """
    connection = None
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset='utf8mb4',
            cursorclass=DictCursor,
            connect_timeout=5
        )
        yield connection
    except pymysql.MySQLError as e:
        logger.error(f"MySQL 연결 오류: {str(e)}")
        if connection:
            connection.rollback()
        raise
    finally:
        if connection:
            connection.close()

def fetch_posts_from_db(limit: int = MAX_POSTS_TO_FETCH, use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    MySQL 데이터베이스에서 게시글 데이터를 가져옴
    
    Args:
        limit (int): 가져올 최대 게시글 수
        use_cache (bool): 캐시 사용 여부
        
    Returns:
        List[Dict[str, Any]]: 게시글 데이터 목록
    """
    global _cached_posts, _last_cache_time
    
    # 캐시 유효성 확인
    current_time = time.time()
    if use_cache and _cached_posts is not None and (current_time - _last_cache_time) < CACHE_EXPIRE_TIME:
        logger.info(f"캐시에서 {len(_cached_posts)} 개의 게시글 데이터 사용")
        return _cached_posts
    
    logger.info(f"MySQL에서 최대 {limit}개의 게시글 데이터를 가져오는 중...")
    
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                # 게시글 데이터 쿼리
                # 예시 쿼리: 실제 데이터베이스 스키마에 맞게 수정 필요
                query = """
                SELECT 
                    f.found_item_id as id, 
                    f.title, 
                    f.content, 
                    f.create_date as created_at, 
                    f.category, 
                    f.name as item_name, 
                    f.color, 
                    f.place as location, 
                    f.photo_url as image_url
                FROM 
                    found_item f
                WHERE 
                    f.is_completed = 0
                ORDER BY 
                    f.create_date DESC
                LIMIT %s
                """
                
                cursor.execute(query, (limit,))
                posts = cursor.fetchall()
                
                # 게시글 데이터 정제
                processed_posts = []
                for post in posts:
                    # 필요한 전처리 작업 수행
                    processed_post = {
                        'id': post['id'],
                        'title': post['title'],
                        'content': post['content'],
                        'category': post['category'],
                        'item_name': post['item_name'],
                        'color': post['color'],
                        'location': post['location'],
                        'image_url': post['image_url'],
                        'created_at': post['created_at'].isoformat() if post['created_at'] else None
                    }
                    processed_posts.append(processed_post)
                
                # 캐시 업데이트
                _cached_posts = processed_posts
                _last_cache_time = current_time
                
                logger.info(f"MySQL에서 {len(processed_posts)}개의 게시글을 성공적으로 가져왔습니다")
                return processed_posts
                
    except Exception as e:
        logger.error(f"게시글 데이터 가져오기 오류: {str(e)}")
        # 캐시가 있는 경우 캐시 반환
        if _cached_posts is not None:
            logger.warning(f"DB 연결 오류로 인해 {len(_cached_posts)}개의 캐시된 게시글 사용")
            return _cached_posts
        return []

def fetch_posts_in_batches(batch_size: int = BATCH_SIZE) -> pd.DataFrame:
    """
    대량의 게시글 데이터를 배치로 가져와 DataFrame으로 변환
    Spark 처리를 위한 준비
    
    Args:
        batch_size (int): 배치 크기
        
    Returns:
        pd.DataFrame: 게시글 데이터 DataFrame
    """
    all_posts = []
    offset = 0
    
    logger.info(f"배치 크기 {batch_size}로 게시글 데이터 로드 중...")
    
    try:
        with get_db_connection() as connection:
            while True:
                with connection.cursor() as cursor:
                    # 배치 쿼리 실행
                    query = """
                    SELECT 
                        f.found_item_id as id, 
                        f.title, 
                        f.content, 
                        f.create_date as created_at, 
                        f.category, 
                        f.name as item_name, 
                        f.color, 
                        f.place as location, 
                        f.photo_url as image_url
                    FROM 
                        found_item f
                    WHERE 
                        f.is_completed = 0
                    ORDER BY 
                        f.found_item_id
                    LIMIT %s OFFSET %s
                    """
                    
                    cursor.execute(query, (batch_size, offset))
                    batch = cursor.fetchall()
                    
                    if not batch:
                        break
                    
                    all_posts.extend(batch)
                    offset += batch_size
                    logger.info(f"{len(all_posts)}개의 게시글 로드됨...")
                    
                    # 최대 건수 제한 (선택적)
                    if len(all_posts) >= MAX_POSTS_TO_FETCH:
                        break
        
        # DataFrame으로 변환
        df = pd.DataFrame(all_posts)
        logger.info(f"총 {len(df)}개의 게시글을 DataFrame으로 변환 완료")
        return df
    
    except Exception as e:
        logger.error(f"배치 데이터 로드 오류: {str(e)}")
        # 부분적으로 로드된 데이터 반환
        if all_posts:
            logger.warning(f"부분 데이터 {len(all_posts)}개로 DataFrame 생성")
            return pd.DataFrame(all_posts)
        raise

def save_matching_result(lost_item_id, found_item_id, similarity_score):
    """
    유사도 계산 결과를 matched_item 테이블에 저장
    
    Args:
        lost_item_id (int): 분실물 ID
        found_item_id (int): 습득물 ID
        similarity_score (float): 유사도 점수
        
    Returns:
        int: 생성된 매칭 ID
    """
    try:
        with get_db_connection() as connection:
            with connection.cursor() as cursor:
                # 이미 존재하는 매칭 확인
                check_query = """
                SELECT matched_item_id FROM matched_item 
                WHERE lost_item_id = %s AND found_item_id = %s
                """
                cursor.execute(check_query, (lost_item_id, found_item_id))
                existing = cursor.fetchone()
                
                if existing:
                    # 기존 매칭 업데이트
                    update_query = """
                    UPDATE matched_item
                    SET similarity = %s, update_date = NOW()
                    WHERE lost_item_id = %s AND found_item_id = %s
                    """
                    cursor.execute(update_query, (similarity_score, lost_item_id, found_item_id))
                    match_id = existing['matched_item_id']
                    logger.info(f"매칭 결과가 업데이트되었습니다. ID: {match_id}")
                else:
                    # 새 매칭 삽입
                    insert_query = """
                    INSERT INTO matched_item (
                        lost_item_id, found_item_id, similarity, create_date
                    ) VALUES (
                        %s, %s, %s, NOW()
                    )
                    """
                    cursor.execute(insert_query, (lost_item_id, found_item_id, similarity_score))
                    match_id = cursor.lastrowid
                    logger.info(f"새 매칭 결과가 저장되었습니다. ID: {match_id}")
                
                connection.commit()
                return match_id
                
    except Exception as e:
        logger.error(f"매칭 결과 저장 오류: {str(e)}")
        # 실패 시 오류를 전파하지 않고 None 반환
        return None

# 모듈 테스트용 코드
if __name__ == "__main__":
    try:
        # MySQL 연결 테스트
        with get_db_connection() as conn:
            logger.info("MySQL 연결 성공!")
        
        # 게시글 데이터 가져오기 테스트
        posts = fetch_posts_from_db(limit=5)
        logger.info(f"가져온 게시글 수: {len(posts)}")
        
        if posts:
            # 첫 번째 게시글 정보 출력
            first_post = posts[0]
            logger.info(f"첫 번째 게시글: {first_post['title']}")
            
        # 배치 로드 테스트
        df = fetch_posts_in_batches(batch_size=100)
        logger.info(f"배치 로드된 게시글 수: {len(df)}")
        logger.info(f"DataFrame 컬럼: {df.columns.tolist()}")
        
    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}")