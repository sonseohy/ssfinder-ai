import mysql.connector
import logging
from typing import List, Dict, Any, Optional
from mysql.connector.pooling import MySQLConnectionPool
import os
from contextlib import contextmanager

# 로깅 설정
logger = logging.getLogger(__name__)

# 연결 풀 (선택적)
pool = None

def init_connection_pool(host: str, port: int, user: str, password: str, 
                        database: str, pool_size: int = 5) -> None:
    """
    MySQL 연결 풀을 초기화합니다.
    
    Args:
        host (str): MySQL 호스트
        port (int): MySQL 포트
        user (str): MySQL 사용자
        password (str): MySQL 비밀번호
        database (str): 데이터베이스 이름
        pool_size (int): 연결 풀 크기
    """
    global pool
    
    try:
        pool = MySQLConnectionPool(
            pool_name="mypool",
            pool_size=pool_size,
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        logger.info("MySQL 연결 풀이 초기화되었습니다.")
    except Exception as e:
        logger.error(f"MySQL 연결 풀 초기화 오류: {str(e)}")
        raise


@contextmanager
def get_connection():
    """
    데이터베이스 연결을 제공하는 컨텍스트 매니저
    """
    conn = None
    try:
        if pool:
            conn = pool.get_connection()
            logger.debug("풀에서 연결을 가져왔습니다.")
        else:
            # 환경 변수 또는 config에서 설정을 가져올 수도 있습니다
            from config import DB_CONFIG
            conn = mysql.connector.connect(**DB_CONFIG)
            logger.debug("새 데이터베이스 연결을 생성했습니다.")
        
        yield conn
    except Exception as e:
        logger.error(f"데이터베이스 연결 오류: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()
            logger.debug("데이터베이스 연결을 닫았습니다.")


def fetch_found_items(host: str = None, port: int = None, user: str = None, 
                     password: str = None, database: str = None, limit: int = 100) -> List[Dict[str, Any]]:
    """
    MySQL에서 습득물 데이터를 가져오는 함수
    
    Args:
        host (str, optional): MySQL 호스트 (연결 풀 사용 시 불필요)
        port (int, optional): MySQL 포트 (연결 풀 사용 시 불필요)
        user (str, optional): MySQL 사용자 (연결 풀 사용 시 불필요)
        password (str, optional): MySQL 비밀번호 (연결 풀 사용 시 불필요)
        database (str, optional): 데이터베이스 이름 (연결 풀 사용 시 불필요)
        limit (int): 가져올 항목 수
        
    Returns:
        List[Dict]: 습득물 데이터 목록
    """
    found_items = []
    
    try:
        # 연결 풀이 초기화되었으면 사용, 아니면 직접 연결
        if pool:
            with get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                _execute_found_items_query(cursor, limit)
                found_items = cursor.fetchall()
        else:
            # 직접 연결 파라미터가 없으면 config에서 가져오기
            if not all([host, port, user, password, database]):
                try:
                    from config import DB_CONFIG
                    host = DB_CONFIG.get('host')
                    port = DB_CONFIG.get('port')
                    user = DB_CONFIG.get('user')
                    password = DB_CONFIG.get('password')
                    database = DB_CONFIG.get('database')
                except (ImportError, AttributeError):
                    logger.error("데이터베이스 설정을 찾을 수 없습니다.")
                    return []
            
            conn = mysql.connector.connect(
                host=host,
                port=port,
                user=user,
                password=password,
                database=database
            )
            
            cursor = conn.cursor(dictionary=True)
            _execute_found_items_query(cursor, limit)
            found_items = cursor.fetchall()
            
            cursor.close()
            conn.close()
        
        # 날짜/시간 객체를 문자열로 변환 (JSON 직렬화를 위해)
        for item in found_items:
            if 'created_at' in item and item['created_at']:
                item['created_at'] = item['created_at'].isoformat()
        
        logger.info(f"{len(found_items)}개의 습득물 데이터를 가져왔습니다.")
        
    except Exception as e:
        logger.error(f"MySQL 연결 또는 쿼리 실행 오류: {str(e)}")
        
    return found_items


def _execute_found_items_query(cursor, limit: int) -> None:
    """
    습득물 데이터 조회 쿼리를 실행하는 내부 함수
    
    Args:
        cursor: MySQL 커서 객체
        limit (int): 가져올 항목 수
    """
    query = """
    SELECT 
        id, item_category_id, name, color, detail, 
        location, image, created_at
    FROM 
        found_item
    ORDER BY 
        created_at DESC
    LIMIT %s
    """
    
    cursor.execute(query, (limit,))


def fetch_found_item_by_id(item_id: int) -> Optional[Dict[str, Any]]:
    """
    ID로 특정 습득물 항목을 조회합니다.
    
    Args:
        item_id (int): 조회할 항목의 ID
        
    Returns:
        Optional[Dict]: 습득물 데이터 또는 None
    """
    try:
        with get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            query = """
            SELECT 
                id, item_category_id, name, color, detail, 
                location, image, created_at
            FROM 
                found_item
            WHERE 
                id = %s
            """
            
            cursor.execute(query, (item_id,))
            item = cursor.fetchone()
            
            if item and 'created_at' in item and item['created_at']:
                item['created_at'] = item['created_at'].isoformat()
            
            return item
            
    except Exception as e:
        logger.error(f"항목 조회 오류 (ID: {item_id}): {str(e)}")
        return None