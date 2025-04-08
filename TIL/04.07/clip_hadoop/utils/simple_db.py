# utils/simple_db.py 파일 생성

import mysql.connector
import logging
from typing import List, Dict, Any

# 로깅 설정
logger = logging.getLogger(__name__)

def fetch_found_items(host, port, user, password, database, limit=100):
    """
    MySQL에서 습득물 데이터를 가져오는 간단한 함수
    
    Args:
        host (str): MySQL 호스트
        port (int): MySQL 포트
        user (str): MySQL 사용자
        password (str): MySQL 비밀번호
        database (str): 데이터베이스 이름
        limit (int): 가져올 항목 수
        
    Returns:
        List[Dict]: 습득물 데이터 목록
    """
    found_items = []
    
    try:
        # 직접 연결
        conn = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database
        )
        
        cursor = conn.cursor(dictionary=True)
        
        # 습득물 데이터 조회 쿼리
        query = """
        SELECT 
            id, category, item_name, color, content, 
            location, image_url, created_at
        FROM 
            found_item
        ORDER BY 
            created_at DESC
        LIMIT %s
        """
        
        cursor.execute(query, (limit,))
        found_items = cursor.fetchall()
        
        # 날짜/시간 객체를 문자열로 변환 (JSON 직렬화를 위해)
        for item in found_items:
            if 'created_at' in item and item['created_at']:
                item['created_at'] = item['created_at'].isoformat()
        
        logger.info(f"{len(found_items)}개의 습득물 데이터를 가져왔습니다.")
        
    except Exception as e:
        logger.error(f"MySQL 연결 또는 쿼리 실행 오류: {str(e)}")
        
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
    
    return found_items