# utils/test_db.py
import logging
import sys
from utils.simple_db import fetch_found_items

# 로깅 설정
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def test_db_connection():
    # DB 연결 정보 - 실제 값으로 대체하세요
    host = "j12c105.p.ssafy.io"  # 또는 실제 DB 호스트
    port = 3306         # 또는 실제 포트
    user = "ssafy"  # 실제 사용자명
    password = "tnatnavkdlsejssafyc!)%"  # 실제 비밀번호
    database = "ssfinder"  # 실제 DB 이름
    
    print("데이터베이스 연결 테스트 중...")
    
    try:
        # 연결 테스트 및 데이터 가져오기
        items = fetch_found_items(host, port, user, password, database, limit=5)
        
        if items:
            print(f"성공! {len(items)}개의 항목을 가져왔습니다:")
            for item in items:
                print(f"ID: {item.get('id')}, 이름: {item.get('name')}")
        else:
            print("연결은 성공했지만 데이터가 없습니다.")
            
        return True
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_db_connection()
    sys.exit(0 if success else 1)