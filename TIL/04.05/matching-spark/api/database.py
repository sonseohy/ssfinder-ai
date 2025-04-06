"""
데이터베이스 연결 관련 유틸리티 함수
"""
import os
import sys
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 추가하여 config.py 임포트 가능하게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from config import MYSQL_HOST, MYSQL_PORT, MYSQL_DB, MYSQL_USER, MYSQL_PASSWORD
except ImportError:
    logger.error("config.py에서 MySQL 설정을 가져올 수 없습니다.")
    # 기본값 설정
    MYSQL_HOST = "localhost"
    MYSQL_PORT = "3306"
    MYSQL_DB = "mydatabase"
    MYSQL_USER = "username"
    MYSQL_PASSWORD = "password"

# MySQL 연결 URL 생성
SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

try:
    # 데이터베이스 엔진 생성
    engine = create_engine(SQLALCHEMY_DATABASE_URL)
    
    # 세션 클래스 생성
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Base 클래스 생성 (ORM 모델 기본 클래스)
    Base = declarative_base()
    
    logger.info(f"MySQL 데이터베이스 연결 초기화 완료 (host: {MYSQL_HOST}, db: {MYSQL_DB})")
    
except Exception as e:
    logger.error(f"데이터베이스 연결 초기화 실패: {str(e)}")
    raise

# 데이터베이스 세션 관리 함수
def get_db():
    """
    데이터베이스 세션을 제공하는 함수 (FastAPI 의존성 주입용)
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_posts(limit=1000):
    """
    게시글 데이터를 가져오는 함수
    
    Args:
        limit (int): 가져올 최대 게시글 수
        
    Returns:
        list: 게시글 목록
    """
    try:
        with engine.connect() as conn:
            query = f"""
            SELECT * FROM posts LIMIT {limit}
            """
            result = conn.execute(text(query))
            posts = [dict(row) for row in result]
            
            logger.info(f"{len(posts)}개 게시글 조회됨")
            return posts
    
    except Exception as e:
        logger.error(f"게시글 조회 실패: {str(e)}")
        return []