"""
시스템 설정 파일
"""
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# API 설정
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

# 모델 설정
CLIP_MODEL_NAME = "Bingsu/clip-vit-large-patch14-ko"  # 한국어 CLIP 모델
DEVICE = "cuda" if os.getenv('USE_GPU', 'False').lower() == 'true' and os.getenv('CUDA_VISIBLE_DEVICES') else "cpu"

# 유사도 설정
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.5'))  # 50% 이상 유사도를 가진 아이템을 반환
TEXT_WEIGHT = float(os.getenv('TEXT_WEIGHT', '0.7'))  # 텍스트 유사도 가중치
IMAGE_WEIGHT = float(os.getenv('IMAGE_WEIGHT', '0.3'))  # 이미지 유사도 가중치

# 텍스트 유사도 계산 설정
CATEGORY_WEIGHT = float(os.getenv('CATEGORY_WEIGHT', '0.5'))  # 카테고리 매칭 가중치
ITEM_NAME_WEIGHT = float(os.getenv('ITEM_NAME_WEIGHT', '0.3'))  # 물품명 매칭 가중치
COLOR_WEIGHT = float(os.getenv('COLOR_WEIGHT', '0.1'))  # 색상 매칭 가중치
CONTENT_WEIGHT = float(os.getenv('CONTENT_WEIGHT', '0.1'))  # 기타 내용 매칭 가중치

# MySQL 설정
MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
MYSQL_PORT = os.getenv('MYSQL_PORT', '3306')
MYSQL_DB = os.getenv('MYSQL_DB', 'mydatabase')
MYSQL_USER = os.getenv('MYSQL_USER', 'username')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', 'password')
MYSQL_URL = f"jdbc:mysql://{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"

# Spark 설정
SPARK_MASTER = os.getenv('SPARK_MASTER', 'local[*]')  # 로컬 테스트용. 실제 Spark 클러스터는 spark://host:port
SPARK_APP_NAME = os.getenv('SPARK_APP_NAME', 'MySQL-Spark-CLIP')
SPARK_EXECUTOR_MEMORY = os.getenv('SPARK_EXECUTOR_MEMORY', '4g')
SPARK_DRIVER_MEMORY = os.getenv('SPARK_DRIVER_MEMORY', '4g')
NUM_CLIP_SERVICES = int(os.getenv('NUM_CLIP_SERVICES', '2'))  # Ray 서비스 인스턴스 수

# 처리 설정
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '100'))  # 한 번에 처리할 데이터 수