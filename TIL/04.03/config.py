"""
Lost and Found AI 시스템 설정 파일 (경량화 모델 지원)
"""
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# ===== 데이터베이스 설정 =====
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = int(os.getenv('DB_PORT', '3306'))
DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_NAME = os.getenv('DB_NAME', 'lostfound')

# ===== API 설정 =====
FASTAPI_HOST = os.getenv('FASTAPI_HOST', 'localhost')
FASTAPI_PORT = int(os.getenv('FASTAPI_PORT', '5000'))

# ===== Spark 설정 =====
SPARK_MASTER = os.getenv('SPARK_MASTER', 'local[*]')  # 로컬 모드 또는 'yarn'
SPARK_EXECUTOR_MEMORY = os.getenv('SPARK_EXECUTOR_MEMORY', '4g')
SPARK_DRIVER_MEMORY = os.getenv('SPARK_DRIVER_MEMORY', '4g')
SPARK_EXECUTOR_CORES = int(os.getenv('SPARK_EXECUTOR_CORES', '2'))

# ===== 모델 설정 =====
CLIP_MODEL_NAME = os.getenv('CLIP_MODEL_NAME', "Bingsu/clip-vit-large-patch14-ko")  # 한국어 CLIP 모델
DEVICE = "cuda" if os.getenv('USE_GPU', 'False').lower() == 'true' and os.getenv('CUDA_VISIBLE_DEVICES') else "cpu"

# 경량화 모델 경로
ONNX_MODEL_PATH = os.getenv('ONNX_MODEL_PATH', "models/onnx")
USE_ONNX_MODEL = os.getenv('USE_ONNX_MODEL', 'True').lower() == 'true'

# ===== 유사도 설정 =====
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.5'))  # 50% 이상 유사도를 가진 아이템을 반환
TEXT_WEIGHT = float(os.getenv('TEXT_WEIGHT', '0.7'))  # 텍스트 유사도 가중치
IMAGE_WEIGHT = float(os.getenv('IMAGE_WEIGHT', '0.3'))  # 이미지 유사도 가중치

# 텍스트 유사도 계산 설정
CATEGORY_WEIGHT = float(os.getenv('CATEGORY_WEIGHT', '0.5'))  # 카테고리 매칭 가중치
ITEM_NAME_WEIGHT = float(os.getenv('ITEM_NAME_WEIGHT', '0.3'))  # 물품명 매칭 가중치
COLOR_WEIGHT = float(os.getenv('COLOR_WEIGHT', '0.1'))  # 색상 매칭 가중치
CONTENT_WEIGHT = float(os.getenv('CONTENT_WEIGHT', '0.1'))  # 기타 내용 매칭 가중치

# ===== 데이터 설정 =====
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '100'))  # 한 번에 처리할 습득물 데이터 수
MAX_POSTS_TO_FETCH = int(os.getenv('MAX_POSTS_TO_FETCH', '10000'))  # 최대 가져올 게시글 수
CACHE_EXPIRE_TIME = int(os.getenv('CACHE_EXPIRE_TIME', '300'))  # 캐시 만료 시간 (초)