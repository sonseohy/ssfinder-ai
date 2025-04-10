"""
AI 임베딩 생성 및 유사도 비교 서비스 설정 파일
"""
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 모델 설정
MODEL_NAME = os.getenv('MODEL_NAME', 'Bingsu/clip-vit-large-patch14-ko')  # 한국어 CLIP 모델
DEVICE = "cuda" if os.getenv('CUDA_VISIBLE_DEVICES') else "cpu"

# 임베딩 캐시 디렉토리
EMBEDDING_CACHE_DIR = os.getenv('EMBEDDING_CACHE_DIR', 'data/embeddings')

# 유사도 설정
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))  # 70% 이상 유사도를 가진 아이템을 반환
TEXT_WEIGHT = float(os.getenv('TEXT_WEIGHT', '0.7'))  # 텍스트 유사도 가중치
IMAGE_WEIGHT = float(os.getenv('IMAGE_WEIGHT', '0.3'))  # 이미지 유사도 가중치

# 텍스트 유사도 계산 설정
CATEGORY_WEIGHT = float(os.getenv('CATEGORY_WEIGHT', '0.5'))  # 카테고리 매칭 가중치
ITEM_NAME_WEIGHT = float(os.getenv('ITEM_NAME_WEIGHT', '0.3'))  # 물품명 매칭 가중치
COLOR_WEIGHT = float(os.getenv('COLOR_WEIGHT', '0.1'))  # 색상 매칭 가중치
CONTENT_WEIGHT = float(os.getenv('CONTENT_WEIGHT', '0.1'))  # 기타 내용 매칭 가중치

# 업로드 설정
UPLOAD_DIR = os.getenv('UPLOAD_DIR', 'data/uploads')
MAX_IMAGE_SIZE = int(os.getenv('MAX_IMAGE_SIZE', '10485760'))  # 10MB

# 로깅 설정
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')