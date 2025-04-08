"""
Lost and Found AI 시스템 설정 파일
"""
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# API 설정
POLICE_API_SERVICE_KEY = os.getenv('POLICE_API_SERVICE_KEY')
API_BASE_URL = 'http://apis.data.go.kr/1320000/LosfundInfoInqireService/getLosfundInfoAccToClAreaPd'

# 모델 설정
CLIP_MODEL_NAME = "Bingsu/clip-vit-large-patch14-ko"  # 한국어 CLIP 모델
DEVICE = "cuda" if os.getenv('USE_GPU', 'False').lower() == 'true' and os.getenv('CUDA_VISIBLE_DEVICES') else "cpu"

# 유사도 설정
SIMILARITY_THRESHOLD = 0.5  # 70% 이상 유사도를 가진 아이템을 반환
TEXT_WEIGHT = 0.7  # 텍스트 유사도 가중치
IMAGE_WEIGHT = 0.3  # 이미지 유사도 가중치

# 텍스트 유사도 계산 설정
CATEGORY_WEIGHT = 0.5  # 카테고리 매칭 가중치
ITEM_NAME_WEIGHT = 0.3  # 물품명 매칭 가중치
COLOR_WEIGHT = 0.1  # 색상 매칭 가중치
CONTENT_WEIGHT = 0.1  # 기타 내용 매칭 가중치

# 데이터 설정
BATCH_SIZE = 100  # 한 번에 처리할, 습득물 데이터 수

# Hadoop 설정
HADOOP_HOST = os.getenv('HADOOP_HOST')  # Hadoop 네임노드 호스트 주소
HADOOP_PORT = int(os.getenv('HADOOP_PORT'))  # WebHDFS 포트
HADOOP_USER = os.getenv('HADOOP_USER')  # HDFS 사용자
