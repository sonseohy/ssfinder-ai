"""
게시글 임베딩 생성 시스템 설정 파일
"""
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 모델 설정
CLIP_MODEL_NAME = "Bingsu/clip-vit-large-patch14-ko"  # 한국어 CLIP 모델
DEVICE = "cuda" if os.getenv('USE_GPU', 'False').lower() == 'true' and os.getenv('CUDA_VISIBLE_DEVICES') else "cpu"

# 임베딩 설정
EMBEDDING_DIMENSION = 768  # CLIP 임베딩 차원 수 (모델에 따라 다를 수 있음)
COMBINE_METHOD = "mean"    # 텍스트와 이미지 임베딩 결합 방법 (mean, concat)

# 데이터 설정
BATCH_SIZE = 32  # 배치 크기