"""
한국어 CLIP 모델 구현
이 모듈은 HuggingFace의 한국어 CLIP 모델을 사용하여 텍스트와 이미지의 임베딩을 생성
"""
import os
import sys
import logging
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import time


# 캐시 디렉토리 설정
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'
os.environ['HF_HOME'] = '/tmp/huggingface_cache'

# 디렉토리 생성
os.makedirs('/tmp/huggingface_cache', exist_ok=True)
os.makedirs('/tmp/uploads', exist_ok=True)

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 모델 설정 - 환경 변수에서 가져오거나 기본값 사용
CLIP_MODEL_NAME = os.getenv('CLIP_MODEL_NAME', 'Bingsu/clip-vit-large-patch14-ko')
DEVICE = "cuda" if torch.cuda.is_available() and os.getenv('USE_GPU', 'True').lower() == 'true' else "cpu"

def preload_clip_model():
    """CLIP 모델을 사전에 다운로드하고 캐시"""
    try:
        start_time = time.time()
        logger.info(f"CLIP 모델 사전 다운로드 시작: {CLIP_MODEL_NAME}")
        
        # 모델과 프로세서 사전 다운로드
        CLIPModel.from_pretrained(
            CLIP_MODEL_NAME, 
            cache_dir='/tmp/huggingface_cache',
            low_cpu_mem_usage=True  # 메모리 사용 최적화
        )
        
        CLIPProcessor.from_pretrained(
            CLIP_MODEL_NAME, 
            cache_dir='/tmp/huggingface_cache'
        )
        
        logger.info(f"✅ CLIP 모델 사전 다운로드 완료 (소요시간: {time.time() - start_time:.2f}초)")
    except Exception as e:
        logger.error(f"❌ CLIP 모델 사전 다운로드 실패: {str(e)}")

class KoreanCLIPModel:
    """
    한국어 CLIP 모델 클래스
    텍스트와 이미지를 임베딩하고 유사도를 계산하는 기능 제공
    """
    
    def __init__(self, model_name=CLIP_MODEL_NAME, device=DEVICE):
        """CLIP 모델 초기화 - 메모리 최적화"""
        self.device = device
        self.model_name = model_name
        
        logger.info(f"CLIP 모델 '{model_name}' 로드 중 (device: {device})...")
        
        try:
            # 캐시 디렉토리 설정
            os.environ["TRANSFORMERS_CACHE"] = "/tmp/transformers_cache"
            os.makedirs("/tmp/transformers_cache", exist_ok=True)
            
            # 메모리 최적화 옵션 추가
            self.model = CLIPModel.from_pretrained(
                model_name,
                cache_dir='/tmp/huggingface_cache',
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16  # 반정밀도 사용
            ).to(device)
            
            self.processor = CLIPProcessor.from_pretrained(
                model_name,
                cache_dir='/tmp/huggingface_cache'
            )
            logger.info("CLIP 모델 로드 완료")
        except Exception as e:
            logger.error(f"CLIP 모델 로드 실패: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
            
    def encode_text(self, text):
        """
        텍스트를 임베딩 벡터로 변환
        
        Args:
            text (str or list): 인코딩할 텍스트 또는 텍스트 리스트
            
        Returns:
            numpy.ndarray: 임베딩 벡터
        """
        if isinstance(text, str):
            text = [text]
            
        try:
            with torch.no_grad():
                # 텍스트 인코딩
                inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                text_features = self.model.get_text_features(**inputs)
                
                # 텍스트 특성 정규화
                text_embeddings = text_features / text_features.norm(dim=1, keepdim=True)
                
            return text_embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"텍스트 인코딩 중 오류 발생: {str(e)}")
            return np.zeros((len(text), self.model.text_embed_dim))
            
    def encode_image(self, image_source):
        """
        이미지를 임베딩 벡터로 변환
        
        Args:
            image_source: 인코딩할 이미지 (PIL Image, URL 또는 이미지 경로)
            
        Returns:
            numpy.ndarray: 임베딩 벡터
        """
        try:
            # 이미지 로드 (URL, 파일 경로, PIL 이미지 객체 또는 Base64)
            if isinstance(image_source, str):
                if image_source.startswith('http'):
                    # URL에서 이미지 로드
                    response = requests.get(image_source)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                else:
                    # 로컬 파일에서 이미지 로드
                    image = Image.open(image_source).convert('RGB')
            else:
                # 이미 PIL 이미지 객체인 경우
                image = image_source.convert('RGB')
                
            with torch.no_grad():
                # 이미지 인코딩
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                image_features = self.model.get_image_features(**inputs)
                
                # 이미지 특성 정규화
                image_embeddings = image_features / image_features.norm(dim=1, keepdim=True)
                
            return image_embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"이미지 인코딩 중 오류 발생: {str(e)}")
            return np.zeros((1, self.model.vision_embed_dim))
    
    def calculate_similarity(self, text_embedding, image_embedding=None):
        """
        텍스트와 이미지 임베딩 간의 유사도 계산
        
        Args:
            text_embedding (numpy.ndarray): 텍스트 임베딩
            image_embedding (numpy.ndarray, optional): 이미지 임베딩 (없으면 텍스트만 비교)
            
        Returns:
            float: 유사도 점수 (0~1 사이)
        """
        if image_embedding is None:
            # 텍스트-텍스트 유사도 계산 (코사인 유사도)
            similarity = np.dot(text_embedding, text_embedding.T)[0, 0]
        else:
            # 텍스트-이미지 유사도 계산 (코사인 유사도)
            similarity = np.dot(text_embedding, image_embedding.T)[0, 0]
            
        # 유사도를 0~1 범위로 정규화
        similarity = (similarity + 1) / 2
        return float(similarity)
        
    def encode_batch_texts(self, texts):
        """
        여러 텍스트를 한 번에 임베딩
        
        Args:
            texts (list): 텍스트 목록
            
        Returns:
            numpy.ndarray: 임베딩 벡터 배열
        """
        # 배치 처리를 위한 코드
        # 실제 구현에서는 메모리 크기에 따라 적절한 배치 크기 조정 필요
        return self.encode_text(texts)

# 모듈 테스트용 코드
if __name__ == "__main__":
    # 모델 초기화
    clip_model = KoreanCLIPModel()
    
    # 샘플 텍스트 인코딩
    sample_text = "검은색 지갑을 잃어버렸습니다. 현금과 카드가 들어있어요."
    text_embedding = clip_model.encode_text(sample_text)
    
    print(f"텍스트 임베딩 shape: {text_embedding.shape}")
    
    # 유사도 계산 (텍스트만)
    sample_text2 = "검은색 지갑을 찾았습니다. 안에 현금과 카드가 있습니다."
    text_embedding2 = clip_model.encode_text(sample_text2)
    
    similarity = clip_model.calculate_similarity(text_embedding, text_embedding2)
    print(f"텍스트 간 유사도: {similarity:.4f}")