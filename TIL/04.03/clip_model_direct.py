"""
PyTorch 기반 CLIP 모델 직접 사용 (ONNX 변환 없이)
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

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLIP_MODEL_NAME, DEVICE

class DirectPytorchCLIPModel:
    """
    PyTorch 기반 CLIP 모델 직접 사용
    """
    
    def __init__(self, model_name=CLIP_MODEL_NAME, device=DEVICE):
        self.device = device
        self.model_name = model_name
        
        logger.info(f"CLIP 모델 '{model_name}' 로드 중 (device: {device})...")
        
        try:
            self.model = CLIPModel.from_pretrained(model_name).to(device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            logger.info("CLIP 모델 로드 완료")
        except Exception as e:
            logger.error(f"CLIP 모델 로드 실패: {str(e)}")
            raise
            
    def encode_text(self, text):
        """텍스트를 임베딩 벡터로 변환"""
        if isinstance(text, str):
            text = [text]
            
        try:
            start_time = time.time()
            
            with torch.no_grad():
                inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                text_features = self.model.get_text_features(**inputs)
                text_embeddings = text_features / text_features.norm(dim=1, keepdim=True)
                
            process_time = time.time() - start_time
            logger.debug(f"텍스트 인코딩 시간: {process_time:.4f}초")
            
            return text_embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"텍스트 인코딩 중 오류 발생: {str(e)}")
            return np.zeros((len(text), self.model.text_embed_dim))
            
    def encode_image(self, image_source):
        """이미지를 임베딩 벡터로 변환"""
        try:
            start_time = time.time()
            
            # 이미지 로드
            if isinstance(image_source, str):
                if image_source.startswith('http'):
                    response = requests.get(image_source)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                else:
                    image = Image.open(image_source).convert('RGB')
            else:
                image = image_source.convert('RGB')
                
            with torch.no_grad():
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                image_features = self.model.get_image_features(**inputs)
                image_embeddings = image_features / image_features.norm(dim=1, keepdim=True)
                
            process_time = time.time() - start_time
            logger.debug(f"이미지 인코딩 시간: {process_time:.4f}초")
            
            return image_embeddings.cpu().numpy()
        except Exception as e:
            logger.error(f"이미지 인코딩 중 오류 발생: {str(e)}")
            return np.zeros((1, self.model.vision_embed_dim))
    
    def calculate_similarity(self, text_embedding, image_embedding=None):
        """텍스트와 이미지 임베딩 간의 유사도 계산"""
        if image_embedding is None:
            # 텍스트-텍스트 유사도 계산
            similarity = np.dot(text_embedding, text_embedding.T)[0, 0]
        else:
            # 텍스트-이미지 유사도 계산
            similarity = np.dot(text_embedding, image_embedding.T)[0, 0]
            
        # 유사도를 0~1 범위로 정규화
        similarity = (similarity + 1) / 2
        return float(similarity)
        
    def encode_batch_texts(self, texts, batch_size=32):
        """여러 텍스트를 배치로 임베딩"""
        all_embeddings = []
        
        # 배치 처리
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.encode_text(batch_texts)
            all_embeddings.append(batch_embeddings)
            
        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            return np.array([])

# 모듈 테스트
if __name__ == "__main__":
    clip_model = DirectPytorchCLIPModel()
    
    sample_text = "검은색 지갑을 잃어버렸습니다. 현금과 카드가 들어있어요."
    text_embedding = clip_model.encode_text(sample_text)
    
    print(f"텍스트 임베딩 shape: {text_embedding.shape}")
    
    sample_text2 = "검은색 지갑을 찾았습니다. 안에 현금과 카드가 있습니다."
    text_embedding2 = clip_model.encode_text(sample_text2)
    
    similarity = clip_model.calculate_similarity(text_embedding, text_embedding2)
    print(f"텍스트 간 유사도: {similarity:.4f}")