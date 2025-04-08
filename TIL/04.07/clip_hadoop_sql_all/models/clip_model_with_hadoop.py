"""
한국어 CLIP 모델 구현 - 하둡 저장 기능 추가
이 모듈은 HuggingFace의 한국어 CLIP 모델을 사용하여 텍스트와 이미지의 임베딩을 생성하고
생성된 임베딩을 Hadoop에 저장하는 기능을 제공
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
from typing import Dict, List, Optional, Union, Any

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 추가하여 config.py 및 hadoop_storage.py 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLIP_MODEL_NAME, DEVICE
from models.hadoop_storage import HadoopStorage

class KoreanCLIPModelWithHadoop:
    """
    한국어 CLIP 모델 클래스 (하둡 저장 기능 추가)
    텍스트와 이미지를 임베딩하고, 유사도를 계산하며, 하둡에 저장하는 기능 제공
    """
    
    def __init__(self, model_name=CLIP_MODEL_NAME, device=DEVICE, hdfs_host=None, hdfs_port=50070, hdfs_user="hadoop"):
        """
        CLIP 모델 및 하둡 스토리지 초기화
        
        Args:
            model_name (str): 사용할 CLIP 모델 이름 또는 경로
            device (str): 사용할 장치 ('cuda' 또는 'cpu')
            hdfs_host (str, optional): Hadoop 네임노드 호스트 주소
            hdfs_port (int): WebHDFS 포트 (기본값: 50070)
            hdfs_user (str): HDFS 사용자 이름 (기본값: 'hadoop')
        """
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
        
        # 하둡 스토리지 초기화 (호스트가 제공된 경우)
        self.hadoop_storage = None
        if hdfs_host:
            try:
                self.hadoop_storage = HadoopStorage(
                    hdfs_host=hdfs_host, 
                    hdfs_port=hdfs_port, 
                    hdfs_user=hdfs_user
                )
                logger.info(f"하둡 스토리지 초기화 완료 (host: {hdfs_host})")
            except Exception as e:
                logger.error(f"하둡 스토리지 초기화 실패: {str(e)}")
                # 하둡 스토리지 초기화 실패해도 모델은 사용 가능하게 함
    
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
            # 이미지 로드 (URL, 파일 경로 또는 PIL 이미지 객체)
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
    
    def calculate_similarity(self, embedding1, embedding2=None):
        """
        두 임베딩 간의 유사도 계산
        
        Args:
            embedding1 (numpy.ndarray): 첫 번째 임베딩
            embedding2 (numpy.ndarray, optional): 두 번째 임베딩 (없으면 자기 자신과 비교)
            
        Returns:
            float: 유사도 점수 (0~1 사이)
        """
        if embedding2 is None:
            # 자기 자신과의 유사도 계산
            similarity = np.dot(embedding1, embedding1.T)[0, 0]
        else:
            # 두 임베딩 간의 유사도 계산 (코사인 유사도)
            similarity = np.dot(embedding1, embedding2.T)[0, 0]
            
        # 유사도를 0~1 범위로 정규화
        similarity = (similarity + 1) / 2
        return float(similarity)
    
    def save_embedding_to_hadoop(self, item_id, text=None, image_source=None, metadata=None):
        """
        텍스트와 이미지를 인코딩하고 임베딩을 하둡에 저장
        
        Args:
            item_id (str): 항목 ID (파일명으로 사용)
            text (str, optional): 인코딩할 텍스트
            image_source (str or PIL.Image, optional): 인코딩할 이미지
            metadata (dict, optional): 저장할 추가 메타데이터
            
        Returns:
            str: 저장된 파일 경로
        """
        if not self.hadoop_storage:
            raise ValueError("하둡 스토리지가 초기화되지 않았습니다. 하둡 호스트를 제공하세요.")
        
        if not text and not image_source:
            raise ValueError("텍스트나 이미지 중 하나는 제공해야 합니다.")
        
        text_embedding = None
        image_embedding = None
        
        # 텍스트 인코딩
        if text:
            text_embedding = self.encode_text(text)
        
        # 이미지 인코딩
        if image_source:
            image_embedding = self.encode_image(image_source)
        
        # 메타데이터가 없으면 기본 정보 추가
        if metadata is None:
            metadata = {}
        
        if text:
            metadata['text'] = text
        
        if isinstance(image_source, str):
            metadata['image_source'] = image_source
        
        # 하둡에 저장
        file_path = self.hadoop_storage.save_embedding(
            item_id=item_id,
            text_embedding=text_embedding if text_embedding is not None else np.zeros((1, self.model.text_embed_dim)),
            image_embedding=image_embedding,
            metadata=metadata
        )
        
        return file_path
    
    def load_embedding_from_hadoop(self, item_id):
        """
        하둡에서 임베딩 데이터 로드
        
        Args:
            item_id (str): 항목 ID
            
        Returns:
            dict: 로드된 임베딩 데이터
        """
        if not self.hadoop_storage:
            raise ValueError("하둡 스토리지가 초기화되지 않았습니다.")
        
        return self.hadoop_storage.load_embedding(item_id)
    
    def search_similar_items_in_hadoop(self, 
                                       query_text=None, 
                                       query_image=None, 
                                       threshold=0.5, 
                                       limit=10):
        """
        하둡에 저장된 임베딩 중에서 유사한 항목 검색
        
        Args:
            query_text (str, optional): 검색할 텍스트
            query_image (str or PIL.Image, optional): 검색할 이미지
            threshold (float): 유사도 임계값
            limit (int): 반환할 최대 결과 수
            
        Returns:
            list: 유사한 항목 목록
        """
        if not self.hadoop_storage:
            raise ValueError("하둡 스토리지가 초기화되지 않았습니다.")
        
        if not query_text and not query_image:
            raise ValueError("텍스트나 이미지 쿼리 중 하나는 제공해야 합니다.")
        
        # 텍스트 기반 검색
        if query_text:
            query_embedding = self.encode_text(query_text)
            return self.hadoop_storage.search_similar_embeddings(
                query_embedding=query_embedding,
                threshold=threshold,
                limit=limit,
                embedding_type="text_embedding"
            )
        
        # 이미지 기반 검색
        if query_image:
            query_embedding = self.encode_image(query_image)
            return self.hadoop_storage.search_similar_embeddings(
                query_embedding=query_embedding,
                threshold=threshold,
                limit=limit,
                embedding_type="image_embedding"
            )

# 모듈 테스트용 코드
if __name__ == "__main__":
    # 테스트 코드
    import os
    
    # 테스트용 환경 변수 설정
    os.environ["HADOOP_HOST"] = "localhost"  # EC2 인스턴스의 IP로 변경 필요
    
    # CLIP 모델 초기화 (하둡 스토리지 포함)
    try:
        clip_model = KoreanCLIPModelWithHadoop(
            hdfs_host=os.environ.get("HADOOP_HOST")
        )
        
        # 샘플 텍스트 및 이미지 인코딩 및 저장
        sample_text = "검은색 지갑을 잃어버렸습니다. 현금과 카드가 들어있어요."
        # sample_image_path = "path/to/sample_image.jpg"  # 실제 이미지 경로로 변경 필요
        
        # 하둡에 임베딩 저장
        # file_path = clip_model.save_embedding_to_hadoop(
        #     item_id="sample_item_001",
        #     text=sample_text,
        #     image_source=sample_image_path,
        #     metadata={
        #         "category": "지갑",
        #         "item_name": "검은색 가죽 지갑"
        #     }
        # )
        # print(f"임베딩이 하둡에 저장됨: {file_path}")
        
        # 유사 항목 검색
        # similar_items = clip_model.search_similar_items_in_hadoop(
        #     query_text="검은색 가죽 지갑 분실",
        #     threshold=0.5,
        #     limit=5
        # )
        # print(f"유사한 항목 {len(similar_items)}개 찾음:")
        # for item in similar_items:
        #     print(f"  - {item['id']}: 유사도 {item['similarity']:.4f}")
        
        print("테스트 코드는 주석 처리됨: 실제 사용 시 주석 해제 필요")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")