"""
분실물 매칭 서비스의 유틸리티 함수
"""
import numpy as np
import torch
from PIL import Image
import io
import logging
import os
from datetime import datetime
import pyarrow.hdfs as hdfs
from transformers import CLIPProcessor, CLIPModel

# 로깅 설정
def setup_logging(log_dir="./logs", level=logging.INFO):
    """로깅 설정"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"lost_found_{timestamp}.log")
    
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger("lost_found")

# CLIP 모델 관련 함수
def load_clip_model(model_name="openai/clip-vit-base-patch32"):
    """CLIP 모델 로드"""
    try:
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        return model, processor
    except Exception as e:
        logging.error(f"CLIP 모델 로드 실패: {str(e)}")
        return None, None

def create_image_embedding(image_path, model, processor):
    """이미지에서 CLIP 임베딩 생성"""
    try:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            image_features = model.get_image_features(inputs["pixel_values"])
            
        # 정규화된 임베딩 반환
        embedding = image_features.cpu().numpy()[0]
        normalized_embedding = embedding / np.linalg.norm(embedding)
        return normalized_embedding
    except Exception as e:
        logging.error(f"이미지 임베딩 생성 실패: {str(e)}")
        return None

def create_text_embedding(text, model, processor):
    """텍스트에서 CLIP 임베딩 생성"""
    try:
        inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            text_features = model.get_text_features(inputs["input_ids"])
            
        # 정규화된 임베딩 반환
        embedding = text_features.cpu().numpy()[0]
        normalized_embedding = embedding / np.linalg.norm(embedding)
        return normalized_embedding
    except Exception as e:
        logging.error(f"텍스트 임베딩 생성 실패: {str(e)}")
        return None

# HDFS 관련 유틸리티
def connect_to_hdfs(hdfs_host="nn1", hdfs_port=9870, user="hadoop"):
    """HDFS 연결"""
    try:
        fs = hdfs.connect(host=hdfs_host, port=hdfs_port, user=user)
        return fs
    except Exception as e:
        logging.error(f"HDFS 연결 실패: {str(e)}")
        return None

def save_embedding_to_hdfs(fs, embedding, metadata, hdfs_path):
    """임베딩을 HDFS에 저장"""
    try:
        # 메타데이터와 임베딩을 함께 저장
        data = {
            "metadata": metadata,
            "embedding": embedding.tolist()
        }
        
        # 파일명 생성
        filename = f"{metadata['id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}.npy"
        full_path = os.path.join(hdfs_path, filename)
        
        # NumPy 배열로 변환하여 저장
        with fs.open(full_path, 'wb') as f:
            np.save(f, np.array(data))
            
        return full_path
    except Exception as e:
        logging.error(f"임베딩 HDFS 저장 실패: {str(e)}")
        return None

# 유사도 관련 유틸리티
def cosine_similarity(vec1, vec2):
    """코사인 유사도 계산"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return float(dot_product / (norm1 * norm2))

def euclidean_distance(vec1, vec2):
    """유클리드 거리 계산"""
    return float(np.linalg.norm(vec1 - vec2))

def euclidean_to_similarity(distance, max_distance=2.0):
    """유클리드 거리를 유사도(0~1)로 변환"""
    # 거리가 0이면 유사도는 1
    # 거리가 max_distance 이상이면 유사도는 0
    similarity = max(0, 1 - (distance / max_distance))
    return similarity

# 결과 포매팅 유틸리티
def format_results(similar_items, include_embedding=False):
    """결과를 포매팅하여 반환"""
    formatted_results = []
    
    for item in similar_items:
        result = {
            "id": item["id"],
            "title": item["title"],
            "description": item["description"],
            "image_path": item["image_path"],
            "similarity_score": f"{item['similarity']:.2f}%",
            "match_quality": get_match_quality(item["similarity"])
        }
        
        if include_embedding and "embedding" in item:
            result["embedding"] = item["embedding"]
            
        formatted_results.append(result)
        
    return formatted_results

def get_match_quality(similarity_percentage):
    """유사도 점수를 기반으로 매치 품질 텍스트 반환"""
    if similarity_percentage >= 95:
        return "매우 높음"
    elif similarity_percentage >= 85:
        return "높음"
    elif similarity_percentage >= 75:
        return "중간"
    elif similarity_percentage >= 65:
        return "낮음"
    else:
        return "매우 낮음"