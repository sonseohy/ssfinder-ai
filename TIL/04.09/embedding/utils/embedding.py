"""
임베딩 생성 및 관련 유틸리티 함수
텍스트와 이미지로부터 CLIP 임베딩을 생성하고 통합하는 기능 제공
"""
import os
import sys
import logging
import numpy as np
from typing import Dict, Any, Optional, Union
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 추가하여 config.py 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import COMBINE_METHOD, EMBEDDING_DIMENSION

def preprocess_post_content(post: Dict[str, Any]) -> str:
    """
    게시글 내용 전처리 함수
    
    Args:
        post (dict): 게시글 정보
        
    Returns:
        str: 전처리된 텍스트
    """
    # 카테고리, 제목, 내용을 결합하여 텍스트 임베딩 생성
    processed_text = f"{post.get('category', '')} {post.get('title', '')} {post.get('content', '')}"
    
    # 추가 전처리가 필요한 경우 여기에 구현
    return processed_text.strip()

def generate_post_embedding(post: Dict[str, Any], clip_model) -> Dict[str, Any]:
    """
    게시글로부터 임베딩 생성
    
    Args:
        post (dict): 게시글 정보 
        clip_model: CLIP 모델 인스턴스
        
    Returns:
        dict: 생성된 임베딩 정보
    """
    logger.info(f"게시글 임베딩 생성 시작: {post.get('title', '제목 없음')}")
    
    # 결과 초기화
    result = {
        "post_id": post.get("id", None),  # 게시글 ID (있는 경우)
        "has_image": False,
        "embeddings": {},
        "metadata": {
            "title": post.get("title", ""),
            "category": post.get("category", ""),
            "content_length": len(post.get("content", "")),
            "model_name": clip_model.model_name if clip_model else "unknown"
        }
    }
    
    # 1. 텍스트 임베딩 생성
    try:
        processed_text = preprocess_post_content(post)
        if clip_model and processed_text:
            text_embedding = clip_model.encode_text(processed_text)
            result["embeddings"]["text"] = text_embedding
            logger.debug(f"텍스트 임베딩 생성 완료: shape={text_embedding.shape}")
        else:
            logger.warning("텍스트 임베딩을 생성할 수 없습니다: 텍스트 부재 또는 모델 없음")
            result["embeddings"]["text"] = np.zeros((1, EMBEDDING_DIMENSION))
    except Exception as e:
        logger.error(f"텍스트 임베딩 생성 중 오류: {str(e)}")
        result["embeddings"]["text"] = np.zeros((1, EMBEDDING_DIMENSION))
    
    # 2. 이미지 임베딩 생성 (이미지가 있는 경우)
    try:
        image_url = post.get("image_url")
        if clip_model and image_url:
            image_embedding = clip_model.encode_image(image_url)
            result["embeddings"]["image"] = image_embedding
            result["has_image"] = True
            logger.debug(f"이미지 임베딩 생성 완료: shape={image_embedding.shape}")
        else:
            logger.info("이미지 임베딩 생성 건너뜀: 이미지 없음 또는 모델 없음")
            if "text" in result["embeddings"]:
                # 텍스트 임베딩과 같은 크기로 초기화
                result["embeddings"]["image"] = np.zeros_like(result["embeddings"]["text"])
            else:
                result["embeddings"]["image"] = np.zeros((1, EMBEDDING_DIMENSION))
    except Exception as e:
        logger.error(f"이미지 임베딩 생성 중 오류: {str(e)}")
        if "text" in result["embeddings"]:
            result["embeddings"]["image"] = np.zeros_like(result["embeddings"]["text"])
        else:
            result["embeddings"]["image"] = np.zeros((1, EMBEDDING_DIMENSION))
    
    # 3. 통합 임베딩 생성 (텍스트 + 이미지)
    try:
        text_emb = result["embeddings"]["text"]
        image_emb = result["embeddings"]["image"]
        
        if COMBINE_METHOD == "mean":
            # 평균 임베딩 계산
            if result["has_image"]:
                combined_embedding = (text_emb + image_emb) / 2
            else:
                combined_embedding = text_emb
                
        elif COMBINE_METHOD == "concat":
            # 텍스트와 이미지 임베딩 연결 (차원 증가)
            combined_embedding = np.concatenate([text_emb, image_emb], axis=1)
            
        else:
            # 기본적으로 텍스트 임베딩 사용
            combined_embedding = text_emb
            
        result["embeddings"]["combined"] = combined_embedding
        logger.debug(f"통합 임베딩 생성 완료: shape={combined_embedding.shape}")
    except Exception as e:
        logger.error(f"통합 임베딩 생성 중 오류: {str(e)}")
        # 텍스트 임베딩을 기본값으로 사용
        result["embeddings"]["combined"] = result["embeddings"]["text"]
    
    logger.info(f"게시글 임베딩 생성 완료: {post.get('title', '제목 없음')}")
    
    return result

def save_embedding_to_file(embedding_result: Dict[str, Any], file_path: str) -> None:
    """
    생성된 임베딩을 파일로 저장
    
    Args:
        embedding_result (dict): 임베딩 결과
        file_path (str): 저장할 파일 경로
    """
    try:
        # NumPy 배열을 리스트로 변환
        serializable_result = embedding_result.copy()
        for key, value in serializable_result["embeddings"].items():
            if isinstance(value, np.ndarray):
                serializable_result["embeddings"][key] = value.tolist()
        
        # 디렉토리가 없으면 생성
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # JSON 파일로 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, ensure_ascii=False, indent=2)
            
        logger.info(f"임베딩이 파일에 저장되었습니다: {file_path}")
    except Exception as e:
        logger.error(f"임베딩 저장 중 오류 발생: {str(e)}")

def load_embedding_from_file(file_path: str) -> Dict[str, Any]:
    """
    파일에서 임베딩 로드
    
    Args:
        file_path (str): 로드할 파일 경로
        
    Returns:
        dict: 로드된 임베딩 정보
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            embedding_result = json.load(f)
        
        # 리스트를 NumPy 배열로 변환
        for key, value in embedding_result["embeddings"].items():
            embedding_result["embeddings"][key] = np.array(value)
            
        logger.info(f"임베딩이 파일에서 로드되었습니다: {file_path}")
        return embedding_result
    except Exception as e:
        logger.error(f"임베딩 로드 중 오류 발생: {str(e)}")
        return {
            "has_image": False,
            "embeddings": {
                "text": np.zeros((1, EMBEDDING_DIMENSION)),
                "image": np.zeros((1, EMBEDDING_DIMENSION)),
                "combined": np.zeros((1, EMBEDDING_DIMENSION))
            },
            "metadata": {}
        }

def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    두 임베딩 벡터 간의 코사인 유사도 계산
    
    Args:
        embedding1 (np.ndarray): 첫 번째 임베딩 벡터
        embedding2 (np.ndarray): 두 번째 임베딩 벡터
        
    Returns:
        float: 코사인 유사도 (0~1 사이)
    """
    # 벡터 정규화
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # 코사인 유사도 계산
    cos_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    # 결과를 0~1 범위로 조정
    return float((cos_sim + 1) / 2)

def find_similar_posts(query_embedding: np.ndarray, post_embeddings: Dict[str, np.ndarray], 
                      threshold: float = 0.7, limit: int = 10) -> list:
    """
    쿼리 임베딩과 유사한 게시글 임베딩 찾기
    
    Args:
        query_embedding (np.ndarray): 쿼리 임베딩 벡터
        post_embeddings (Dict[str, np.ndarray]): 게시글 ID와 임베딩 벡터의 사전
        threshold (float): 유사도 임계값 (기본값: 0.7)
        limit (int): 반환할 최대 결과 수 (기본값: 10)
        
    Returns:
        list: 유사도가 임계값 이상인 게시글 ID와 유사도의 리스트 (유사도 높은 순)
    """
    results = []
    
    for post_id, embedding in post_embeddings.items():
        similarity = calculate_similarity(query_embedding, embedding)
        
        if similarity >= threshold:
            results.append({
                "post_id": post_id,
                "similarity": similarity
            })
    
    # 유사도 높은 순으로 정렬하고 제한
    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:limit]

# 모듈 테스트용 코드
if __name__ == "__main__":
    # 샘플 게시글
    sample_post = {
        "category": "질문",
        "title": "CLIP 모델에 대해 질문합니다",
        "content": "CLIP 모델은 어떤 방식으로 텍스트와 이미지를 같은 임베딩 공간에 매핑하나요?"
    }
    
    # 임베딩 테스트 (모델 없이)
    print("임베딩 테스트:")
    embedding_result = generate_post_embedding(sample_post, None)
    print(f"결과 키: {list(embedding_result.keys())}")
    print(f"메타데이터: {embedding_result['metadata']}")
    
    # 유사도 테스트
    print("\n유사도 테스트:")
    vec1 = np.random.rand(1, EMBEDDING_DIMENSION)
    vec2 = np.random.rand(1, EMBEDDING_DIMENSION)
    sim = calculate_similarity(vec1, vec2)
    print(f"랜덤 벡터 간 유사도: {sim:.4f}")