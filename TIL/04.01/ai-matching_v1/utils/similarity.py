import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Union, Tuple, Optional
import logging
import re

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cosine_sim(emb1: Union[torch.Tensor, np.ndarray], emb2: Union[torch.Tensor, np.ndarray]) -> float:
    """
    두 임베딩 간의 코사인 유사도 계산
    
    Args:
        emb1: 첫 번째 임베딩
        emb2: 두 번째 임베딩
        
    Returns:
        float: 코사인 유사도 (0-1 범위)
    """
    # 텐서를 numpy 배열로 변환
    if isinstance(emb1, torch.Tensor):
        emb1 = emb1.detach().cpu().numpy()
    if isinstance(emb2, torch.Tensor):
        emb2 = emb2.detach().cpu().numpy()
    
    # 형태 확인 및 조정
    if len(emb1.shape) == 1:
        emb1 = emb1.reshape(1, -1)
    if len(emb2.shape) == 1:
        emb2 = emb2.reshape(1, -1)
    
    # 코사인 유사도 계산
    sim = cosine_similarity(emb1, emb2)[0][0]
    
    # 유사도를 0-1 범위로 정규화 (코사인 유사도는 -1에서 1 사이)
    return (sim + 1) / 2

def normalized_dot_product(emb1: Union[torch.Tensor, np.ndarray], emb2: Union[torch.Tensor, np.ndarray]) -> float:
    """
    정규화된 내적 계산 (코사인 유사도와 유사하지만 약간 다름)
    
    Args:
        emb1: 첫 번째 임베딩
        emb2: 두 번째 임베딩
        
    Returns:
        float: 정규화된 내적 (0-1 범위)
    """
    # 텐서를 numpy 배열로 변환
    if isinstance(emb1, torch.Tensor):
        emb1 = emb1.detach().cpu().numpy()
    if isinstance(emb2, torch.Tensor):
        emb2 = emb2.detach().cpu().numpy()
    
    # 형태 확인 및 조정
    if len(emb1.shape) == 1:
        emb1 = emb1.reshape(-1)
    if len(emb2.shape) == 1:
        emb2 = emb2.reshape(-1)
    
    # L2 정규화
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    
    # 내적 계산
    similarity = np.dot(emb1_norm, emb2_norm)
    
    # 유사도 정규화 (0-1 범위)
    return (similarity + 1) / 2

def text_similarity(text1: str, text2: str) -> float:
    """
    두 텍스트 간의 문자열 유사도 계산
    
    Args:
        text1: 첫 번째 텍스트
        text2: 두 번째 텍스트
        
    Returns:
        float: 텍스트 유사도 (0-1 범위)
    """
    if not text1 or not text2:
        return 0.0
    
    # 전처리
    text1 = clean_text(text1).lower()
    text2 = clean_text(text2).lower()
    
    # 정확히 일치하면 1.0
    if text1 == text2:
        return 1.0
    
    # 부분 문자열 매칭
    if text1 in text2 or text2 in text1:
        return 0.8
    
    # 단어 수준의 유사도
    words1 = set(text1.split())
    words2 = set(text2.split())
    
    # 특수 케이스: 매우 짧은 텍스트
    if len(words1) < 2 or len(words2) < 2:
        # 단어 단위로 일치 확인 
        for w1 in words1:
            for w2 in words2:
                if w1 == w2:
                    return 0.7
                if w1 in w2 or w2 in w1:
                    return 0.5
        return 0.1
    
    # Jaccard 유사도
    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def clean_text(text: str) -> str:
    """
    텍스트 정리 및 정규화
    
    Args:
        text: 정리할 텍스트
        
    Returns:
        str: 정리된 텍스트
    """
    if not text:
        return ""
    
    # 특수 문자 제거, 공백 정규화
    cleaned = re.sub(r'[^\w\s가-힣]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned

def attribute_similarity(attrs1: Dict[str, str], attrs2: Dict[str, str]) -> Dict[str, float]:
    """
    두 속성 집합 간의 유사도 계산
    
    Args:
        attrs1: 첫 번째 속성 사전
        attrs2: 두 번째 속성 사전
        
    Returns:
        Dict[str, float]: 각 속성별 유사도
    """
    results = {}
    
    # 공통 속성에 대해 유사도 계산
    for key in set(attrs1.keys()) & set(attrs2.keys()):
        results[key] = text_similarity(attrs1[key], attrs2[key])
    
    # 누락된 속성에 대해 0 설정
    for key in set(attrs1.keys()) | set(attrs2.keys()):
        if key not in results:
            results[key] = 0.0
    
    return results

def weighted_attribute_similarity(attrs1: Dict[str, str], attrs2: Dict[str, str], 
                                weights: Optional[Dict[str, float]] = None) -> float:
    """
    가중치가 적용된 속성 유사도 계산
    
    Args:
        attrs1: 첫 번째 속성 사전
        attrs2: 두 번째 속성 사전
        weights: 속성별 가중치 (기본값: 모두 동일)
        
    Returns:
        float: 가중 평균 유사도
    """
    # 기본 가중치 설정
    if weights is None:
        weights = {
            'color': 0.2,
            'brand': 0.3,
            'type': 0.3,
            'material': 0.1,
            'condition': 0.1
        }
    
    # 각 속성별 유사도 계산
    sim_scores = attribute_similarity(attrs1, attrs2)
    
    # 가중 평균 계산
    weighted_sum = 0.0
    total_weight = 0.0
    
    for attr, weight in weights.items():
        if attr in sim_scores:
            weighted_sum += sim_scores[attr] * weight
            total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return weighted_sum / total_weight

def combined_similarity(query_item: Dict[str, Any], candidate_item: Dict[str, Any], 
                      weights: Optional[Dict[str, float]] = None) -> float:
    """
    여러 유사도 측정값을 결합한 종합 유사도 계산
    
    Args:
        query_item: 쿼리 아이템
        candidate_item: 후보 아이템
        weights: 각 유사도 측정값에 대한 가중치
        
    Returns:
        float: 종합 유사도 점수
    """
    # 기본 가중치 설정
    if weights is None:
        weights = {
            'clip_similarity': 0.4,
            'attribute_similarity': 0.4,
            'text_similarity': 0.2
        }
    
    similarities = {}
    
    # CLIP 임베딩 유사도
    if 'clip_embedding' in query_item and 'clip_embedding' in candidate_item:
        similarities['clip_similarity'] = cosine_sim(
            query_item['clip_embedding'], 
            candidate_item['clip_embedding']
        )
    else:
        similarities['clip_similarity'] = 0.0
    
    # 속성 유사도
    if 'blip_attributes' in query_item and 'blip_attributes' in candidate_item:
        query_attrs = query_item['blip_attributes'].get('attributes', {})
        candidate_attrs = candidate_item['blip_attributes'].get('attributes', {})
        
        attr_weights = {
            'color': 0.2,
            'brand': 0.3,
            'type': 0.3,
            'material': 0.1,
            'condition': 0.1
        }
        
        similarities['attribute_similarity'] = weighted_attribute_similarity(
            query_attrs, candidate_attrs, attr_weights
        )
    else:
        similarities['attribute_similarity'] = 0.0
    
    # 텍스트 설명 유사도
    query_text = ""
    candidate_text = ""
    
    # 텍스트 필드 수집
    text_fields = ['caption', 'clean_subject', 'clean_product_name', 'fdSbjt', 'fdPrdtNm']
    
    for field in text_fields:
        if field in query_item:
            query_text += " " + str(query_item[field])
        if field in query_item.get('blip_attributes', {}):
            query_text += " " + str(query_item['blip_attributes'][field])
        
        if field in candidate_item:
            candidate_text += " " + str(candidate_item[field])
        if field in candidate_item.get('blip_attributes', {}):
            candidate_text += " " + str(candidate_item['blip_attributes'][field])
    
    similarities['text_similarity'] = text_similarity(query_text, candidate_text)
    
    # 종합 유사도 계산
    weighted_sum = 0.0
    total_weight = 0.0
    
    for sim_type, weight in weights.items():
        if sim_type in similarities:
            weighted_sum += similarities[sim_type] * weight
            total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return weighted_sum / total_weight

if __name__ == "__main__":
    # 간단한 테스트
    
    # 임베딩 유사도 테스트
    emb1 = np.random.rand(512)
    emb2 = np.random.rand(512)
    sim = cosine_sim(emb1, emb2)
    print(f"랜덤 임베딩 간 코사인 유사도: {sim:.4f}")
    
    # 유사 임베딩
    emb3 = emb1 + 0.1 * np.random.rand(512)  # 약간의 노이즈 추가
    sim = cosine_sim(emb1, emb3)
    print(f"유사 임베딩 간 코사인 유사도: {sim:.4f}")
    
    # 텍스트 유사도 테스트
    text1 = "검정색 갤럭시 스마트폰"
    text2 = "검정 갤럭시 휴대폰"
    text3 = "흰색 아이폰"
    
    print(f"텍스트 유사도 (유사): {text_similarity(text1, text2):.4f}")
    print(f"텍스트 유사도 (다름): {text_similarity(text1, text3):.4f}")
    
    # 속성 유사도 테스트
    attrs1 = {'color': '검정', 'brand': '삼성', 'type': '휴대폰'}
    attrs2 = {'color': '검정색', 'brand': '갤럭시', 'type': '스마트폰'}
    attrs3 = {'color': '흰색', 'brand': '애플', 'type': '아이폰'}
    
    print(f"속성 유사도 (유사):")
    for k, v in attribute_similarity(attrs1, attrs2).items():
        print(f"  {k}: {v:.4f}")
    
    print(f"속성 유사도 (다름):")
    for k, v in attribute_similarity(attrs1, attrs3).items():
        print(f"  {k}: {v:.4f}")
    
    print(f"가중 속성 유사도 (유사): {weighted_attribute_similarity(attrs1, attrs2):.4f}")
    print(f"가중 속성 유사도 (다름): {weighted_attribute_similarity(attrs1, attrs3):.4f}")