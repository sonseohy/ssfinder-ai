from typing import Dict, List, Tuple, Any, Optional
import torch
import numpy as np

from .category_matcher import CategoryMatcher
from .image_comparator import ImageComparator
from config import Config

class SimilarityCalculator:
    """
    분실물 게시글 간 종합적인 유사도 계산을 위한 클래스
    """
    def __init__(self):
        """
        유사도 계산기 초기화
        """
        self.category_matcher = CategoryMatcher()
        self.image_comparator = ImageComparator()
    
    def extract_post_features(self, post: Dict[str, Any]) -> Dict[str, Any]:
        """
        게시글에서 특성 추출
        
        Args:
            post: 게시글 정보 (제목, 내용, 이미지 경로 등)
            
        Returns:
            Dict[str, Any]: 추출된 특성
        """
        features = {}
        
        # 텍스트 관련 특성 추출
        if 'title' in post:
            features['title_keywords'] = self.category_matcher.extract_keywords_from_text(post['title'])
        
        if 'content' in post:
            features['content_keywords'] = self.category_matcher.extract_keywords_from_text(post['content'])
        
        # 카테고리 추출
        text_for_category = ' '.join([post.get('title', ''), post.get('content', '')])
        category, category_score = self.category_matcher.get_category_from_text(text_for_category)
        features['category'] = category
        features['category_score'] = category_score
        
        # 이미지 관련 특성 추출
        if 'image_path' in post and post['image_path']:
            image_features = self.image_comparator.classify_image(post['image_path'])
            
            features['clip_results'] = image_features.get('clip_classification', {})
            features['blip_results'] = image_features.get('blip_analysis', {})
            features['color_results'] = image_features.get('color_analysis', {})
            features['primary_category'] = image_features.get('primary_category', None)
            features['primary_caption'] = image_features.get('primary_caption', '')
            features['primary_color'] = image_features.get('primary_color', '')
        
        return features
    
    def calculate_post_similarity(self, query_post: Dict[str, Any], 
                                 candidate_post: Dict[str, Any]) -> Dict[str, float]:
        """
        두 게시글 간의 유사도 계산
        
        Args:
            query_post: 질의 게시글
            candidate_post: 비교할 게시글
            
        Returns:
            Dict[str, float]: 유사도 계산 결과
        """
        similarity_scores = {}
        
        # 카테고리 유사도
        query_category = query_post.get('category', None)
        candidate_category = candidate_post.get('category', None)
        
        category_match = False
        if query_category and candidate_category:
            category_match = (query_category == candidate_category)
            similarity_scores['category_similarity'] = 1.0 if category_match else 0.0
        else:
            similarity_scores['category_similarity'] = 0.0
        
        # 키워드 유사도
        query_text = ' '.join([query_post.get('title', ''), query_post.get('content', '')])
        candidate_text = ' '.join([candidate_post.get('title', ''), candidate_post.get('content', '')])
        
        keyword_similarity = self.category_matcher.calculate_keyword_similarity(query_text, candidate_text)
        similarity_scores['keyword_similarity'] = keyword_similarity
        
        # 이미지 유사도
        if 'image_path' in query_post and 'image_path' in candidate_post:
            image_similarities = self.image_comparator.calculate_combined_similarity(
                query_post['image_path'],
                candidate_post['image_path'],
                category_match
            )
            
            similarity_scores.update(image_similarities)
        else:
            similarity_scores['clip_similarity'] = 0.0
            similarity_scores['blip_similarity'] = 0.0
            similarity_scores['color_similarity'] = 0.0
            similarity_scores['combined_similarity'] = 0.0
        
        # 최종 유사도 계산
        weights = Config.WEIGHTS
        
        final_similarity = (
            weights['category_match'] * similarity_scores['category_similarity'] +
            weights['keyword_match'] * similarity_scores['keyword_similarity'] +
            weights['clip_similarity'] * similarity_scores.get('clip_similarity', 0.0) +
            weights['blip_similarity'] * similarity_scores.get('blip_similarity', 0.0) +
            weights['color_similarity'] * similarity_scores.get('color_similarity', 0.0)
        )
        
        similarity_scores['final_similarity'] = final_similarity
        
        return similarity_scores
    
    def find_similar_posts(self, query_post: Dict[str, Any], 
                          candidate_posts: List[Dict[str, Any]],
                          threshold: float = None,
                          max_results: int = None) -> List[Tuple[Dict[str, Any], float]]:
        """
        질의 게시글과 유사한 게시글 찾기
        
        Args:
            query_post: 질의 게시글
            candidate_posts: 후보 게시글 목록
            threshold: 유사도 임계값 (기본값은 Config에서 가져옴)
            max_results: 최대 결과 수 (기본값은 Config에서 가져옴)
            
        Returns:
            List[Tuple[Dict[str, Any], float]]: (게시글, 유사도) 튜플 목록
        """
        threshold = threshold or Config.SIMILARITY_THRESHOLD
        max_results = max_results or Config.MAX_RECOMMENDATIONS
        
        # 질의 게시글 특성 추출
        query_features = self.extract_post_features(query_post)
        query_post_with_features = {**query_post, **query_features}
        
        results = []
        
        for candidate in candidate_posts:
            # 후보 게시글 특성 추출
            candidate_features = self.extract_post_features(candidate)
            candidate_with_features = {**candidate, **candidate_features}
            
            # 유사도 계산
            similarity = self.calculate_post_similarity(query_post_with_features, candidate_with_features)
            final_similarity = similarity['final_similarity']
            
            # 임계값 이상인 경우만 결과에 추가
            if final_similarity >= threshold:
                results.append((candidate, final_similarity))
        
        # 유사도 내림차순 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        
        # 최대 결과 수 제한
        return results[:max_results]