import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional
import cv2

from models import CLIPModel, BLIPModel
from models.color_analyzer import ColorAnalyzer
from config import Config

class ImageComparator:
    """
    이미지 간 유사도를 비교하는 클래스
    """
    def __init__(self):
        """
        이미지 비교기 초기화
        """
        self.clip_model = CLIPModel()
        self.blip_model = BLIPModel()
        self.color_analyzer = ColorAnalyzer()
    
    def get_clip_features(self, image_path: str) -> torch.Tensor:
        """
        이미지의 CLIP 특성 추출
        
        Args:
            image_path: 이미지 경로
            
        Returns:
            torch.Tensor: CLIP 특성 벡터
        """
        return self.clip_model.get_image_features(image_path)
    
    def get_blip_features(self, image_path: str) -> torch.Tensor:
        """
        이미지의 BLIP 특성 추출
        
        Args:
            image_path: 이미지 경로
            
        Returns:
            torch.Tensor: BLIP 특성 벡터
        """
        return self.blip_model.get_caption_features(image_path)
    
    def calculate_clip_similarity(self, image1_path: str, image2_path: str) -> float:
        """
        CLIP 모델을 사용한 이미지 유사도 계산
        
        Args:
            image1_path: 첫 번째 이미지 경로
            image2_path: 두 번째 이미지 경로
            
        Returns:
            float: 유사도 점수 (0~1)
        """
        features1 = self.get_clip_features(image1_path)
        features2 = self.get_clip_features(image2_path)
        
        if features1 is None or features2 is None:
            return 0.0
        
        # 코사인 유사도 계산
        similarity = torch.nn.functional.cosine_similarity(
            features1.unsqueeze(0), features2.unsqueeze(0)
        ).item()
        
        return max(0, similarity)  # 음수 방지
    
    def calculate_blip_similarity(self, image1_path: str, image2_path: str) -> float:
        """
        BLIP 모델을 사용한 이미지 유사도 계산
        
        Args:
            image1_path: 첫 번째 이미지 경로
            image2_path: 두 번째 이미지 경로
            
        Returns:
            float: 유사도 점수 (0~1)
        """
        return self.blip_model.calculate_visual_textual_similarity(image1_path, image2_path)
    
    def calculate_color_similarity(self, image1_path: str, image2_path: str) -> float:
        """
        캡션 기반 색상 유사도 계산 (BLIP 캡션 활용)
        
        Args:
            image1_path: 첫 번째 이미지 경로
            image2_path: 두 번째 이미지 경로
            
        Returns:
            float: 유사도 점수 (0~1)
        """
        # 각 이미지의 캡션 생성
        captions1 = self.blip_model.generate_captions(image1_path)
        captions2 = self.blip_model.generate_captions(image2_path)
        
        if not captions1 or not captions2:
            return 0.0
        
        # 캡션에서 색상 분석
        colors1 = self.color_analyzer.analyze_color_from_captions(captions1)
        colors2 = self.color_analyzer.analyze_color_from_captions(captions2)
        
        # 색상 유사도 계산
        return self.color_analyzer.calculate_color_similarity(colors1, colors2)
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """
        이미지 분류 및 특성 추출
        
        Args:
            image_path: 이미지 경로
            
        Returns:
            Dict[str, Any]: 이미지 분류 결과 및 특성
        """
        # CLIP을 사용한 이미지 분류
        clip_results = self.clip_model.classify_image(image_path, topk=5)
        
        # BLIP을 사용한 이미지 내용 분석
        blip_results = self.blip_model.analyze_image_content(image_path)
        
        # BLIP 캡션에서 색상 추출
        captions = []
        if isinstance(blip_results, dict) and 'captions' in blip_results:
            captions = blip_results['captions']
        elif isinstance(blip_results, list):
            captions = blip_results
        
        color_results = self.color_analyzer.analyze_color_from_captions(captions)
        
        # 결과 종합
        return {
            'clip_classification': clip_results,
            'blip_analysis': blip_results,
            'color_analysis': color_results,
            'primary_category': next(iter(clip_results.items()))[0] if clip_results else None,
            'primary_caption': blip_results.get('primary_caption', '') if isinstance(blip_results, dict) else '',
            'primary_color': color_results.get('primary_color', '')
        }
    
    def calculate_combined_similarity(self, image1_path: str, image2_path: str,
                                     category_match: bool = False) -> Dict[str, float]:
        """
        여러 방법을 조합한 이미지 유사도 계산
        
        Args:
            image1_path: 첫 번째 이미지 경로
            image2_path: 두 번째 이미지 경로
            category_match: 카테고리 일치 여부
            
        Returns:
            Dict[str, float]: 유사도 계산 결과
        """
        # 각 모델별 유사도 계산
        clip_similarity = self.calculate_clip_similarity(image1_path, image2_path)
        blip_similarity = self.calculate_blip_similarity(image1_path, image2_path)
        color_similarity = self.calculate_color_similarity(image1_path, image2_path)
        
        # 가중치 설정
        weights = Config.WEIGHTS.copy()
        
        # 카테고리가 일치하면 색상 가중치 증가
        if category_match:
            weights['color_similarity'] += Config.COLOR_WEIGHT_BOOST
            # 가중치 합이 1이 되도록 조정
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
        
        # 가중 평균으로 최종 유사도 계산
        combined_similarity = (
            weights['clip_similarity'] * clip_similarity +
            weights['blip_similarity'] * blip_similarity +
            weights['color_similarity'] * color_similarity
        )
        
        return {
            'clip_similarity': clip_similarity,
            'blip_similarity': blip_similarity,
            'color_similarity': color_similarity,
            'combined_similarity': combined_similarity
        }