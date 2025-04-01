import re
import torch
from PIL import Image
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set

from config import Config

class ColorAnalyzer:
    """
    CLIP 및 BLIP 모델의 캡션 정보를 활용한 스마트 색상 분석기
    """
    def __init__(self):
        """
        스마트 색상 분석기 초기화
        """
        # 색상 사전 정의 (영어)
        self.color_names_en = {
            'red': ['red', 'crimson', 'maroon', 'burgundy', 'scarlet', 'ruby'],
            'orange': ['orange', 'amber', 'coral', 'peach'],
            'yellow': ['yellow', 'gold', 'golden', 'lemon', 'mustard'],
            'green': ['green', 'olive', 'lime', 'emerald', 'mint', 'jade', 'teal'],
            'blue': ['blue', 'navy', 'azure', 'cyan', 'teal', 'turquoise', 'indigo'],
            'purple': ['purple', 'violet', 'magenta', 'lavender', 'lilac', 'mauve', 'plum'],
            'pink': ['pink', 'rose', 'salmon', 'fuchsia', 'hot pink'],
            'brown': ['brown', 'chocolate', 'coffee', 'tan', 'bronze', 'caramel', 'khaki', 'beige'],
            'white': ['white', 'ivory', 'cream', 'eggshell', 'pearl', 'snow'],
            'gray': ['gray', 'grey', 'silver', 'ash', 'charcoal', 'slate'],
            'black': ['black', 'ebony', 'jet', 'onyx', 'charcoal']
        }
        
        # 색상 사전 정의 (한국어)
        self.color_names_ko = {
            'red': ['빨간', '빨강', '빨간색', '적색', '붉은', '자주', '진홍색', '주홍색'],
            'orange': ['주황', '주황색', '오렌지', '오렌지색', '귤색'],
            'yellow': ['노란', '노랑', '노란색', '황색', '금색'],
            'green': ['초록', '초록색', '녹색', '연두', '연두색', '올리브', '비취', '민트', '민트색'],
            'blue': ['파란', '파랑', '파란색', '청색', '남색', '쪽색', '하늘색', '하늘', '옥색'],
            'purple': ['보라', '보라색', '자주', '자주색', '퍼플', '라벤더', '라벤더색'],
            'pink': ['분홍', '분홍색', '핑크', '핑크색', '살구색'],
            'brown': ['갈색', '밤색', '초콜릿색', '황갈색', '카키색', '브라운'],
            'white': ['흰', '하얀', '하얀색', '흰색', '백색', '화이트'],
            'gray': ['회색', '그레이', '회', '회갈색', '실버', '은색'],
            'black': ['검정', '검은', '검정색', '검은색', '흑색', '블랙']
        }
        
        # 흔한 색상 목록 (검정, 흰색, 회색)
        self.common_colors = {'black', 'white', 'gray'}
        
    def extract_colors_from_caption(self, caption: str) -> List[Tuple[str, float]]:
        """
        캡션에서 색상 정보 추출
        
        Args:
            caption: 이미지 캡션 텍스트
            
        Returns:
            List[Tuple[str, float]]: (색상, 신뢰도) 목록
        """
        caption = caption.lower()
        
        # 한국어 텍스트 여부 확인
        is_korean = any('\uAC00' <= char <= '\uD7A3' for char in caption)
        
        # 언어에 따라 색상 사전 선택
        color_dict = self.color_names_ko if is_korean else self.color_names_en
        
        # 발견된 색상 저장
        found_colors = {}
        
        for primary_color, variants in color_dict.items():
            for color in variants:
                # 색상 단어가 캡션에 있는지 확인
                if re.search(r'\b{}\b'.format(color), caption):
                    # 기본 신뢰도 설정
                    confidence = 0.8
                    
                    # 신뢰도 조정 요인
                    
                    # 1. 색상이 캡션 앞 부분에 등장하면 신뢰도 상승
                    position = caption.find(color) / len(caption)
                    if position < 0.3:
                        confidence += 0.1
                    
                    # 2. 색상 단어 앞에 "is", "in", "a" 등의 단어가 있으면 신뢰도 상승
                    if re.search(r'\b(is|in|a|the|with|and|of)\s+{}\b'.format(color), caption):
                        confidence += 0.05
                    
                    # 3. 영어의 경우 색상 단어가 형용사로 사용되면 신뢰도 상승
                    if not is_korean and re.search(r'\b{}\s+(object|item|thing|device|phone|bag)\b'.format(color), caption):
                        confidence += 0.1
                    
                    # 기존 색상보다 높은 신뢰도만 저장
                    if primary_color not in found_colors or confidence > found_colors[primary_color]:
                        found_colors[primary_color] = confidence
        
        # 발견된 색상을 신뢰도 내림차순으로 정렬
        return sorted(found_colors.items(), key=lambda x: x[1], reverse=True)
    
    def analyze_color_from_captions(self, captions: List[str]) -> Dict[str, Any]:
        """
        여러 캡션에서 색상 정보 분석
        
        Args:
            captions: 이미지 캡션 목록
            
        Returns:
            Dict[str, Any]: 색상 분석 결과
        """
        all_colors = []
        
        # 모든 캡션에서 색상 추출
        for caption in captions:
            colors = self.extract_colors_from_caption(caption)
            all_colors.extend(colors)
        
        # 색상별 평균 신뢰도 계산
        color_confidences = {}
        for color, confidence in all_colors:
            if color in color_confidences:
                color_confidences[color].append(confidence)
            else:
                color_confidences[color] = [confidence]
        
        # 평균 신뢰도로 변환
        avg_confidences = {
            color: sum(confidences) / len(confidences) 
            for color, confidences in color_confidences.items()
        }
        
        # 결과 포맷팅
        colors_sorted = sorted(avg_confidences.items(), key=lambda x: x[1], reverse=True)
        
        # 주요 색상이 흔한 색상인지 확인
        primary_color = colors_sorted[0][0] if colors_sorted else None
        is_common_color = primary_color in self.common_colors if primary_color else False
        
        return {
            'primary_color': primary_color,
            'is_common_color': is_common_color,
            'all_colors': [
                {'name': color, 'confidence': confidence} 
                for color, confidence in colors_sorted
            ]
        }
    
    def calculate_color_similarity(self, colors1: Dict[str, Any], colors2: Dict[str, Any]) -> float:
        """
        두 이미지의 색상 분석 결과 간 유사도 계산
        
        Args:
            colors1: 첫 번째 이미지의 색상 분석 결과
            colors2: 두 번째 이미지의 색상 분석 결과
            
        Returns:
            float: 색상 유사도 점수 (0~1)
        """
        # 주요 색상이 같으면 높은 유사도
        if (colors1.get('primary_color') == colors2.get('primary_color') and 
            colors1.get('primary_color') is not None):
            
            # 두 이미지 모두 흔한 색상이면 유사도 감소
            if colors1.get('is_common_color', False) and colors2.get('is_common_color', False):
                return 0.7 * Config.COLOR_WEIGHT_REDUCTION
            else:
                return 0.9  # 흔하지 않은 동일한 주요 색상
        
        # 공통 색상 찾기
        colors1_set = {c['name'] for c in colors1.get('all_colors', [])}
        colors2_set = {c['name'] for c in colors2.get('all_colors', [])}
        
        common_colors = colors1_set.intersection(colors2_set)
        all_colors = colors1_set.union(colors2_set)
        
        # 자카드 유사도 (공통 색상 / 전체 색상)
        if not all_colors:
            return 0.0
            
        jaccard = len(common_colors) / len(all_colors)
        
        # 유사도 조정
        if jaccard > 0:
            # 색상 집합이 모두 흔한 색상만 포함하면 유사도 감소
            common_color_ratio = len(common_colors.intersection(self.common_colors)) / len(common_colors) if common_colors else 0
            
            if common_color_ratio > 0.5:
                jaccard *= Config.COLOR_WEIGHT_REDUCTION
        
        return jaccard