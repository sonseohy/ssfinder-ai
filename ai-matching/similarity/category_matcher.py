from typing import Dict, List, Tuple, Set, Optional
import re
import string
from collections import Counter

from config import Config

class CategoryMatcher:
    """
    게시글 카테고리 및 키워드 매칭을 위한 클래스
    """
    def __init__(self):
        """
        카테고리 매처 초기화
        """
        self.categories = Config.CATEGORY_GROUPS
        self.all_keywords = self._extract_all_keywords()
        
    def _extract_all_keywords(self) -> Set[str]:
        """
        모든 카테고리에서 키워드 추출
        
        Returns:
            Set[str]: 모든 키워드 집합
        """
        keywords = set()
        for category, items in self.categories.items():
            keywords.update(items)
        return keywords
    
    def _normalize_text(self, text: str) -> str:
        """
        텍스트 정규화 (소문자 변환, 특수문자 제거 등)
        
        Args:
            text: 정규화할 텍스트
            
        Returns:
            str: 정규화된 텍스트
        """
        # 소문자 변환
        text = text.lower()
        
        # 영어 텍스트의 경우 구두점 제거 (한국어는 유지)
        if not any('\uAC00' <= char <= '\uD7A3' for char in text):
            # 구두점 제거 (영어)
            translator = str.maketrans('', '', string.punctuation)
            text = text.translate(translator)
        
        return text
    
    def extract_keywords_from_text(self, text: str) -> List[str]:
        """
        텍스트에서 의미 있는 키워드 추출
        
        Args:
            text: 키워드를 추출할 텍스트
            
        Returns:
            List[str]: 추출된 키워드 목록
        """
        # 텍스트 정규화
        normalized_text = self._normalize_text(text)
        
        # 단어 분리
        is_korean = any('\uAC00' <= char <= '\uD7A3' for char in text)
        
        if is_korean:
            try:
                # 한국어 형태소 분석 (konlpy가 설치된 경우)
                from konlpy.tag import Okt
                okt = Okt()
                words = okt.nouns(normalized_text)
            except:
                # konlpy가 없는 경우 기본 처리
                words = re.findall(r'\w+', normalized_text)
        else:
            # 영어 단어 추출
            words = normalized_text.split()
        
        return words
    
    def find_matching_categories(self, text: str) -> Dict[str, float]:
        """
        텍스트에 가장 적합한 카테고리 찾기
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            Dict[str, float]: 카테고리별 매칭 점수 (0~1)
        """
        # 텍스트에서 키워드 추출
        keywords = self.extract_keywords_from_text(text)
        
        if not keywords:
            return {}
        
        # 카테고리별 일치하는 키워드 수 계산
        category_matches = {}
        
        for category, category_keywords in self.categories.items():
            # 카테고리 키워드를 정규화
            normalized_category_keywords = [self._normalize_text(k) for k in category_keywords]
            
            # 일치하는 키워드 찾기
            matches = 0
            for word in keywords:
                for category_word in normalized_category_keywords:
                    # 부분 일치도 허용 (최소 3글자 이상 겹치는 경우)
                    if len(word) >= 3 and len(category_word) >= 3:
                        if word in category_word or category_word in word:
                            matches += 1
                            break
                    # 정확히 일치하는 경우
                    elif word == category_word:
                        matches += 1
                        break
            
            # 매칭 점수 계산 (정규화)
            if matches > 0:
                # 매칭 점수 = 일치 키워드 수 / 최대(텍스트 키워드 수, 카테고리 키워드 수)
                score = matches / max(len(keywords), len(category_keywords))
                category_matches[category] = score
        
        return category_matches
    
    def calculate_keyword_similarity(self, text1: str, text2: str) -> float:
        """
        두 텍스트 사이의 키워드 유사도 계산
        
        Args:
            text1: 첫 번째 텍스트
            text2: 두 번째 텍스트
            
        Returns:
            float: 키워드 유사도 점수 (0~1)
        """
        # 키워드 추출
        keywords1 = self.extract_keywords_from_text(text1)
        keywords2 = self.extract_keywords_from_text(text2)
        
        if not keywords1 or not keywords2:
            return 0.0
        
        # 키워드 빈도수 계산
        counter1 = Counter(keywords1)
        counter2 = Counter(keywords2)
        
        # 공통 키워드 찾기
        common_keywords = set(counter1.keys()) & set(counter2.keys())
        
        if not common_keywords:
            # 부분 일치 검사
            partial_matches = 0
            for word1 in counter1.keys():
                for word2 in counter2.keys():
                    # 최소 3글자 이상 겹치는 경우
                    if len(word1) >= 3 and len(word2) >= 3:
                        if word1 in word2 or word2 in word1:
                            partial_matches += 1
            
            if partial_matches > 0:
                return partial_matches / max(len(counter1), len(counter2))
            return 0.0
        
        # 자카드 유사도 계산
        intersection = sum(min(counter1[k], counter2[k]) for k in common_keywords)
        union = sum(counter1.values()) + sum(counter2.values()) - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def get_category_from_text(self, text: str) -> Tuple[str, float]:
        """
        텍스트에 가장 적합한 카테고리 반환
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            Tuple[str, float]: (카테고리명, 신뢰도)
        """
        category_matches = self.find_matching_categories(text)
        
        if not category_matches:
            return ("unknown", 0.0)
        
        # 가장 높은 점수의 카테고리 선택
        best_category = max(category_matches.items(), key=lambda x: x[1])
        return best_category