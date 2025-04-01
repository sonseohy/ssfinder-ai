import torch
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import re
from sklearn.metrics.pairwise import cosine_similarity

from config.config import WEIGHTS, TOP_K_CANDIDATES, SIMILARITY_THRESHOLD

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LostItemMatcher:
    """습득물(분실물) 간의 매칭을 위한 알고리즘"""
    
    def __init__(self, weights: Dict[str, float] = WEIGHTS):
        """
        매처 초기화
        
        Args:
            weights (Dict[str, float]): 각 매칭 요소에 대한 가중치
        """
        self.weights = weights
        
        # 가중치 검증
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.001:
            logger.warning(f"가중치 합계가 1.0이 아닙니다 (현재: {total_weight:.4f}). 정규화합니다.")
            self.weights = {k: v / total_weight for k, v in weights.items()}
            
        logger.info(f"매칭 가중치 설정: {self.weights}")
    
    def match_items(self, query_item: Dict[str, Any], candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        쿼리 아이템과 후보 아이템 간의 매칭 수행
        
        Args:
            query_item (Dict[str, Any]): 쿼리 아이템 (사용자 분실물)
            candidates (List[Dict[str, Any]]): 후보 아이템 리스트 (습득물 데이터베이스)
            
        Returns:
            List[Dict[str, Any]]: 매칭 점수가 포함된 후보 아이템, 점수 내림차순 정렬
        """
        logger.info(f"쿼리 아이템과 {len(candidates)}개 후보 아이템 매칭 시작")
        
        matched_candidates = []
        
        for candidate in candidates:
            # 각 후보와의 유사도 점수 계산
            similarity_scores = self._calculate_similarity_scores(query_item, candidate)
            
            # 가중 종합 점수 계산
            weighted_score = self._calculate_weighted_score(similarity_scores)
            
            # 임계값보다 높은 점수만 유지
            if weighted_score >= SIMILARITY_THRESHOLD:
                # 결과 정보 추가
                candidate_with_score = candidate.copy()
                candidate_with_score['matching'] = {
                    'total_score': weighted_score,
                    'detailed_scores': similarity_scores,
                    'match_explanation': self._generate_match_explanation(query_item, candidate, similarity_scores)
                }
                matched_candidates.append(candidate_with_score)
        
        # 총점으로 정렬
        matched_candidates.sort(key=lambda x: x['matching']['total_score'], reverse=True)
        
        # 상위 후보만 유지
        top_candidates = matched_candidates[:TOP_K_CANDIDATES]
        
        logger.info(f"매칭 완료. {len(matched_candidates)}개 일치 항목 중 상위 {len(top_candidates)}개 반환")
        return top_candidates
    
    def _calculate_similarity_scores(self, query_item: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, float]:
        """
        다양한 기준에 따른 쿼리 아이템과 후보 아이템 간의 유사도 점수 계산
        
        Args:
            query_item (Dict[str, Any]): 쿼리 아이템
            candidate (Dict[str, Any]): 후보 아이템
            
        Returns:
            Dict[str, float]: 각 기준별 유사도 점수
        """
        scores = {}
        
        # 1. 객체 클래스 일치 점수
        scores['object_class_match'] = self._calculate_class_similarity(query_item, candidate)
        
        # 2. CLIP 이미지-텍스트 유사도
        if 'clip_embedding' in query_item and 'clip_embedding' in candidate:
            scores['clip_similarity'] = self._calculate_clip_similarity(query_item, candidate)
        else:
            scores['clip_similarity'] = 0.0
        
        # 3. 속성 일치 점수 (BLIP으로 추출)
        scores['attribute_match'] = self._calculate_attribute_similarity(query_item, candidate)
        
        # 4. 메타데이터 일치 점수
        scores['metadata_match'] = self._calculate_metadata_similarity(query_item, candidate)
        
        return scores
    
    def _calculate_weighted_score(self, similarity_scores: Dict[str, float]) -> float:
        """
        가중치를 적용한 종합 점수 계산
        
        Args:
            similarity_scores (Dict[str, float]): 각 기준별 유사도 점수
            
        Returns:
            float: 가중 종합 점수
        """
        weighted_score = 0.0
        
        for score_type, score in similarity_scores.items():
            if score_type in self.weights:
                weighted_score += score * self.weights[score_type]
        
        return weighted_score
    
    def _calculate_class_similarity(self, query_item: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """
        객체 클래스 유사도 계산
        
        Args:
            query_item (Dict[str, Any]): 쿼리 아이템
            candidate (Dict[str, Any]): 후보 아이템
            
        Returns:
            float: 클래스 유사도 점수 (0-1)
        """
        # 클래스 정보 추출
        query_class = self._extract_class_info(query_item)
        candidate_class = self._extract_class_info(candidate)
        
        if not query_class or not candidate_class:
            return 0.0
        
        # 클래스 일치 여부 확인
        if query_class == candidate_class:
            return 1.0
        
        # 클래스 그룹 정의 (유사 클래스)
        class_groups = [
            # 휴대폰 그룹
            {"smartphone", "mobile phone", "iphone", "galaxy", "휴대폰", "스마트폰", "아이폰", "갤럭시", "휴대전화"},
            # 애플 기기 그룹
            {"iphone", "ipad", "macbook", "airpods", "earpods", "apple watch"},
            # 삼성 기기 그룹
            {"samsung phone", "galaxy phone", "galaxy s", "galaxy note", "galaxy tab", "galaxy watch", "galaxy buds"},
            # 삼성 폴더블 그룹
            {"galaxy z flip", "galaxy z fold", "samsung foldable phone", "foldable phone", "samsung folding phone"},
            # 가방 그룹
            {"bag", "backpack", "handbag", "shopping bag", "여성용 가방", "남성용 가방", "쇼핑백"},
            # 지갑/카드 그룹
            {"wallet", "purse", "credit card", "id card", "transportation card", "membership card", "지갑"},
            # 귀금속/액세서리 그룹
            {"jewelry", "ring", "necklace", "earring", "watch", "반지", "목걸이", "귀걸이", "시계"}
        ]
        
        # 같은 그룹에 속하는지 확인
        for group in class_groups:
            if query_class in group and candidate_class in group:
                return 0.8  # 같은 그룹 내 클래스는 높은 유사도
        
        # 부분 문자열 매칭 (예: "iphone" in "apple iphone")
        if query_class in candidate_class or candidate_class in query_class:
            return 0.6
        
        # 완전히 다른 클래스는 낮은 점수
        return 0.1
    
    def _extract_class_info(self, item: Dict[str, Any]) -> Optional[str]:
        """
        아이템에서 클래스 정보 추출
        
        Args:
            item (Dict[str, Any]): 분석할 아이템
            
        Returns:
            Optional[str]: 추출된 클래스 또는 None
        """
        # CLIP 분류 결과
        if 'clip_classification' in item and item['clip_classification']:
            if isinstance(item['clip_classification'], list) and len(item['clip_classification']) > 0:
                if isinstance(item['clip_classification'][0], dict) and 'class' in item['clip_classification'][0]:
                    return item['clip_classification'][0]['class'].lower()
                return str(item['clip_classification'][0]).lower()
            elif isinstance(item['clip_classification'], dict) and 'class' in item['clip_classification']:
                return item['clip_classification']['class'].lower()
        
        # 직접 클래스 필드
        if 'class' in item:
            return str(item['class']).lower()
            
        # BLIP 속성에서 type 필드
        if 'blip_attributes' in item and 'attributes' in item['blip_attributes']:
            attrs = item['blip_attributes']['attributes']
            if 'type' in attrs:
                return attrs['type'].lower()
        
        # 제품명에서 유추
        if 'fdPrdtNm' in item and item['fdPrdtNm']:
            return item['fdPrdtNm'].lower()
            
        # 설명에서 유추
        if 'clean_subject' in item and item['clean_subject']:
            # 알려진 물품 키워드 확인
            categories = [
                "휴대폰", "스마트폰", "아이폰", "갤럭시", "지갑", "가방", 
                "핸드백", "카드", "신분증", "열쇠", "안경", "시계", "의류"
            ]
            
            for category in categories:
                if category in item['clean_subject']:
                    return category
        
        # 제목에서 유추
        if 'clean_product_name' in item and item['clean_product_name']:
            return item['clean_product_name'].lower()
        
        return None
    
    def _calculate_clip_similarity(self, query_item: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """
        CLIP 임베딩 간 유사도 계산
        
        Args:
            query_item (Dict[str, Any]): 쿼리 아이템
            candidate (Dict[str, Any]): 후보 아이템
            
        Returns:
            float: CLIP 유사도 점수 (0-1)
        """
        # 임베딩 확인
        if 'clip_embedding' not in query_item or 'clip_embedding' not in candidate:
            return 0.0
        
        # 임베딩을 numpy 배열로 변환
        if isinstance(query_item['clip_embedding'], torch.Tensor):
            query_emb = query_item['clip_embedding'].cpu().numpy()
        else:
            query_emb = np.array(query_item['clip_embedding'])
            
        if isinstance(candidate['clip_embedding'], torch.Tensor):
            candidate_emb = candidate['clip_embedding'].cpu().numpy()
        else:
            candidate_emb = np.array(candidate['clip_embedding'])
        
        # 임베딩 형태 조정
        if len(query_emb.shape) == 1:
            query_emb = query_emb.reshape(1, -1)
        if len(candidate_emb.shape) == 1:
            candidate_emb = candidate_emb.reshape(1, -1)
        
        # 코사인 유사도 계산
        similarity = cosine_similarity(query_emb, candidate_emb)[0][0]
        
        # 결과를 0-1 범위로 정규화 (코사인 유사도는 -1에서 1 사이)
        normalized_similarity = (similarity + 1) / 2
        
        return normalized_similarity
    
    def _calculate_attribute_similarity(self, query_item: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """
        BLIP으로 추출한 속성 간 유사도 계산
        
        Args:
            query_item (Dict[str, Any]): 쿼리 아이템
            candidate (Dict[str, Any]): 후보 아이템
            
        Returns:
            float: 속성 유사도 점수 (0-1)
        """
        # 속성 정보 확인
        query_attrs = self._extract_attribute_info(query_item)
        candidate_attrs = self._extract_attribute_info(candidate)
        
        if not query_attrs or not candidate_attrs:
            return 0.0
        
        # 속성별 가중치 설정 (중요도에 따라 조정)
        attr_weights = {
            'color': 0.2,      # 색상은 중요하지만 너무 많은 검정/흰색 물건이 있음
            'brand': 0.35,     # 브랜드는 매우 중요
            'type': 0.25,      # 물건 종류도 중요
            'material': 0.1,   # 재질은 덜 중요
            'condition': 0.1   # 상태도 덜 중요
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        # 각 속성별 유사도 계산
        for attr, weight in attr_weights.items():
            if attr in query_attrs and attr in candidate_attrs:
                attr_similarity = self._compare_attribute_values(
                    query_attrs[attr], candidate_attrs[attr], attr_type=attr
                )
                total_score += attr_similarity * weight
                total_weight += weight
                
        # 가중 평균 계산
        if total_weight > 0:
            return total_score / total_weight
        return 0.0
    
    def _extract_attribute_info(self, item: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        아이템에서 속성 정보 추출
        
        Args:
            item (Dict[str, Any]): 분석할 아이템
            
        Returns:
            Optional[Dict[str, str]]: 추출된 속성 또는 None
        """
        attributes = {}
        
        # BLIP 속성 직접 사용
        if 'blip_attributes' in item and 'attributes' in item['blip_attributes']:
            return item['blip_attributes']['attributes']
        
        # 개별 필드에서 속성 추출
        
        # 색상 추출
        if 'extracted_colors' in item and item['extracted_colors']:
            attributes['color'] = item['extracted_colors'][0]
        elif 'FD_COL_NM' in item and item['FD_COL_NM']:
            attributes['color'] = item['FD_COL_NM']
        
        # 브랜드 추출
        if 'extracted_brands' in item and item['extracted_brands']:
            attributes['brand'] = item['extracted_brands'][0]
        elif 'brand_detection' in item and item['brand_detection'].get('brand'):
            attributes['brand'] = item['brand_detection']['brand']
        
        # 타입 추출
        if 'class' in item:
            attributes['type'] = item['class']
        elif 'fdPrdtNm' in item:
            attributes['type'] = item['fdPrdtNm']
        
        return attributes if attributes else None
    
    def _compare_attribute_values(self, query_value: str, candidate_value: str, attr_type: str) -> float:
        """
        두 속성 값의 유사도 계산
        
        Args:
            query_value (str): 쿼리 속성 값
            candidate_value (str): 후보 속성 값
            attr_type (str): 속성 유형
            
        Returns:
            float: 속성 유사도 (0-1)
        """
        if not query_value or not candidate_value:
            return 0.0
            
        query_value = str(query_value).lower()
        candidate_value = str(candidate_value).lower()
        
        # 정확히 일치하면 1.0
        if query_value == candidate_value:
            return 1.0
        
        # 속성 유형별 특수 처리
        if attr_type == 'color':
            # 색상 그룹
            color_groups = [
                {'검정', '검은', '블랙', 'black'},
                {'흰', '흰색', '화이트', 'white'},
                {'빨간', '빨강', '레드', 'red'},
                {'파란', '파랑', '블루', 'blue'},
                {'노란', '노랑', '옐로우', 'yellow'},
                {'초록', '녹색', '그린', 'green'},
                {'회색', '그레이', 'gray', 'grey'},
                {'보라', '퍼플', 'purple'},
                {'분홍', '핑크', 'pink'},
                {'갈색', '브라운', 'brown'},
                {'금색', '골드', 'gold'},
                {'은색', '실버', 'silver'}
            ]
            
            # 같은 색상 그룹에 속하는지 확인
            for group in color_groups:
                if query_value in group and candidate_value in group:
                    return 0.9  # 같은 색상 그룹
                    
            # 부분 문자열 매칭
            if query_value in candidate_value or candidate_value in query_value:
                return 0.7
                
            return 0.0  # 다른 색상
            
        elif attr_type == 'brand':
            # 브랜드 그룹
            brand_groups = [
                {'apple', 'iphone', 'macbook', 'ipad', 'airpods', '애플', '아이폰', '맥북', '아이패드', '에어팟'},
                {'samsung', 'galaxy', '삼성', '갤럭시'}
            ]
            
            # 같은 브랜드 그룹에 속하는지 확인
            for group in brand_groups:
                if query_value in group and candidate_value in group:
                    return 0.9  # 같은 브랜드 그룹
            
            # 부분 문자열 매칭
            if query_value in candidate_value or candidate_value in query_value:
                return 0.7
                
            return 0.0  # 다른 브랜드
        
        else:
            # 기본 텍스트 유사도
            # 두 문자열 중 하나가 다른 하나에 포함되는 경우
            if query_value in candidate_value or candidate_value in query_value:
                return 0.7
            
            # 단어 수준의 일치 여부 확인
            query_words = set(query_value.split())
            candidate_words = set(candidate_value.split())
            
            common_words = query_words.intersection(candidate_words)
            
            if common_words:
                # 공통 단어 비율 계산
                similarity = len(common_words) / max(len(query_words), len(candidate_words))
                return min(0.8, similarity)  # 최대 0.8
            
            return 0.1  # 최소 유사도
    
    def _calculate_metadata_similarity(self, query_item: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """
        메타데이터(시간, 장소 등) 유사도 계산
        
        Args:
            query_item (Dict[str, Any]): 쿼리 아이템
            candidate (Dict[str, Any]): 후보 아이템
            
        Returns:
            float: 메타데이터 유사도 점수 (0-1)
        """
        # 시간 유사도: 분실/습득 시간이 가까울수록 높은 점수
        time_similarity = self._calculate_time_similarity(query_item, candidate)
        
        # 장소 유사도: 분실/습득 장소가 가까울수록 높은 점수
        location_similarity = self._calculate_location_similarity(query_item, candidate)
        
        # 메타데이터 유사도 결합 (시간 70%, 장소 30%)
        metadata_similarity = 0.7 * time_similarity + 0.3 * location_similarity
        
        return metadata_similarity
    
    def _calculate_time_similarity(self, query_item: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """
        시간 유사도 계산 (분실/습득 시간 차이 기반)
        
        Args:
            query_item (Dict[str, Any]): 쿼리 아이템
            candidate (Dict[str, Any]): 후보 아이템
            
        Returns:
            float: 시간 유사도 (0-1)
        """
        # 분실 시간과 습득 시간 추출
        query_time = self._extract_datetime(query_item)
        candidate_time = self._extract_datetime(candidate)
        
        if not query_time or not candidate_time:
            return 0.5  # 시간 정보가 없으면 중간 점수
        
        # 시간 차이 계산 (일 단위)
        time_diff = abs((query_time - candidate_time).days)
        
        # 시간 차이가 적을수록 높은 점수 (최대 30일 차이까지 고려)
        if time_diff == 0:
            return 1.0
        elif time_diff <= 1:
            return 0.9
        elif time_diff <= 3:
            return 0.8
        elif time_diff <= 7:
            return 0.7
        elif time_diff <= 14:
            return 0.5
        elif time_diff <= 30:
            return 0.3
        else:
            return 0.1
    
    def _extract_datetime(self, item: Dict[str, Any]) -> Optional[datetime]:
        """
        아이템에서 날짜/시간 정보 추출
        
        Args:
            item (Dict[str, Any]): 분석할 아이템
            
        Returns:
            Optional[datetime]: 추출된 날짜/시간 또는 None
        """
        # 이미 처리된 datetime 객체 사용
        date_fields = ['fdYmd_datetime', 'lstYmd_datetime', 'lstPrdtYmd_datetime']
        for field in date_fields:
            if field in item and item[field]:
                return item[field]
        
        # 날짜 문자열 처리
        date_fields = ['fdYmd', 'lstYmd', 'lstPrdtYmd']
        for field in date_fields:
            if field in item and item[field]:
                try:
                    # YYYYMMDD 형식 파싱
                    date_str = str(item[field])
                    if len(date_str) == 8:
                        return datetime.strptime(date_str, '%Y%m%d')
                except Exception as e:
                    logger.warning(f"날짜 파싱 오류: {e}")
        
        return None
    
    def _calculate_location_similarity(self, query_item: Dict[str, Any], candidate: Dict[str, Any]) -> float:
        """
        장소 유사도 계산
        
        Args:
            query_item (Dict[str, Any]): 쿼리 아이템
            candidate (Dict[str, Any]): 후보 아이템
            
        Returns:
            float: 장소 유사도 (0-1)
        """
        # 장소 코드 또는 이름 추출
        query_location = self._extract_location(query_item)
        candidate_location = self._extract_location(candidate)
        
        if not query_location or not candidate_location:
            return 0.5  # 장소 정보가 없으면 중간 점수
        
        # 정확히 일치하면 1.0
        if query_location == candidate_location:
            return 1.0
        
        # 부분 일치 확인 (예: "서울특별시" vs "서울특별시 강남구")
        if query_location in candidate_location or candidate_location in query_location:
            return 0.8
        
        # 지역 그룹 (예: 서울 내 다른 구)
        if self._are_in_same_region(query_location, candidate_location):
            return 0.6
        
        # 일치하지 않으면 낮은 점수
        return 0.2
    
    def _extract_location(self, item: Dict[str, Any]) -> Optional[str]:
        """
        아이템에서 장소 정보 추출
        
        Args:
            item (Dict[str, Any]): 분석할 아이템
            
        Returns:
            Optional[str]: 추출된 장소 또는 None
        """
        # 장소 필드 확인
        location_fields = ['fdPlaceNm', 'N_FD_LCT_NM', 'lstPlace']
        
        for field in location_fields:
            if field in item and item[field]:
                return str(item[field])
        
        return None
    
    def _are_in_same_region(self, location1: str, location2: str) -> bool:
        """
        두 장소가 같은 지역에 있는지 확인
        
        Args:
            location1 (str): 첫 번째 장소
            location2 (str): 두 번째 장소
            
        Returns:
            bool: 같은 지역이면 True
        """
        # 지역 매칭을 위한 간단한 패턴
        # 서울, 부산, 대구 등 주요 도시
        city_patterns = [
            r'서울[시특별]',
            r'부산[시광역]',
            r'대구[시광역]',
            r'인천[시광역]',
            r'광주[시광역]',
            r'대전[시광역]',
            r'울산[시광역]',
            r'세종[시특별]',
            r'경기[도]',
            r'강원[도]',
            r'충[북남][도]',
            r'전[북남][도]',
            r'경[북남][도]',
            r'제주[도특별]'
        ]
        
        # 각 지역 패턴에 대해 두 장소가 같은 패턴과 일치하는지 확인
        for pattern in city_patterns:
            if re.search(pattern, location1) and re.search(pattern, location2):
                return True
        
        return False
    
    def _generate_match_explanation(self, query_item: Dict[str, Any], candidate: Dict[str, Any], 
                                  similarity_scores: Dict[str, float]) -> str:
        """
        매칭 설명 생성 (왜 이 아이템이 일치하는지)
        
        Args:
            query_item (Dict[str, Any]): 쿼리 아이템
            candidate (Dict[str, Any]): 후보 아이템
            similarity_scores (Dict[str, float]): 유사도 점수
            
        Returns:
            str: 매칭 설명
        """
        explanations = []
        
        # 클래스 유사도 설명
        if similarity_scores.get('object_class_match', 0) > 0.7:
            query_class = self._extract_class_info(query_item) or "알 수 없는 물품"
            candidate_class = self._extract_class_info(candidate) or "알 수 없는 물품"
            
            if query_class == candidate_class:
                explanations.append(f"물품 종류가 동일합니다 ('{query_class}').")
            else:
                explanations.append(f"물품 종류가 유사합니다 ('{query_class}' vs '{candidate_class}').")
        
        # 속성 유사도 설명
        query_attrs = self._extract_attribute_info(query_item) or {}
        candidate_attrs = self._extract_attribute_info(candidate) or {}
        
        # 색상 설명
        if 'color' in query_attrs and 'color' in candidate_attrs:
            query_color = query_attrs['color']
            candidate_color = candidate_attrs['color']
            
            if query_color == candidate_color:
                explanations.append(f"색상이 일치합니다 ('{query_color}').")
            elif self._compare_attribute_values(query_color, candidate_color, 'color') > 0.7:
                explanations.append(f"색상이 유사합니다 ('{query_color}' vs '{candidate_color}').")
        
        # 브랜드 설명
        if 'brand' in query_attrs and 'brand' in candidate_attrs:
            query_brand = query_attrs['brand']
            candidate_brand = candidate_attrs['brand']
            
            if query_brand == candidate_brand:
                explanations.append(f"브랜드가 일치합니다 ('{query_brand}').")
            elif self._compare_attribute_values(query_brand, candidate_brand, 'brand') > 0.7:
                explanations.append(f"브랜드가 유사합니다 ('{query_brand}' vs '{candidate_brand}').")
        
        # 시간/장소 설명
        if similarity_scores.get('metadata_match', 0) > 0.7:
            query_time = self._extract_datetime(query_item)
            candidate_time = self._extract_datetime(candidate)
            
            if query_time and candidate_time:
                time_diff = abs((query_time - candidate_time).days)
                if time_diff <= 3:
                    explanations.append(f"분실/습득 시간이 매우 가깝습니다 (차이: {time_diff}일).")
                elif time_diff <= 7:
                    explanations.append(f"분실/습득 시간이 1주일 이내입니다 (차이: {time_diff}일).")
        
        # CLIP 유사도 설명
        if similarity_scores.get('clip_similarity', 0) > 0.8:
            explanations.append("시각적으로 매우 유사합니다.")
        elif similarity_scores.get('clip_similarity', 0) > 0.6:
            explanations.append("시각적으로 유사한 특징이 있습니다.")
        
        # 설명이 없으면 기본 메시지
        if not explanations:
            explanations.append("일부 특성이 일치합니다.")
        
        return " ".join(explanations)

if __name__ == "__main__":
    # 간단한 테스트
    matcher = LostItemMatcher()
    
    # 테스트 데이터
    query_item = {
        'clip_classification': [{'class': 'iphone', 'score': 0.95}],
        'blip_attributes': {
            'caption': '검정색 아이폰이 책상 위에 놓여있다.',
            'attributes': {
                'color': '검정',
                'brand': '애플',
                'type': '아이폰',
                'material': '유리와 메탈',
                'condition': '양호'
            }
        },
        'fdYmd': '20250215'
    }
    
    candidates = [
        {
            'clip_classification': [{'class': 'iphone', 'score': 0.90}],
            'blip_attributes': {
                'caption': '검정색 아이폰이 발견되었다.',
                'attributes': {
                    'color': '검정',
                    'brand': '애플',
                    'type': '아이폰',
                    'material': '유리와 메탈',
                    'condition': '양호'
                }
            },
            'fdYmd': '20250216',
            'fdPlaceNm': '서울특별시 강남구'
        },
        {
            'clip_classification': [{'class': 'samsung phone', 'score': 0.85}],
            'blip_attributes': {
                'caption': '검정색 갤럭시 폰이 발견되었다.',
                'attributes': {
                    'color': '검정',
                    'brand': '삼성',
                    'type': '갤럭시',
                    'material': '유리와 메탈',
                    'condition': '양호'
                }
            },
            'fdYmd': '20250214',
            'fdPlaceNm': '서울특별시 서초구'
        }
    ]
    
    # 매칭 수행
    results = matcher.match_items(query_item, candidates)
    
    # 결과 출력
    print(f"쿼리 아이템: {query_item['blip_attributes']['caption']}")
    print(f"\n매칭 결과 ({len(results)}개):")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. 점수: {result['matching']['total_score']:.4f}")
        print(f"   설명: {result['matching']['match_explanation']}")
        print(f"   상세 점수: {result['matching']['detailed_scores']}")
        print(f"   캡션: {result['blip_attributes']['caption']}")