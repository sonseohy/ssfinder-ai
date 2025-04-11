"""
유사도 계산 및 관련 유틸리티 함수
Kiwi 형태소 분석기를 사용하여 한국어 텍스트 분석 개선
"""
import os
import sys
import logging
import numpy as np
import re
from collections import Counter
from kiwipiepy import Kiwi

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Kiwi 형태소 분석기 초기화
kiwi = Kiwi()

# 설정 값 (환경변수 또는 기본값)
SIMILARITY_THRESHOLD = float(os.getenv('SIMILARITY_THRESHOLD', '0.7'))
TEXT_WEIGHT = float(os.getenv('TEXT_WEIGHT', '0.7'))
IMAGE_WEIGHT = float(os.getenv('IMAGE_WEIGHT', '0.3'))
CATEGORY_WEIGHT = float(os.getenv('CATEGORY_WEIGHT', '0.5'))
ITEM_NAME_WEIGHT = float(os.getenv('ITEM_NAME_WEIGHT', '0.3'))
COLOR_WEIGHT = float(os.getenv('COLOR_WEIGHT', '0.1'))
CONTENT_WEIGHT = float(os.getenv('CONTENT_WEIGHT', '0.1'))

def preprocess_text(text):
    """
    텍스트 전처리 함수
    
    Args:
        text (str): 전처리할 텍스트
        
    Returns:
        str: 전처리된 텍스트
    """
    if not text:
        return ""

    if not isinstance(text, str):
        text = str(text)
        
    # 소문자 변환 (영어의 경우)
    text = text.lower()
    
    # 불필요한 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 특수 문자 제거 (단, 한글, 영문, 숫자는 유지)
    text = re.sub(r'[^\w\s가-힣ㄱ-ㅎㅏ-ㅣ]', ' ', text)
    
    return text

def extract_keywords(text):
    """
    Kiwi 형태소 분석기를 사용하여 텍스트에서 중요 키워드 추출
    
    Args:
        text (str): 키워드를 추출할 텍스트
        
    Returns:
        list: 키워드 리스트 (주로 명사와 형용사)
    """
    if not text:
        return []
    
    # 텍스트 전처리
    processed_text = preprocess_text(text)
    
    try:
        # Kiwi 형태소 분석 수행
        result = kiwi.analyze(processed_text)
        
        # 중요 키워드 추출 (명사, 형용사 등)
        keywords = []
        for token in result[0][0]:
            # NNG: 일반명사, NNP: 고유명사, VA: 형용사, VV: 동사, SL: 외국어(영어 등)
            if token.tag in ['NNG', 'NNP', 'VA', 'SL']:
                # 한 글자 명사는 중요도 낮을 수 있어 필터링 (선택적)
                if len(token.form) > 1 or token.tag in ['SL']:
                    keywords.append(token.form)
        
        logger.debug(f"키워드 추출 결과: {keywords}")
        return keywords
    
    except Exception as e:
        logger.warning(f"형태소 분석 오류: {str(e)}, 기본 분리 방식으로 대체")
        # 오류 발생 시 기본 방식으로 대체
        words = processed_text.split()
        return words

def calculate_text_similarity(text1, text2, weights=None):
    """
    두 텍스트 간의 유사도 계산 (Kiwi 형태소 분석 활용)
    
    Args:
        text1 (str): 첫 번째 텍스트
        text2 (str): 두 번째 텍스트
        weights (dict, optional): 각 부분에 대한 가중치
        
    Returns:
        float: 유사도 점수 (0~1 사이)
    """
    if not text1 or not text2:
        return 0.0
    
    # 기본 가중치 설정
    if weights is None:
        weights = {
            'common_words': 0.7,  # 공통 단어 비율의 가중치 증가
            'length_ratio': 0.15,
            'word_order': 0.15
        }
    
    # 텍스트에서 키워드 추출 (Kiwi 형태소 분석기 사용)
    keywords1 = extract_keywords(text1)
    keywords2 = extract_keywords(text2)
    
    if not keywords1 or not keywords2:
        return 0.0
    
    # 1. 공통 단어 비율 계산
    common_words = set(keywords1) & set(keywords2)
    common_ratio = len(common_words) / max(1, min(len(set(keywords1)), len(set(keywords2))))
    
    # 2. 텍스트 길이 유사도
    length_ratio = min(len(keywords1), len(keywords2)) / max(1, max(len(keywords1), len(keywords2)))
    
    # 3. 단어 순서 유사도 (선택적)
    word_order_sim = 0.0
    if common_words:
        # 공통 단어의 위치 차이 기반 유사도
        positions1 = {word: i for i, word in enumerate(keywords1) if word in common_words}
        positions2 = {word: i for i, word in enumerate(keywords2) if word in common_words}
        
        if positions1 and positions2:
            pos_diff_sum = sum(abs(positions1[word] - positions2[word]) for word in common_words if word in positions1 and word in positions2)
            max_diff = len(keywords1) + len(keywords2)
            word_order_sim = 1.0 - (pos_diff_sum / max(1, max_diff))
    
    # 가중치 적용하여 최종 유사도 계산
    similarity = (
        weights['common_words'] * common_ratio + 
        weights['length_ratio'] * length_ratio + 
        weights['word_order'] * word_order_sim
    )
    
    return min(1.0, max(0.0, similarity))

def calculate_category_similarity(category1, category2):
    """
    두 카테고리 간의 유사도 계산 (기타 카테고리 고려)
    
    Args:
        category1 (str): 첫 번째 카테고리
        category2 (str): 두 번째 카테고리
        
    Returns:
        float: 유사도 점수 (0~1 사이)
    """
    if not category1 or not category2:
        return 0.0
    
    # 카테고리 전처리
    cat1 = preprocess_text(str(category1))
    cat2 = preprocess_text(str(category2))
    
    # 정확히 일치하는 경우
    if cat1 == cat2:
        return 1.0
    
    # Kiwi로 키워드 추출
    keywords1 = set(extract_keywords(cat1))
    keywords2 = set(extract_keywords(cat2))
    
    # '기타' 카테고리 처리
    if '기타' in cat1 or '기타' in cat2:
        # 키워드 추출 및 교집합 비교
        if not keywords1 or not keywords2:
            return 0.3  # 기타 카테고리는 기본 유사도 부여
        
        # 교집합 단어가 있으면 높은 유사도
        common_words = keywords1 & keywords2
        if common_words:
            return 0.7
        
        return 0.3  # 기타 카테고리지만 공통 키워드 없음
    
    # 일반 카테고리 유사도
    return calculate_text_similarity(cat1, cat2)

def calculate_similarity(user_post, lost_item, clip_model=None):
    """
    사용자 게시글과 습득물 항목 간의 종합 유사도 계산
    Spring Boot와 호환되도록 필드명 매핑 추가
    
    Args:
        user_post (dict): 사용자 게시글 정보
        lost_item (dict): 습득물 데이터
        clip_model (KoreanCLIPModel, optional): CLIP 모델 인스턴스
        
    Returns:
        float: 유사도 점수 (0~1 사이)
        dict: 세부 유사도 정보
    """
    # 텍스트 유사도 계산
    text_similarities = {}
    
    # 1. 카테고리 유사도 (Spring Boot에서는 category 또는 itemCategoryId)
    category_sim = 0.0
    user_category = user_post.get('category', user_post.get('itemCategoryId'))
    lost_category = lost_item.get('category', lost_item.get('item_category_id'))
    
    if user_category is not None and lost_category is not None:
        category_sim = calculate_category_similarity(user_category, lost_category)
    text_similarities['category'] = category_sim
    
    # 2. 물품명 유사도 (Spring Boot에서는 title)
    item_name_sim = 0.0
    user_item_name = user_post.get('item_name', user_post.get('title', ''))
    lost_item_name = lost_item.get('item_name', lost_item.get('title', ''))
    
    if user_item_name and lost_item_name:
        item_name_sim = calculate_text_similarity(user_item_name, lost_item_name)
    text_similarities['item_name'] = item_name_sim
    
    # 3. 색상 유사도
    color_sim = 0.0
    user_color = user_post.get('color', '')
    lost_color = lost_item.get('color', '')
    
    if user_color and lost_color:
        color_sim = calculate_text_similarity(user_color, lost_color)
    text_similarities['color'] = color_sim
    
    # 4. 내용 유사도 (Spring Boot에서는 detail)
    content_sim = 0.0
    user_content = user_post.get('content', user_post.get('detail', ''))
    lost_content = lost_item.get('content', lost_item.get('detail', ''))
    
    if user_content and lost_content:
        content_sim = calculate_text_similarity(user_content, lost_content)
    text_similarities['content'] = content_sim
    
    # 텍스트 종합 유사도 계산 (가중치 적용)
    text_similarity = (
        CATEGORY_WEIGHT * category_sim +
        ITEM_NAME_WEIGHT * item_name_sim +
        COLOR_WEIGHT * color_sim +
        CONTENT_WEIGHT * content_sim
    )
    
    # CLIP 모델을 사용한 이미지-텍스트 유사도 계산
    image_similarity = 0.0
    has_image = False
    
    if clip_model is not None:
        # 사용자 게시글과 습득물에 모두 이미지가 있는 경우
        user_image = user_post.get('image', None) or user_post.get('image_url', None)
        lost_image = lost_item.get('image', None) or lost_item.get('image_url', None)
        
        if user_image and lost_image:
            try:
                # CLIP 모델을 사용한 유사도 계산
                user_text_embedding = clip_model.encode_text(user_content or user_item_name)
                user_image_embedding = clip_model.encode_image(user_image)
                
                item_text_embedding = clip_model.encode_text(lost_content or lost_item_name)
                item_image_embedding = clip_model.encode_image(lost_image)
                
                # 텍스트-이미지 교차 유사도 계산
                text_to_image_sim = clip_model.calculate_similarity(user_text_embedding, item_image_embedding)
                image_to_text_sim = clip_model.calculate_similarity(item_text_embedding, user_image_embedding)
                image_to_image_sim = clip_model.calculate_similarity(user_image_embedding, item_image_embedding)
                
                image_similarity = (text_to_image_sim + image_to_text_sim + image_to_image_sim) / 3
                has_image = True
            except Exception as e:
                logger.warning(f"이미지 유사도 계산 중 오류 발생: {str(e)}")
    
    # 최종 유사도 계산 (텍스트와 이미지 가중치 적용)
    if has_image:
        final_similarity = TEXT_WEIGHT * text_similarity + IMAGE_WEIGHT * image_similarity
    else:
        final_similarity = text_similarity
    
    # 세부 유사도 정보
    similarity_details = {
        'text_similarity': text_similarity,
        'image_similarity': image_similarity if has_image else None,
        'final_similarity': final_similarity,
        'details': text_similarities
    }
    
    return final_similarity, similarity_details

def find_similar_items(user_post, lost_items, threshold=SIMILARITY_THRESHOLD, clip_model=None):
    """
    사용자 게시글과 유사한 습득물 목록 찾기
    
    Args:
        user_post (dict): 사용자 게시글 정보
        lost_items (list): 습득물 데이터 목록
        threshold (float): 유사도 임계값 (기본값: config에서 설정)
        clip_model (KoreanCLIPModel, optional): CLIP 모델 인스턴스
        
    Returns:
        list: 유사도가 임계값 이상인 습득물 목록 (유사도 높은 순)
    """
    similar_items = []
    
    logger.info(f"사용자 게시글과 {len(lost_items)}개 습득물 비교 중...")
    
    for item in lost_items:
        similarity, details = calculate_similarity(user_post, item, clip_model)
        
        if similarity >= threshold:
            similar_items.append({
                'item': item,
                'similarity': similarity,
                'details': details
            })
    
    # 유사도 높은 순으로 정렬
    similar_items.sort(key=lambda x: x['similarity'], reverse=True)
    
    logger.info(f"유사도 {threshold} 이상인 습득물 {len(similar_items)}개 발견")
    
    return similar_items

# 모듈 테스트용 코드
if __name__ == "__main__":
    # 텍스트 유사도 테스트
    text1 = "검은색 가죽 지갑을 잃어버렸습니다."
    text2 = "검정 가죽 지갑을 찾았습니다."
    text3 = "노트북을 분실했습니다."
    
    # 키워드 추출 테스트
    print("[ 키워드 추출 테스트 ]")
    print(f"텍스트 1: '{text1}'")
    print(f"추출된 키워드: {extract_keywords(text1)}")
    print(f"텍스트 2: '{text2}'")
    print(f"추출된 키워드: {extract_keywords(text2)}")
    
    # 유사도 테스트
    sim12 = calculate_text_similarity(text1, text2)
    sim13 = calculate_text_similarity(text1, text3)
    
    print("\n[ 유사도 테스트 ]")
    print(f"텍스트 1-2 유사도: {sim12:.4f}")
    print(f"텍스트 1-3 유사도: {sim13:.4f}")
    
    # 카테고리 유사도 테스트
    cat1 = "지갑"
    cat2 = "가방/지갑"
    cat3 = "기타"
    
    cat_sim12 = calculate_category_similarity(cat1, cat2)
    cat_sim13 = calculate_category_similarity(cat1, cat3)
    
    print("\n[ 카테고리 유사도 테스트 ]")
    print(f"카테고리 1-2 유사도: {cat_sim12:.4f}")
    print(f"카테고리 1-3 유사도: {cat_sim13:.4f}")