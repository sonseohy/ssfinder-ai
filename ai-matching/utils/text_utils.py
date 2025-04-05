import re
import string
from typing import List, Set, Dict, Any, Optional
import unicodedata

from config import Config

def normalize_text(text: str) -> str:
    """
    텍스트 정규화 (소문자 변환, 특수문자 제거, 공백 정리 등)
    
    Args:
        text: 정규화할 텍스트
        
    Returns:
        str: 정규화된 텍스트
    """
    # 소문자 변환
    text = text.lower()
    
    # 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 한국어 텍스트 여부 확인
    is_korean = any('\uAC00' <= char <= '\uD7A3' for char in text)
    
    if not is_korean:
        # 영어 텍스트 정규화
        # 구두점 제거 (영어)
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)
    
    # 유니코드 정규화 (NFC 형식)
    text = unicodedata.normalize('NFC', text)
    
    return text

def extract_keywords(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    텍스트에서 키워드 추출
    
    Args:
        text: 키워드를 추출할 텍스트
        remove_stopwords: 불용어 제거 여부
        
    Returns:
        List[str]: 추출된 키워드 목록
    """
    if not text:
        return []
    
    # 텍스트 정규화
    text = normalize_text(text)
    
    # 한국어 텍스트 여부 확인
    is_korean = any('\uAC00' <= char <= '\uD7A3' for char in text)
    
    # 한국어 텍스트 처리
    if is_korean:
        try:
            # konlpy를 사용한 한국어 형태소 분석
            from konlpy.tag import Okt
            okt = Okt()
            
            # 명사 추출
            nouns = okt.nouns(text)
            
            # 불용어 제거
            if remove_stopwords:
                korean_stopwords = {'이', '그', '저', '것', '수', '등', '들', '및', '에서', '에게', '로', '으로', '에', '을', '를', '이다', '있다'}
                nouns = [word for word in nouns if word not in korean_stopwords and len(word) > 1]
            
            return nouns
        except ImportError:
            # konlpy가 설치되지 않은 경우 기본 처리
            words = re.findall(r'\w+', text)
            return [w for w in words if len(w) > 1]
    
    # 영어 텍스트 처리
    else:
        # 단어 분리
        words = text.split()
        
        # 불용어 제거
        if remove_stopwords:
            english_stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
                               'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 'from'}
            words = [word for word in words if word not in english_stopwords and len(word) > 2]
        
        return words

def get_text_language(text: str) -> str:
    """
    텍스트의 주요 언어 감지
    
    Args:
        text: 언어를 감지할 텍스트
        
    Returns:
        str: 언어 코드 ('ko', 'en', 'unknown')
    """
    if not text:
        return 'unknown'
    
    # 한글 문자 비율 계산
    korean_chars = sum(1 for char in text if '\uAC00' <= char <= '\uD7A3')
    korean_ratio = korean_chars / len(text)
    
    # 영문 문자 비율 계산
    english_chars = sum(1 for char in text if 'a' <= char.lower() <= 'z')
    english_ratio = english_chars / len(text)
    
    # 비율에 따라 언어 판단
    if korean_ratio > 0.3:
        return 'ko'
    elif english_ratio > 0.3:
        return 'en'
    else:
        return 'unknown'

def translate_keywords(keywords: List[str], target_lang: str = 'en') -> Dict[str, str]:
    """
    키워드 번역 (한영/영한)
    
    Args:
        keywords: 번역할 키워드 목록
        target_lang: 목표 언어 ('ko' 또는 'en')
        
    Returns:
        Dict[str, str]: 원본 키워드와 번역된 키워드 매핑
    """
    # 기본 번역 사전 (확장 가능)
    ko_to_en = {
        '휴대폰': 'smartphone',
        '핸드폰': 'smartphone',
        '스마트폰': 'smartphone',
        '아이폰': 'iphone',
        '갤럭시': 'galaxy',
        '지갑': 'wallet',
        '가방': 'bag',
        '열쇠': 'key',
        '안경': 'glasses',
        '선글라스': 'sunglasses',
        '우산': 'umbrella',
        '카드': 'card',
        '신용카드': 'credit card',
        '현금': 'cash',
        '돈': 'money',
        '책': 'book'
    }
    
    # 영한 변환용 사전 생성
    en_to_ko = {v: k for k, v in ko_to_en.items()}
    
    # 번역 사전 선택
    translation_dict = ko_to_en if target_lang == 'en' else en_to_ko
    
    # 키워드 번역
    translated = {}
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in translation_dict:
            translated[keyword] = translation_dict[keyword_lower]
        else:
            translated[keyword] = keyword  # 번역 없으면 원본 유지
    
    return translated

def clean_text_for_comparison(text1: str, text2: str) -> Tuple[List[str], List[str]]:
    """
    두 텍스트를 비교를 위해 정규화하고 키워드 추출
    
    Args:
        text1: 첫 번째 텍스트
        text2: 두 번째 텍스트
        
    Returns:
        Tuple[List[str], List[str]]: 추출된 키워드 목록 (text1, text2)
    """
    # 키워드 추출
    keywords1 = extract_keywords(text1)
    keywords2 = extract_keywords(text2)
    
    # 언어 감지
    lang1 = get_text_language(text1)
    lang2 = get_text_language(text2)
    
    # 다른 언어인 경우 번역
    if lang1 != lang2 and lang1 != 'unknown' and lang2 != 'unknown':
        if lang1 == 'ko':
            # 한글 -> 영어 번역
            trans = translate_keywords(keywords1, 'en')
            keywords1 = list(trans.values())
        else:
            # 영어 -> 한글 번역
            trans = translate_keywords(keywords2, 'ko')
            keywords2 = list(trans.values())
    
    return keywords1, keywords2