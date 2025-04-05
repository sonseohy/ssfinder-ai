import os
from pathlib import Path

class Config:
    """
    시스템 전반의 설정을 관리하는 클래스
    """
    # 프로젝트 기본 경로
    BASE_DIR = Path(__file__).parent.parent
    
    # 모델 관련 설정
    CLIP_MODEL_NAME = "ViT-B/32"
    BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-large"
    
    # 가중치 설정
    WEIGHTS = {
        'category_match': 0.40,      # 카테고리 일치 가중치
        'keyword_match': 0.25,       # 키워드 일치 가중치
        'clip_similarity': 0.15,     # CLIP 모델 유사도 가중치
        'blip_similarity': 0.15,     # BLIP 모델 유사도 가중치
        'color_similarity': 0.05,    # 색상 유사도 가중치 (기본)
    }
    
    # 색상 가중치 조정
    COLOR_WEIGHT_BOOST = 0.15        # 카테고리/키워드 일치 시 색상 가중치 증가
    
    # 색상 가중치 감소 대상 색상 (흔한 색상)
    COMMON_COLORS = ['black', 'white', 'gray']
    COLOR_WEIGHT_REDUCTION = 0.5     # 흔한 색상에 대한 가중치 감소 계수
    
    # 유사도 임계값
    SIMILARITY_THRESHOLD = 0.60      # 이 값 이상의 유사도를 가진 항목만 추천
    
    # 언어 설정
    SUPPORTED_LANGUAGES = ['ko', 'en']
    PRIMARY_LANGUAGE = 'ko'
    
    # 카테고리 및 키워드 설정
    CATEGORY_GROUPS = {
        'electronics_apple': ["iphone", "ipad", "macbook", "airpods", "earpods", "apple watch"],
        'electronics_samsung': ["samsung phone", "galaxy phone", "galaxy s", "galaxy note", "galaxy tab", 
                               "galaxy watch", "galaxy buds", "samsung tablet", "samsung earbuds"],
        'electronics_samsung_foldable': ["galaxy z flip", "galaxy z fold", "samsung foldable phone", "foldable phone",
                                       "samsung folding phone", "폴더블 스마트폰", "접이식 휴대폰", "갤럭시 Z 플립", "갤럭시 Z 폴드"],
        'electronics_general': ["smartphone", "phone", "tablet", "wireless earbuds", "headphones", "headset",
                              "laptop", "smart watch", "earbuds", "camera", "태블릿", "휴대폰", "노트북", "카메라"],
        'money_items': ["money", "cash", "korean won", "dollar", "euro", "현금", "지폐", "동전", "한국 돈"],
        'car_items': ["car", "car key", "car license", "자동차", "차키", "자동차등록증", "운전면허증"],
        'wallet_items': ["wallet", "purse", "credit card", "id card", "transportation card", "membership card", 
                        "지갑", "여성용 지갑", "남성용 지갑", "카드", "신용카드", "교통카드", "멤버십카드"],
        'lost_items_general': ["umbrella", "key", "glasses", "sunglasses", "water bottle", "우산", "열쇠", "안경", "선글라스", "물병"]
    }
    
    # 처리할 이미지 크기 설정
    IMAGE_SIZE = (224, 224)
    
    # BLIP 모델의 캡션 생성 설정
    BLIP_MAX_LENGTH = 50
    BLIP_NUM_BEAMS = 5
    BLIP_MIN_LENGTH = 5
    
    # 유사도 검색 결과 제한
    MAX_RECOMMENDATIONS = 5