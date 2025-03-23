import os
import json
import random
from pathlib import Path

# 카테고리 구조 정의 (이미지에서 확인된 카테고리)
CATEGORIES = {
    "가방": ["여성용가방", "남성용가방", "기타가방"],
    "귀금속": ["반지 목걸이", "귀걸이 시계", "기타"],
    "도서용품": ["학습서적 소설", "컴퓨터서적", "만화책 기타서적"],
    "서류": ["서류 기타문류"],
    "산업용품": ["기타용품"],
    "쇼핑백": ["쇼핑백"],
    "스포츠용품": ["스포츠용품"],
    "악기": ["건반악기 관악기", "타악기 현악기", "기타악기"],
    "유가증권": ["여행 상품권", "채권 기타"],
    "의류": ["여성의류", "남성의류", "아기의류 모자", "신발 기타의류"],
    "자동차": ["자동차열쇠", "네비게이션", "자동차번호판", "임시번호판", "기타용품"],
    "전자기기": ["태블릿", "스마트워치", "무선이어폰", "카메라 기타용품"],
    "지갑": ["여성용 지갑", "남성용 지갑", "기타 지갑"],
    "증명서": ["신분증 면허증", "여권 기타"],
    "컴퓨터": ["삼성노트북", "LG노트북", "애플노트북 기타"],
    "카드": ["신용(체크)카드", "일반카드", "교통카드", "기타카드"],
    "현금": ["현금 수표 외화", "기타"],
    "휴대폰": ["삼성휴대폰", "LG휴대폰", "아이폰", "기타휴대폰", "기타통신기기"],
    "기타물품": ["안경 선글라스", "매장문화재 기타"],
    "유류품": ["무연고유류품", "유류품"]
}

# 색상 목록
COLORS = ["검정색", "흰색", "회색", "빨간색", "파란색", "녹색", "노란색", "갈색", "분홍색", "보라색", "주황색", "기타"]

# 설명 템플릿
FOUND_DESCRIPTIONS = [
    "{color} {category}를 {location}에서 발견했습니다.",
    "{location}에서 {color} {subcategory}를 주웠습니다.",
    "{color}색 {category}, {feature} 특징이 있습니다.",
    "{location} 근처에서 {color} {subcategory}를 습득했습니다.",
    "{feature} 특징의 {color} {category}입니다."
]

LOST_DESCRIPTIONS = [
    "{color} {category}를 {location}에서 잃어버렸습니다.",
    "{location}에서 {color} {subcategory}를 두고 내렸습니다.",
    "{color}색 {category}, {feature} 특징이 있습니다.",
    "{location} 근처에서 {color} {subcategory}를 분실했습니다.",
    "{feature} 특징의 {color} {category}를 찾고 있습니다."
]

# 장소 목록
LOCATIONS = ["지하철", "버스", "카페", "공원", "쇼핑몰", "도서관", "학교", "식당", "회사", "영화관"]

# 특징 목록
FEATURES = {
    "가방": ["손잡이가 긴", "지퍼가 고장난", "로고가 있는", "작은 사이즈의", "큰 사이즈의"],
    "귀금속": ["빛나는", "소형", "독특한 디자인의", "오래된", "새것 같은"],
    "도서용품": ["표지가 찢어진", "필기가 있는", "새책 같은", "오래된", "책갈피가 꽂힌"],
    "전자기기": ["케이스가 있는", "화면이 깨진", "배터리가 적은", "새것 같은", "스크래치가 있는"],
    "지갑": ["가죽", "천", "지퍼형", "접이식", "카드 수납공간이 많은"]
}

# 기본 특징 (특정 카테고리에 없는 경우 사용)
DEFAULT_FEATURES = ["작은", "큰", "중간 크기의", "새것 같은", "오래된", "깨끗한", "약간 손상된"]

def create_test_data(num_found_items=100, num_lost_items=20):
    """테스트용 습득물/분실물 데이터 생성"""
    
    # 데이터 저장 디렉토리 생성
    data_dir = Path("test_data")
    data_dir.mkdir(exist_ok=True)
    
    # 습득물 데이터 생성
    found_items = []
    for i in range(1, num_found_items + 1):
        main_category = random.choice(list(CATEGORIES.keys()))
        sub_categories = CATEGORIES[main_category]
        subcategory = random.choice(sub_categories)
        color = random.choice(COLORS)
        location = random.choice(LOCATIONS)
        
        # 카테고리별 특징 선택 또는 기본 특징 사용
        feature_list = FEATURES.get(main_category, DEFAULT_FEATURES)
        feature = random.choice(feature_list)
        
        # 설명 생성
        description_template = random.choice(FOUND_DESCRIPTIONS)
        description = description_template.format(
            color=color,
            category=main_category,
            subcategory=subcategory,
            location=location,
            feature=feature
        )
        
        # 이미지 경로 (실제 파일은 없고 경로만 생성)
        # 실제 구현 시에는 이미지를 다운로드하거나 준비해야 함
        has_image = random.random() > 0.3  # 70% 확률로 이미지 존재
        image_path = f"test_data/found_{i}.jpg" if has_image else None
        
        found_items.append({
            "id": i,
            "main_category": main_category,
            "subcategory": subcategory,
            "color": color,
            "description": description,
            "location": location,
            "image_path": image_path,
            "date_found": f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        })
    
    # 분실물 데이터 생성
    lost_items = []
    for i in range(1, num_lost_items + 1):
        main_category = random.choice(list(CATEGORIES.keys()))
        sub_categories = CATEGORIES[main_category]
        subcategory = random.choice(sub_categories)
        color = random.choice(COLORS)
        location = random.choice(LOCATIONS)
        
        feature_list = FEATURES.get(main_category, DEFAULT_FEATURES)
        feature = random.choice(feature_list)
        
        description_template = random.choice(LOST_DESCRIPTIONS)
        description = description_template.format(
            color=color,
            category=main_category,
            subcategory=subcategory,
            location=location,
            feature=feature
        )
        
        has_image = random.random() > 0.3
        image_path = f"test_data/lost_{i}.jpg" if has_image else None
        
        lost_items.append({
            "id": i,
            "main_category": main_category,
            "subcategory": subcategory,
            "color": color,
            "description": description,
            "location": location,
            "image_path": image_path,
            "date_lost": f"2023-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}"
        })
    
    # JSON 파일로 저장
    with open(data_dir / "found_items.json", "w", encoding="utf-8") as f:
        json.dump(found_items, f, ensure_ascii=False, indent=2)
    
    with open(data_dir / "lost_items.json", "w", encoding="utf-8") as f:
        json.dump(lost_items, f, ensure_ascii=False, indent=2)
    
    print(f"생성된 데이터: {num_found_items}개의 습득물, {num_lost_items}개의 분실물")
    print(f"데이터 저장 위치: {data_dir.absolute()}")
    
    # 샘플 이미지 생성에 대한 메시지
    print("\n주의: 실제 이미지 파일은 생성되지 않았습니다.")
    print("테스트를 위해 실제 이미지 데이터셋을 준비하거나, 생성형 AI로 카테고리별 샘플 이미지를 생성해야 합니다.")
    
    return found_items, lost_items

if __name__ == "__main__":
    # 샘플 데이터 생성 (100개의 습득물, 20개의 분실물)
    found_items, lost_items = create_test_data(100, 20)
    
    # 샘플 출력
    print("\n습득물 샘플:")
    for item in found_items[:3]:
        print(f"- {item['main_category']} ({item['subcategory']}): {item['description']}")
    
    print("\n분실물 샘플:")
    for item in lost_items[:3]:
        print(f"- {item['main_category']} ({item['subcategory']}): {item['description']}")