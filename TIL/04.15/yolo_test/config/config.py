import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# 모델 설정
CAPTION_MODEL = "Salesforce/blip-image-captioning-large"
VQA_MODEL = "Salesforce/blip-vqa-capfilt-large"

# 파파고 API 설정
NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# 이미지 처리 설정
MAX_IMAGE_SIZE = 1000

# 카테고리 정의 (영어)
CATEGORIES = [
    "electronics", "clothing", "bag", "wallet", "jewelry", "card", "id", "computer", "cash", "phone",
    "umbrella", "cosmetics", "sports equipment", "books", "others", "documents", "industrial goods", 
    "shopping bag", "musical instrument", "car", "miscellaneous", "earbuds", "smartwatch", "accessories"
]

# 카테고리 한영 매핑
CATEGORY_TRANSLATION = {
    "electronics": "전자기기",
    "clothing": "의류",
    "bag": "가방",
    "wallet": "지갑",
    "jewelry": "보석/액세서리",
    "card": "카드",
    "id": "신분증",
    "computer": "컴퓨터",
    "cash": "현금",
    "phone": "휴대폰",
    "phone case": "휴대폰 케이스",
    "umbrella": "우산",
    "cosmetics": "화장품",
    "sports equipment": "스포츠용품",
    "books": "도서",
    "others": "기타",
    "documents": "서류",
    "industrial goods": "산업용품",
    "shopping bag": "쇼핑백",
    "musical instrument": "악기",
    "car": "자동차",
    "miscellaneous": "기타",
    "earbuds": "무선이어버드",
    "smartwatch": "스마트워치",
    "accessories": "액세서리"
}

# 카테고리 매핑 (캡션의 일반적인 단어를 카테고리로 매핑)
CATEGORY_MAPPING = {
    "umbrella": "umbrella",
    "phone": "phone", 
    "smartphone": "phone",
    "cellphone": "phone",
    "mobile phone": "phone",
    "mobile device": "phone",
    "wallet": "wallet",
    "purse": "wallet",
    "billfold": "wallet",
    "money holder": "wallet",
    "card holder": "wallet",
    "bag": "bag",
    "handbag": "bag",
    "backpack": "bag",
    "tote": "bag",
    "crossbody bag": "bag",
    "computer": "computer",
    "laptop": "computer",
    "tablet": "electronics",
    "watch": "smartwatch",
    "smartwatch": "smartwatch",
    "wearable": "smartwatch",
    "smart watch": "smartwatch",
    "book": "books",
    "notebook": "books",
    "textbook": "books",
    "diary": "books",
    "headphones": "electronics",
    "earphones": "electronics",
    "earbuds": "earbuds",
    "wireless earbuds": "earbuds", 
    "galaxy buds": "earbuds",
    "bluetooth earbuds": "earbuds",
    "earpiece": "earbuds",
    "card": "card",
    "credit card": "card",
    "id card": "id",
    "identification": "id",
    "id": "id",
    "key": "others",
    "keys": "others",
    "glasses": "others",
    "sunglasses": "others",
    "camera": "electronics",
    "jewelry": "jewelry",
    "necklace": "jewelry",
    "bracelet": "jewelry",
    "ring": "jewelry",
    "accessories": "accessories",
    "charger": "electronics",
    "power bank": "electronics",
    "flash drive": "electronics",
    "usb drive": "electronics"
}

# 색상 목록 정의
COLORS = [
    "red", "blue", "green", "yellow", "black", "white", "gray", "grey", "brown", "purple", 
    "pink", "orange", "silver", "gold", "navy", "beige", "transparent", "multicolor", "teal",
    "turquoise", "maroon", "olive", "cyan", "magenta", "lavender", "indigo", "violet", "tan",
    "bronze", "copper", "cream", "burgundy", "khaki", "charcoal", "rose gold"
]

# 재질 목록 정의
MATERIALS = [
    "plastic", "metal", "leather", "fabric", "paper", "wood", "glass", "ceramic", "rubber",
    "cotton", "polyester", "nylon", "carbon fiber", "stone", "silicone", "aluminium", "steel",
    "cloth", "textile", "canvas", "denim", "wool", "synthetic", "composite", "unknown",
    "vinyl", "polycarbonate", "genuine leather", "faux leather", "pleather", "polyurethane",
    "mesh", "chrome", "titanium", "velvet", "suede", "stainless steel"
]

# 브랜드 연관 매핑 (제품 -> 브랜드 연결)
BRAND_ASSOCIATION = {
    # 삼성 제품
    "galaxy": "samsung",
    "samsung": "samsung",
    "galaxy s": "samsung",
    "galaxy note": "samsung",
    "galaxy tab": "samsung",
    "galaxy watch": "samsung",
    "galaxy buds": "samsung",
    "buds": "samsung",
    "galaxy buds": "samsung", 
    "wireless earbuds": "samsung",
    "bluetooth earbuds": "samsung",
    "earbuds": "samsung",
    "gear": "samsung",
    "galaxy gear": "samsung",
    "galaxy fold": "samsung",
    "galaxy flip": "samsung",
    "galaxy z": "samsung",
    "galaxy a": "samsung",
    "s series": "samsung",
    "note series": "samsung",
    "samsung phone": "samsung",
    "samsung tablet": "samsung",
    "samsung watch": "samsung",
    "samsung earbuds": "samsung",
    
    # LG 제품
    "gram": "lg",
    "lg phone": "lg",
    "lg g": "lg",
    "lg v": "lg",
    "optimus": "lg",
    
    # 지갑 브랜드
    "louis vuitton wallet": "louis vuitton",
    "lv wallet": "louis vuitton",
    "gucci wallet": "gucci",
    "prada wallet": "prada",
    "hermès wallet": "hermes",
    "hermes wallet": "hermes",
    "chanel wallet": "chanel",
    "coach wallet": "coach",
    "cartier wallet": "cartier",
    "montblanc wallet": "montblanc",
    "mont blanc wallet": "montblanc",
    "bottega veneta wallet": "bottega veneta",
    "ferragamo wallet": "ferragamo",
    "fendi wallet": "fendi",
    "tory burch wallet": "tory burch",
    "michael kors wallet": "michael kors",
    "burberry wallet": "burberry",
    "mulberry wallet": "mulberry",
    "dior wallet": "dior",
    "balenciaga wallet": "balenciaga",
    "saint laurent wallet": "saint laurent",
    "ysl wallet": "saint laurent",
    "versace wallet": "versace",
    "goyard wallet": "goyard",
    "celine wallet": "celine",
    "givenchy wallet": "givenchy",
    
    # 스포츠 브랜드
    "airmax": "nike",
    "air max": "nike",
    "air jordan": "nike",
    "adidas": "adidas",
    "ultraboost": "adidas",
    "yeezy": "adidas",
    "puma": "puma",
    "under armour": "under armour",
    "reebok": "reebok",
    "new balance": "new balance",
    "asics": "asics",
    
    # 전자제품 브랜드
    "sony": "sony",
    "jbl": "jbl",
    "bose": "bose",
    "beats": "beats",
    "lenovo": "lenovo",
    "thinkpad": "lenovo",
    "hp": "hp",
    "dell": "dell",
    "surface": "microsoft",
    "logitech": "logitech"
}

# 브랜드 번역 매핑
BRAND_TRANSLATION = {
    "samsung": "삼성",
    "lg": "엘지",
    "sony": "소니",
    "nike": "나이키",
    "adidas": "아디다스",
    "puma": "푸마",
    "reebok": "리복",
    "louis vuitton": "루이비통",
    "gucci": "구찌",
    "chanel": "샤넬",
    "prada": "프라다",
    "hermes": "에르메스",
    "coach": "코치",
    "dell": "델",
    "hp": "에이치피",
    "lenovo": "레노버",
    "asus": "아수스",
    "acer": "에이서",
    "timex": "타이맥스",
    "casio": "카시오",
    "seiko": "세이코",
    "citizen": "시티즌",
    "logitech": "로지텍",
    "microsoft": "마이크로소프트",
    "canon": "캐논",
    "nikon": "니콘",
    "jbl": "제이비엘",
    "bose": "보스",
    "sennheiser": "젠하이저",
    "samsonite": "쌤소나이트",
    "tumi": "투미",
    "kindle": "킨들",
    "google": "구글",
    "montblanc": "몽블랑",
    "bottega veneta": "보테가 베네타",
    "ferragamo": "페라가모",
    "fendi": "펜디",
    "tory burch": "토리버치",
    "michael kors": "마이클코어스",
    "burberry": "버버리",
    "mulberry": "멀버리",
    "dior": "디올",
    "balenciaga": "발렌시아가",
    "saint laurent": "생로랑",
    "versace": "베르사체",
    "goyard": "고야드",
    "celine": "셀린느",
    "givenchy": "지방시",
    "cartier": "까르띠에",
    "beats": "비츠",
    "under armour": "언더아머",
    "new balance": "뉴발란스",
    "asics": "아식스",
    "unknown": "알 수 없음"
}

# 자주 사용되는 제품 이름 한국어 매핑 (제목 생성용)
PRODUCT_TRANSLATION = {
    "phone case": "휴대폰 케이스",
    "phone": "휴대폰", 
    "umbrella": "우산",
    "wallet": "지갑",
    "purse": "지갑",
    "bag": "가방",
    "handbag": "핸드백",
    "backpack": "백팩",
    "laptop": "노트북",
    "computer": "컴퓨터",
    "watch": "시계",
    "smartwatch": "스마트워치",
    "book": "책",
    "headphones": "헤드셋",
    "earphones": "이어폰",
    "earbuds": "이어버드",
    "buds": "버즈",
    "galaxy buds": "갤럭시 버즈",
    "galaxy watch": "갤럭시 워치",
    "galaxy phone": "갤럭시 휴대폰",
    "camera": "카메라",
    "glasses": "안경",
    "tablet": "태블릿",
    "key": "열쇠",
    "keys": "열쇠",
    "card": "카드",
    "id card": "신분증",
    "bracelet": "팔찌",
    "necklace": "목걸이",
    "ring": "반지",
    "charger": "충전기",
    "power bank": "보조배터리"
}

# API 질문 템플릿
QUESTIONS = {
    "category": "What type of item is this? Choose from the following categories: {categories}",
    "color": "What is the main color of this item? Be specific and mention only the color.",
    "material": "What material is this item made of? If unknown, say 'unknown'.",
    "distinctive_features": "What are the distinctive features or unique aspects of this item?",
    "brand": "What is the brand of this item? If unknown, just say 'unknown'."
}