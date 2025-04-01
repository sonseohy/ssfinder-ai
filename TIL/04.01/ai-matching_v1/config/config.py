import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# API 설정
POLICE_API_SERVICE_KEY = os.getenv('POLICE_API_SERVICE_KEY')
POLICE_API_URL = 'http://apis.data.go.kr/1320000/LosfundInfoInqireService/getLosfundInfoAccToClAreaPd'

# 모델 설정
CLIP_MODEL_NAME = "ViT-B/32"
BLIP_MODEL_NAME = "blip2_opt"
BLIP_MODEL_SIZE = "pretrain_opt2.7b"

# 장치 설정
DEVICE = "cuda" if os.getenv('USE_GPU', 'True').lower() == 'true' and os.getenv('CUDA_VISIBLE_DEVICES') else "cpu"

# 데이터 처리
IMAGE_SIZE = 224
BATCH_SIZE = 32
MAX_TOKENS = 77

# 매칭 설정
SIMILARITY_THRESHOLD = 0.7
TOP_K_CANDIDATES = 50
MAX_RESULTS = 10

# 랭킹 알고리즘을 위한 가중치 파라미터
WEIGHTS = {
    'object_class_match': 0.3,
    'clip_similarity': 0.25,
    'attribute_match': 0.35,
    'metadata_match': 0.1
}

# 사용자 정의 클래스 (CLIP 분류용)
CUSTOM_CLASSES = [
    # 가방 관련
    "bag", "backpack", "handbag", "shopping bag", "여성용 가방", "남성용 가방", "쇼핑백",
    
    # 귀금속/액세서리
    "jewelry", "ring", "necklace", "earring", "watch", "반지", "목걸이", "귀걸이", "시계",
    
    # 도서
    "book", "textbook", "novel", "comic book", "학습서적", "소설", "만화책", "컴퓨터서적",
    
    # 서류
    "document", "certificate", "contract", "identification", "license", "서류", "증명서", "계약서", "신분증", "면허증",
    
    # 산업용품/공구
    "tool", "hammer", "screwdriver", "wrench", "공구", "망치", "드라이버", "렌치",
    
    # 쇼핑백/쇼핑물품
    "shopping bag", "shopping item", "쇼핑백", "쇼핑물품",
    
    # 스포츠용품
    "sports equipment", "baseball", "soccer ball", "tennis racket", "스포츠용품", "야구공", "축구공", "테니스라켓",
    
    # 악기
    "musical instrument", "guitar", "piano", "violin", "drum", "악기", "기타", "피아노", "바이올린", "드럼",
    
    # 유가증권
    "stock certificate", "bond", "check", "receipt", "주식", "채권", "수표", "영수증",
    
    # 의류
    "clothing", "shirt", "pants", "jacket", "coat", "hat", "dress", "suit", "uniform", 
    "옷", "셔츠", "바지", "재킷", "코트", "모자", "드레스", "정장", "유니폼",
    
    # 자동차 관련
    "car", "car key", "car license", "자동차", "차키", "자동차등록증", "운전면허증",
    
    # 전자기기 (애플 제품)
    "iphone", "ipad", "macbook", "airpods", "earpods", "apple watch",
    
    # 전자기기 (삼성 제품)
    "samsung phone", "galaxy phone", "galaxy s", "galaxy note", "galaxy tab", 
    "galaxy watch", "galaxy buds", "samsung tablet", "samsung earbuds",
    
    # 전자기기 (삼성 폴더블)
    "galaxy z flip", "galaxy z fold", "samsung foldable phone", "foldable phone",
    "samsung folding phone", "폴더블 스마트폰", "접이식 휴대폰", "갤럭시 Z 플립", "갤럭시 Z 폴드",
    
    # 전자기기 (일반)
    "smartphone", "phone", "tablet", "wireless earbuds", "headphones", "headset",
    "laptop", "camera", "smart watch", "earbuds", "태블릿", "휴대폰", "노트북", "카메라",
    
    # 지갑/카드
    "wallet", "purse", "credit card", "id card", "transportation card", "membership card", 
    "지갑", "여성용 지갑", "남성용 지갑", "카드", "신용카드", "교통카드", "멤버십카드",
    
    # 컴퓨터 관련
    "computer", "keyboard", "mouse", "monitor", "노트북", "키보드", "마우스", "모니터",
    
    # 현금/화폐
    "money", "cash", "korean won", "10000 won", "50000 won", "dollar", "euro", 
    "현금", "지폐", "동전", "한국 돈", "만원", "오만원", "천원", "오천원", "백원", "오백원",
    
    # 휴대폰
    "smartphone", "mobile phone", "iphone", "galaxy", "휴대폰", "스마트폰", "아이폰", "갤럭시", "휴대전화",
    
    # 기타 분실물
    "umbrella", "key", "glasses", "sunglasses", "water bottle", "우산", "열쇠", "안경", "선글라스", "물병"
]

# 속성 추출을 위한 VQA 질문
ATTRIBUTE_QUESTIONS = {
    'color': [
        "What is the color of this item?",
        "What color is this object?",
        "Describe the color of this item."
    ],
    'brand': [
        "What brand is this item?",
        "Can you identify the brand of this object?",
        "Is there a visible brand or logo on this item?"
    ],
    'material': [
        "What material is this item made of?",
        "What is the main material of this object?",
        "Is this item made of plastic, metal, or other material?"
    ],
    'condition': [
        "What is the condition of this item?",
        "Does this item look new or used?",
        "Is this item in good condition or damaged?"
    ],
    'type': [
        "What type of item is this?",
        "What is this object specifically?",
        "Can you identify what this item is?"
    ]
}

# 경찰청 API 분류 코드
POLICE_CATEGORY_CODES = {
    "가방": "PRA000",
    "귀금속": "PRB000",
    "도서": "PRC000",
    "서류": "PRD000",
    "산업용품": "PRE000",
    "쇼핑물": "PRF000",
    "스포츠용품": "PRG000",
    "악기": "PRH000",
    "유가증권": "PRI000",
    "의류": "PRJ000",
    "자동차": "PRK000",
    "전자기기": "PRL000",
    "지갑": "PRM000",
    "컴퓨터": "PRN000",
    "현금": "PRO000",
    "휴대폰": "PRP000",
    "기타물품": "PRZ000"
}

POLICE_COLOR_CODES = {
    "검정색": "CL1001",
    "흰색": "CL1002",
    "빨간색": "CL1003",
    "주황색": "CL1004",
    "노란색": "CL1005",
    "녹색": "CL1006",
    "파란색": "CL1007",
    "남색": "CL1008",
    "보라색": "CL1009",
    "분홍색": "CL1010",
    "갈색": "CL1011",
    "회색": "CL1012",
    "은색": "CL1013",
    "금색": "CL1014",
    "투명": "CL1015",
    "기타": "CL1016"
}