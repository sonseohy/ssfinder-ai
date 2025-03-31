import os
import requests
import xml.etree.ElementTree as ET
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import math
from collections import defaultdict
from dotenv import load_dotenv
from tqdm import tqdm
import re

# CLIP 모델 관련 라이브러리 임포트
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

# 환경 변수 로드
load_dotenv()

# 설정 값
TEXT_SIMILARITY_THRESHOLD = 0.5  # 텍스트 유사도 임계값 (조금 낮춤)
IMAGE_TEXT_CONSISTENCY_THRESHOLD = 0.3  # 이미지와 텍스트 일치성 임계값
NO_IMAGE_URL = "https://www.lost112.go.kr/lostnfs/images/sub/img02_no_img.gif"  # 이미지 없음 URL
KOREAN_CLIP_MODEL = "Bingsu/clip-vit-base-patch32-ko"  # 한국어 특화 CLIP 모델

# 이미지와 텍스트 기본 가중치 (동적으로 조정됨)
BASE_IMAGE_WEIGHT = 0.3  # 이미지 30%, 텍스트 70% 가중치

# 유사도 보너스/페널티 가중치
CATEGORY_MATCH_BONUS = 0.2  # 카테고리 일치 시 보너스
BRAND_MATCH_BONUS = 0.15   # 브랜드 일치 시 보너스
BRAND_MISMATCH_PENALTY = -0.15  # 브랜드 불일치 시 페널티
COLOR_MATCH_BONUS = 0.1    # 색상 일치 시 보너스
KEYWORD_MATCH_BONUS = 0.05  # 키워드 일치 시 보너스

# 카테고리 매핑
CATEGORY_MAPPING = {
    '휴대폰': ['휴대폰', '핸드폰', '스마트폰', '피처폰', '모바일폰', '폰', '전화기'],
    '지갑': ['지갑', '카드지갑', '명함지갑', '반지갑', '장지갑', '동전지갑', '머니클립'],
    '태블릿': ['태블릿', '패드', '탭'],
    '노트북': ['노트북', '랩탑', '컴퓨터', 'PC', '넷북'],
    '이어폰': ['이어폰', '헤드폰', '이어버드', '블루투스이어폰', '에어팟', '버즈'],
    '시계': ['시계', '워치', '손목시계', '스마트워치', '스마트밴드'],
    '카메라': ['카메라', 'DSLR', '디카', '디지털카메라', '렌즈', '캠코더'],
    '의류': ['의류', '옷', '상의', '하의', '자켓', '코트', '티셔츠', '바지', '셔츠', '니트', '스웨터', '후드', '패딩', '점퍼'],
    '신발': ['신발', '운동화', '구두', '슬리퍼', '슈즈', '샌들', '부츠', '워커', '로퍼', '스니커즈'],
    '가방': ['가방', '백팩', '핸드백', '크로스백', '숄더백', '파우치', '에코백', '클러치', '메신저백', '여행가방', '캐리어'],
    '악세서리': ['악세서리', '목걸이', '귀걸이', '팔찌', '반지', '시계', '머리핀', '헤어밴드', '모자', '장갑', '스카프', '넥타이', '벨트'],
    '전자기기': ['전자기기', '충전기', '보조배터리', '스피커', '웨어러블', '거치대', '케이블', '마우스', '키보드', 'USB'],
    '우산': ['우산', '양산', '접이식우산', '장우산', '자동우산', '골프우산', '3단우산'],
    '도서': ['도서', '책', '교재', '만화책', '소설', '문제집', '교과서', '잡지'],
    '안경': ['안경', '선글라스', '안경테', '안경케이스', '안경닦이', '콘택트렌즈'],
    '문서': ['문서', '카드', '신분증', '면허증', '여권', '통장', '기프티콘', '영수증', '티켓', '상품권']
}

# 색상 매핑
COLOR_MAPPING = {
    '검정': ['검정', '블랙', '흑색', '검은색', '검정색', '까만', '까만색', '블랙컬러', 'black'],
    '흰색': ['흰색', '화이트', '하얀', '하얀색', '화이트컬러', '백색', 'white'],
    '빨강': ['빨강', '빨간', '빨간색', '레드', '적색', '레드컬러', 'red'],
    '파랑': ['파랑', '파란', '파란색', '블루', '청색', '네이비', '남색', '진청색', '블루컬러', 'blue', 'navy'],
    '노랑': ['노랑', '노란', '노란색', '옐로우', '황색', '옐로우컬러', 'yellow'],
    '초록': ['초록', '초록색', '그린', '녹색', '그린컬러', 'green'],
    '주황': ['주황', '주황색', '오렌지', '오렌지색', '오렌지컬러', 'orange'],
    '보라': ['보라', '보라색', '퍼플', '바이올렛', '자주색', '퍼플컬러', 'purple', 'violet'],
    '분홍': ['분홍', '분홍색', '핑크', '핑크색', '핑크컬러', 'pink'],
    '갈색': ['갈색', '브라운', '베이지', '탄색', '브라운컬러', 'brown', 'beige'],
    '회색': ['회색', '그레이', '그레이컬러', '실버', '은색', 'gray', 'grey', 'silver'],
    '금색': ['금색', '골드', '골드컬러', 'gold']
}

# 브랜드 목록 - 다양한 제품을 생산하는 브랜드 (확장)
BRAND_LIST = [
    # 전자기기 브랜드
    '삼성', '애플', '엘지', 'LG', '샤오미', '화웨이', '소니', '파나소닉', '필립스', '캐논', '니콘', 
    '로지텍', '레노버', '에이수스', '마이크로소프트', 'MS', '인텔', 'AMD', '모토로라', '비보', '오포',
    '갤럭시', '아이폰', '아이패드', '맥북', '아이맥',
    
    # 패션 브랜드
    '나이키', '아디다스', '퓨마', '리복', '뉴발란스', '컨버스', '반스', '아식스', '언더아머', '휠라',
    
    # 명품 브랜드
    '구찌', '루이비통', '샤넬', '프라다', '에르메스', '발렌시아가', '생로랑', '페라가모', '롤렉스',
    '오메가', '타이틀리스트', '까르띠에', '스와로브스키', '몽블랑', '듀퐁', '꼼데가르송', '버버리',
    
    # 기타 브랜드
    '코치', '마이클코어스', 'MCM', '닥터마틴', '폴로', '네파', '블랙야크', '노스페이스', '코오롱', 
    '라코스테', '무신사', '유니클로', '자라', 'H&M', 'CK', '타미힐피거', '디젤', '리바이스'
]

# 특정 브랜드 제품명
SPECIFIC_PRODUCTS = {
    '휴대폰': [
        '갤럭시', '아이폰', '픽셀폰', 'v시리즈', 'g시리즈', 'iphone', 'galaxy s', 'galaxy note', 'galaxy z', 
        'galaxy flip', 'galaxy fold', 'mi', 'redmi', 'poco', 'honor', 'nova', 'xperia', 'oppo'
    ],
    '태블릿': [
        '아이패드', '갤럭시탭', '갤탭', 'ipad', 'galaxy tab', 'xiaomi pad', 'mi pad', 'surface', 'zenpad'
    ],
    '노트북': [
        '맥북', '갤럭시북', 'macbook', 'galaxy book', 'thinkpad', 'zenbook', 'ideapad', 'gram', '그램', 
        'surface', '서피스', 'pavilion', 'spectre', 'xps', 'chromebook', '크롬북', 'alienware'
    ],
    '이어폰': [
        '에어팟', '갤럭시버즈', '버즈', 'airpods', 'galaxy buds', 'freebuds', 'wf', 'qc', 'quietcomfort', 
        'soundcore', 'airdots', 'liberty', 'jabra', 'sony wf', 'sony wh', 'soundsport'
    ],
    '스마트워치': [
        '애플워치', 'apple watch', '갤럭시워치', 'galaxy watch', 'mi band', 'amazfit', 'huawei watch', 
        'fitbit', '핏빗', 'garmin', '가민'
    ]
}

# 기타 카테고리 관련 키워드
MISC_CATEGORIES = ['기타', '기타물품', '기타 물품', '기타 아이템', '미분류', '분류없음', '미상']

# CLIP 모델 로드
def load_clip_model():
    """
    한국어 CLIP 모델과 프로세서를 로드하는 함수
    """
    try:
        # 한국어 특화 CLIP 모델 로드 시도
        print(f"한국어 특화 CLIP 모델 로드 중: {KOREAN_CLIP_MODEL}")
        
        # 트랜스포머 캐시 디렉토리 설정
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
        
        # 모델 및 프로세서 로드
        model = CLIPModel.from_pretrained(KOREAN_CLIP_MODEL)
        processor = CLIPProcessor.from_pretrained(KOREAN_CLIP_MODEL)
        model_type = "Korean"
        
        # 모델 차원 확인 및 출력
        if hasattr(model.text_model.config, "hidden_size"):
            dim = model.text_model.config.hidden_size
            print(f"모델 임베딩 차원: {dim}")
        else:
            print("경고: 모델 차원을 확인할 수 없습니다.")
        
    except Exception as e:
        # 실패 시 기본 CLIP 모델 사용
        print(f"한국어 모델 로드 실패, 기본 CLIP 모델로 대체합니다: {str(e)}")
        
        model_name = "openai/clip-vit-base-patch32"
        
        try:
            model = CLIPModel.from_pretrained(model_name)
            processor = CLIPProcessor.from_pretrained(model_name)
            model_type = "English"
            
            # 모델 차원 확인 및 출력
            if hasattr(model.text_model.config, "hidden_size"):
                dim = model.text_model.config.hidden_size
                print(f"모델 임베딩 차원: {dim}")
            else:
                print("경고: 모델 차원을 확인할 수 없습니다.")
                
        except Exception as e2:
            print(f"기본 CLIP 모델 로드에도 실패했습니다: {str(e2)}")
            print("간단한 임베딩 모델로 대체합니다.")
            
            # 더 간단한 임베딩 모델 사용
            class SimpleEmbedding(nn.Module):
                def __init__(self, dim=512):
                    super().__init__()
                    self.embedding = nn.Embedding(10000, dim)
                    self.dim = dim
                
                def get_text_features(self, input_ids, **kwargs):
                    return self.embedding(input_ids[:, 0:1])
                
                def get_image_features(self, pixel_values, **kwargs):
                    batch_size = pixel_values.shape[0]
                    return torch.randn(batch_size, self.dim)
            
            model = SimpleEmbedding(dim=512)
            
            tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            
            class SimpleProcessor:
                def __init__(self, tokenizer):
                    self.tokenizer = tokenizer
                
                def __call__(self, text=None, images=None, **kwargs):
                    inputs = {}
                    if text is not None:
                        inputs.update(self.tokenizer(text, **kwargs))
                    if images is not None:
                        # 간단히 더미 값 반환
                        inputs["pixel_values"] = torch.ones((len(images), 3, 224, 224))
                    return inputs
                
                def image_processor(self, images, **kwargs):
                    return {"pixel_values": torch.ones((len(images), 3, 224, 224))}
            
            processor = SimpleProcessor(tokenizer)
            model_type = "Simple"
    
    # GPU 사용 가능하면 GPU로 모델 이동
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"{model_type} CLIP model loaded on {device}")
    return model, processor, device, model_type

# 경찰청 API에서 습득물 데이터 가져오기
def fetch_police_lost_items(service_key, num_items=5):
    """
    경찰청 API를 통해 최신 습득물 데이터를 가져옴
    """
    url = 'http://apis.data.go.kr/1320000/LosfundInfoInqireService/getLosfundInfoAccToClAreaPd'
    
    # 현재 날짜 기준으로 최신 데이터를 가져오기 위한 파라미터 설정
    params = {
        'serviceKey': service_key,
        'START_YMD': '20250101',    # 시작 날짜
        'END_YMD': '20250322',      # 종료 날짜
        'pageNo': '1',              # 첫 페이지부터 조회
        'numOfRows': str(num_items),# 가져올 항목 수
        'sort': 'DESC',             # 내림차순 정렬 (최신순)
        'sortField': 'fdYmd'        # 습득일자 기준으로 정렬
    }
    
    try:
        print(f"API 요청 URL: {url}")
        
        response = requests.get(url, params=params)
        
        print(f"응답 상태 코드: {response.status_code}")
        
        # API 응답 상태 확인
        if response.status_code == 200:
            try:
                root = ET.fromstring(response.content)
                
                # 응답에 오류 코드가 있는지 확인
                result_code = root.find('.//resultCode')
                result_msg = root.find('.//resultMsg')
                
                if result_code is not None and result_code.text != '00':
                    print(f"API 오류 코드: {result_code.text}, 메시지: {result_msg.text if result_msg is not None else '없음'}")
                    return []
                
                items = []
                
                # items 태그 아래의 item 요소들 추출
                item_elements = root.findall('.//item')
                if not item_elements:
                    print("응답에 item 요소가 없습니다.")
                
                for item in item_elements:
                    item_data = {}
                    
                    # 각 필드 추출
                    for child in item:
                        item_data[child.tag] = child.text
                    
                    items.append(item_data)
                
                print(f"API에서 {len(items)}개의 최신 습득물 데이터를 성공적으로 가져왔습니다.")
                return items
            except ET.ParseError as e:
                print(f"XML 파싱 오류: {str(e)}")
                print("응답 내용:")
                print(response.content.decode('utf-8', errors='replace'))
                return []
        else:
            print(f"API 호출 실패: {response.status_code}")
            print("응답 내용:")
            print(response.content.decode('utf-8', errors='replace'))
            return []
            
    except Exception as e:
        print(f"API 호출 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

# 습득물 이미지 다운로드
def download_image(url):
    """
    URL에서 이미지를 다운로드하여 PIL 이미지로 변환
    """
    # 기본 "이미지 없음" URL인 경우 건너뛰기
    if url == NO_IMAGE_URL:
        return None
        
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return Image.open(BytesIO(response.content)).convert('RGB')
        else:
            print(f"이미지 다운로드 실패: {response.status_code}")
            return None
    except Exception as e:
        print(f"이미지 다운로드 중 오류: {str(e)}")
        return None

# Base64 인코딩된 이미지 디코딩
def decode_base64_image(base64_string):
    """
    Base64 인코딩된 이미지를 PIL 이미지로 변환
    """
    try:
        # Base64 접두사 제거
        if "base64," in base64_string:
            base64_string = base64_string.split("base64,")[1]
            
        # 디코딩 및 이미지 로드
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        return image
    except Exception as e:
        print(f"Base64 이미지 디코딩 오류: {str(e)}")
        return None

# 텍스트에서 카테고리 추출
def extract_category(text, item_category=None):
    """
    텍스트와 기존 카테고리 정보를 사용하여 카테고리 추출
    
    Args:
        text (str): 물품 이름이나 설명이 포함된 텍스트
        item_category (str): 기존에 정의된 카테고리 정보 (있는 경우)
        
    Returns:
        str: 추출된 카테고리, None: 카테고리를 결정할 수 없음
    """
    if not text:
        return None
    
    # 텍스트를 소문자로 변환하고 전처리
    text = text.lower()
    
    # 1. 이미 카테고리 정보가 있으면 확인
    if item_category:
        # '기타' 카테고리 확인
        for misc in MISC_CATEGORIES:
            if misc.lower() in item_category.lower():
                # 기타 카테고리라도 텍스트에서 카테고리 정보를 추출해본다
                break
        
        # 카테고리 매핑에 있는지 확인
        for category, keywords in CATEGORY_MAPPING.items():
            for keyword in keywords:
                if keyword.lower() in item_category.lower():
                    return category
        
        # 카테고리 정보가 있지만 매핑되지 않은 경우 그대로 반환
        return item_category
    
    # 2. 특정 제품명 확인 (가장 높은 우선순위)
    for category, products in SPECIFIC_PRODUCTS.items():
        for product in products:
            if product.lower() in text.lower():
                return category
    
    # 3. 일반 카테고리 키워드 확인
    for category, keywords in CATEGORY_MAPPING.items():
        for keyword in keywords:
            if keyword.lower() in text.lower():
                return category
    
    # 4. 카테고리를 찾을 수 없는 경우
    return None

# 텍스트에서 브랜드 추출
def extract_brand(text):
    """
    텍스트에서 브랜드 정보 추출
    
    Args:
        text (str): 물품 이름이나 설명이 포함된 텍스트
        
    Returns:
        str: 추출된 브랜드, None: 브랜드를 찾을 수 없음
    """
    if not text:
        return None
    
    text = text.lower()
    
    # 브랜드 목록에서 일치하는 것 찾기
    for brand in BRAND_LIST:
        # 브랜드명이 다른 단어의 일부가 아니라 독립된 단어인지 확인 (정규식 사용)
        pattern = r'\b' + re.escape(brand.lower()) + r'\b'
        if re.search(pattern, text.lower()):
            return brand
    
    return None

# 텍스트에서 색상 추출
def extract_color(text):
    """
    텍스트에서 색상 정보 추출
    
    Args:
        text (str): 물품 이름이나 설명이 포함된 텍스트
        
    Returns:
        str: 표준화된 색상명, None: 색상을 찾을 수 없음
    """
    if not text:
        return None
    
    text = text.lower()
    
    # 색상 매핑에서 일치하는 것 찾기
    for standard_color, variants in COLOR_MAPPING.items():
        for color_variant in variants:
            if color_variant.lower() in text.lower():
                return standard_color
    
    return None

# 습득물 데이터에서 텍스트 정보 추출
def extract_text_from_item(item, model_type="Korean"):
    """
    습득물 데이터에서 관련 텍스트 정보를 추출하여 하나의 문자열로 결합
    """
    text_parts = []
    
    # 필드 라벨 설정 (모델 타입에 따라 한글 또는 영어로)
    if model_type == "Korean":
        fields_to_extract = {
            'fdPrdtNm': '물품명',       # 물품명
            'fdPrdtClNm': '분류',      # 분류명
            'prdtClNm': '분류',        # 대체 분류명
            'fdSbjt': '제목',          # 제목
            'clrNm': '색상',           # 색상
            'fdPlace': '습득장소',      # 습득 장소
            'depPlace': '보관장소',     # 보관 장소
            'fdYmd': '습득일자',        # 습득 일자
            'fdHor': '습득시간',        # 습득 시간
            'uniq': '특이사항',         # 특이사항
            'csteSbjt': '내용',        # 내용
            'atchFileId': '파일ID',    # 첨부파일 ID
            'fndKeepOrgnSeNm': '보관기관' # 보관 기관명
        }
    else:  # English 모델
        fields_to_extract = {
            'fdPrdtNm': 'Item name',       # 물품명
            'fdPrdtClNm': 'Category',      # 분류명
            'prdtClNm': 'Category',        # 대체 분류명
            'fdSbjt': 'Title',             # 제목
            'clrNm': 'Color',              # 색상
            'fdPlace': 'Found place',      # 습득 장소
            'depPlace': 'Storage place',   # 보관 장소
            'fdYmd': 'Found date',         # 습득 일자
            'fdHor': 'Found time',         # 습득 시간
            'uniq': 'Unique features',     # 특이사항
            'csteSbjt': 'Details',         # 내용
            'atchFileId': 'File ID',       # 첨부파일 ID
            'fndKeepOrgnSeNm': 'Organization' # 보관 기관명
        }
    
    for field, label in fields_to_extract.items():
        if field in item and item[field]:
            text_parts.append(f"{label}: {item[field]}")
    
    # 결합된 텍스트 반환
    combined_text = " ".join(text_parts)
    
    # 텍스트가 너무 길면 잘라내기 (CLIP의 토큰 제한 고려)
    if len(combined_text) > 200:
        combined_text = combined_text[:200]
        
    return combined_text if combined_text else "정보 없음" if model_type == "Korean" else "No item information"

# 텍스트 임베딩 생성
def generate_text_embedding(model, processor, text, device):
    """
    텍스트에 대한 임베딩 생성
    """
    try:
        with torch.no_grad():
            # 텍스트가 너무 길면 잘라내기
            if len(text) > 200:
                text = text[:200]
            
            # 텍스트 토큰화 및 임베딩 생성
            text_inputs = processor.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(device)
            
            text_features = model.get_text_features(**text_inputs)
            
            # 정규화
            text_embedding = text_features / text_features.norm(dim=1, keepdim=True)
            
            # 임베딩을 numpy 배열로 변환
            embedding = text_embedding[0].cpu().numpy()
            return embedding
    
    except Exception as e:
        print(f"텍스트 임베딩 생성 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # 오류 발생 시 랜덤 임베딩 반환
        try:
            if hasattr(model.text_model.config, "hidden_size"):
                dim = model.text_model.config.hidden_size
            else:
                dim = 768
        except:
            dim = 768
        
        print(f"랜덤 텍스트 임베딩 생성 (차원: {dim})")
        random_embedding = np.random.random(dim)
        return random_embedding / np.linalg.norm(random_embedding)

# 이미지 임베딩 생성
def generate_image_embedding(model, processor, image, device):
    """
    이미지에 대한 임베딩 생성
    """
    if image is None:
        return None
    
    try:
        with torch.no_grad():
            # 이미지 전처리 및 임베딩 생성
            image_inputs = processor.image_processor(
                images=[image], 
                return_tensors="pt"
            ).to(device)
            
            image_features = model.get_image_features(**image_inputs)
            
            # 정규화
            image_embedding = image_features / image_features.norm(dim=1, keepdim=True)
            
            # 임베딩을 numpy 배열로 변환
            embedding = image_embedding[0].cpu().numpy()
            return embedding
    
    except Exception as e:
        print(f"이미지 임베딩 생성 오류: {str(e)}")
        return None

# 순수한 코사인 유사도 계산
def pure_cosine_similarity(embedding1, embedding2):
    """
    두 임베딩 벡터 간의 순수한 코사인 유사도 계산
    
    Args:
        embedding1 (numpy.ndarray): 첫 번째 임베딩 벡터
        embedding2 (numpy.ndarray): 두 번째 임베딩 벡터
        
    Returns:
        float: 코사인 유사도 (-1~1 사이 값)
    """
    if embedding1 is None or embedding2 is None:
        return 0.0
    
    # 벡터 차원 확인 및 조정
    if embedding1.shape != embedding2.shape:
        min_dim = min(embedding1.shape[0], embedding2.shape[0])
        embedding1 = embedding1[:min_dim]
        embedding2 = embedding2[:min_dim]
    
    # NaN 또는 Inf 값 처리
    embedding1 = np.nan_to_num(embedding1)
    embedding2 = np.nan_to_num(embedding2)
    
    # 영벡터 체크
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # 정규화
    embedding1 = embedding1 / norm1
    embedding2 = embedding2 / norm2
    
    # 코사인 유사도 계산
    similarity = np.dot(embedding1, embedding2)
    
    # 부동소수점 오차 방지 (-1~1 사이로 제한)
    similarity = max(-1.0, min(1.0, similarity))
    
    return similarity

# 이미지-텍스트 일관성 검사
def check_image_text_consistency(item_image_embedding, item_text_embedding):
    """
    아이템의 이미지와 텍스트가 얼마나 일관성이 있는지 검사
    """
    if item_image_embedding is None or item_text_embedding is None:
        return 0.0  # 이미지가 없으면 일관성이 없음
    
    # 이미지와 텍스트 임베딩 간의 코사인 유사도 계산
    consistency = pure_cosine_similarity(item_image_embedding, item_text_embedding)
    return consistency

# 아이템에서 카테고리 정보 추출
def get_category_from_item(item):
    """
    아이템에서 카테고리 정보 추출
    
    Args:
        item (dict): 아이템 정보
    
    Returns:
        str: 카테고리 정보 또는 None
    """
    # 1. 공식 카테고리 필드 확인
    category_from_field = None
    for field in ['fdPrdtClNm', 'prdtClNm']:
        if field in item and item[field]:
            category_from_field = item[field]
            break
    
    # 2. 물품명에서 추출
    product_name = item.get('fdPrdtNm', '')
    
    # 3. 카테고리 추출 (기존 카테고리 정보 참고)
    return extract_category(product_name, category_from_field)

# 유사도 계산을 위한 아이템 메타데이터 추출
def extract_item_metadata(item, text):
    """
    아이템에서 유사도 계산에 사용할 메타데이터(카테고리, 브랜드, 색상) 추출
    
    Args:
        item (dict): 원본 아이템 데이터
        text (str): 아이템 설명 텍스트
        
    Returns:
        dict: 추출된 메타데이터
    """
    # 카테고리 추출
    category_from_field = None
    for field in ['fdPrdtClNm', 'prdtClNm']:
        if field in item and item[field]:
            category_from_field = item[field]
            break
    
    product_name = item.get('fdPrdtNm', '')
    category = extract_category(product_name, category_from_field)
    
    # 브랜드 추출 (물품명과 텍스트 모두에서 시도)
    brand = extract_brand(product_name) or extract_brand(text)
    
    # 색상 추출 (색상 필드 또는 텍스트에서)
    color_from_field = item.get('clrNm', '')
    color = extract_color(color_from_field) or extract_color(text) or extract_color(product_name)
    
    return {
        'category': category,
        'brand': brand,
        'color': color,
        'is_miscellaneous': category_from_field in MISC_CATEGORIES if category_from_field else False
    }

# 계층적 유사도 분석
def analyze_hierarchical_similarity(user_item, found_item):
    """
    사용자 분실물과 습득물 간의 계층적 유사도 분석:
    1. 순수 코사인 유사도 (기본)
    2. 카테고리 매칭 (최우선)
    3. 브랜드 매칭/불일치 (중요)
    4. 색상 매칭 (보조)
    
    Args:
        user_item (dict): 사용자 분실물 정보
        found_item (dict): 습득물 정보
        
    Returns:
        dict: 유사도 분석 결과
    """
    item_id = found_item['original'].get('atcId', 'unknown')
    print(f"\n===== 아이템 ID: {item_id} 계층적 유사도 분석 시작 =====")
    print(f"아이템 텍스트: {found_item.get('text', '')}")
    
    # 1. 기본 유사도: 순수 텍스트 코사인 유사도 계산
    base_text_similarity = pure_cosine_similarity(user_item['text_embedding'], found_item['text_embedding'])
    print(f"기본 텍스트 코사인 유사도: {base_text_similarity:.4f}")
    
    # 2. 이미지 코사인 유사도 계산 (이미지가 있는 경우)
    image_similarity = 0.0
    if user_item['image_embedding'] is not None and found_item['image_embedding'] is not None:
        image_similarity = pure_cosine_similarity(user_item['image_embedding'], found_item['image_embedding'])
        print(f"이미지 코사인 유사도: {image_similarity:.4f}")
        
        # 이미지-텍스트 일관성 검사
        user_consistency = check_image_text_consistency(user_item['image_embedding'], user_item['text_embedding'])
        found_consistency = check_image_text_consistency(found_item['image_embedding'], found_item['text_embedding'])
        
        print(f"사용자 이미지-텍스트 일관성: {user_consistency:.4f}")
        print(f"습득물 이미지-텍스트 일관성: {found_consistency:.4f}")
        
        # 이미지 가중치를 일관성에 따라 동적 조정
        image_weight = BASE_IMAGE_WEIGHT
        if user_consistency > IMAGE_TEXT_CONSISTENCY_THRESHOLD and found_consistency > IMAGE_TEXT_CONSISTENCY_THRESHOLD:
            # 양쪽 모두 일관성이 높으면 이미지 가중치 증가
            image_weight = min(0.5, BASE_IMAGE_WEIGHT * 1.5)  # 최대 50%까지 증가
            print(f"이미지-텍스트 일관성이 높아 이미지 가중치 증가: {image_weight:.2f}")
    else:
        print("이미지 정보가 없어 이미지 유사도는 계산하지 않습니다.")
        image_weight = 0.0
    
    # 3. 메타데이터 추출
    user_metadata = extract_item_metadata(user_item['original'], user_item.get('text', ''))
    found_metadata = extract_item_metadata(found_item['original'], found_item.get('text', ''))
    
    print(f"사용자 메타데이터: {user_metadata}")
    print(f"습득물 메타데이터: {found_metadata}")
    
    # 4. 메타데이터 유사도 보너스/페널티 계산
    similarity_adjustments = []
    adjusted_similarity = base_text_similarity
    
    # 4.1 카테고리 매칭 (가장 중요)
    category_match = False
    if user_metadata['category'] and found_metadata['category']:
        if user_metadata['category'] == found_metadata['category']:
            category_match = True
            bonus = CATEGORY_MATCH_BONUS
            adjusted_similarity += bonus
            similarity_adjustments.append(f"카테고리 일치 보너스: +{bonus:.2f}")
        elif not found_metadata['is_miscellaneous']:  # 기타 카테고리가 아닌 경우에만 불일치 판단
            # 카테고리 불일치는 큰 페널티 없이 기록만 함
            similarity_adjustments.append("카테고리 불일치")
    
    # 4.2 브랜드 매칭 (중요)
    if user_metadata['brand'] and found_metadata['brand']:
        if user_metadata['brand'] == found_metadata['brand']:
            bonus = BRAND_MATCH_BONUS
            adjusted_similarity += bonus
            similarity_adjustments.append(f"브랜드 일치 보너스: +{bonus:.2f}")
        else:
            # 브랜드 불일치는 큰 페널티 적용 (브랜드가 다르면 다른 제품)
            penalty = BRAND_MISMATCH_PENALTY
            adjusted_similarity += penalty
            similarity_adjustments.append(f"브랜드 불일치 페널티: {penalty:.2f}")
    
    # 4.3 색상 매칭 (보조적)
    if user_metadata['color'] and found_metadata['color']:
        if user_metadata['color'] == found_metadata['color']:
            bonus = COLOR_MATCH_BONUS
            adjusted_similarity += bonus
            similarity_adjustments.append(f"색상 일치 보너스: +{bonus:.2f}")
    
    # 5. 최종 유사도 계산 (텍스트 + 이미지 + 메타데이터 조정)
    final_similarity = adjusted_similarity
    if image_similarity > 0:
        # 텍스트+메타데이터 조정 유사도와 이미지 유사도의 가중 평균
        final_similarity = (adjusted_similarity * (1 - image_weight)) + (image_similarity * image_weight)
        similarity_adjustments.append(f"이미지 유사도 반영 (가중치: {image_weight:.2f})")
    
    # 최종 유사도 스코어 범위 조정 (-1~1)
    final_similarity = max(-1.0, min(1.0, final_similarity))
    
    print(f"조정된 유사도: {adjusted_similarity:.4f}")
    print(f"최종 유사도: {final_similarity:.4f}")
    print(f"조정 내역: {similarity_adjustments}")
    
    # 결과 구성
    result = {
        'base_similarity': base_text_similarity,
        'image_similarity': image_similarity,
        'adjusted_similarity': adjusted_similarity,
        'final_similarity': final_similarity,
        'category_match': category_match,
        'user_metadata': user_metadata,
        'found_metadata': found_metadata,
        'adjustments': similarity_adjustments
    }
    
    print("===== 유사도 분석 완료 =====\n")
    
    return result

# 계층적 유사도 결과 처리
def process_hierarchical_results(similarity_results):
    """
    계층적 유사도 결과 처리
    - 카테고리별로 그룹화
    - 각 카테고리 내에서 최종 유사도로 정렬
    
    Args:
        similarity_results (list): (item, similarity_result) 튜플의 리스트
        
    Returns:
        dict: 처리된 결과
    """
    # 1. 결과를 최종 유사도로 정렬
    sorted_by_similarity = sorted(
        similarity_results, 
        key=lambda x: x[1]['final_similarity'], 
        reverse=True
    )
    
    # 2. 임계값 이상인 결과만 필터링
    filtered_results = [
        (item, result) for item, result in sorted_by_similarity 
        if result['final_similarity'] >= TEXT_SIMILARITY_THRESHOLD
    ]
    
    # 3. 카테고리별로 그룹화
    category_groups = defaultdict(list)
    uncategorized = []
    
    for item, result in filtered_results:
        found_category = result['found_metadata']['category']
        if found_category:
            category_groups[found_category].append((item, result))
        else:
            uncategorized.append((item, result))
    
    # 4. 사용자 카테고리 결정
    user_category = None
    user_brand = None
    user_color = None
    
    for _, result in filtered_results:
        if result['user_metadata']['category']:
            user_category = result['user_metadata']['category']
        if result['user_metadata']['brand']:
            user_brand = result['user_metadata']['brand']
        if result['user_metadata']['color']:
            user_color = result['user_metadata']['color']
        
        if user_category and user_brand and user_color:
            break
    
    # 5. 결과 구성
    final_results = {
        'user_metadata': {
            'category': user_category,
            'brand': user_brand,
            'color': user_color
        },
        'category_groups': dict(category_groups),
        'uncategorized': uncategorized,
        'all_sorted': filtered_results
    }
    
    return final_results

# 유사도 결과 시각화
def visualize_similarities(user_image, similar_items, similarities, color_values=None):
    """
    유사도 결과를 시각적으로 표시
    
    Args:
        user_image: 사용자 이미지
        similar_items: 유사한 습득물 목록
        similarities: 유사도 점수 목록
        color_values: 색상 표시를 위한 값 (옵션)
    """
    n = min(5, len(similar_items))  # 최대 5개까지 표시
    
    plt.figure(figsize=(15, 10))
    
    # 한글 폰트 설정 (가능한 경우)
    try:
        plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우용 한글 폰트
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
    except:
        print("한글 폰트 설정 실패, 기본 폰트를 사용합니다.")
    
    # 사용자 이미지 표시
    plt.subplot(1, n+1, 1)
    if user_image:
        plt.imshow(user_image)
        plt.title("사용자 이미지", fontsize=12)
    else:
        plt.text(0.5, 0.5, "이미지 없음", ha='center', va='center', fontsize=12)
        plt.title("사용자 분실물", fontsize=12)
    plt.axis('off')
    
    # 유사한 습득물 이미지 표시
    for i in range(n):
        plt.subplot(1, n+1, i+2)
        
        # 이미지가 있으면 표시, 없으면 텍스트 표시
        item = similar_items[i]
        if 'image' in item and item['image'] is not None:
            plt.imshow(item['image'])
        else:
            plt.text(0.5, 0.5, "이미지 없음", ha='center', va='center', fontsize=12)
        
        # 유사도와 물품명 표시
        title_text = f"유사도: {similarities[i]:.2f}\n"
        
        # 색상 정보 추가 (있으면)
        if color_values and i < len(color_values) and color_values[i]:
            color_info = color_values[i]
            # 색상 정보를 표시할 형식 지정
            if isinstance(color_info, dict):
                color_text = ", ".join([f"{k}: {v}" for k, v in color_info.items() if v])
                title_text += f"{color_text}\n"
        
        # 물품명 추가
        if isinstance(item, dict) and 'original' in item and 'fdPrdtNm' in item['original'] and item['original']['fdPrdtNm']:
            title_text += f"{item['original']['fdPrdtNm']}"
        elif isinstance(item, dict) and 'fdPrdtNm' in item and item['fdPrdtNm']:
            title_text += f"{item['fdPrdtNm']}"
        else:
            title_text += "정보 없음"
            
        plt.title(title_text, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 사용자 카테고리 수동 설정 (하드코딩된 분실물 테스트용)
def set_user_category(user_description, manual_category=None):
    """
    사용자 분실물 설명에서 카테고리를 추출하거나 수동으로 설정
    
    Args:
        user_description (str): 사용자 분실물 설명
        manual_category (str): 수동으로 설정할 카테고리 (옵션)
        
    Returns:
        str: 설정된 카테고리
    """
    if manual_category:
        print(f"사용자 카테고리 수동 설정: {manual_category}")
        return manual_category
    
    # 자동 추출 시도
    extracted_category = extract_category(user_description)
    if extracted_category:
        print(f"사용자 카테고리 자동 추출: {extracted_category}")
        return extracted_category
    
    print("카테고리를 추출할 수 없습니다.")
    return None

# 메인 함수: 사용자 분실물과 경찰청 습득물 유사도 비교 (계층적 접근)
def compare_lost_and_found(user_image_path, user_description, manual_category=None, num_items=10, analyze_top_n=3):
    """
    사용자 분실물과 경찰청 습득물 데이터의 유사도 비교
    (계층적 접근: 카테고리 > 브랜드 > 텍스트 유사도 > 색상)
    
    Args:
        user_image_path (str): 사용자 이미지 파일 경로 또는 Base64 인코딩 문자열
        user_description (str): 사용자 분실물 설명
        manual_category (str): 사용자가 수동으로 지정한 카테고리 (옵션)
        num_items (int): 가져올 습득물 데이터 수
        analyze_top_n (int): 상세 분석할 상위 아이템 수
        
    Returns:
        dict: 계층적 접근으로 처리된 결과
    """
    # 1. CLIP 모델 로드
    model, processor, device, model_type = load_clip_model()
    
    # 2. 사용자 이미지 로드
    user_image = None
    if user_image_path:
        if user_image_path.startswith('data:image') or user_image_path.startswith('http'):
            # Base64 이미지 또는 URL
            user_image = decode_base64_image(user_image_path) if user_image_path.startswith('data:image') else download_image(user_image_path)
        else:
            # 로컬 파일 경로
            try:
                user_image = Image.open(user_image_path).convert('RGB')
            except Exception as e:
                print(f"사용자 이미지 로드 오류: {str(e)}")
                print("텍스트만으로 진행합니다.")
    
    # 3. 사용자 분실물 텍스트 및 이미지 임베딩 생성
    user_text_embedding = generate_text_embedding(model, processor, user_description, device)
    user_image_embedding = generate_image_embedding(model, processor, user_image, device) if user_image else None
    
    # 4. 사용자 분실물 정보 및 메타데이터 설정
    # 사용자 카테고리 설정 (수동 또는 자동)
    user_category = set_user_category(user_description, manual_category)
    
    # 사용자 브랜드 추출
    user_brand = extract_brand(user_description)
    if user_brand:
        print(f"사용자 브랜드 추출: {user_brand}")
    
    # 사용자 색상 추출
    user_color = extract_color(user_description)
    if user_color:
        print(f"사용자 색상 추출: {user_color}")
    
    # 사용자 분실물 메타데이터 구성
    user_metadata = {
        'fdPrdtNm': user_description.split(',')[0] if ',' in user_description else user_description,
        'fdPrdtClNm': user_category
    }
    
    # 사용자 분실물 정보 저장
    user_item = {
        'original': user_metadata,
        'text': user_description,
        'image': user_image,
        'text_embedding': user_text_embedding,
        'image_embedding': user_image_embedding
    }
    
    # 5. 경찰청 API에서 습득물 데이터 가져오기
    service_key = os.getenv('POLICE_API_SERVICE_KEY')
    if not service_key:
        print("에러: POLICE_API_SERVICE_KEY 환경 변수가 설정되지 않았습니다.")
        return {}
    
    found_items = fetch_police_lost_items(service_key, num_items)
    
    # 6. 각 습득물 데이터의 임베딩 생성 및 유사도 계산
    processed_items = []
    
    print(f"\n총 {len(found_items)}개의 습득물 데이터 처리 중...")
    for item in tqdm(found_items):
        # 텍스트 정보 추출
        text = extract_text_from_item(item, model_type)
        
        # 이미지 다운로드 (있는 경우)
        image = None
        if 'fdFilePathImg' in item and item['fdFilePathImg'] and item['fdFilePathImg'] != NO_IMAGE_URL:
            image = download_image(item['fdFilePathImg'])
        
        # 텍스트 임베딩 생성
        text_embedding = generate_text_embedding(model, processor, text, device)
        
        # 이미지 임베딩 생성 (이미지가 있는 경우)
        image_embedding = generate_image_embedding(model, processor, image, device) if image else None
        
        # 처리된 아이템 정보 저장
        processed_item = {
            'original': item,
            'text': text,
            'image': image,
            'text_embedding': text_embedding,
            'image_embedding': image_embedding
        }
        
        processed_items.append(processed_item)
    
    # 7. 각 습득물에 대해 계층적 유사도 점수 계산
    similarity_results = []
    
    for found_item in processed_items:
        # 계층적 유사도 분석
        similarity_result = analyze_hierarchical_similarity(user_item, found_item)
        
        # 결과 저장
        similarity_results.append((found_item, similarity_result))
    
    # 8. 계층적 접근으로 결과 처리
    hierarchical_results = process_hierarchical_results(similarity_results)
    
    # 9. 결과 출력
    all_sorted = hierarchical_results['all_sorted']
    category_groups = hierarchical_results['category_groups']
    user_metadata = hierarchical_results['user_metadata']
    
    print(f"\n총 {len(processed_items)}개 중 {len(all_sorted)}개의 유사한 습득물을 찾았습니다.")
    
    if all_sorted:
        print("\n사용자 메타데이터:")
        print(f"  카테고리: {user_metadata['category'] if user_metadata['category'] else '알 수 없음'}")
        print(f"  브랜드: {user_metadata['brand'] if user_metadata['brand'] else '알 수 없음'}")
        print(f"  색상: {user_metadata['color'] if user_metadata['color'] else '알 수 없음'}")
        
        # 사용자 카테고리와 일치하는 그룹 먼저 출력
        user_category = user_metadata['category']
        if user_category and user_category in category_groups:
            matching_group = category_groups[user_category]
            print(f"\n▶ 일치하는 카테고리({user_category})의 아이템: {len(matching_group)}개")
            
            for i, (item, result) in enumerate(matching_group[:analyze_top_n]):
                print(f"\n{i+1}. 최종 유사도: {result['final_similarity']:.4f} (카테고리: {user_category})")
                print(f"   조정 내역: {', '.join(result['adjustments'])}")
                print_item_details(item['original'])
        
        # 나머지 카테고리 그룹 출력
        for category, items in category_groups.items():
            if category != user_category:  # 사용자 카테고리와 일치하는 그룹은 이미 출력했으므로 건너뜀
                print(f"\n▶ 카테고리({category})의 아이템: {len(items)}개")
                
                for i, (item, result) in enumerate(items[:analyze_top_n]):
                    print(f"\n{i+1}. 최종 유사도: {result['final_similarity']:.4f} (카테고리: {category})")
                    print(f"   조정 내역: {', '.join(result['adjustments'])}")
                    print_item_details(item['original'])
                
                # 더 많은 아이템이 있는 경우
                if len(items) > analyze_top_n:
                    print(f"... 외 {len(items) - analyze_top_n}개 아이템")
        
        # 카테고리를 알 수 없는 아이템 출력
        if hierarchical_results['uncategorized']:
            print(f"\n▶ 카테고리를 알 수 없는 아이템: {len(hierarchical_results['uncategorized'])}개")
            
            for i, (item, result) in enumerate(hierarchical_results['uncategorized'][:analyze_top_n]):
                print(f"\n{i+1}. 최종 유사도: {result['final_similarity']:.4f} (카테고리: 알 수 없음)")
                print(f"   조정 내역: {', '.join(result['adjustments'])}")
                print_item_details(item['original'])
    else:
        print("유사한 습득물을 찾지 못했습니다.")
    
    # 10. 시각화 (사용자 이미지가 있을 경우)
    if all_sorted:
        # 우선 순위: 같은 카테고리 → 높은 유사도
        visual_items = []
        similarity_values = []
        metadata_values = []
        
        # 사용자 카테고리와 일치하는 아이템 우선 추가
        if user_category and user_category in category_groups:
            for item, result in category_groups[user_category][:5]:
                # 원본 아이템과 이미지 함께 추가
                item_data = item.copy()  # 복사본 사용
                visual_items.append(item_data)
                similarity_values.append(result['final_similarity'])
                metadata_values.append({
                    '카테고리': result['found_metadata']['category'] or '없음',
                    '브랜드': result['found_metadata']['brand'] or '없음',
                    '색상': result['found_metadata']['color'] or '없음'
                })
        
        # 남은 자리는 유사도 순으로 채움
        remaining_slots = 5 - len(visual_items)
        if remaining_slots > 0:
            used_ids = set(item['original'].get('atcId', '') for item in visual_items)
            
            for item, result in all_sorted:
                if len(visual_items) >= 5:
                    break
                    
                item_id = item['original'].get('atcId', '')
                if item_id not in used_ids:
                    item_data = item.copy()  # 복사본 사용
                    visual_items.append(item_data)
                    similarity_values.append(result['final_similarity'])
                    metadata_values.append({
                        '카테고리': result['found_metadata']['category'] or '없음',
                        '브랜드': result['found_metadata']['brand'] or '없음',
                        '색상': result['found_metadata']['color'] or '없음'
                    })
                    used_ids.add(item_id)
        
        # 시각화 실행
        visualize_similarities(user_image, visual_items, similarity_values, metadata_values)
    
    # 11. 결과 반환
    return hierarchical_results

# 아이템 상세 정보 출력 헬퍼 함수
def print_item_details(item):
    """
    아이템의 상세 정보 출력
    """
    important_fields = [
        ('fdPrdtNm', '물품명'),
        ('fdPrdtClNm', '분류'),
        ('prdtClNm', '분류'),
        ('fdSbjt', '제목'),
        ('clrNm', '색상'),
        ('fdPlace', '습득장소'),
        ('depPlace', '보관장소'),
        ('fdYmd', '습득일자'),
        ('fndKeepOrgnSeNm', '보관기관')
    ]
    
    for field, label in important_fields:
        if field in item and item[field]:
            print(f"   {label}: {item[field]}")
    
    has_image = "있음" if item.get('fdFilePathImg') and item.get('fdFilePathImg') != NO_IMAGE_URL else "없음"
    print(f"   이미지: {has_image}")

# 메인 실행 블록
if __name__ == "__main__":
    # 하드코딩된 값으로 테스트 실행
    print("분실물-습득물 계층적 유사도 비교 테스트를 시작합니다.")
    
    # 하드코딩된 분실물 정보 (실제 경로나 URL로 수정 필요)
    user_image_path = "phone.jpg"  # 로컬 파일 경로
    # 또는 URL 사용: user_image_path = "https://example.com/images/wallet.jpg"
    
    # 한국어 설명 사용 (한국어 모델 사용 시 더 효과적)
    user_description = "검정색 삼성 갤럭시폰폰"
    
    # 사용자 카테고리 수동 설정 (자동 추출 실패 시 사용)
    manual_category = "휴대폰"  # None으로 설정하면 자동 추출 시도
    
    # 유사도 비교 실행 - 계층적 접근 방식
    # analyze_top_n: 상세 분석할 상위 아이템 수
    compare_lost_and_found(
        user_image_path,
        user_description,
        manual_category=manual_category,
        num_items=20,
        analyze_top_n=3
    )