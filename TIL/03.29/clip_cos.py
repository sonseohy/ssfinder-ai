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
from dotenv import load_dotenv
from tqdm import tqdm

# CLIP 모델 관련 라이브러리 임포트
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer

# 환경 변수 로드
load_dotenv()

# 설정 값
TEXT_SIMILARITY_THRESHOLD = 0.6  # 텍스트 유사도 임계값
IMAGE_TEXT_CONSISTENCY_THRESHOLD = 0.3  # 이미지와 텍스트 일치성 임계값
NO_IMAGE_URL = "https://www.lost112.go.kr/lostnfs/images/sub/img02_no_img.gif"  # 이미지 없음 URL
KOREAN_CLIP_MODEL = "Bingsu/clip-vit-base-patch32-ko"  # 한국어 특화 CLIP 모델 (base 버전으로 변경)

# 이미지 가중치 (텍스트와 이미지 유사도 결합 시 사용)
IMAGE_WEIGHT = 0.3  # 이미지 30%, 텍스트 70% 가중치

# 카테고리 매핑 - 참고용 (순수 코사인 유사도에는 사용하지 않음)
CATEGORY_MAPPING = {
    '휴대폰': ['휴대폰', '핸드폰', '스마트폰', '피처폰', '모바일폰'],
    '지갑': ['지갑', '카드지갑', '명함지갑', '반지갑', '장지갑', '동전지갑'],
    '태블릿': ['태블릿', '패드'],
    '노트북': ['노트북', '랩탑', '컴퓨터'],
    '이어폰': ['이어폰', '헤드폰', '이어버드'],
    '시계': ['시계', '워치', '손목시계'],
    '카메라': ['카메라', 'DSLR', '디카'],
    '의류': ['의류', '옷', '상의', '하의', '자켓', '코트', '티셔츠', '바지'],
    '신발': ['신발', '운동화', '구두', '슬리퍼'],
    '가방': ['가방', '백팩', '핸드백', '크로스백', '숄더백', '파우치']
}

# 브랜드 목록 - 다양한 제품을 생산하는 브랜드
BRAND_LIST = ['삼성', '애플', '엘지', 'LG', '샤오미', '화웨이', '소니', '파나소닉', 
              '나이키', '아디다스', '퓨마', '리복', '뉴발란스', 
              '구찌', '루이비통', '샤넬', '프라다', '에르메스']

# 특정 브랜드 제품명
SPECIFIC_PRODUCTS = {
    '휴대폰': ['갤럭시폰', '아이폰', '픽셀폰', 'v30', 'g7', 'iphone', 'galaxy s', 'galaxy note', 'galaxy z'],
    '태블릿': ['아이패드', '갤럭시탭', '갤탭', 'ipad', 'galaxy tab'],
    '노트북': ['맥북', '갤럭시북', 'macbook', 'galaxy book', 'thinkpad', 'zenbook'],
    '이어폰': ['에어팟', '갤럭시버즈', '버즈', 'airpods', 'galaxy buds']
}

# 기타 카테고리 관련 키워드
MISC_CATEGORIES = ['기타', '기타물품', '기타 물품', '기타 아이템', '미분류', '분류없음']

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

# 텍스트에서 카테고리 추출 (참고용, 순수 코사인 유사도에는 사용되지 않음)
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
    
    text = text.lower()
    
    # 1. 이미 카테고리 정보가 있으면 확인
    if item_category:
        # '기타' 카테고리 확인 - 예외 처리
        for misc in MISC_CATEGORIES:
            if misc.lower() in item_category.lower():
                return None  # 기타 카테고리는 패널티나 보너스를 적용하지 않음
        
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
                # 브랜드 이름이 있는지 체크 (특정 브랜드만 있고 제품군이 없는 경우는 카테고리 결정 보류)
                has_only_brand = False
                for brand in BRAND_LIST:
                    if brand.lower() in text.lower():
                        # 브랜드만 있고 카테고리 키워드가 없는 경우
                        if keyword.lower() not in text.lower():
                            has_only_brand = True
                
                if not has_only_brand:
                    return category
    
    # 4. 카테고리를 찾을 수 없는 경우
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
        print(f"임베딩 차원 불일치: {embedding1.shape} vs {embedding2.shape}")
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

# 아이템에서 카테고리 정보 추출 (참고용)
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

# 유사도 상세 분석
def analyze_similarity_detailed(user_item, found_item):
    """
    사용자 분실물과 습득물 간의 유사도를 상세히 분석
    (순수한 코사인 유사도 사용)
    """
    item_id = found_item['original'].get('atcId', 'unknown')
    print(f"\n===== 아이템 ID: {item_id} 유사도 분석 시작 =====")
    print(f"아이템 텍스트: {found_item['text']}")
    
    # 순수 텍스트 코사인 유사도 계산
    text_similarity = pure_cosine_similarity(user_item['text_embedding'], found_item['text_embedding'])
    print(f"텍스트 코사인 유사도: {text_similarity:.4f}")
    
    # 순수 이미지 코사인 유사도 계산 (이미지가 있는 경우)
    image_similarity = 0.0
    if user_item['image_embedding'] is not None and found_item['image_embedding'] is not None:
        image_similarity = pure_cosine_similarity(user_item['image_embedding'], found_item['image_embedding'])
        print(f"이미지 코사인 유사도: {image_similarity:.4f}")
        
        # 사용자 분실물 이미지-텍스트 일관성
        user_consistency = check_image_text_consistency(user_item['image_embedding'], user_item['text_embedding'])
        print(f"사용자 분실물 이미지-텍스트 일관성: {user_consistency:.4f}")
        
        # 습득물 이미지-텍스트 일관성
        found_consistency = check_image_text_consistency(found_item['image_embedding'], found_item['text_embedding'])
        print(f"습득물 이미지-텍스트 일관성: {found_consistency:.4f}")
    else:
        print("이미지 정보가 없어 이미지 유사도는 계산하지 않습니다.")
    
    # 참고용 카테고리 정보 출력 (점수에는 반영하지 않음)
    user_product_name = user_item['original'].get('fdPrdtNm', '')
    found_product_name = found_item['original'].get('fdPrdtNm', '')
    
    found_category_field = None
    for field in ['fdPrdtClNm', 'prdtClNm']:
        if field in found_item['original'] and found_item['original'][field]:
            found_category_field = found_item['original'][field]
            break
    
    user_category = extract_category(user_item.get('text', ''))
    found_category = extract_category(found_product_name, found_category_field)
    
    print(f"사용자 물품명: {user_product_name}")
    print(f"습득물 물품명: {found_product_name}")
    print(f"습득물 카테고리 필드: {found_category_field}")
    print(f"추출된 사용자 카테고리: {user_category}")
    print(f"추출된 습득물 카테고리: {found_category}")
    
    # 카테고리 일치 여부 (참고용)
    if user_category and found_category:
        category_match = user_category == found_category
        print(f"카테고리 일치 여부: {category_match}")
    else:
        print("카테고리 정보 부족: 일치 여부 판단 불가")
    
    # 순수 코사인 유사도 최종 계산 (이미지+텍스트)
    final_similarity = text_similarity
    if image_similarity > 0:
        # 이미지 유사도가 있을 경우 가중 평균 적용 (범위: -1~1 유지)
        final_similarity = (text_similarity * (1 - IMAGE_WEIGHT)) + (image_similarity * IMAGE_WEIGHT)
    
    print(f"최종 순수 코사인 유사도: {final_similarity:.4f}")
    print("===== 유사도 분석 완료 =====\n")
    
    return final_similarity

# 유사도 결과 시각화
def visualize_similarities(user_image, similar_items, similarities):
    """
    유사도 결과를 시각적으로 표시
    """
    n = min(5, len(similar_items))  # 최대 5개까지 표시
    
    plt.figure(figsize=(15, 8))
    
    # 사용자 이미지 표시
    plt.subplot(1, n+1, 1)
    plt.imshow(user_image)
    plt.title("User Image")
    plt.axis('off')
    
    # 유사한 습득물 이미지 표시
    for i in range(n):
        plt.subplot(1, n+1, i+2)
        
        # 이미지가 있으면 표시, 없으면 텍스트 표시
        item = similar_items[i]
        if 'image' in item and item['image'] is not None:
            plt.imshow(item['image'])
        else:
            plt.text(0.5, 0.5, "No Image", ha='center', va='center')
        
        # 유사도와 물품명 표시
        title_text = f"Similarity: {similarities[i]:.2f}\n"
        
        # 물품명 추가
        if 'fdPrdtNm' in item and item['fdPrdtNm']:
            title_text += f"{item['fdPrdtNm']}"
        else:
            title_text += "No info"
            
        plt.title(title_text)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 메인 함수: 사용자 분실물과 경찰청 습득물 유사도 비교
def compare_lost_and_found(user_image_path, user_description, num_items=10, analyze_top_n=3):
    """
    사용자 분실물과 경찰청 습득물 데이터의 유사도 비교
    (순수 코사인 유사도 사용)
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
    
    # 사용자 분실물 정보 저장
    user_item = {
        'original': {'fdPrdtNm': user_description.split(',')[0] if ',' in user_description else user_description},
        'text': user_description,
        'image': user_image,
        'text_embedding': user_text_embedding,
        'image_embedding': user_image_embedding
    }
    
    # 4. 경찰청 API에서 습득물 데이터 가져오기
    service_key = os.getenv('POLICE_API_SERVICE_KEY')
    if not service_key:
        print("에러: POLICE_API_SERVICE_KEY 환경 변수가 설정되지 않았습니다.")
        return []
    
    found_items = fetch_police_lost_items(service_key, num_items)
    
    # 5. 각 습득물 데이터의 임베딩 생성 및 유사도 계산
    processed_items = []
    
    print("습득물 데이터 처리 중...")
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
    
    # 6. 각 습득물에 대해 유사도 점수 계산
    similarity_results = []
    
    for found_item in processed_items:
        # 순수 코사인 유사도 계산
        final_similarity = analyze_similarity_detailed(user_item, found_item)
        
        # 결과 저장
        similarity_results.append((found_item, final_similarity))
    
    # 7. 유사도에 따라 정렬
    similarity_results.sort(key=lambda x: x[1], reverse=True)
    
    # 8. 유사도 임계값 이상인 결과만 필터링
    filtered_results = [(item, sim) for item, sim in similarity_results if sim >= TEXT_SIMILARITY_THRESHOLD]
    
    # 9. 결과 출력
    print(f"\n총 {len(processed_items)}개 중 {len(filtered_results)}개의 유사한 습득물을 찾았습니다.")
    
    if filtered_results:
        print("\n유사도 높은 아이템:")
        
        # 상위 N개 아이템 출력
        for i, (item, similarity) in enumerate(filtered_results[:analyze_top_n]):
            print(f"\n{i+1}. 유사도: {similarity:.4f}")
            
            # 물품 정보 출력
            item_original = item['original']
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
                if field in item_original and item_original[field]:
                    print(f"   {label}: {item_original[field]}")
            
            has_image = "있음" if item['image'] is not None else "없음"
            print(f"   이미지: {has_image}")
        
        # 나머지 아이템은 간략하게 표시
        if len(filtered_results) > analyze_top_n:
            print(f"\n나머지 {len(filtered_results) - analyze_top_n}개 아이템:")
            for i, (item, similarity) in enumerate(filtered_results[analyze_top_n:], start=analyze_top_n+1):
                product_name = item['original'].get('fdPrdtNm', '정보 없음')
                category = item['original'].get('fdPrdtClNm', '')
                print(f"{i}. 유사도: {similarity:.4f} - 물품명: {product_name}" + (f" (분류: {category})" if category else ""))
    else:
        print("유사한 습득물을 찾지 못했습니다.")
    
    # 10. 시각화 (사용자 이미지가 있을 경우)
    if user_image and filtered_results:
        similar_items = [item['original'] for item, _ in filtered_results[:5]]
        for i, item in enumerate(similar_items):
            item['image'] = filtered_results[i][0]['image']
        similarity_values = [sim for _, sim in filtered_results[:5]]
        visualize_similarities(user_image, similar_items, similarity_values)
    
    # 11. 결과 반환
    return [(item['original'], sim) for item, sim in filtered_results]

# 메인 실행 블록
if __name__ == "__main__":
    # 하드코딩된 값으로 테스트 실행
    print("분실물-습득물 유사도 비교 테스트를 시작합니다.")
    
    # 하드코딩된 분실물 정보 (실제 경로나 URL로 수정 필요)
    user_image_path = "phone.jpg"  # 로컬 파일 경로
    # 또는 URL 사용: user_image_path = "https://example.com/images/wallet.jpg"
    
    # 한국어 설명 사용 (한국어 모델 사용 시 더 효과적)
    user_description = "검정색 갤럭시 스마트폰"
    
    # 유사도 비교 실행 - 이미지가 없는 경우에도 텍스트만으로 비교
    # analyze_top_n: 상세 분석할 상위 아이템 수
    compare_lost_and_found(user_image_path, user_description, num_items=20, analyze_top_n=3)