import os
import requests
import xml.etree.ElementTree as ET
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from dotenv import load_dotenv
from tqdm import tqdm

# CLIP 모델 관련 라이브러리 임포트
from transformers import CLIPProcessor, CLIPModel

# 환경 변수 로드
load_dotenv()

# 설정 값
SIMILARITY_THRESHOLD = 0.75  # 유사도 임계값
NO_IMAGE_URL = "https://www.lost112.go.kr/lostnfs/images/sub/img02_no_img.gif"  # 이미지 없음 URL
KOREAN_CLIP_MODEL = "Bingsu/clip-vit-large-patch14-ko"  # 한국어 특화 CLIP 모델

# CLIP 모델 로드
def load_clip_model():
    """
    한국어 CLIP 모델과 프로세서를 로드하는 함수
    """
    try:
        # 한국어 특화 CLIP 모델 로드 시도
        print(f"한국어 특화 CLIP 모델 로드 중: {KOREAN_CLIP_MODEL}")
        model = CLIPModel.from_pretrained(KOREAN_CLIP_MODEL)
        processor = CLIPProcessor.from_pretrained(KOREAN_CLIP_MODEL)
        model_type = "Korean"
    except Exception as e:
        # 실패 시 기본 CLIP 모델 사용
        print(f"한국어 모델 로드 실패, 기본 CLIP 모델로 대체합니다: {str(e)}")
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        model_type = "English"
    
    # GPU 사용 가능하면 GPU로 모델 이동
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"{model_type} CLIP model loaded on {device}")
    return model, processor, device, model_type

# 경찰청 API에서 습득물 데이터 가져오기
def fetch_police_lost_items(service_key, num_items=5):
    """
    경찰청 API를 통해 최신 습득물 데이터를 가져옴
    
    Args:
        service_key (str): 경찰청 API 서비스 키
        num_items (int): 가져올 아이템 수
        
    Returns:
        list: 습득물 데이터 리스트
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
        response = requests.get(url, params=params)
        
        # API 응답 상태 확인
        if response.status_code == 200:
            # 디버깅: API 원본 응답 출력 (필요시 주석 해제)
            # print("API 원본 응답:")
            # print(response.content.decode('utf-8', errors='replace'))
            
            root = ET.fromstring(response.content)
            items = []
            
            # items 태그 아래의 item 요소들 추출
            for item in root.findall('.//item'):
                item_data = {}
                
                # 각 필드 추출
                for child in item:
                    item_data[child.tag] = child.text
                
                items.append(item_data)
                
            print(f"API에서 {len(items)}개의 최신 습득물 데이터를 성공적으로 가져왔습니다.")
            return items
        else:
            print(f"API 호출 실패: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"API 호출 중 오류 발생: {str(e)}")
        return []

# 습득물 이미지 다운로드
def download_image(url):
    """
    URL에서 이미지를 다운로드하여 PIL 이미지로 변환
    
    Args:
        url (str): 이미지 URL
        
    Returns:
        PIL.Image: 다운로드한 이미지 (실패 시 None)
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
    
    Args:
        base64_string (str): Base64 인코딩된 이미지 문자열
        
    Returns:
        PIL.Image: 디코딩된 이미지
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

# 습득물 데이터에서 텍스트 정보 추출
def extract_text_from_item(item, model_type="Korean"):
    """
    습득물 데이터에서 관련 텍스트 정보를 추출하여 하나의 문자열로 결합
    
    Args:
        item (dict): 습득물 데이터 딕셔너리
        model_type (str): 모델 타입 ('Korean' 또는 'English')
        
    Returns:
        str: 결합된 텍스트 정보
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

# CLIP 모델을 사용한 임베딩 생성
def generate_clip_embedding(model, processor, image, text, device, image_weight=0.5):
    """
    CLIP 모델을 사용하여 이미지와 텍스트의 결합 임베딩 생성
    
    Args:
        model: CLIP 모델
        processor: CLIP 프로세서
        image (PIL.Image): 이미지
        text (str): 텍스트
        device (str): 'cuda' 또는 'cpu'
        image_weight (float): 이미지 임베딩 가중치 (0~1 사이)
        
    Returns:
        numpy.ndarray: 임베딩 벡터
    """
    try:
        with torch.no_grad():
            # 텍스트가 너무 길면 잘라내기
            if len(text) > 200:
                text = text[:200]
                
            # 이미지와 텍스트 전처리
            inputs = processor(
                text=[text],
                images=[image] if image else None,
                return_tensors="pt",
                padding=True,
                truncation=True,  # 텍스트 자동 잘라내기
                max_length=77     # CLIP의 최대 토큰 길이
            ).to(device)
            
            # 이미지와 텍스트의 특징 추출
            outputs = model(**inputs)
            
            # 이미지 특징이 있으면 이미지와 텍스트 특징의 가중 평균, 없으면 텍스트 특징만 사용
            if image is not None:
                image_features = outputs.image_embeds
                text_features = outputs.text_embeds
                combined_features = (image_features * image_weight + text_features * (1 - image_weight))
            else:
                combined_features = outputs.text_embeds
                
            # 임베딩 정규화
            embedding = combined_features[0].cpu().numpy()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
    except Exception as e:
        print(f"임베딩 생성 오류: {str(e)}")
        # 오류 발생 시 랜덤 임베딩 반환 (실제 서비스에서는 더 나은 대안이 필요)
        random_embedding = np.random.random(512)
        return random_embedding / np.linalg.norm(random_embedding)

# 코사인 유사도 계산
def cosine_similarity(embedding1, embedding2):
    """
    두 임베딩 벡터 간의 코사인 유사도 계산
    
    Args:
        embedding1 (numpy.ndarray): 첫 번째 임베딩 벡터
        embedding2 (numpy.ndarray): 두 번째 임베딩 벡터
        
    Returns:
        float: 코사인 유사도 (-1~1 사이 값, 1에 가까울수록 유사)
    """
    return np.dot(embedding1, embedding2)

# 유사도 결과 시각화
def visualize_similarities(user_image, similar_items, similarities):
    """
    유사도 결과를 시각적으로 표시
    
    Args:
        user_image (PIL.Image): 사용자 이미지
        similar_items (list): 유사한 습득물 아이템 리스트
        similarities (list): 유사도 값 리스트
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
def compare_lost_and_found(user_image_path, user_description, num_items=10, image_weight=0.5):
    """
    사용자 분실물과 경찰청 습득물 데이터의 유사도 비교
    
    Args:
        user_image_path (str): 사용자 이미지 파일 경로 또는 Base64 인코딩 문자열
        user_description (str): 사용자 분실물 설명
        num_items (int): 가져올 습득물 데이터 수
        image_weight (float): 이미지 임베딩 가중치 (0~1 사이)
        
    Returns:
        list: 유사도 높은 습득물 리스트
    """
    # 1. CLIP 모델 로드
    model, processor, device, model_type = load_clip_model()
    
    # 2. 사용자 이미지 로드
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
            user_image = None
    
    # 3. 사용자 분실물 임베딩 생성
    user_embedding = generate_clip_embedding(model, processor, user_image, user_description, device, image_weight)
    
    # 4. 경찰청 API에서 습득물 데이터 가져오기
    service_key = os.getenv('POLICE_API_SERVICE_KEY')
    if not service_key:
        print("에러: POLICE_API_SERVICE_KEY 환경 변수가 설정되지 않았습니다.")
        return []
    
    found_items = fetch_police_lost_items(service_key, num_items)
    
    # 5. 각 습득물 데이터의 임베딩 생성 및 유사도 계산
    similarities = []
    
    print("습득물 데이터 처리 중...")
    for item in tqdm(found_items):
        # 텍스트 정보 추출 (모델 타입에 맞게)
        text = extract_text_from_item(item, model_type)
        
        # 이미지 URL이 있고 기본 이미지가 아닌 경우에만 다운로드
        image = None
        if 'fdFilePathImg' in item and item['fdFilePathImg'] and item['fdFilePathImg'] != NO_IMAGE_URL:
            image = download_image(item['fdFilePathImg'])
        
        # 임베딩 생성 (이미지가 없어도 텍스트만으로 생성)
        item_embedding = generate_clip_embedding(model, processor, image, text, device, image_weight)
        
        # 유사도 계산
        similarity = cosine_similarity(user_embedding, item_embedding)
        
        # 이미지 저장 (나중에 시각화할 때 사용)
        item['image'] = image
        
        # 결과 저장
        similarities.append((item, similarity))
    
    # 6. 유사도에 따라 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # 7. 유사도 임계값 이상인 결과만 필터링
    filtered_results = [(item, sim) for item, sim in similarities if sim >= SIMILARITY_THRESHOLD]
    
    # 8. 결과 출력
    print(f"\n총 {len(found_items)}개 중 {len(filtered_results)}개의 유사한 습득물을 찾았습니다.")
    
    if filtered_results:
        print("\n유사도 높은 아이템:")
        for i, (item, similarity) in enumerate(filtered_results[:5]):
            print(f"\n{i+1}. 유사도: {similarity:.4f}")
            
            # 모든 중요 필드 출력 (사용 가능한 경우)
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
            
            has_image = "있음" if item.get('image') is not None else "없음"
            print(f"   이미지: {has_image}")
    else:
        print("유사한 습득물을 찾지 못했습니다.")
    
    # 9. 시각화 (사용자 이미지가 있을 경우)
    if user_image and filtered_results:
        similar_items = [item for item, _ in filtered_results[:5]]
        similarity_values = [sim for _, sim in filtered_results[:5]]
        visualize_similarities(user_image, similar_items, similarity_values)
    
    # 10. 결과 반환
    return filtered_results

# 메인 실행 블록
if __name__ == "__main__":
    # 하드코딩된 값으로 테스트 실행
    print("분실물-습득물 유사도 비교 테스트를 시작합니다.")
    
    # 하드코딩된 분실물 정보 (실제 경로나 URL로 수정 필요)
    user_image_path = "galtab.jpg"  # 로컬 파일 경로
    # 또는 URL 사용: user_image_path = "https://example.com/images/wallet.jpg"
    
    # 한국어 설명 사용 (한국어 모델 사용 시 더 효과적)
    user_description = "회색 갤럭시 탭, 태블릿"
    
    # 유사도 비교 실행 - 이미지 가중치 조정 가능 (0.5: 이미지와 텍스트 동등 고려)
    compare_lost_and_found(user_image_path, user_description, num_items=40, image_weight=0.5)