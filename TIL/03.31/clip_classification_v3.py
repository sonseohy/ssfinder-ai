import torch
import clip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# CLIP 모델과 전처리기(preprocessor) 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 사용자 정의 클래스 정의 (한국어 및 영어)
custom_classes = [
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

# 요구사항 2-1: 라벨 정보 토큰화하기
def tokenize_labels(classes):
    # "a photo of a {label}" 형식으로 텍스트 프롬프트 생성
    text_inputs = []
    expanded_class_list = []
    
    for c in classes:
        # 한국어/영어 처리
        is_korean = any('\uAC00' <= char <= '\uD7A3' for char in c)
        
        if is_korean:
            # 한국어 프롬프트
            prompts = [
                f"{c}의 사진",
                f"{c} 이미지",
                f"{c}"
            ]
        else:
            # 영어 프롬프트 - 기본 관사 처리 (a/an)
            article = "an" if c[0].lower() in "aeiou" else "a"
            
            # 특수 케이스 처리
            if c in ["airpods", "earpods", "headphones", "glasses", "sunglasses", "keys", "gloves", "galaxy buds", "samsung earbuds", "earbuds"]:
                # 복수형 단어는 "a pair of"로 시작
                prompts = [
                    f"a photo of {c}",
                    f"a picture of {c}",
                    f"a pair of {c}",
                    f"the {c}"
                ]
            else:
                prompts = [
                    f"a photo of {article} {c}",
                    f"a picture of {article} {c}",
                    f"{article} {c}",
                    f"the {c}"
                ]
            
            # 한국 돈/화폐 특화 프롬프트
            if c in ["korean won", "10000 won", "50000 won", "현금", "지폐", "동전", "한국 돈", "만원", "오만원", "천원", "오천원", "백원", "오백원"]:
                prompts.append(f"korean money")
                prompts.append(f"korean currency")
                if "won" in c:
                    prompts.append(f"korean {c}")
            
            # 차키 특화 프롬프트
            if c in ["car key", "차키"]:
                prompts.append(f"automobile key")
                prompts.append(f"vehicle key")
                prompts.append(f"key fob")
                prompts.append(f"remote car key")
            
            # 제조사별 특화된 프롬프트 추가
            # 애플 제품
            if c in ["iphone", "ipad", "macbook", "airpods", "earpods", "apple watch"]:
                prompts.append(f"an apple {c}")
                prompts.append(f"an apple device")
            
            # 삼성 제품
            if c in ["samsung phone", "galaxy phone", "galaxy s", "galaxy note", "galaxy tab", "galaxy watch", "galaxy buds", "samsung tablet", "samsung earbuds"]:
                if "galaxy" in c:
                    prompts.append(f"a samsung {c}")
                prompts.append(f"a samsung device")
                
                # 갤럭시 워치 특화
                if c == "galaxy watch":
                    prompts.append(f"a samsung smartwatch")
                    prompts.append(f"a round smartwatch")
                
                # 갤럭시 버즈 특화
                if c == "galaxy buds" or c == "samsung earbuds":
                    prompts.append(f"samsung wireless earbuds")
            
            # 삼성 폴더블 제품 특화
            if c in ["galaxy z flip", "galaxy z fold", "samsung foldable phone", "foldable phone", 
                    "samsung folding phone", "폴더블 스마트폰", "접이식 휴대폰", "갤럭시 Z 플립", "갤럭시 Z 폴드"]:
                
                is_flip = "flip" in c.lower() or "플립" in c
                is_fold = "fold" in c.lower() or "폴드" in c
                
                prompts.append(f"a samsung foldable phone")
                prompts.append(f"a foldable smartphone")
                
                if is_flip:
                    prompts.append(f"a samsung galaxy z flip")
                    prompts.append(f"a flip phone that folds horizontally")
                    prompts.append(f"a clamshell folding phone")
                
                if is_fold:
                    prompts.append(f"a samsung galaxy z fold")
                    prompts.append(f"a phone that folds like a book")
                    prompts.append(f"a fold phone that opens into a tablet")
                
                # 삼성 브랜드와 폴더블 두 가지 특성 모두 강조
                prompts.append(f"a samsung device that folds")
                prompts.append(f"a premium samsung folding phone")
            
            # 전자기기 일반
            if c in ["smartphone", "phone", "tablet", "wireless earbuds", "headphones", "headset", 
                     "laptop", "smartwatch", "smart watch", "earbuds", "smart device", "camera"]:
                prompts.append(f"a modern {c}")
                prompts.append(f"an electronic {c}")
            
            # 분실물 관련
            if c in ["wallet", "purse", "credit card", "id card", "card", "key", "keys", "backpack", 
                     "bag", "glasses", "sunglasses", "book", "umbrella", "water bottle"]:
                prompts.append(f"a lost {c}")
                prompts.append(f"a personal {c}")
            
        for p in prompts:
            text_inputs.append(clip.tokenize(p))
            expanded_class_list.append(c)
    
    text_inputs = torch.cat(text_inputs).to(device)
    
    # 토큰화된 텍스트의 CLIP 특성 추출
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    return text_features, expanded_class_list

# 토큰화된 라벨 특성 얻기
text_features, expanded_classes = tokenize_labels(custom_classes)
print(f"Number of original classes: {len(custom_classes)}")
print(f"Number of expanded prompts: {len(expanded_classes)}")

# 요구사항 2-2: 이미지와 라벨 텍스트 간 CLIP feature 유사도 계산하기
def calculate_similarity(image, text_features, topk=10):
    # 이미지를 CLIP 전처리기를 통해 전처리
    image_input = preprocess(image).unsqueeze(0).to(device)
    
    # 이미지의 CLIP 특성 추출
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # 이미지 특성과 텍스트 특성 간의 코사인 유사도 계산
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    # Top-K 유사도 값과 해당 인덱스 추출
    values, indices = similarity[0].topk(topk)
    
    # 결과를 원래 클래스와 매핑
    result_values = []
    result_indices = []
    seen_classes = set()
    
    for v, idx in zip(values, indices):
        class_name = expanded_classes[idx]
        if class_name not in seen_classes:
            seen_classes.add(class_name)
            original_idx = custom_classes.index(class_name) if class_name in custom_classes else -1
            result_values.append(v)
            result_indices.append(original_idx)
            
            if len(result_values) >= topk:
                break
    
    return torch.tensor(result_values), torch.tensor(result_indices)

# CLIP을 사용한 이미지 분류기 함수 (종합 버전)
def classify_image_with_clip(image_path=None, image=None, topk=10, display_image=True, category_filter=None):
    # 이미지 로드 (경로 또는 이미지 객체 사용)
    if image_path is not None:
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"이미지를 불러오는 데 오류가 발생했습니다: {e}")
            return None, None
    elif image is None:
        print("이미지 경로나 이미지 객체가 필요합니다.")
        return None, None
    
    # 유사도 계산
    values, indices = calculate_similarity(image, text_features, topk=topk if category_filter is None else topk*2)
    
    # 카테고리 필터 적용 (예: 전자기기만 표시)
    if category_filter is not None:
        filtered_values = []
        filtered_indices = []
        
        electronics_apple = ["iphone", "ipad", "macbook", "airpods", "earpods", "apple watch"]
        electronics_samsung = ["samsung phone", "galaxy phone", "galaxy s", "galaxy note", "galaxy tab", 
                              "galaxy watch", "galaxy buds", "samsung tablet", "samsung earbuds"]
        electronics_samsung_foldable = ["galaxy z flip", "galaxy z fold", "samsung foldable phone", "foldable phone",
                                       "samsung folding phone", "폴더블 스마트폰", "접이식 휴대폰", "갤럭시 Z 플립", "갤럭시 Z 폴드"]
        electronics_general = ["smartphone", "phone", "tablet", "wireless earbuds", "headphones", "headset",
                              "laptop", "smartwatch", "smart watch", "earbuds", "smart device", "camera",
                              "태블릿", "휴대폰", "노트북", "카메라"]
        
        money_items = ["money", "cash", "korean won", "10000 won", "50000 won", "dollar", "euro", 
                       "현금", "지폐", "동전", "한국 돈", "만원", "오만원", "천원", "오천원", "백원", "오백원"]
        
        car_items = ["car", "car key", "car license", "자동차", "차키", "자동차등록증", "운전면허증"]
        
        wallet_items = ["wallet", "purse", "credit card", "id card", "transportation card", "membership card", 
                        "지갑", "여성용 지갑", "남성용 지갑", "카드", "신용카드", "교통카드", "멤버십카드"]
        
        lost_items = ["wallet", "purse", "credit card", "id card", "card", "key", "keys", "backpack", 
                     "bag", "glasses", "sunglasses", "book", "umbrella", "water bottle",
                     "지갑", "우산", "열쇠", "안경", "선글라스", "물병"]
        
        filter_list = []
        if "electronics_apple" in category_filter:
            filter_list.extend(electronics_apple)
        if "electronics_samsung" in category_filter:
            filter_list.extend(electronics_samsung)
        if "electronics_samsung_foldable" in category_filter:
            filter_list.extend(electronics_samsung_foldable)
        if "electronics_general" in category_filter:
            filter_list.extend(electronics_general)
        if "electronics" in category_filter:
            filter_list.extend(electronics_apple + electronics_samsung + electronics_samsung_foldable + electronics_general)
        if "money" in category_filter:
            filter_list.extend(money_items)
        if "car" in category_filter:
            filter_list.extend(car_items)
        if "wallet" in category_filter:
            filter_list.extend(wallet_items)
        if "lost_items" in category_filter:
            filter_list.extend(lost_items)
        if "all" in category_filter:
            filter_list = custom_classes
            
        for v, idx in zip(values, indices):
            class_name = custom_classes[idx.item()]
            if class_name in filter_list:
                filtered_values.append(v)
                filtered_indices.append(idx)
                
                if len(filtered_values) >= topk:
                    break
        
        if filtered_values:
            values = torch.stack(filtered_values)
            indices = torch.stack(filtered_indices)
    
    # 결과 출력
    print(f"Top {len(values)} Classification Results:")
    for i, (v, idx) in enumerate(zip(values, indices)):
        class_name = custom_classes[idx.item()]
        print(f"{class_name:<15}: {v.item()*100:.2f}%")
    
    # 이미지 표시
    if display_image and len(indices) > 0:
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(f"Predicted: {custom_classes[indices[0].item()]}")
        plt.axis('off')
        plt.show()
    
    if len(indices) > 0:
        return custom_classes[indices[0].item()], values[0].item()
    else:
        return None, None

# 로고 감지를 위한 함수 추가
def detect_brand_logo(image_path):
    """
    이미지에서 애플 또는 삼성 로고를 감지하는 기능 개선 함수
    
    매개변수:
        image_path (str): 분석할 이미지 경로
        
    반환값:
        brand (str): 감지된 브랜드 ('apple', 'samsung', 또는 None)
        confidence (float): 감지 신뢰도 (0~1)
    """
    # 이미지 로드
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # 로고 인식을 위한 CLIP 특화 프롬프트 - 더 구체적인 로고 관련 프롬프트 추가
    logo_prompts = [
        "the apple logo, a bitten apple",
        "the iconic apple symbol, bitten apple",
        "the apple brand logo on a device",
        "the samsung logo text",
        "the samsung brand name text",
        "the samsung logo on a device",
        "no visible brand logo",
        "a plain device with no logo"
    ]
    
    # 텍스트 토큰화 및 특성 추출
    text_inputs = torch.cat([clip.tokenize(p) for p in logo_prompts]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 이미지 전처리 및 특성 추출
    preprocessed_image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(preprocessed_image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # 유사도 계산
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(2)  # 상위 2개 결과 확인
    
    top_index = indices[0].item()
    top_confidence = values[0].item()
    second_index = indices[1].item()
    second_confidence = values[1].item()
    
    # 애플 로고 감지를 위한 특화된 처리
    # iPhone 카메라 모듈 감지 - 아이폰의 특징적인 카메라 배열 감지
    is_iphone_camera = detect_iphone_camera(image_np)
    
    # 삼성 로고 텍스트 감지를 위한 특화된 처리
    is_samsung_text = detect_samsung_text(image_np)
    
    # 결과 결정 - 로고 및 장치 특성 기반
    if top_index < 3:  # 애플 관련 프롬프트
        if is_iphone_camera:  # 아이폰 카메라 모듈이 감지되면 확신도 증가
            return "apple", max(top_confidence, 0.8)
        else:
            return "apple", top_confidence
    elif top_index < 6:  # 삼성 관련 프롬프트
        if is_samsung_text:  # 삼성 텍스트 로고가 감지되면 확신도 증가
            return "samsung", max(top_confidence, 0.8)
        else:
            return "samsung", top_confidence
    else:
        # 로고가 명확하지 않으면 장치 특성 기반으로 판단
        if is_iphone_camera:
            return "apple", 0.7  # 카메라 모듈 기반 애플 판단
        elif is_samsung_text:
            return "samsung", 0.7  # 텍스트 기반 삼성 판단
        else:
            return None, top_confidence  # 명확한 브랜드 없음

def detect_iphone_camera(image_np):
    """아이폰 카메라 모듈 감지 함수"""
    # 그레이스케일 변환
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # 블러 처리로 노이즈 감소
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 원 감지를 위한 허프 변환 (아이폰 카메라 렌즈는 원형)
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=20, 
        param1=50, 
        param2=30, 
        minRadius=10, 
        maxRadius=50
    )
    
    # 아이폰 특유의 카메라 배열 패턴 확인 (정사각형/삼각형 배열의 2-3개 원)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        
        # 원이 2-3개 있고, 특정 패턴으로 배열되어 있는지 확인
        if 2 <= len(circles[0]) <= 3:
            # 원들 간의 거리 계산을 통해 아이폰 카메라 배열 패턴 확인 가능
            # 여기서는 단순화를 위해 원 개수만 체크
            return True
    
    return False

def detect_samsung_text(image_np):
    """삼성 로고 텍스트 감지 함수"""
    # 텍스트 감지를 위해 이미지 전처리
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # 이진화
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 윤곽선 검출
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # SAMSUNG 로고는 보통 직사각형 형태의 텍스트
    # 특정 비율의 직사각형 윤곽선을 찾음
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        # SAMSUNG 텍스트의 일반적인 가로세로 비율 (가로가 세로보다 약 3-5배 긺)
        if 3 <= aspect_ratio <= 5 and w > 50:
            # 이 부분에서 더 정확한 텍스트 인식을 위해 OCR을 사용할 수 있음
            # 간단한 구현을 위해 비율만 확인
            return True
    
    return False

# 브랜드별 이미지 색상 분석
def analyze_image_colors(image_path):
    """
    이미지의 색상 구성을 분석하여 애플 또는 삼성 제품일 가능성을 평가
    
    매개변수:
        image_path (str): 분석할 이미지 경로
        
    반환값:
        dict: 각 브랜드에 대한 색상 유사성 점수
    """
    # 이미지 로드
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)
    
    # 색상 히스토그램 계산
    hist_r = cv2.calcHist([image_np], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([image_np], [1], None, [256], [0, 256])
    hist_b = cv2.calcHist([image_np], [2], None, [256], [0, 256])
    
    # 정규화
    cv2.normalize(hist_r, hist_r, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_g, hist_g, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
    
    # 애플 색상 특성 (흰색, 검정색, 회색, 금색) - 수정된 계산 방식
    apple_white = np.sum(hist_r[200:256] * hist_g[200:256] * hist_b[200:256])
    apple_black = np.sum(hist_r[0:50] * hist_g[0:50] * hist_b[0:50])
    apple_gray = np.sum(hist_r[100:150] * hist_g[100:150] * hist_b[100:150])
    
    # 금색 계산 수정 (범위를 같게 맞춤)
    gold_range = 50  # 모든 채널에 동일한 범위 적용
    apple_gold = np.sum(hist_r[180:180+gold_range] * hist_g[150:150+gold_range] * hist_b[50:50+gold_range])
    
    # 삼성 색상 특성 (흰색, 검정색, 파란색)
    samsung_white = apple_white  # 흰색은 동일
    samsung_black = apple_black  # 검정색은 동일
    samsung_blue = np.sum(hist_r[0:100] * hist_g[0:100] * hist_b[150:250])  # 범위 수정
    
    # 색상 점수 계산
    apple_score = (apple_white + apple_black + apple_gray + apple_gold) / 4
    samsung_score = (samsung_white + samsung_black + samsung_blue) / 3
    
    return {
        "apple": float(apple_score),
        "samsung": float(samsung_score)
    }

# CLIP을 사용한 이미지 분류기 함수 (로고 감지 기능 통합)
def advanced_clip_classification(image_path, topk=10, display_image=True):
    """
    개선된 로고 감지를 활용한 이미지 분류 함수
    
    매개변수:
        image_path (str): 분석할 이미지 경로
        topk (int): 표시할 상위 결과 수
        display_image (bool): 이미지 표시 여부
        
    반환값:
        tuple: (최종 클래스, 신뢰도)
    """
    # 일반 CLIP 분류 수행
    clip_class, clip_confidence = classify_image_with_clip(image_path, display_image=False)
    
    # 브랜드 로고 감지 - 개선된 함수 사용
    brand, logo_confidence = detect_brand_logo(image_path)
    
    # 결과 수정
    final_class = clip_class
    final_confidence = clip_confidence
    
    # 애플 제품 관련 클래스 목록
    apple_classes = ["iphone", "ipad", "macbook", "airpods", "earpods", "apple watch"]
    
    # 삼성 제품 관련 클래스 목록 (일반 + 폴더블)
    samsung_classes = [
        "samsung phone", "galaxy phone", "galaxy s", "galaxy note", "galaxy tab",
        "galaxy watch", "galaxy buds", "samsung tablet", "samsung earbuds",
        "galaxy z flip", "galaxy z fold", "samsung foldable phone", "foldable phone",
        "samsung folding phone"
    ]
    
    # 로고 감지 결과와 CLIP 결과를 비교하여 최종 판단
    if brand == "apple" and logo_confidence > 0.5:
        # CLIP 결과가 애플 제품이 아니면 수정
        if not any(apple_term in clip_class.lower() for apple_term in apple_classes):
            print(f"로고 감지: 애플 로고 감지됨 (신뢰도: {logo_confidence:.2f})")
            
            # 이미지 형태 분석하여 구체적인 애플 제품 판단
            # 여기서는 간단히 아이폰으로 설정
            final_class = "iphone"
            
            # 애플 로고가 명확하게 감지된 경우 높은 신뢰도 부여
            if logo_confidence > 0.7:
                final_confidence = logo_confidence
            else:
                final_confidence = (clip_confidence + logo_confidence) / 2
                
    elif brand == "samsung" and logo_confidence > 0.5:
        # CLIP 결과가 삼성 제품이 아니면 수정
        if not any(samsung_term in clip_class.lower() for samsung_term in samsung_classes):
            print(f"로고 감지: 삼성 로고 감지됨 (신뢰도: {logo_confidence:.2f})")
            
            # 이미지 형태 분석으로 폴더블 여부 판단
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            
            # 폴더블 부분만 감지했을 경우를 위한 로직
            is_foldable = detect_fold_hinge(np.array(image))
            
            if is_foldable:
                if width > height * 1.2:  # 가로가 세로보다 20% 이상 길면
                    final_class = "galaxy z fold"
                else:
                    final_class = "galaxy z flip"
            else:
                # 폴더블 특성이 없으면 일반 삼성폰
                final_class = "samsung phone"
            
            # 삼성 로고가 명확하게 감지된 경우 높은 신뢰도 부여
            if logo_confidence > 0.7:
                final_confidence = logo_confidence
            else:
                final_confidence = (clip_confidence + logo_confidence) / 2
    
    # CLIP과 로고 감지 결과가 충돌할 경우 (애플 vs 삼성)
    elif clip_class in apple_classes and brand == "samsung":
        # 신뢰도 비교하여 선택
        if logo_confidence > clip_confidence * 1.5:  # 로고 신뢰도가 훨씬 높으면
            print(f"로고 감지(삼성)가 CLIP 결과(애플)보다 더 신뢰할 수 있습니다")
            final_class = "samsung phone"
            final_confidence = logo_confidence
        else:
            print(f"CLIP 결과(애플)를 유지합니다")
            # CLIP 결과 유지
    
    elif clip_class in samsung_classes and brand == "apple":
        # 신뢰도 비교하여 선택
        if logo_confidence > clip_confidence * 1.5:  # 로고 신뢰도가 훨씬 높으면
            print(f"로고 감지(애플)가 CLIP 결과(삼성)보다 더 신뢰할 수 있습니다")
            final_class = "iphone"  # 기본 아이폰으로 설정
            final_confidence = logo_confidence
        else:
            print(f"CLIP 결과(삼성)를 유지합니다")
            # CLIP 결과 유지
    
    # 최종 결과 표시
    if display_image:
        image = Image.open(image_path).convert("RGB")
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.title(f"Final Prediction: {final_class} ({final_confidence*100:.2f}%)")
        plt.axis('off')
        
        # 추가 정보 표시
        info_text = f"CLIP 예측: {clip_class} ({clip_confidence*100:.2f}%)\n"
        if brand:
            info_text += f"로고 감지: {brand} ({logo_confidence*100:.2f}%)\n"
        
        plt.figtext(0.5, 0.01, info_text, ha="center", fontsize=10, 
                   bbox={"facecolor":"white", "alpha":0.8, "pad":5})
        plt.tight_layout()
        plt.show()
    
    return final_class, final_confidence

def detect_fold_hinge(image_np):
    """폴더블 폰의 힌지(접히는 부분) 감지 함수"""
    # 그레이스케일 변환
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    
    # 에지 감지
    edges = cv2.Canny(gray, 50, 150)
    
    # 직선 감지 (힌지는 보통 직선으로 나타남)
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=100, 
        minLineLength=max(image_np.shape[0], image_np.shape[1]) // 4,  # 이미지 크기의 1/4 이상 길이
        maxLineGap=20
    )
    
    if lines is not None:
        height, width = image_np.shape[:2]
        center_y = height // 2
        
        # 중앙 부근에 긴 수직/수평선이 있는지 확인
        for line in lines:
            for x1, y1, x2, y2 in line:
                # 선의 길이 계산
                line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                
                # 선의 방향 (수직 또는 수평)
                is_vertical = abs(x2 - x1) < abs(y2 - y1)
                is_horizontal = not is_vertical
                
                # 이미지 중앙 부근에 있는 긴 선인지 확인
                is_center_region = (
                    (is_vertical and abs((x1 + x2) / 2 - width / 2) < width * 0.2) or
                    (is_horizontal and abs((y1 + y2) / 2 - height / 2) < height * 0.2)
                )
                
                # 충분히 긴 선이고 중앙 부근에 있으면 힌지로 간주
                if line_length > min(height, width) * 0.25 and is_center_region:
                    return True
    
    return False

# 메인 실행 부분 수정
if __name__ == "__main__":
    print("CLIP 이미지 분류기를 실행합니다.")
    print("로고 감지 + 고급 이미지 분류 기능이 자동으로 적용됩니다.")
    
    # 이미지 경로 입력받기
    image_path = input("이미지 경로를 입력하세요: ").strip()
    
    # 이미지 분류 실행
    if image_path:
        print(f"분석할 이미지: {image_path}")
        final_class, final_confidence = advanced_clip_classification(image_path)
        print(f"최종 예측: {final_class}, 신뢰도: {final_confidence*100:.2f}%")
    else:
        print("이미지 경로가 입력되지 않았습니다. 프로그램을 종료합니다.")