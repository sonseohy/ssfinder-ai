import torch
import clip
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

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
print(f"Result of tokenization (for random 3 labels):")
print(f"tensor: {clip.tokenize([f'a photo of a {custom_classes[0]}', f'a photo of a {custom_classes[1]}', f'a photo of a {custom_classes[2]}'])}")

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

# 테스트 이미지에 대한 분류 결과 시각화
def visualize_classification_results(image_path, topk=10):
    # 이미지 로드
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"이미지를 불러오는 데 오류가 발생했습니다: {e}")
        return
        
    # 유사도 계산
    values, indices = calculate_similarity(image, text_features, topk=topk)
    
    # 시각화
    plt.figure(figsize=(12, 6))
    
    # 이미지 표시
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    if len(indices) > 0:
        plt.title(f"Predicted: {custom_classes[indices[0].item()]}")
    plt.axis('off')
    
    # 결과 바 그래프
    plt.subplot(1, 2, 2)
    
    if len(indices) > 0:
        # 결과가 있을 때만 그래프 그리기
        plt.barh(range(len(values)), [v.item()*100 for v in values], color='skyblue')
        plt.yticks(range(len(values)), [custom_classes[idx.item()] for idx in indices])
        plt.xlabel('Confidence (%)')
        plt.title(f"Top {len(values)} Classification Results")
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        # 결과 출력
        print(f"Top {len(values)} Classification Results:")
        for i, (v, idx) in enumerate(zip(values, indices)):
            class_name = custom_classes[idx.item()]
            print(f"{class_name:<15}: {v.item()*100:.2f}%")
    else:
        plt.text(0.5, 0.5, "No results found", horizontalalignment='center', verticalalignment='center')
    
    plt.tight_layout()
    plt.show()

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
        if "electronics_general" in category_filter:
            filter_list.extend(electronics_general)
        if "electronics" in category_filter:
            filter_list.extend(electronics_apple + electronics_samsung + electronics_general)
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

# 사용자 입력으로 이미지 테스트하기
def test_with_user_image(image_path=None):
    # 이미지 경로가 직접 제공되지 않은 경우 파일 선택 다이얼로그 사용
    if image_path is None:
        import tkinter as tk
        from tkinter import filedialog
        
        # 파일 선택 다이얼로그
        root = tk.Tk()
        root.withdraw()  # GUI 창 숨기기
        
        print("이미지 파일을 선택해주세요...")
        file_path = filedialog.askopenfilename(
            title="이미지 파일 선택",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if not file_path:
            print("이미지가 선택되지 않았습니다.")
            return
        
        image_path = file_path
    
    # 이미지 분류 및 결과 시각화
    visualize_classification_results(image_path, topk=10)

# 메인 실행 부분
if __name__ == "__main__":
    print("CLIP 이미지 분류기를 실행합니다.")
    print("1. 사용자 이미지로 테스트")
    print("2. 여러 이미지 경로 입력하여 테스트")
    print("3. 인식 가능한 클래스 정보 보기")
    choice = input("선택하세요 (1, 2, 또는 3): ")
    
    if choice == '1':
        # 사용자 이미지로 테스트
        image_path = input("이미지 경로를 입력하거나 Enter 키를 눌러 파일 선택 대화상자 열기: ").strip()
        if image_path:
            test_with_user_image(image_path)
        else:
            test_with_user_image()
    elif choice == '2':
        # 여러 이미지 경로 입력하여 테스트
        print("테스트할 이미지 경로를 한 줄에 하나씩 입력하세요. 입력을 마치려면 빈 줄을 입력하세요.")
        image_paths = []
        while True:
            path = input("이미지 경로: ").strip()
            if not path:
                break
            image_paths.append(path)
        
        if not image_paths:
            print("입력된 이미지가 없습니다.")
        else:
            for path in image_paths:
                print(f"\n===== {path} 분석 결과 =====")
                try:
                    visualize_classification_results(path)
                except Exception as e:
                    print(f"이미지 분석 중 오류 발생: {e}")
    elif choice == '3':
        # 모델 정보 출력
        print("\n=== CLIP 이미지 분류기 정보 ===")
        
        # 카테고리 목록
        categories = {
            "가방 관련": [c for c in custom_classes if c in ["bag", "backpack", "handbag", "shopping bag", "여성용 가방", "남성용 가방", "쇼핑백"]],
            "귀금속/액세서리": [c for c in custom_classes if c in ["jewelry", "ring", "necklace", "earring", "watch", "반지", "목걸이", "귀걸이", "시계"]],
            "도서": [c for c in custom_classes if c in ["book", "textbook", "novel", "comic book", "학습서적", "소설", "만화책", "컴퓨터서적"]],
            "서류": [c for c in custom_classes if c in ["document", "certificate", "contract", "identification", "license", "서류", "증명서", "계약서", "신분증", "면허증"]],
            "산업용품/공구": [c for c in custom_classes if c in ["tool", "hammer", "screwdriver", "wrench", "공구", "망치", "드라이버", "렌치"]],
            "의류": [c for c in custom_classes if c in ["clothing", "shirt", "pants", "jacket", "coat", "hat", "dress", "suit", "uniform", "옷", "셔츠", "바지", "재킷", "코트", "모자", "드레스", "정장", "유니폼"]],
            "자동차 관련": [c for c in custom_classes if c in ["car", "car key", "car license", "자동차", "차키", "자동차등록증", "운전면허증"]],
            "전자기기 (애플)": [c for c in custom_classes if c in ["iphone", "ipad", "macbook", "airpods", "earpods", "apple watch"]],
            "전자기기 (삼성)": [c for c in custom_classes if c in ["samsung phone", "galaxy phone", "galaxy s", "galaxy note", "galaxy tab", "galaxy watch", "galaxy buds", "samsung tablet", "samsung earbuds"]],
            "전자기기 (일반)": [c for c in custom_classes if c in ["smartphone", "phone", "tablet", "wireless earbuds", "headphones", "headset", "laptop", "camera", "smart watch", "earbuds", "태블릿", "휴대폰", "노트북", "카메라"]],
            "지갑/카드": [c for c in custom_classes if c in ["wallet", "purse", "credit card", "id card", "transportation card", "membership card", "지갑", "여성용 지갑", "남성용 지갑", "카드", "신용카드", "교통카드", "멤버십카드"]],
            "현금/화폐": [c for c in custom_classes if c in ["money", "cash", "korean won", "10000 won", "50000 won", "dollar", "euro", "현금", "지폐", "동전", "한국 돈", "만원", "오만원", "천원", "오천원", "백원", "오백원"]],
            "기타 분실물": [c for c in custom_classes if c in ["umbrella", "key", "glasses", "sunglasses", "water bottle", "우산", "열쇠", "안경", "선글라스", "물병"]]
        }
        
        # 카테고리별 클래스 출력
        total_classes = 0
        for category, items in categories.items():
            print(f"\n{category} ({len(items)}개):")
            for item in items:
                print(f"- {item}")
            total_classes += len(items)
            
        print(f"\n총 인식 가능 클래스 수: {total_classes}개")
    else:
        print("잘못된 선택입니다. 프로그램을 종료합니다.")