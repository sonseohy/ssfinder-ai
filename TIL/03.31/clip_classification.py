import torch
import clip
import numpy as np
from PIL import Image
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# CLIP 모델과 전처리기(preprocessor) 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# CIFAR-100 데이터셋 로드
cifar100 = datasets.CIFAR100(root='./data', download=True, train=False)
classes = cifar100.classes  # CIFAR-100의 클래스(라벨) 목록

# 추가 클래스 정의 (CIFAR-100에 없는 클래스들)
additional_classes = [
    # 전자기기 - 애플 제품
    "iphone", "ipad", "macbook", "airpods", "earpods", "apple watch",
    
    # 전자기기 - 삼성 제품
    "samsung phone", "galaxy phone", "galaxy s", "galaxy note", "galaxy tab", 
    "galaxy watch", "galaxy buds", "samsung tablet", "samsung earbuds",
    
    # 전자기기 - 일반
    "smartphone", "phone", "tablet", "wireless earbuds", "headphones", "headset",
    "laptop", "smartwatch", "smart watch", "earbuds", "smart device",
    
    # 분실물로 자주 발견되는 물건들
    "wallet", "purse", "credit card", "id card", "card", "key", "keys", "backpack", 
    "bag", "glasses", "sunglasses", "book", "notebook", "pen", "jacket", "coat", 
    "scarf", "hat", "gloves", "umbrella", "water bottle"
]
all_classes = classes + additional_classes

# 요구사항 2-1: 라벨 정보 토큰화하기
def tokenize_labels(classes):
    # "a photo of a {label}" 형식으로 텍스트 프롬프트 생성
    text_inputs = []
    expanded_class_list = []
    
    for c in classes:
        # CIFAR-100 클래스는 기본 프롬프트만 사용
        if c in cifar100.classes:
            text_inputs.append(clip.tokenize(f"a photo of a {c}"))
            expanded_class_list.append(c)
        # 추가 클래스에 대해서는 다양한 프롬프트 변형 제공
        else:
            # 기본 관사 처리 (a/an)
            article = "an" if c[0].lower() in "aeiou" else "a"
            
            # 특수 케이스 처리
            if c == "airpods" or c == "earpods" or c == "headphones" or c == "glasses" or c == "sunglasses" or c == "keys" or c == "gloves" or c == "galaxy buds" or c == "samsung earbuds" or c == "earbuds":
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
                     "laptop", "smartwatch", "smart watch", "earbuds", "smart device"]:
                prompts.append(f"a modern {c}")
                prompts.append(f"an electronic {c}")
            
            # 분실물 관련
            if c in ["wallet", "purse", "credit card", "id card", "card", "key", "keys", "backpack", 
                      "bag", "glasses", "sunglasses", "book", "notebook", "pen", "jacket", "coat", 
                      "scarf", "hat", "gloves", "umbrella", "water bottle"]:
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
text_features, expanded_classes = tokenize_labels(all_classes)
print(f"Number of original classes: {len(all_classes)}")
print(f"Number of expanded prompts: {len(expanded_classes)}")
print(f"Result of tokenization (for random 3 labels):")
print(f"tensor: {clip.tokenize([f'a photo of a {all_classes[0]}', f'a photo of a {all_classes[1]}', f'a photo of a {all_classes[2]}'])}")

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
            original_idx = all_classes.index(class_name) if class_name in all_classes else -1
            result_values.append(v)
            result_indices.append(original_idx)
            
            if len(result_values) >= topk:
                break
    
    return torch.tensor(result_values), torch.tensor(result_indices)

# 테스트 이미지에 대한 분류 결과 시각화
def visualize_classification_results(image_indices, topk=10):
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(image_indices):
        # 이미지와 정답 라벨 가져오기
        image, label = cifar100[idx]
        
        # 유사도 계산
        values, indices = calculate_similarity(image, text_features, topk=topk)
        
        # 시각화
        plt.subplot(len(image_indices), 2, 2*i+1)
        plt.imshow(image)
        plt.title(f"Label (answer): {classes[label]}")
        plt.axis('off')
        
        plt.subplot(len(image_indices), 2, 2*i+2)
        plt.bar(range(len(values)), values.cpu().numpy())
        plt.xticks(range(len(values)), [all_classes[idx.item()] for idx in indices], rotation=45, ha='right')
        plt.title(f"Top {len(values)} Classification Results:")
        for j, (v, idx) in enumerate(zip(values, indices)):
            print(f"{all_classes[idx.item()]:<15}: {v.item()*100:.2f}%")
        
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
                              "laptop", "smartwatch", "smart watch", "earbuds", "smart device"]
        lost_items = ["wallet", "purse", "credit card", "id card", "card", "key", "keys", "backpack", 
                     "bag", "glasses", "sunglasses", "book", "notebook", "pen", "jacket", "coat", 
                     "scarf", "hat", "gloves", "umbrella", "water bottle"]
        
        filter_list = []
        if "electronics_apple" in category_filter:
            filter_list.extend(electronics_apple)
        if "electronics_samsung" in category_filter:
            filter_list.extend(electronics_samsung)
        if "electronics_general" in category_filter:
            filter_list.extend(electronics_general)
        if "electronics" in category_filter:
            filter_list.extend(electronics_apple + electronics_samsung + electronics_general)
        if "lost_items" in category_filter:
            filter_list.extend(lost_items)
        if "all" in category_filter:
            filter_list = all_classes
            
        for v, idx in zip(values, indices):
            class_name = all_classes[idx.item()]
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
        class_name = all_classes[idx.item()]
        print(f"{class_name:<15}: {v.item()*100:.2f}%")
    
    # 이미지 표시
    if display_image:
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        if len(indices) > 0:
            plt.title(f"Predicted: {all_classes[indices[0].item()]}")
        plt.axis('off')
        plt.show()
    
    if len(indices) > 0:
        return all_classes[indices[0].item()], values[0].item()
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
    
    print(f"선택한 이미지: {image_path}")
    
    # 이미지 분류 실행 (모든 카테고리 사용)
    top_class, confidence = classify_image_with_clip(
        image_path=image_path,
        category_filter=["all"]  # 항상 모든 카테고리 사용
    )
    
    if top_class is not None:
        print(f"예측 클래스: {top_class}, 확률: {confidence*100:.2f}%")
    else:
        print("이미지 분류에 실패했습니다.")

# 메인 실행 부분
if __name__ == "__main__":
    print("CLIP 이미지 분류기를 실행합니다.")
    print("1. 예제 이미지로 테스트")
    print("2. 사용자 이미지로 테스트")
    print("3. 인식 가능한 클래스 정보 보기")
    choice = input("선택하세요 (1, 2, 또는 3): ")
    
    if choice == '1':
        # 실행 결과 예시: CIFAR-100의 몇 개의 테스트 이미지에 대해 분류 결과 시각화
        test_indices = [10, 20, 30]  # 테스트할 이미지 인덱스
        print(f"CIFAR-100 테스트 이미지 인덱스 {test_indices}에 대한 분류 결과:")
        visualize_classification_results(test_indices)
    elif choice == '2':
        # 사용자 이미지로 테스트
        image_path = input("이미지 경로를 입력하거나 Enter 키를 눌러 파일 선택 대화상자 열기: ").strip()
        if image_path:
            test_with_user_image(image_path)
        else:
            test_with_user_image()
    elif choice == '3':
        # 모델 정보 출력
        print("\n=== CLIP 이미지 분류기 정보 ===")
        print(f"CIFAR-100 클래스 수: {len(classes)}")
        print(f"추가된 클래스 수: {len(additional_classes)}")
        print(f"총 인식 가능 클래스 수: {len(all_classes)}")
        
        # 카테고리별 클래스 출력
        print("\n전자기기 클래스:")
        electronics = ["cellphone", "smartphone", "phone", "tablet", "ipad", "airpods", "earpods", 
                      "wireless earbuds", "headphones", "headset", "laptop", "smartwatch", "smart watch", 
                      "apple watch"]
        for item in electronics:
            if item in additional_classes:
                print(f"- {item}")
                
        print("\n분실물 관련 클래스:")
        lost_items = ["wallet", "purse", "credit card", "id card", "card", "key", "keys", "backpack", 
                     "bag", "glasses", "sunglasses", "book", "notebook", "pen", "jacket", "coat", 
                     "scarf", "hat", "gloves", "umbrella", "water bottle"]
        for item in lost_items:
            if item in additional_classes:
                print(f"- {item}")
    else:
        print("잘못된 선택입니다. 프로그램을 종료합니다.")