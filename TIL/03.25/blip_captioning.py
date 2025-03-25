# 코드를 실행할 때 이미지 파일명만 지정
# python 스크립트.py --image_name bag.jpg

# 다른 디렉토리에 이미지를 저장했다면 다음과 같이 지정
# python 스크립트.py --image_name bag.jpg --images_dir 다른_디렉토리

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import time
import os

def test_blip_english_captioning(image_name=None, images_dir="test_images"):
    """
    BLIP 모델을 사용하여 영어로 이미지 캡션을 생성하는 테스트 코드
    
    Args:
        image_name (str): 이미지 파일 이름 (예: "phone.jpg")
        images_dir (str): 이미지 파일이 위치한 디렉토리 경로
    """
    # 시작 시간 기록
    start_time = time.time()
    
    # 모델 및 프로세서 로드
    print("BLIP 모델 로딩 중...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    
    # GPU 사용 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"모델이 {device}에 로드되었습니다.")
    
    # 이미지 로드
    if image_name:
        image_path = os.path.join(images_dir, image_name)
        print(f"이미지 로드 중: {image_path}")
        if os.path.exists(image_path):
            raw_image = Image.open(image_path).convert('RGB')
        else:
            print(f"에러: 파일 '{image_path}'을 찾을 수 없습니다.")
            return None
    else:
        # 기본 테스트 이미지 사용
        default_image = "phone.jpg"  # 기본 이미지 파일명
        image_path = os.path.join(images_dir, default_image)
        
        if os.path.exists(image_path):
            print(f"기본 테스트 이미지 사용: {image_path}")
            raw_image = Image.open(image_path).convert('RGB')
        else:
            print(f"에러: 기본 이미지 '{image_path}'를 찾을 수 없습니다.")
            return None
    
    # 이미지 전처리 및 캡셔닝 실행
    print("이미지 캡셔닝 실행 중...")
    
    # 기본 캡션 생성
    inputs = processor(raw_image, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_length=50)
    caption = processor.decode(output[0], skip_special_tokens=True)
    
    # 상세 캡션 생성 (프롬프트 사용)
    text = "a photograph of"
    inputs = processor(raw_image, text, return_tensors="pt").to(device)
    detailed_output = model.generate(**inputs, max_length=100)
    detailed_caption = processor.decode(detailed_output[0], skip_special_tokens=True)
    
    # 실행 시간 계산
    execution_time = time.time() - start_time
    
    # 결과 출력
    print("\n===== BLIP 영어 이미지 캡셔닝 결과 =====")
    print(f"이미지: {image_path}")
    print(f"기본 캡션: {caption}")
    print(f"상세 캡션: {detailed_caption}")
    print(f"실행 시간: {execution_time:.2f}초")
    print("========================================\n")
    
    return {
        "image_path": image_path,
        "basic_caption": caption,
        "detailed_caption": detailed_caption,
        "execution_time": execution_time
    }

if __name__ == "__main__":
    # 명령줄에서 실행 시 인자 파싱
    import argparse
    parser = argparse.ArgumentParser(description="BLIP 영어 이미지 캡셔닝 테스트")
    parser.add_argument("--image_name", type=str, help="이미지 파일 이름 (예: phone.jpg)")
    parser.add_argument("--images_dir", type=str, default="test_images", help="이미지 파일이 위치한 디렉토리 경로")
    args = parser.parse_args()
    
    # 테스트 실행
    result = test_blip_english_captioning(args.image_name, args.images_dir)