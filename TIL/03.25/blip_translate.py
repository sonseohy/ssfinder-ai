import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time
import os

def test_blip_with_translation(image_name=None, images_dir="test_images", translator_model_name="facebook/nllb-200-distilled-600M"):
    """
    BLIP 모델로 영어 캡션을 생성하고 번역 모델로 한국어로 변환하는 테스트 코드
    
    Args:
        image_name (str): 이미지 파일 이름 (예: "phone.jpg")
        images_dir (str): 이미지 파일이 위치한 디렉토리 경로
        translator_model_name (str): 번역에 사용할 모델 이름
    """
    # 시작 시간 기록
    start_time = time.time()
    
    # BLIP 모델 로드
    print("BLIP 모델 로딩 중...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    
    # 번역 모델 로드
    print(f"번역 모델 로딩 중: {translator_model_name}")
    translator_tokenizer = AutoTokenizer.from_pretrained(translator_model_name)
    translator_model = AutoModelForSeq2SeqLM.from_pretrained(translator_model_name)
    
    # GPU 사용 설정
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blip_model.to(device)
    translator_model.to(device)
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
    
    # 영어 캡션 생성
    print("BLIP으로 영어 캡션 생성 중...")
    inputs = blip_processor(raw_image, return_tensors="pt").to(device)
    blip_output = blip_model.generate(**inputs, max_length=75)
    english_caption = blip_processor.decode(blip_output[0], skip_special_tokens=True)
    
    # 상세 영어 캡션 생성 (프롬프트 사용)
    text = "a detailed description of this image"
    inputs = blip_processor(raw_image, text, return_tensors="pt").to(device)
    detailed_blip_output = blip_model.generate(**inputs, max_length=150)
    detailed_english_caption = blip_processor.decode(detailed_blip_output[0], skip_special_tokens=True)
    
    # 한국어로 번역
    print("영어 캡션을 한국어로 번역 중...")
    
    # 기본 캡션 번역
    if "nllb" in translator_model_name.lower():
        # NLLB 모델용 번역 코드
        translator_inputs = translator_tokenizer(english_caption, return_tensors="pt", src_lang="eng_Latn").to(device)
        translated_ids = translator_model.generate(
            **translator_inputs, 
            forced_bos_token_id=translator_tokenizer.lang_code_to_id["kor_Hang"], 
            max_length=100
        )
        korean_caption = translator_tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
        
        # 상세 캡션 번역
        detailed_translator_inputs = translator_tokenizer(detailed_english_caption, return_tensors="pt", src_lang="eng_Latn").to(device)
        detailed_translated_ids = translator_model.generate(
            **detailed_translator_inputs, 
            forced_bos_token_id=translator_tokenizer.lang_code_to_id["kor_Hang"], 
            max_length=200
        )
        detailed_korean_caption = translator_tokenizer.batch_decode(detailed_translated_ids, skip_special_tokens=True)[0]
    else:
        # M2M100 또는 다른 번역 모델용 코드
        translator_inputs = translator_tokenizer(english_caption, return_tensors="pt").to(device)
        translated_ids = translator_model.generate(
            **translator_inputs, 
            forced_bos_token_id=translator_tokenizer.get_lang_id("ko"), 
            max_length=100
        )
        korean_caption = translator_tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
        
        # 상세 캡션 번역
        detailed_translator_inputs = translator_tokenizer(detailed_english_caption, return_tensors="pt").to(device)
        detailed_translated_ids = translator_model.generate(
            **detailed_translator_inputs, 
            forced_bos_token_id=translator_tokenizer.get_lang_id("ko"), 
            max_length=200
        )
        detailed_korean_caption = translator_tokenizer.batch_decode(detailed_translated_ids, skip_special_tokens=True)[0]
    
    # 실행 시간 계산
    execution_time = time.time() - start_time
    
    # 결과 출력
    print("\n===== BLIP + 번역 파이프라인 결과 =====")
    print(f"이미지: {image_path}")
    print(f"영어 캡션: {english_caption}")
    print(f"한국어 번역: {korean_caption}")
    print(f"\n상세 영어 캡션: {detailed_english_caption}")
    print(f"상세 한국어 번역: {detailed_korean_caption}")
    print(f"실행 시간: {execution_time:.2f}초")
    print("========================================\n")
    
    return {
        "image_path": image_path,
        "english_caption": english_caption,
        "korean_caption": korean_caption,
        "detailed_english_caption": detailed_english_caption,
        "detailed_korean_caption": detailed_korean_caption,
        "execution_time": execution_time
    }

if __name__ == "__main__":
    # 명령줄에서 실행 시 인자 파싱
    import argparse
    parser = argparse.ArgumentParser(description="BLIP + 번역 파이프라인 테스트")
    parser.add_argument("--image_name", type=str, help="이미지 파일 이름 (예: phone.jpg)")
    parser.add_argument("--images_dir", type=str, default="test_images", help="이미지 파일이 위치한 디렉토리 경로")
    parser.add_argument("--translator", type=str, default="facebook/nllb-200-distilled-600M", 
                        help="번역 모델 이름 (facebook/nllb-200-distilled-600M 또는 facebook/m2m100_418M)")
    args = parser.parse_args()
    
    # 테스트 실행
    result = test_blip_with_translation(args.image_name, args.images_dir, args.translator)