import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse

def generate_captions(image_path):
    """
    BLIP 모델을 사용하여 영어 이미지 캡션을 생성하고 
    NLLB 모델을 사용하여 한국어로 번역합니다.
    
    Args:
        image_path (str): 이미지 파일 경로
        
    Returns:
        tuple: 영어 캡션과 한국어 캡션
    """
    # BLIP 모델 로딩
    print("BLIP 모델 로딩 중...")
    blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    
    # NLLB 번역 모델 로딩
    print("NLLB 모델 로딩 중...")
    translator_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B")
    translator_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")
    
    # 이미지 로딩 및 처리
    print(f"이미지 처리 중: {image_path}")
    image = Image.open(image_path).convert('RGB')
    inputs = blip_processor(image, return_tensors="pt")
    
    # 영어 캡션 생성
    print("영어 캡션 생성 중...")
    output = blip_model.generate(**inputs, max_length=50)
    english_caption = blip_processor.decode(output[0], skip_special_tokens=True)
    
    # NLLB 모델로 한국어 번역
    print("NLLB 모델로 한국어 번역 중...")
    # 영어(eng_Latn)에서 한국어(kor_Hang)로 번역
    inputs = translator_tokenizer(english_caption, return_tensors="pt")
    
    # 번역 대상 언어 설정 (한국어)
    forced_bos_token_id = translator_tokenizer.lang_code_to_id["kor_Hang"]
    
    translated_tokens = translator_model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        max_length=100,
        num_beams=5,
        early_stopping=True
    )
    
    korean_caption = translator_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    
    return english_caption, korean_caption

def show_example_with_image():
    """
    GUI로 이미지 선택 및 캡션 생성을 보여주는 예제 함수
    """
    import tkinter as tk
    from tkinter import filedialog
    from PIL import ImageTk
    import io
    
    # GUI 초기화
    root = tk.Tk()
    root.title("BLIP 이미지 캡셔닝 데모")
    root.geometry("800x600")
    
    # 변수 초기화
    selected_image_path = tk.StringVar()
    english_result = tk.StringVar()
    korean_result = tk.StringVar()
    
    # 이미지 표시 영역
    image_frame = tk.Frame(root, width=400, height=300)
    image_frame.pack(pady=10)
    image_label = tk.Label(image_frame)
    image_label.pack()
    
    # 결과 표시 영역
    result_frame = tk.Frame(root)
    result_frame.pack(pady=10, fill=tk.X, padx=20)
    
    tk.Label(result_frame, text="영어 캡션:").grid(row=0, column=0, sticky=tk.W, pady=5)
    tk.Label(result_frame, textvariable=english_result, wraplength=700).grid(row=0, column=1, sticky=tk.W, pady=5)
    
    tk.Label(result_frame, text="한국어 캡션:").grid(row=1, column=0, sticky=tk.W, pady=5)
    tk.Label(result_frame, textvariable=korean_result, wraplength=700).grid(row=1, column=1, sticky=tk.W, pady=5)
    
    def select_image():
        file_path = filedialog.askopenfilename(
            title="이미지 선택",
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            selected_image_path.set(file_path)
            display_image(file_path)
    
    def display_image(file_path):
        img = Image.open(file_path)
        # 이미지 크기 조정
        img.thumbnail((400, 300))
        photo_img = ImageTk.PhotoImage(img)
        image_label.config(image=photo_img)
        image_label.image = photo_img  # 참조 유지
    
    def process_selected_image():
        if selected_image_path.get():
            english, korean = generate_captions(selected_image_path.get())
            english_result.set(english)
            korean_result.set(korean)
    
    # 버튼 영역
    button_frame = tk.Frame(root)
    button_frame.pack(pady=10)
    
    select_button = tk.Button(button_frame, text="이미지 선택", command=select_image)
    select_button.grid(row=0, column=0, padx=10)
    
    process_button = tk.Button(button_frame, text="캡션 생성", command=process_selected_image)
    process_button.grid(row=0, column=1, padx=10)
    
    # 상태 표시 레이블
    status_label = tk.Label(root, text="", bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    # 모델 불러오기 메모리 경고
    memory_warning = tk.Label(root, text="※ 주의: NLLB-200-3.3B 모델은 크기가 매우 큽니다 (약 6GB). 충분한 메모리가 필요합니다.", fg="red")
    memory_warning.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BLIP 이미지 캡셔닝 및 한국어 번역')
    parser.add_argument('--image_path', type=str, help='이미지 파일 경로')
    parser.add_argument('--gui', action='store_true', help='GUI 인터페이스 사용')
    
    args = parser.parse_args()
    
    if args.gui:
        show_example_with_image()
    elif args.image_path:
        english, korean = generate_captions(args.image_path)
        
        print("\n===== 결과 =====")
        print("영어 캡션:", english)
        print("한국어 캡션:", korean)
    else:
        print("이미지 경로를 입력하거나 GUI 모드를 선택하세요.")
        print("사용법: python script.py --image_path 이미지경로.jpg")
        print("또는:  python script.py --gui")