import torch
from PIL import Image
import numpy as np
import time
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from typing import Dict, List, Any, Tuple
from models.translator import Translator
from config import config
from ultralytics import YOLO
import cv2
import clip
import io

# 이미지 분석 및 캡셔닝
class ImageAnalyzer:
    def __init__(self):
        # GPU 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 캡셔닝용 BLIP 모델 로드 (GPU로)
        self.caption_processor = BlipProcessor.from_pretrained(config.CAPTION_MODEL)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(config.CAPTION_MODEL).to(self.device)
        
        # VQA용 BLIP 모델 로드 (GPU로)
        self.vqa_processor = BlipProcessor.from_pretrained(config.VQA_MODEL)
        self.vqa_model = BlipForQuestionAnswering.from_pretrained(config.VQA_MODEL).to(self.device)
        
        # YOLO 모델 로드
        try:
            self.yolo_model = YOLO('yolov8m-oiv7.pt')
            print("YOLO 모델 로드 성공")
        except Exception as e:
            print(f"YOLO 모델 로드 실패: {e}")
            self.yolo_model = None
        
        # CLIP 모델 로드
        try:
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
            print("CLIP 모델 로드 성공")
        except Exception as e:
            print(f"CLIP 모델 로드 실패: {e}")
            self.clip_model = None
            self.clip_preprocess = None

    def preprocess_image(self, image_path: str) -> Image.Image:
        """이미지 전처리 및 크기 조정 메서드"""
        # 이미지 로드 및 크기 제한
        image = Image.open(image_path).convert("RGB")
        
        # 이미지 크기 조정 (최대 크기 제한)
        width, height = image.size
        max_size = config.MAX_IMAGE_SIZE
        
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        return image
    
    def generate_caption(self, image: Image.Image) -> str:
        """이미지 캡션 생성 메서드"""
        # 이미지를 모델 입력으로 처리 (GPU로)
        inputs = self.caption_processor(image, return_tensors="pt").to(self.device)
        
        # 캡션 생성 (GPU 최적화)
        with torch.no_grad(), torch.cuda.amp.autocast():
            out = self.caption_model.generate(
                **inputs,
                max_length=75,
                num_beams=5,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.5
            )
        
        # 토큰을 텍스트로 디코딩
        caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
        return caption

    def ask_question(self, image: Image.Image, question: str) -> str:
        """이미지에 대한 질문 응답 메서드"""
        # 이미지와 질문을 모델 입력으로 처리 (GPU로)
        inputs = self.vqa_processor(image, question, return_tensors="pt").to(self.device)
        
        # 질문에 대한 답변 생성 (GPU 최적화)
        with torch.no_grad(), torch.cuda.amp.autocast():
            out = self.vqa_model.generate(
                **inputs,
                max_length=20,
                num_beams=5,
                do_sample=True,
                top_p=0.9
            )
        
        # 토큰을 텍스트로 디코딩
        answer = self.vqa_processor.decode(out[0], skip_special_tokens=True)
        
        return answer
    
    def detect_objects(self, image_path: str) -> Tuple[List[Dict], Image.Image]:
        """YOLO를 사용하여 객체 탐지"""
        if self.yolo_model is None:
            print("YOLO 모델이 로드되지 않았습니다.")
            return [], Image.open(image_path).convert("RGB")
            
        # 이미지 로드
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"이미지 로드 실패: {image_path}")
            
            # YOLO 객체 탐지 실행
            results = self.yolo_model(image)
            
            # 결과 정보 추출
            objects = []
            for result in results:
                boxes = result.boxes
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(val) for val in box.xyxy[0].tolist()]
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    # 신뢰도가 0.4 이상인 객체만 추가
                    if confidence >= 0.4:
                        objects.append({
                            'box': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name
                        })
            
            # PIL 이미지로 변환
            pil_image = Image.open(image_path).convert("RGB")
            
            return objects, pil_image
        except Exception as e:
            print(f"객체 탐지 중 오류: {e}")
            return [], Image.open(image_path).convert("RGB")
    
    def clip_analyze(self, image: Image.Image, categories: List[str]) -> Dict[str, float]:
        """CLIP 모델로 이미지 범주 분석"""
        if self.clip_model is None or self.clip_preprocess is None:
            print("CLIP 모델이 로드되지 않았습니다.")
            return {}
            
        try:
            # 이미지 전처리
            image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)
            
            # 텍스트 전처리
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in categories]).to(self.device)
            
            # 예측 실행
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_inputs)
                
                # 정규화
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # 유사도 계산
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                
            # 결과를 딕셔너리로 변환
            results = {}
            for i, category in enumerate(categories):
                results[category] = float(similarity[0][i].item())
                
            return results
        except Exception as e:
            print(f"CLIP 분석 중 오류: {e}")
            return {}
    
    def crop_object(self, image: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
        """객체 영역 잘라내기"""
        try:
            x1, y1, x2, y2 = box
            return image.crop((x1, y1, x2, y2))
        except Exception as e:
            print(f"객체 잘라내기 오류: {e}")
            return image


class LostItemAnalyzer:
    def __init__(self):
        # 이미지 분석기 초기화
        self.image_analyzer = ImageAnalyzer()
        
        # 번역기 초기화
        self.translator = Translator()
        
        # 브랜드 매칭을 적용할 카테고리 목록
        self.brand_applicable_categories = [
            "electronics", "phone", "computer", "wallet", "earbuds", "smartwatch", 
            "jewelry", "bag", "accessories", "clothing"
        ]
    
    def extract_category_from_caption(self, caption: str) -> str:
        """캡션에서 카테고리 추출"""
        caption_lower = caption.lower()
        words = caption_lower.split()
        
        # 키워드를 길이 순으로 정렬 (긴 것이 먼저)
        sorted_keywords = sorted(config.CATEGORY_MAPPING.keys(), key=len, reverse=True)
        
        # 각 매핑된 키워드를 확인
        for keyword in sorted_keywords:
            # 전체 단어 매칭 또는 복합어 검사
            if keyword in words or (len(keyword.split()) > 1 and keyword in caption_lower):
                return config.CATEGORY_MAPPING[keyword]
        
        return ""
    
    def is_valid_color(self, text: str) -> bool:
        """색상의 유효성 검사"""
        text = text.lower()
        return any(color in text for color in config.COLORS)
    
    def is_valid_material(self, text: str) -> bool:
        """재질의 유효성 검사"""
        text = text.lower()
        return any(material in text for material in config.MATERIALS)
    
    def should_apply_brand_matching(self, category: str, caption: str) -> bool:
        """현재 항목에 브랜드 매칭을 적용해야 하는지 확인"""
        # 카테고리가 브랜드 적용 가능 목록에 있는지 확인
        if any(cat in category.lower() for cat in self.brand_applicable_categories):
            return True
            
        # 캡션에 브랜드 적용 카테고리 키워드가 있는지 확인
        caption_lower = caption.lower()
        brand_keywords = [
            "phone", "smartphone", "laptop", "computer", "watch", "smartwatch",
            "earbuds", "headphones", "wallet", "purse", "jewelry", "bag", "handbag",
            "tablet", "electronics", "gadget", "device", "accessory"
        ]
        
        if any(keyword in caption_lower for keyword in brand_keywords):
            return True
            
        return False
    
    def extract_brand(self, text_list: List[str]) -> str:
        """텍스트 목록에서 브랜드 추출 (개선된 버전)"""
        # 직접적인 브랜드 언급 확인 (대소문자 구분 없이)
        common_brands = [k for k in config.BRAND_TRANSLATION.keys() if k != "unknown"]
        
        # 삼성 제품 키워드
        samsung_keywords = ["galaxy", "samsung", "갤럭시", "삼성", "buds", "버즈", 
                            "gear", "watch", "워치", "note", "노트", "fold", "flip", "s series"]
        
        # 지갑 브랜드 키워드
        wallet_keywords = ["wallet", "지갑", "purse", "billfold", "cardholder", "money holder"]
        
        # 결합된 키워드 생성 (예: "samsung phone", "galaxy buds", "louis vuitton wallet")
        combined_brand_keywords = []
        for text in text_list:
            text_lower = text.lower()
            
            # 삼성 제품 감지
            for keyword in samsung_keywords:
                if keyword in text_lower:
                    return "samsung"
            
            # 지갑 브랜드 감지
            for brand in ["louis vuitton", "lv", "gucci", "prada", "hermes", "chanel", 
                        "coach", "montblanc", "bottega", "fendi", "burberry", "dior"]:
                if brand in text_lower:
                    # 지갑과 브랜드가 모두 언급된 경우
                    for wallet_word in wallet_keywords:
                        if wallet_word in text_lower:
                            return brand.replace("lv", "louis vuitton")
                    
                    # 지갑이 명확히 언급되지 않았어도 고급 브랜드는 지갑일 가능성이 높음
                    return brand.replace("lv", "louis vuitton")
        
        for text in text_list:
            text_lower = text.lower()
            
            # 직접적인 브랜드 언급 확인 (전체 문자열 검색)
            for brand in common_brands:
                if brand in text_lower:
                    return brand
            
            # 브랜드 연관 매핑 확인 (정확한 제품-브랜드 연결)
            for product, brand in config.BRAND_ASSOCIATION.items():
                if product in text_lower:
                    return brand
        
        return ""
    
    def is_galaxy_buds(self, caption: str, answers: List[str]) -> bool:
        """이미지가 갤럭시 버즈인지 확인"""
        # 캡션과 모든 응답을 결합하여 검사
        all_text = caption.lower() + " " + " ".join(answer.lower() for answer in answers if answer)
        
        # 갤럭시 버즈 키워드
        buds_keywords = ["galaxy buds", "samsung buds", "galaxy earbuds", "samsung earbuds", 
                        "buds", "wireless earbuds", "wireless earphones"]
                        
        # iPod 키워드 제외
        if "ipod" in all_text:
            # iPod 키워드가 있지만 갤럭시 버즈 키워드도 있는지 확인
            for keyword in buds_keywords:
                if keyword in all_text:
                    return True
            return False
            
        # 이어버드/이어폰 관련 키워드가 있고 삼성 관련 키워드도 있는지 확인
        has_earbuds_keyword = any(keyword in all_text for keyword in ["earbuds", "earphones", "headphones", "buds"])
        has_samsung_keyword = any(keyword in all_text for keyword in ["samsung", "galaxy"])
        
        if has_earbuds_keyword and has_samsung_keyword:
            return True
            
        # 명시적인 갤럭시 버즈 키워드 확인
        for keyword in buds_keywords:
            if keyword in all_text:
                return True
                
        return False
    
    def generate_title(self, answers: Dict[str, str], caption: str, category: str, brand: str) -> str:
        """답변 기반 게시글 제목 생성"""
        # 색상 추출
        color = answers["color"].lower()
        
        # 상품 이름 추출 시도 (캡션에서 핵심 단어 추출)
        product_name = ""
        
        # 정확한 단어 매칭을 위해 캡션을 단어로 분리
        caption_words = caption.lower().split()
        
        common_items = ["headphones", "earphones", "ipad", "iphone", "macbook", "laptop", "phone", 
                   "tablet", "watch", "airpods", "wallet", "bag", "umbrella", "camera", 
                   "book", "glasses"]
        
        # 단어 단위로 정확하게 매칭
        for item in common_items:
            if item in caption_words:
                product_name = item
                break
        
        # 제목 생성
        if product_name:
            # 제품명이 발견되면 이를 사용
            title = f"{color} {product_name}"
        elif category:
            # 카테고리 사용
            title = f"{color} {category}"
        else:
            # 둘 다 없는 경우 일반적 항목으로
            title = f"{color} item"
            
        # 브랜드 추가 (있는 경우)
        if brand and brand.lower() not in title.lower():
            title = f"{brand} {title}"
        
        return title
    
    def generate_description(self, caption: str, answers: Dict[str, str]) -> str:
        """캡션과 답변 기반 상세 설명 생성"""
        description = caption + "\n\n"
        
        # 재질, 특징과 같은 추가 정보 포함
        if answers["material"] and "unknown" not in answers["material"].lower():
            description += f"Material: {answers['material']}\n"
            
        if answers["distinctive_features"] and "none" not in answers["distinctive_features"].lower():
            description += f"Distinctive features: {answers['distinctive_features']}\n"
            
        return description.strip()
    
    def extract_features_with_yolo_clip(self, image_path: str) -> Dict[str, Any]:
        """YOLO로 객체 탐지 후 CLIP 모델로 특징 추출"""
        start_time = time.time()
        
        # YOLO 객체 탐지
        yolo_start = time.time()
        objects, original_image = self.image_analyzer.detect_objects(image_path)
        yolo_time = time.time() - yolo_start
        
        # 객체가 없는 경우, 전체 이미지로 진행
        if not objects:
            print("객체 탐지 결과 없음. 전체 이미지로 분석합니다.")
            return self.extract_features(original_image, timing_info={"yolo_time": yolo_time})
        
        # 가장 큰 객체 선택 (면적 기준)
        largest_object = max(objects, key=lambda obj: (obj['box'][2] - obj['box'][0]) * (obj['box'][3] - obj['box'][1]))
        
        # 객체 잘라내기
        cropped_image = self.image_analyzer.crop_object(original_image, largest_object['box'])
        
        # CLIP으로 객체 분석
        clip_start = time.time()
        clip_categories = config.CATEGORIES
        clip_results = self.image_analyzer.clip_analyze(cropped_image, clip_categories)
        clip_time = time.time() - clip_start
        
        # YOLO 클래스 이름과 CLIP 분석 결과 조합
        yolo_class = largest_object['class_name']
        print(f"YOLO 탐지 결과: {yolo_class}")
        print(f"CLIP 분석 결과 상위 3개: {sorted(clip_results.items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        # 기본 캡션 생성
        caption = self.image_analyzer.generate_caption(cropped_image)
        
        # 카테고리 결정 (YOLO 결과 우선, 그 다음 CLIP 결과)
        if yolo_class in config.CATEGORY_MAPPING:
            final_category = config.CATEGORY_MAPPING[yolo_class]
        else:
            # CLIP 결과에서 가장 높은 확률의 카테고리 선택
            if clip_results:
                final_category = max(clip_results.items(), key=lambda x: x[1])[0]
            else:
                final_category = ""
        
        # 캡션에서 카테고리 추출 시도
        caption_category = self.extract_category_from_caption(caption)
        if caption_category:
            final_category = caption_category
        
        # 삼성 제품 키워드 확인
        samsung_keywords = ["galaxy", "samsung", "buds", "gear", "watch", "note", "fold", "flip"]
        contains_samsung = any(keyword in caption.lower() for keyword in samsung_keywords)
        
        # 추가 질문 설정 (삼성 제품에 대한 구체적 질문)
        additional_questions = []
        if contains_samsung:
            additional_questions = [
                "Is this a Samsung Galaxy device?",
                "Is this a Samsung Galaxy Buds?",
                "Is this a Samsung Galaxy Watch?",
                "Is this a Samsung smartphone?"
            ]
        
        # 무선이어폰 관련 키워드 확인
        earbuds_keywords = ["earbuds", "earphones", "buds", "headphones"]
        contains_earbuds = any(keyword in caption.lower() for keyword in earbuds_keywords)
        
        # 무선이어폰 관련 추가 질문
        if contains_earbuds:
            additional_questions.extend([
                "Are these Samsung Galaxy Buds?",
                "What brand of wireless earbuds are these?",
                "Are these wireless earbuds or headphones?",
                "Are these Samsung brand earbuds?"
            ])
        
        # 지갑 관련 키워드 확인
        wallet_keywords = ["wallet", "purse", "billfold", "money holder"]
        contains_wallet = any(keyword in caption.lower() for keyword in wallet_keywords)
        
        # 지갑 관련 추가 질문
        if contains_wallet:
            additional_questions.extend([
                "Is this a branded wallet? If yes, what brand?",
                "Does this wallet have any logo or brand marking visible?",
                "What luxury brand is this wallet from?"
            ])
        
        # 다양한 질문으로 특징 추출 (영어로 질문)
        questions = {
            "category": config.QUESTIONS["category"].format(categories=", ".join(config.CATEGORIES)),
            "color": config.QUESTIONS["color"],
            "material": config.QUESTIONS["material"],
            "distinctive_features": config.QUESTIONS["distinctive_features"],
            "brand": "What is the brand of this item? Be very specific about the brand name. If it's Samsung, specify which Samsung product line if possible."
        }
        
        # 각 질문에 대한 답변 수집
        vqa_start = time.time()
        answers = {}
        for key, question in questions.items():
            answers[key] = self.image_analyzer.ask_question(cropped_image, question)
            
            # 색상 검증 및 수정
            if key == "color" and not self.is_valid_color(answers[key]):
                answers[key] = self.image_analyzer.ask_question(cropped_image, 
                    f"What is the main color of this item? Choose from: {', '.join(config.COLORS)}")
            
            # 재질 검증 및 수정
            if key == "material" and not self.is_valid_material(answers[key]):
                answers[key] = self.image_analyzer.ask_question(cropped_image, 
                    f"What material is this item made of? Choose from: {', '.join(config.MATERIALS)}")
        vqa_time = time.time() - vqa_start
        
        # 추가 질문에 대한 답변 수집
        additional_answers = []
        for question in additional_questions:
            answer = self.image_analyzer.ask_question(cropped_image, question)
            additional_answers.append(answer)
        
        # 무선이어폰이 iPod으로 잘못 인식되는 문제 수정
        all_answers = [answers[key] for key in answers] + additional_answers
        
        # 갤럭시 버즈 확인
        if self.is_galaxy_buds(caption, all_answers):
            final_category = "earbuds"
            final_brand = "samsung"
        else:
            # 브랜드 매칭 적용 여부 결정
            should_match_brand = self.should_apply_brand_matching(final_category, caption)
            
            # 기본값으로 브랜드를 빈 문자열로 설정
            final_brand = ""
            
            # 브랜드 매칭이 적용되어야 하는 경우에만 브랜드 추출
            if should_match_brand:
                # 브랜드 추출 강화 (캡션, 특이사항, 응답 결과, 추가 답변에서 모두 찾기)
                brand_sources = [caption, answers["distinctive_features"], answers["brand"]] + additional_answers
                
                # 삼성 제품 우선 확인
                for source in brand_sources:
                    if source:
                        source_lower = source.lower()
                        if "samsung" in source_lower or "galaxy" in source_lower:
                            # 삼성 버즈 확인
                            if "buds" in source_lower or "earbuds" in source_lower or "earbud" in source_lower:
                                final_brand = "samsung"
                                final_category = "earbuds"  # 카테고리 업데이트
                                break
                            # 삼성 워치 확인
                            elif "watch" in source_lower:
                                final_brand = "samsung"
                                final_category = "smartwatch"  # 카테고리 업데이트
                                break
                            # 삼성 폰 확인
                            elif "phone" in source_lower or "smartphone" in source_lower or "s series" in source_lower or "note" in source_lower:
                                final_brand = "samsung"
                                final_category = "phone"  # 카테고리 업데이트
                                break
                            else:
                                final_brand = "samsung"
                                break
                else:
                    # 삼성 제품이 아닌 경우 일반적인 브랜드 추출
                    final_brand = self.extract_brand(brand_sources)
                
                # 지갑인 경우 브랜드 추가 확인
                if "wallet" in final_category.lower() or contains_wallet:
                    wallet_brand_question = "What luxury brand is this wallet from? If you can see any logo or brand marking, please specify."
                    wallet_brand_answer = self.image_analyzer.ask_question(cropped_image, wallet_brand_question)
                    
                    # 지갑 브랜드 연관 매핑에서 찾기
                    for brand_key in config.BRAND_ASSOCIATION:
                        if "wallet" in brand_key and any(brand in wallet_brand_answer.lower() for brand in brand_key.lower().split()):
                            brand_name = config.BRAND_ASSOCIATION[brand_key]
                            if brand_name and final_brand == "":
                                final_brand = brand_name
                            break
        
        # 색상이 여전히 유효하지 않으면 기본값으로 설정
        if not self.is_valid_color(answers["color"]):
            potential_colors = [color for color in config.COLORS if color in caption.lower()]
            if potential_colors:
                answers["color"] = potential_colors[0]
            else:
                answers["color"] = "unknown color"
        
        # 재질이 여전히 유효하지 않으면 기본값으로 설정
        if not self.is_valid_material(answers["material"]):
            potential_materials = [material for material in config.MATERIALS if material in caption.lower()]
            if potential_materials:
                answers["material"] = potential_materials[0]
            else:
                answers["material"] = "unknown material"
        
        # 기본 설명 생성
        description = self.generate_description(caption, answers)
        
        # 제목 생성
        title = self.generate_title(answers, caption, final_category, final_brand)
        
        # 결과 한국어 번역
        translate_start = time.time()
        translated_result = self.translator.translate_results({
            "caption": caption,
            "title": title,
            "description": description,
            "category": final_category,
            "color": answers["color"],
            "material": answers["material"],
            "brand": final_brand,
            "distinctive_features": answers["distinctive_features"]
        })
        translate_time = time.time() - translate_start
        
        # 총 처리 시간
        total_time = time.time() - start_time
        
        # 타이밍 정보
        timing = {
            "total_time": total_time,
            "yolo_time": yolo_time,
            "clip_time": clip_time,
            "vqa_time": vqa_time,
            "translate_time": translate_time
        }
        
        # 결과 구조화
        result = {
            "caption": caption,
            "title": title,
            "description": description,
            "category": final_category,
            "color": answers["color"],
            "material": answers["material"],
            "brand": final_brand,
            "distinctive_features": answers["distinctive_features"],
            "raw_answers": answers,  # 디버깅 및 추가 분석용
            "additional_answers": additional_answers,  # 추가 질문에 대한 답변
            "translated": translated_result,  # 한국어 번역 결과
            "timing": timing,  # 시간 측정 정보
            "yolo_results": {
                "detected_class": yolo_class,
                "confidence": largest_object['confidence'],
                "box": largest_object['box']
            },
            "clip_results": clip_results
        }
        
        return result
    
    def extract_features(self, image: Image.Image, timing_info=None) -> Dict[str, Any]:
        """기존 BLIP 모델만 사용하여 특징 추출"""
        start_time = time.time()
        
        # 기본 캡션 생성
        caption_start = time.time()
        caption = self.image_analyzer.generate_caption(image)
        caption_time = time.time() - caption_start
        
        # 캡션에서 카테고리 추출 시도
        caption_category = self.extract_category_from_caption(caption)
        
        # 삼성 제품 키워드 확인
        samsung_keywords = ["galaxy", "samsung", "buds", "gear", "watch", "note", "fold", "flip"]
        contains_samsung = any(keyword in caption.lower() for keyword in samsung_keywords)
        
        # 추가 질문 설정 (삼성 제품에 대한 구체적 질문)
        additional_questions = []
        if contains_samsung:
            additional_questions = [
                "Is this a Samsung Galaxy device?",
                "Is this a Samsung Galaxy Buds?",
                "Is this a Samsung Galaxy Watch?",
                "Is this a Samsung smartphone?"
            ]
        
        # 무선이어폰 관련 키워드 확인
        earbuds_keywords = ["earbuds", "earphones", "buds", "headphones"]
        contains_earbuds = any(keyword in caption.lower() for keyword in earbuds_keywords)
        
        # 무선이어폰 관련 추가 질문
        if contains_earbuds:
            additional_questions.extend([
                "Are these Samsung Galaxy Buds?",
                "What brand of wireless earbuds are these?",
                "Are these wireless earbuds or headphones?",
                "Are these Samsung brand earbuds?"
            ])
        
        # 지갑 관련 키워드 확인
        wallet_keywords = ["wallet", "purse", "billfold", "money holder"]
        contains_wallet = any(keyword in caption.lower() for keyword in wallet_keywords)
        
        # 지갑 관련 추가 질문
        if contains_wallet:
            additional_questions.extend([
                "Is this a branded wallet? If yes, what brand?",
                "Does this wallet have any logo or brand marking visible?",
                "What luxury brand is this wallet from?"
            ])
        
        # 다양한 질문으로 특징 추출 (영어로 질문)
        questions = {
            "category": config.QUESTIONS["category"].format(categories=", ".join(config.CATEGORIES)),
            "color": config.QUESTIONS["color"],
            "material": config.QUESTIONS["material"],
            "distinctive_features": config.QUESTIONS["distinctive_features"],
            "brand": "What is the brand of this item? Be very specific about the brand name. If it's Samsung, specify which Samsung product line if possible."
        }
        
        # 각 질문에 대한 답변 수집
        vqa_start = time.time()
        answers = {}
        for key, question in questions.items():
            answers[key] = self.image_analyzer.ask_question(image, question)
            
            # 색상 검증 및 수정
            if key == "color" and not self.is_valid_color(answers[key]):
                answers[key] = self.image_analyzer.ask_question(image, 
                    f"What is the main color of this item? Choose from: {', '.join(config.COLORS)}")
            
            # 재질 검증 및 수정
            if key == "material" and not self.is_valid_material(answers[key]):
                answers[key] = self.image_analyzer.ask_question(image, 
                    f"What material is this item made of? Choose from: {', '.join(config.MATERIALS)}")
        vqa_time = time.time() - vqa_start
        
        # 추가 질문에 대한 답변 수집
        additional_answers = []
        for question in additional_questions:
            answer = self.image_analyzer.ask_question(image, question)
            additional_answers.append(answer)
        
        # 카테고리 우선순위 결정 (캡션 > VQA 응답)
        final_category = caption_category if caption_category else answers["category"]
        
        # 무선이어폰이 iPod으로 잘못 인식되는 문제 수정
        all_answers = [answers[key] for key in answers] + additional_answers
        
        # 갤럭시 버즈 확인
        if self.is_galaxy_buds(caption, all_answers):
            final_category = "earbuds"
            final_brand = "samsung"
        else:
            # 브랜드 매칭 적용 여부 결정
            should_match_brand = self.should_apply_brand_matching(final_category, caption)
            
            # 기본값으로 브랜드를 빈 문자열로 설정
            final_brand = ""
            
            # 브랜드 매칭이 적용되어야 하는 경우에만 브랜드 추출
            if should_match_brand:
                # 브랜드 추출 강화 (캡션, 특이사항, 응답 결과, 추가 답변에서 모두 찾기)
                brand_sources = [caption, answers["distinctive_features"], answers["brand"]] + additional_answers
                
                # 삼성 제품 우선 확인
                for source in brand_sources:
                    if source:
                        source_lower = source.lower()
                        if "samsung" in source_lower or "galaxy" in source_lower:
                            # 삼성 버즈 확인
                            if "buds" in source_lower or "earbuds" in source_lower or "earbud" in source_lower:
                                final_brand = "samsung"
                                final_category = "earbuds"  # 카테고리 업데이트
                                break
                            # 삼성 워치 확인
                            elif "watch" in source_lower:
                                final_brand = "samsung"
                                final_category = "smartwatch"  # 카테고리 업데이트
                                break
                            # 삼성 폰 확인
                            elif "phone" in source_lower or "smartphone" in source_lower or "s series" in source_lower or "note" in source_lower:
                                final_brand = "samsung"
                                final_category = "phone"  # 카테고리 업데이트
                                break
                            else:
                                final_brand = "samsung"
                                break
                else:
                    # 삼성 제품이 아닌 경우 일반적인 브랜드 추출
                    final_brand = self.extract_brand(brand_sources)
                
                # 지갑인 경우 브랜드 추가 확인
                if "wallet" in final_category.lower() or contains_wallet:
                    wallet_brand_question = "What luxury brand is this wallet from? If you can see any logo or brand marking, please specify."
                    wallet_brand_answer = self.image_analyzer.ask_question(image, wallet_brand_question)
                    
                    # 지갑 브랜드 연관 매핑에서 찾기
                    for brand_key in config.BRAND_ASSOCIATION:
                        if "wallet" in brand_key and any(brand in wallet_brand_answer.lower() for brand in brand_key.lower().split()):
                            brand_name = config.BRAND_ASSOCIATION[brand_key]
                            if brand_name and final_brand == "":
                                final_brand = brand_name
                            break
        
        # 색상이 여전히 유효하지 않으면 기본값으로 설정
        if not self.is_valid_color(answers["color"]):
            potential_colors = [color for color in config.COLORS if color in caption.lower()]
            if potential_colors:
                answers["color"] = potential_colors[0]
            else:
                answers["color"] = "unknown color"
        
        # 재질이 여전히 유효하지 않으면 기본값으로 설정
        if not self.is_valid_material(answers["material"]):
            potential_materials = [material for material in config.MATERIALS if material in caption.lower()]
            if potential_materials:
                answers["material"] = potential_materials[0]
            else:
                answers["material"] = "unknown material"
        
        # 기본 설명 생성
        description = self.generate_description(caption, answers)
        
        # 제목 생성
        title = self.generate_title(answers, caption, final_category, final_brand)
        
        # 결과 한국어 번역
        translate_start = time.time()
        translated_result = self.translator.translate_results({
            "caption": caption,
            "title": title,
            "description": description,
            "category": final_category,
            "color": answers["color"],
            "material": answers["material"],
            "brand": final_brand,
            "distinctive_features": answers["distinctive_features"]
        })
        translate_time = time.time() - translate_start
        
        # 총 처리 시간
        total_time = time.time() - start_time
        
        # 기존 YOLO 타이밍 정보가 있다면 포함
        timing = timing_info or {}
        timing.update({
            "total_time": total_time,
            "caption_time": caption_time,
            "vqa_time": vqa_time,
            "translate_time": translate_time
        })
        
        # 결과 구조화
        result = {
            "caption": caption,
            "title": title,
            "description": description,
            "category": final_category,
            "color": answers["color"],
            "material": answers["material"],
            "brand": final_brand,
            "distinctive_features": answers["distinctive_features"],
            "raw_answers": answers,  # 디버깅 및 추가 분석용
            "additional_answers": additional_answers,  # 추가 질문에 대한 답변
            "translated": translated_result,  # 한국어 번역 결과
            "timing": timing  # 시간 측정 정보
        }
        
        return result
    
    def analyze_lost_item(self, image_path: str, use_yolo_clip: bool = True) -> Dict[str, Any]:
        """분실물 이미지 분석 메인 함수"""
        try:
            if use_yolo_clip:
                # YOLO+CLIP 방식으로 분석
                print("YOLO+CLIP 방식으로 분석을 시작합니다...")
                start_time = time.time()
                features = self.extract_features_with_yolo_clip(image_path)
                total_time = time.time() - start_time
                print(f"YOLO+CLIP 분석 완료. 총 소요 시간: {total_time:.2f}초")
                
                # 타이밍 세부 정보 출력
                if "timing" in features:
                    timing = features["timing"]
                    print(f"- YOLO 객체 탐지 시간: {timing.get('yolo_time', 0):.2f}초")
                    print(f"- CLIP 분석 시간: {timing.get('clip_time', 0):.2f}초")
                    print(f"- VQA 질의응답 시간: {timing.get('vqa_time', 0):.2f}초")
                    print(f"- 번역 시간: {timing.get('translate_time', 0):.2f}초")
            else:
                # 기존 방식으로 분석 (CLIP 없이)
                print("기존 방식으로 분석을 시작합니다...")
                start_time = time.time()
                image = self.image_analyzer.preprocess_image(image_path)
                features = self.extract_features(image)
                total_time = time.time() - start_time
                print(f"기존 방식 분석 완료. 총 소요 시간: {total_time:.2f}초")
                
                # 타이밍 세부 정보 출력
                if "timing" in features:
                    timing = features["timing"]
                    print(f"- 캡션 생성 시간: {timing.get('caption_time', 0):.2f}초")
                    print(f"- VQA 질의응답 시간: {timing.get('vqa_time', 0):.2f}초")
                    print(f"- 번역 시간: {timing.get('translate_time', 0):.2f}초")
            
            return {
                "success": True,
                "data": features
            }
        except Exception as e:
            import traceback
            print(f"분석 중 오류 발생: {str(e)}")
            traceback.print_exc()
            return {
                "success": False,
                "error": str(e)
            }