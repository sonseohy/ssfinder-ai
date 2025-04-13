import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BlipProcessor, BlipForQuestionAnswering
from typing import Dict, List, Any
from models.translator import Translator
from config import config

class ImageAnalyzer:
    def __init__(self):
        # GPU 설정
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 캡셔닝용 BLIP-2 모델 로드 (GPU로)
        CAPTION_MODEL = "Salesforce/blip2-opt-2.7b"
        self.caption_processor = Blip2Processor.from_pretrained(CAPTION_MODEL)
        self.caption_model = Blip2ForConditionalGeneration.from_pretrained(CAPTION_MODEL).to(self.device)
        
        # VQA용 BLIP 모델 로드 (GPU로)
        VQA_MODEL = "Salesforce/blip-vqa-capfilt-large"
        self.vqa_processor = BlipProcessor.from_pretrained(VQA_MODEL)
        self.vqa_model = BlipForQuestionAnswering.from_pretrained(VQA_MODEL).to(self.device)

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
        inputs = self.caption_processor(image, text="", return_tensors="pt").to(self.device)
        
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
        # 질문 유형 식별
        def identify_question_type(question):
            question_types = {
                "category": ["category", "type", "item"],
                "color": ["color", "main color"],
                "material": ["material", "made of"],
                "brand": ["brand", "manufacturer", "logo"],
                "distinctive_features": ["features", "unique", "distinctive"],
            }
            
            for key, keywords in question_types.items():
                if any(keyword in question.lower() for keyword in keywords):
                    return key
            return "generic"

        question_type = identify_question_type(question)

        # 이미지와 질문을 모델 입력으로 처리 (GPU로)
        inputs = self.vqa_processor(image, question, return_tensors="pt").to(self.device)
        
        # 질문에 대한 답변 생성 (GPU 최적화)
        with torch.no_grad(), torch.amp.autocast('cuda'):
            out = self.vqa_model.generate(
                **inputs,
                max_new_tokens=20,  # 토큰 수 좀 더 제한
                num_beams=3,
                do_sample=True,
                top_p=0.8,
                temperature=0.7,
                repetition_penalty=1.2
            )
        
        # 토큰을 텍스트로 디코딩
        answer = self.vqa_processor.decode(out[0], skip_special_tokens=True)
        
        # 후처리 로직
        def postprocess_answer(answer, question_type):
            # 공통 불용어 및 불필요한 문구 제거
            cleanup_phrases = [
                "the main", "as far as i can see", "based on the image", 
                "from the image", "according to the image", "it appears to be"
            ]
            for phrase in cleanup_phrases:
                answer = answer.replace(phrase, "").strip()
            
            # 질문 유형별 특화된 후처리
            if question_type == "category":
                categories = config.CATEGORIES
                filtered_answer = ' '.join(word for word in answer.split() if word.lower() in map(str.lower, categories))
                return filtered_answer or "others"
            
            elif question_type == "color":
                colors = config.COLORS
                matched_colors = [color for color in colors if color in answer.lower()]
                return matched_colors[0] if matched_colors else "unknown color"
            
            elif question_type == "material":
                materials = config.MATERIALS
                matched_materials = [material for material in materials if material in answer.lower()]
                return matched_materials[0] if matched_materials else "unknown material"
            
            elif question_type == "brand":
                # 브랜드 목록에서 매칭
                brands = list(config.BRAND_TRANSLATION.keys())
                matched_brands = [brand for brand in brands if brand in answer.lower()]
                return matched_brands[0] if matched_brands else "unknown"
            
            return answer.split('.')[0].strip()  # 첫 문장만 사용
        
        # 후처리된 답변 반환
        processed_answer = postprocess_answer(answer, question_type)
        
        # 디버깅용 로그 (필요시 주석 해제)
        print(f"질문 유형: {question_type}")
        print(f"원본 답변: {answer}")
        print(f"처리된 답변: {processed_answer}")
        
        return processed_answer

class LostItemAnalyzer:
    def __init__(self):
        # 이미지 분석기 초기화
        self.image_analyzer = ImageAnalyzer()
        
        # 번역기 초기화
        self.translator = Translator()
    
    def extract_category_from_caption(self, caption: str) -> str:
        """캡션에서 카테고리 추출 (개선된 로직)"""
        caption_lower = caption.lower()
        words = caption_lower.split()
        
        # 키워드를 길이 순으로 정렬 (긴 것이 먼저)
        sorted_keywords = sorted(config.CATEGORY_MAPPING.keys(), key=len, reverse=True)
        
        # 각 매핑된 키워드를 확인
        for keyword in sorted_keywords:
            # 전체 단어 매칭 또는 복합어 검사
            if keyword in words or (len(keyword.split()) > 1 and keyword in caption_lower):
                return config.CATEGORY_MAPPING[keyword]
        
        # 추가 휴리스틱 로직
        additional_mappings = {
            "mobile": "electronics",
            "device": "electronics",
            "garment": "clothing",
            "accessory": "jewelry",
            "document": "documents"
        }
        
        for word, category in additional_mappings.items():
            if word in caption_lower:
                return category
        
        return ""
    
    def is_valid_color(self, text: str) -> bool:
        """색상의 유효성 검사"""
        text = text.lower()
        return any(color in text for color in config.COLORS)
    
    def is_valid_material(self, text: str) -> bool:
        """재질의 유효성 검사"""
        text = text.lower()
        return any(material in text for material in config.MATERIALS)
    
    def extract_brand(self, text_list: List[str]) -> str:
        # 대소문자 구분 없는 브랜드 검색
        common_brands = [k.lower() for k in config.BRAND_TRANSLATION.keys() if k != "unknown"]
        
        for text in text_list:
            text_lower = text.lower()
            
            # 1. 로고 및 브랜드 명시적 추출을 위한 추가 VQA 질문
            logo_question = "Is there a logo or brand name visible in the image?"
            brand_location_question = "Where is the brand logo or name located in the image?"
            
            # 2. 정확한 브랜드 매칭 (완전한 단어)
            for brand in common_brands:
                if f" {brand} " in f" {text_lower} ":  # 단어 경계 체크
                    return brand
            
            # 3. 제품-브랜드 연관 매핑 확인
            for product, brand in config.BRAND_ASSOCIATION.items():
                if product in text_lower:
                    return brand
            
            # 4. 더 유연한 부분 일치 검사
            for brand in common_brands:
                if brand in text_lower.split():
                    return brand
        
        return ""
    
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
    
    def extract_features(self, image: Image.Image) -> Dict[str, Any]:
        """분실물 특징 추출"""
        # 기본 캡션 생성
        caption = self.image_analyzer.generate_caption(image)

        # 로그 추가: 캡션 출력
        print("=== BLIP 이미지 캡셔닝 결과 ===")
        print(f"캡션: {caption}")
        print("===============================")
        
        # 캡션에서 카테고리 추출 시도
        caption_category = self.extract_category_from_caption(caption)
        
        # 다양한 질문으로 특징 추출 (영어로 질문)
        questions = {
            "category": config.QUESTIONS["category"].format(categories=", ".join(config.CATEGORIES)),
            "color": config.QUESTIONS["color"],
            "material": config.QUESTIONS["material"],
            "distinctive_features": config.QUESTIONS["distinctive_features"],
            "brand": config.QUESTIONS["brand"]
        }
        
        # 각 질문에 대한 답변 수집
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

                # 로그 추가: 각 질문별 VQA 응답 출력
            print(f"질문 ({key}): {question}")
            print(f"응답: {answers[key]}\n")
        print("============================")


        # 카테고리 우선순위 결정 (캡션 > VQA 응답)
        final_category = caption_category if caption_category else answers["category"]
        
        # 브랜드 추출 (캡션, 특이사항, 응답 결과에서 모두 찾기)
        brand_sources = [caption, answers["distinctive_features"], answers["brand"]]
        final_brand = self.extract_brand(brand_sources)
        
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
            "translated": translated_result  # 한국어 번역 결과
        }
        
        # 더 상세한 로깅
        print("=== 이미지 분석 상세 로그 ===")
        print(f"캡션: {caption}")
        print(f"추출된 카테고리: {final_category}")
        print(f"브랜드 후보들: {brand_sources}")
        print(f"최종 브랜드: {final_brand}")
        print("원시 답변:")
        for key, value in answers.items():
            print(f"{key}: {value}")
        print("============================")

        return result
    
    def analyze_lost_item(self, image_path: str) -> Dict[str, Any]:
        """분실물 이미지 분석 메인 함수"""
        try:
            # 이미지 전처리
            image = self.image_analyzer.preprocess_image(image_path)
            
            # 특징 추출
            features = self.extract_features(image)

            # 디버깅을 위한 로깅 추가
            print(f"브랜드 분석 결과: {features['brand']}")
            print(f"추출된 캡션: {features['caption']}")
            print(f"원시 답변: {features['raw_answers']}")
            
            return {
                "success": True,
                "data": features
            }
        except Exception as e:
            # 오류 상세 로깅
            print(f"분석 중 오류 발생: {e}")

            return {
                "success": False,
                "error": str(e)
            }