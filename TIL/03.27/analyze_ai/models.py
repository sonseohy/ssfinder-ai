"""
이미지 분석 모델 및 핵심 기능 모듈
"""
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from typing import Dict, List, Any
from translate import Translator
import config

class ImageAnalyzer:
    """
    이미지 분석 및 캡셔닝을 위한 클래스
    """
    def __init__(self):
        # 캡셔닝용 BLIP 모델 로드
        self.caption_processor = BlipProcessor.from_pretrained(config.CAPTION_MODEL)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(config.CAPTION_MODEL)
        
        # VQA용 BLIP 모델 로드
        self.vqa_processor = BlipProcessor.from_pretrained(config.VQA_MODEL)
        self.vqa_model = BlipForQuestionAnswering.from_pretrained(config.VQA_MODEL)
    
    def preprocess_image(self, image_path: str) -> Image.Image:
        """
        이미지 전처리
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            전처리된 PIL 이미지
        """
        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        
        # 이미지 크기 최적화 (너무 큰 경우 리사이즈)
        max_size = config.MAX_IMAGE_SIZE
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            
        return image
    
    def generate_caption(self, image: Image.Image) -> str:
        """
        이미지 캡션 생성
        
        Args:
            image: PIL 이미지 객체
            
        Returns:
            생성된 캡션 텍스트
        """
        # 이미지를 모델 입력으로 처리
        inputs = self.caption_processor(image, return_tensors="pt")
        
        # 캡션 생성
        with torch.no_grad():
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
        """
        이미지에 대한 질문 응답
        
        Args:
            image: PIL 이미지 객체
            question: 질문 텍스트
            
        Returns:
            응답 텍스트
        """
        # 이미지와 질문을 모델 입력으로 처리
        inputs = self.vqa_processor(image, question, return_tensors="pt")
        
        # 질문에 대한 답변 생성
        with torch.no_grad():
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


class LostItemAnalyzer:
    """
    분실물 이미지 분석을 위한 클래스
    """
    def __init__(self):
        # 이미지 분석기 초기화
        self.image_analyzer = ImageAnalyzer()
        
        # 번역기 초기화
        self.translator = Translator()
    
    def extract_category_from_caption(self, caption: str) -> str:
        """
        캡션에서 카테고리 추출
        
        Args:
            caption: 이미지 캡션
            
        Returns:
            추출된 카테고리
        """
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
        """
        유효한 색상인지 확인
        
        Args:
            text: 확인할 텍스트
            
        Returns:
            유효한 색상이면 True
        """
        text = text.lower()
        return any(color in text for color in config.COLORS)
    
    def is_valid_material(self, text: str) -> bool:
        """
        유효한 재질인지 확인
        
        Args:
            text: 확인할 텍스트
            
        Returns:
            유효한 재질이면 True
        """
        text = text.lower()
        return any(material in text for material in config.MATERIALS)
    
    def extract_brand(self, text_list: List[str]) -> str:
        """
        텍스트 목록에서 브랜드 추출
        
        Args:
            text_list: 텍스트 목록
            
        Returns:
            추출된 브랜드명
        """
        # 직접적인 브랜드 언급 확인 (대소문자 구분 없이)
        common_brands = [k for k in config.BRAND_TRANSLATION.keys() if k != "unknown"]
        
        for text in text_list:
            text_lower = text.lower()
            
            # 직접적인 브랜드 언급 확인
            for brand in common_brands:
                if brand in text_lower:
                    return brand
                    
            # 브랜드 연관 매핑 확인
            for product, brand in config.BRAND_ASSOCIATION.items():
                if product in text_lower:
                    return brand
                    
        return ""
    
    def generate_title(self, answers: Dict[str, str], caption: str, category: str, brand: str) -> str:
        """
        답변 기반 게시글 제목 생성
        
        Args:
            answers: 질문 응답 딕셔너리
            caption: 이미지 캡션
            category: 분류된 카테고리
            brand: 감지된 브랜드
            
        Returns:
            생성된 제목
        """
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
        """
        캡션과 답변을 기반으로 상세 설명 생성
        
        Args:
            caption: 기본 캡션
            answers: 질문별 답변 딕셔너리
            
        Returns:
            생성된 설명
        """
        description = caption + "\n\n"
        
        # 추가 정보 포함
        if answers["material"] and "unknown" not in answers["material"].lower():
            description += f"Material: {answers['material']}\n"
            
        if answers["distinctive_features"] and "none" not in answers["distinctive_features"].lower():
            description += f"Distinctive features: {answers['distinctive_features']}\n"
            
        return description.strip()
    
    def extract_features(self, image: Image.Image) -> Dict[str, Any]:
        """
        이미지에서 분실물의 다양한 특징 추출
        
        Args:
            image: PIL 이미지
            
        Returns:
            추출된 특징들을 담은 딕셔너리
        """
        # 기본 캡션 생성
        caption = self.image_analyzer.generate_caption(image)
        
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
        
        return result
    
    def analyze_lost_item(self, image_path: str) -> Dict[str, Any]:
        """
        분실물 이미지 분석 메인 함수
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            분석 결과 딕셔너리
        """
        try:
            # 이미지 전처리
            image = self.image_analyzer.preprocess_image(image_path)
            
            # 특징 추출
            features = self.extract_features(image)
            
            return {
                "success": True,
                "data": features
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }