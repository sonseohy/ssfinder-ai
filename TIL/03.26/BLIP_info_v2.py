import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import re
import json
from typing import Dict, List, Any, Tuple

class LostItemAnalyzer:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        BLIP 기반 분실물 분석기 초기화
        
        Args:
            device: 연산 장치 (GPU 또는 CPU)
        """
        self.device = device
        
        # 캡셔닝용 BLIP 모델 로드
        print("캡셔닝 모델 로딩 중...")
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to(self.device)
        
        # VQA용 BLIP 모델 로드
        print("VQA 모델 로딩 중...")
        self.vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        self.vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large").to(self.device)
        
        # 카테고리 정의 (영어)
        self.categories = [
            "electronics", "clothing", "bag", "wallet", "jewelry", "card", "id", "computer", "cash", "phone",
            "umbrella", "cosmetics", "sports equipment", "books", "others", "documents", "industrial goods", 
            "shopping bag", "musical instrument", "car", "miscellaneous"
        ]
        
        # 카테고리 매핑 (캡션의 일반적인 단어를 카테고리로 매핑)
        self.category_mapping = {
            "umbrella": "umbrella",
            "phone": "phone", 
            "smartphone": "phone",
            "cellphone": "phone",
            "iphone": "phone",
            "wallet": "wallet",
            "bag": "bag",
            "handbag": "bag",
            "backpack": "bag",
            "computer": "computer",
            "laptop": "computer",
            "tablet": "electronics",
            "ipad": "electronics",
            "watch": "jewelry",
            "book": "books",
            "notebook": "books",
            "headphones": "electronics",
            "earphones": "electronics",
            "airpods": "electronics",
            "card": "card",
            "id": "id",
            "key": "others",
            "keys": "others",
            "glasses": "others",
            "sunglasses": "others",
            "camera": "electronics",
            "jewelry": "jewelry"
        }
        
        # 색상 목록 정의
        self.colors = [
            "red", "blue", "green", "yellow", "black", "white", "gray", "grey", "brown", "purple", 
            "pink", "orange", "silver", "gold", "navy", "beige", "transparent", "multicolor", "teal",
            "turquoise", "maroon", "olive", "cyan", "magenta", "lavender", "indigo", "violet", "tan"
        ]
        
        # 재질 목록 정의
        self.materials = [
            "plastic", "metal", "leather", "fabric", "paper", "wood", "glass", "ceramic", "rubber",
            "cotton", "polyester", "nylon", "carbon fiber", "stone", "silicone", "aluminium", "steel",
            "cloth", "textile", "canvas", "denim", "wool", "synthetic", "composite", "unknown"
        ]
        
        # 유명 브랜드 목록 (분실물로 흔한 브랜드)
        self.common_brands = [
            "apple", "samsung", "lg", "sony", "nike", "adidas", "puma", "reebok", "louis vuitton", 
            "gucci", "chanel", "prada", "hermes", "coach", "michael kors", "dell", "hp", "lenovo", 
            "asus", "acer", "timex", "casio", "seiko", "citizen", "logitech", "microsoft", "canon", 
            "nikon", "jbl", "bose", "sennheiser", "sony", "samsonite", "tumi", "kindle", "google"
        ]
        
        print("모델 로딩 완료!")

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
        max_size = 1000
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            
        return image

    def generate_caption(self, image: Image.Image) -> str:
        """
        이미지에서 캡션 생성
        
        Args:
            image: PIL 이미지
            
        Returns:
            생성된 캡션 문자열
        """
        # 이미지를 모델 입력으로 처리
        inputs = self.caption_processor(image, return_tensors="pt").to(self.device)
        
        # 캡션 생성
        with torch.no_grad():
            out = self.caption_model.generate(
                **inputs,
                max_length=75,
                num_beams=5,
                do_sample=True,  # 경고 제거를 위해 샘플링 활성화
                top_p=0.9,
                repetition_penalty=1.5
            )
        
        # 토큰을 텍스트로 디코딩
        caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
        return caption

    def ask_question(self, image: Image.Image, question: str) -> str:
        """
        이미지에 대해 질문하고 답변 받기
        
        Args:
            image: PIL 이미지
            question: 질문 문자열
            
        Returns:
            질문에 대한 답변
        """
        # 이미지와 질문을 모델 입력으로 처리
        inputs = self.vqa_processor(image, question, return_tensors="pt").to(self.device)
        
        # 질문에 대한 답변 생성
        with torch.no_grad():
            out = self.vqa_model.generate(
                **inputs,
                max_length=20,
                num_beams=5,
                do_sample=True,  # 경고 제거를 위해 샘플링 활성화
                top_p=0.9
            )
        
        # 토큰을 텍스트로 디코딩
        answer = self.vqa_processor.decode(out[0], skip_special_tokens=True)
        
        # 로그에 질문과 답변 출력
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        print("-" * 50)
        
        return answer

    def extract_category_from_caption(self, caption: str) -> str:
        """
        캡션에서 카테고리 추출
        
        Args:
            caption: 이미지 캡션
            
        Returns:
            추출된 카테고리
        """
        caption_lower = caption.lower()
        
        # 각 매핑된 키워드를 확인
        for keyword, category in self.category_mapping.items():
            if keyword in caption_lower:
                return category
                
        return ""

    def is_valid_color(self, text: str) -> bool:
        """
        유효한 색상인지 확인
        
        Args:
            text: 확인할 텍스트
            
        Returns:
            유효한 색상이면 True, 아니면 False
        """
        text = text.lower()
        return any(color in text for color in self.colors)

    def is_valid_material(self, text: str) -> bool:
        """
        유효한 재질인지 확인
        
        Args:
            text: 확인할 텍스트
            
        Returns:
            유효한 재질이면 True, 아니면 False
        """
        text = text.lower()
        return any(material in text for material in self.materials)

    def extract_brand_from_caption(self, caption: str) -> str:
        """
        캡션에서 브랜드 추출 (캡션에 있는 경우에만)
        
        Args:
            caption: 이미지 캡션
            
        Returns:
            추출된 브랜드 또는 빈 문자열
        """
        caption_lower = caption.lower()
        
        # 각 브랜드 확인
        for brand in self.common_brands:
            if brand in caption_lower:
                return brand
                
        return ""

    def extract_features(self, image: Image.Image) -> Dict[str, Any]:
        """
        이미지에서 분실물의 다양한 특징 추출
        
        Args:
            image: PIL 이미지
            
        Returns:
            추출된 특징들을 담은 딕셔너리
        """
        # 기본 캡션 생성
        caption = self.generate_caption(image)
        print(f"Generated caption: {caption}")
        print("-" * 50)
        
        # 캡션에서 카테고리 추출 시도
        caption_category = self.extract_category_from_caption(caption)
        
        # 캡션에서 브랜드 추출 시도
        caption_brand = self.extract_brand_from_caption(caption)
        
        # 다양한 질문으로 특징 추출 (영어로 변경)
        questions = {
            "category": f"What type of item is this? Choose from the following categories: {', '.join(self.categories)}",
            "color": "What is the main color of this item? Be specific and mention only the color.",
            "material": "What material is this item made of? If unknown, say 'unknown'.",
            "distinctive_features": "What are the distinctive features or unique aspects of this item?",
        }
        
        # 브랜드 질문은 캡션에서 브랜드가 발견된 경우에만 추가
        if caption_brand:
            questions["brand"] = f"Is this a {caption_brand} product? Answer yes or no."
        else:
            # 캡션에 브랜드가 없으면 브랜드를 묻지 않음
            questions["brand"] = "Is there any visible brand name on this item? If not, just say 'no'."
        
        # 각 질문에 대한 답변 수집
        answers = {}
        for key, question in questions.items():
            print(f"{key} 분석 중...")
            answers[key] = self.ask_question(image, question)
            
            # 색상 검증 및 수정
            if key == "color" and not self.is_valid_color(answers[key]):
                print("Invalid color detected, trying with more specific question...")
                answers[key] = self.ask_question(image, 
                    f"What is the main color of this item? Choose from: {', '.join(self.colors)}")
            
            # 재질 검증 및 수정
            if key == "material" and not self.is_valid_material(answers[key]):
                print("Invalid material detected, trying with more specific question...")
                answers[key] = self.ask_question(image, 
                    f"What material is this item made of? Choose from: {', '.join(self.materials)}")
        
        # 카테고리 우선순위 결정 (캡션 > VQA 응답)
        final_category = ""
        if caption_category:
            # 캡션에서 추출한 카테고리가 있으면 우선 사용
            final_category = caption_category
            print(f"Using category from caption: {final_category}")
        else:
            # 없으면, VQA 응답 사용
            final_category = answers["category"]
        
        # 브랜드 결정 - 캡션에서 추출된 경우에만 사용
        final_brand = ""
        if caption_brand:
            # 캡션에서 브랜드가 발견되었고, VQA 응답이 긍정적이면 사용
            if "yes" in answers["brand"].lower():
                final_brand = caption_brand
                print(f"Confirmed brand from caption: {final_brand}")
        
        # 색상이 여전히 유효하지 않으면 기본값으로 설정
        if not self.is_valid_color(answers["color"]):
            potential_colors = [color for color in self.colors if color in caption.lower()]
            if potential_colors:
                answers["color"] = potential_colors[0]
            else:
                answers["color"] = "unknown color"
        
        # 재질이 여전히 유효하지 않으면 기본값으로 설정
        if not self.is_valid_material(answers["material"]):
            potential_materials = [material for material in self.materials if material in caption.lower()]
            if potential_materials:
                answers["material"] = potential_materials[0]
            else:
                answers["material"] = "unknown material"
        
        # 기본 설명 생성
        description = self._generate_description(caption, answers)
        
        # 제목 생성
        title = self._generate_title(answers, caption, final_category, final_brand)
        
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
            "raw_answers": answers  # 디버깅 및 추가 분석용
        }
        
        return result
    
    def _generate_title(self, answers: Dict[str, str], caption: str, category: str, brand: str) -> str:
        """
        답변을 기반으로 게시글 제목 생성
        
        Args:
            answers: 질문별 답변 딕셔너리
            caption: 이미지 캡션
            category: 결정된 카테고리
            brand: 결정된 브랜드
            
        Returns:
            생성된 제목
        """
        # 색상 추출
        color = answers["color"].lower()
        
        # 상품 이름 추출 시도 (캡션에서 핵심 단어 추출)
        product_name = ""
        common_items = ["ipad", "iphone", "macbook", "laptop", "phone", "tablet", "watch", "airpods", 
                       "wallet", "bag", "umbrella", "headphones", "camera", "book", "glasses"]
        
        for item in common_items:
            if item in caption.lower():
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
    
    def _generate_description(self, caption: str, answers: Dict[str, str]) -> str:
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
    
    def analyze_lost_item(self, image_path: str) -> Dict[str, Any]:
        """
        분실물 이미지 분석 메인 함수
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            분석 결과
        """
        try:
            # 이미지 전처리
            image = self.preprocess_image(image_path)
            
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


# 사용 예시
if __name__ == "__main__":
    # 분석기 초기화
    analyzer = LostItemAnalyzer()
    
    # 테스트 이미지 분석
    image_path = "iphone.jpg"
    result = analyzer.analyze_lost_item(image_path)
    
    if result["success"]:
        # 결과 출력
        print("\n===== 분실물 분석 결과 =====")
        print(f"제목: {result['data']['title']}")
        print(f"카테고리: {result['data']['category']}")
        print(f"색상: {result['data']['color']}")
        print(f"재질: {result['data']['material']}")
        print(f"브랜드: {result['data']['brand']}")
        print(f"설명:\n{result['data']['description']}")
        
        # JSON으로 저장 (API 응답용)
        with open("analysis_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        print(f"오류 발생: {result['error']}")