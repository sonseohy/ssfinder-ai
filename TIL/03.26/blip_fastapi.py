import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
import re
import json
from typing import Dict, List, Any, Tuple
import os
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import io
import uvicorn
from pydantic import BaseModel

class LostItemAnalyzer:
    def __init__(self):
        # 캡셔닝용 BLIP 모델 로드
        print("캡셔닝 모델 로딩 중...")
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        
        # VQA용 BLIP 모델 로드
        print("VQA 모델 로딩 중...")
        self.vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        self.vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-capfilt-large")
        
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
        
        # 브랜드 연관 매핑 (제품 -> 브랜드 연결)
        self.brand_association = {
            # Apple 제품
            "ipad": "apple",
            "iphone": "apple",
            "macbook": "apple",
            "mac": "apple",
            "airpods": "apple",
            "ipod": "apple",
            
            # Samsung 제품
            "galaxy": "samsung",
            
            # 기타 제품들
            "gram": "lg",
            "airmax": "nike",
        }
        
        print("모델 로딩 완료!")

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        이미지 전처리
        
        Args:
            image: PIL 이미지
            
        Returns:
            전처리된 PIL 이미지
        """
        # 이미지 RGB로 변환
        image = image.convert('RGB')
        
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
        이미지에 대해 질문하고 답변 받기
        
        Args:
            image: PIL 이미지
            question: 질문 문자열
            
        Returns:
            질문에 대한 답변
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
        text = text.lower()
        return any(material in text for material in self.materials)

    def extract_brand(self, text_list: List[str]) -> str:

        # 직접적인 브랜드 언급 확인 (대소문자 구분 없이)
        common_brands = [
            "apple", "samsung", "lg", "sony", "nike", "adidas", "puma", "reebok", "louis vuitton", 
            "gucci", "chanel", "prada", "hermes", "coach", "michael kors", "dell", "hp", "lenovo", 
            "asus", "acer", "timex", "casio", "seiko", "citizen", "logitech", "microsoft", "canon", 
            "nikon", "jbl", "bose", "sennheiser", "samsonite", "tumi", "kindle", "google"
        ]
        
        for text in text_list:
            text_lower = text.lower()
            
            # 직접적인 브랜드 언급 확인
            for brand in common_brands:
                if brand in text_lower:
                    return brand
                    
            # 브랜드 연관 매핑 확인
            for product, brand in self.brand_association.items():
                if product in text_lower:
                    return brand
                    
        return ""

    def extract_features(self, image: Image.Image) -> Dict[str, Any]:

        # 기본 캡션 생성
        caption = self.generate_caption(image)
        print(f"Generated caption: {caption}")
        print("-" * 50)
        
        # 캡션에서 카테고리 추출 시도
        caption_category = self.extract_category_from_caption(caption)
        
        # 다양한 질문으로 특징 추출 (영어로 변경)
        questions = {
            "category": f"What type of item is this? Choose from the following categories: {', '.join(self.categories)}",
            "color": "What is the main color of this item? Be specific and mention only the color.",
            "material": "What material is this item made of? If unknown, say 'unknown'.",
            "distinctive_features": "What are the distinctive features or unique aspects of this item?",
        }
        
        # 브랜드 질문은 별도로 수행
        questions["brand"] = "What is the brand of this item? If unknown, just say 'unknown'."
        
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
        
        # 브랜드 추출 (캡션, 특이사항, 응답 결과에서 모두 찾기)
        brand_sources = [caption, answers["distinctive_features"], answers["brand"]]
        final_brand = self.extract_brand(brand_sources)
        
        if final_brand:
            print(f"Detected brand: {final_brand}")
        
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

        description = caption + "\n\n"
        
        # 추가 정보 포함
        if answers["material"] and "unknown" not in answers["material"].lower():
            description += f"Material: {answers['material']}\n"
            
        if answers["distinctive_features"] and "none" not in answers["distinctive_features"].lower():
            description += f"Distinctive features: {answers['distinctive_features']}\n"
            
        return description.strip()
    
    def analyze_image(self, image: Image.Image) -> Dict[str, Any]:

        try:
            # 이미지 전처리
            processed_image = self.preprocess_image(image)
            
            # 특징 추출
            features = self.extract_features(processed_image)
            
            return {
                "success": True,
                "data": features
            }
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }

# FastAPI 애플리케이션 설정
app = FastAPI(title="Lost Item Analyzer API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 환경에서는 특정 도메인으로 제한해야 함
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 분석기 초기화 (서버 시작 시 한 번만 실행)
analyzer = LostItemAnalyzer()

# 상태 확인 엔드포인트
@app.get("/")
async def root():
    return {"status": "OK", "message": "Lost Item Analyzer API is running"}

# 분실물 이미지 분석 엔드포인트
@app.post("/analyze")
async def analyze_lost_item(file: UploadFile = File(...)):
    try:
        # 이미지 파일 읽기
        content = await file.read()
        image = Image.open(io.BytesIO(content))
        
        # 이미지 분석
        result = analyzer.analyze_image(image)
        
        return result
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"success": False, "error": str(e)}

# 서버 실행 (직접 실행 시)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)