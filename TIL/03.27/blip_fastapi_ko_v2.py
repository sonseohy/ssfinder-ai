# 실행
# uvicorn 파일이름:app --host 0.0.0.0 --port 8000 --reload
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering
from typing import Dict, List, Any
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import uvicorn

class LostItemAnalyzer:
    def __init__(self):
        """
        BLIP 기반 분실물 분석기 초기화
        """
        
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
        
        # 카테고리 한영 매핑
        self.category_translation = {
            "electronics": "전자기기",
            "clothing": "의류",
            "bag": "가방",
            "wallet": "지갑",
            "jewelry": "보석/액세서리",
            "card": "카드",
            "id": "신분증",
            "computer": "컴퓨터",
            "cash": "현금",
            "phone": "휴대폰",
            "umbrella": "우산",
            "cosmetics": "화장품",
            "sports equipment": "스포츠용품",
            "books": "도서",
            "others": "기타",
            "documents": "서류",
            "industrial goods": "산업용품",
            "shopping bag": "쇼핑백",
            "musical instrument": "악기",
            "car": "자동차",
            "miscellaneous": "기타"
        }
        
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
        
        # 색상 목록 정의 및 번역
        self.colors = [
            "red", "blue", "green", "yellow", "black", "white", "gray", "grey", "brown", "purple", 
            "pink", "orange", "silver", "gold", "navy", "beige", "transparent", "multicolor", "teal",
            "turquoise", "maroon", "olive", "cyan", "magenta", "lavender", "indigo", "violet", "tan"
        ]
        
        self.color_translation = {
            "red": "빨간색",
            "blue": "파란색",
            "green": "초록색",
            "yellow": "노란색",
            "black": "검은색",
            "white": "흰색",
            "gray": "회색",
            "grey": "회색",
            "brown": "갈색",
            "purple": "보라색",
            "pink": "분홍색",
            "orange": "주황색",
            "silver": "은색",
            "gold": "금색",
            "navy": "네이비색",
            "beige": "베이지색",
            "transparent": "투명한",
            "multicolor": "다색",
            "teal": "청록색",
            "turquoise": "터콰이즈색",
            "maroon": "적갈색",
            "olive": "올리브색",
            "cyan": "시안색",
            "magenta": "자주색",
            "lavender": "라벤더색",
            "indigo": "인디고색",
            "violet": "보라색",
            "tan": "황갈색",
            "unknown color": "알 수 없는 색상"
        }
        
        # 재질 목록 정의 및 번역
        self.materials = [
            "plastic", "metal", "leather", "fabric", "paper", "wood", "glass", "ceramic", "rubber",
            "cotton", "polyester", "nylon", "carbon fiber", "stone", "silicone", "aluminium", "steel",
            "cloth", "textile", "canvas", "denim", "wool", "synthetic", "composite", "unknown"
        ]
        
        self.material_translation = {
            "plastic": "플라스틱",
            "metal": "금속",
            "leather": "가죽",
            "fabric": "천",
            "paper": "종이",
            "wood": "나무",
            "glass": "유리",
            "ceramic": "세라믹",
            "rubber": "고무",
            "cotton": "면",
            "polyester": "폴리에스터",
            "nylon": "나일론",
            "carbon fiber": "탄소섬유",
            "stone": "돌",
            "silicone": "실리콘",
            "aluminium": "알루미늄",
            "steel": "강철",
            "cloth": "천",
            "textile": "직물",
            "canvas": "캔버스",
            "denim": "데님",
            "wool": "울",
            "synthetic": "합성 소재",
            "composite": "복합 소재",
            "unknown": "알 수 없음",
            "unknown material": "알 수 없는 재질"
        }
        
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
        
        # 브랜드 번역 매핑
        self.brand_translation = {
            "apple": "애플",
            "samsung": "삼성",
            "lg": "엘지",
            "sony": "소니",
            "nike": "나이키",
            "adidas": "아디다스",
            "puma": "푸마",
            "reebok": "리복",
            "louis vuitton": "루이비통",
            "gucci": "구찌",
            "chanel": "샤넬",
            "prada": "프라다",
            "hermes": "에르메스",
            "coach": "코치",
            "dell": "델",
            "hp": "에이치피",
            "lenovo": "레노버",
            "asus": "아수스",
            "acer": "에이서",
            "timex": "타이맥스",
            "casio": "카시오",
            "seiko": "세이코",
            "citizen": "시티즌",
            "logitech": "로지텍",
            "microsoft": "마이크로소프트",
            "canon": "캐논",
            "nikon": "니콘",
            "jbl": "제이비엘",
            "bose": "보스",
            "sennheiser": "젠하이저",
            "samsonite": "쌤소나이트",
            "tumi": "투미",
            "kindle": "킨들",
            "google": "구글",
            "unknown": "알 수 없음"
        }
        
        print("모델 로딩 완료!")

    # 이미지 전처리
    def preprocess_image(self, image_path: str) -> Image.Image:

        # 이미지 로드
        image = Image.open(image_path).convert('RGB')
        
        # 이미지 크기 최적화 (너무 큰 경우 리사이즈)
        max_size = 1000
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.LANCZOS)
            
        return image

    # 이미지 캡셔닝
    def generate_caption(self, image: Image.Image) -> str:
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

    # 이미지 질문 답변변
    def ask_question(self, image: Image.Image, question: str) -> str:

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

    # 캡션에서 카테고리 추출출
    def extract_category_from_caption(self, caption):
        caption_lower = caption.lower()
        words = caption_lower.split()
        
        print(f"### DEBUG: Words in caption: {words}")
        
        # 키워드를 길이 순으로 정렬 (긴 것이 먼저)
        sorted_keywords = sorted(self.category_mapping.keys(), key=len, reverse=True)
        
        # 각 매핑된 키워드를 확인
        for keyword in sorted_keywords:
            # 전체 단어 매칭 또는 복합어 검사
            if keyword in words or (len(keyword.split()) > 1 and keyword in caption_lower):
                print(f"### DEBUG: Found keyword '{keyword}' -> category '{self.category_mapping[keyword]}'")
                return self.category_mapping[keyword]
        
        print("### DEBUG: No category found in caption")
        return ""

    # 유효 색상 확인
    def is_valid_color(self, text: str) -> bool:

        text = text.lower()
        return any(color in text for color in self.colors)

    # 유효 재질 확인
    def is_valid_material(self, text: str) -> bool:
        text = text.lower()
        return any(material in text for material in self.materials)

    # 텍스트 목록에서 브랜드 추출출
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
        
        # 다양한 질문으로 특징 추출 (영어로 질문)
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
        
        # 결과 한국어 번역
        translated_result = self._translate_results(
            caption, title, description, final_category, 
            answers["color"], answers["material"], final_brand,
            answers["distinctive_features"]
        )
        
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
    
    # 답변 기반 게시글 제목 생성성
    def _generate_title(self, answers: Dict[str, str], caption: str, category: str, brand: str) -> str:
        # 색상 추출
        color = answers["color"].lower()
        
        # 상품 이름 추출 시도 (캡션에서 핵심 단어 추출)
        product_name = ""
        
        # 디버깅: 제품명 추출 과정 로깅
        print(f"### DEBUG: Extracting product name from caption: {caption}")
        
        # 정확한 단어 매칭을 위해 캡션을 단어로 분리
        caption_words = caption.lower().split()
        
        common_items = ["headphones", "earphones", "ipad", "iphone", "macbook", "laptop", "phone", 
                    "tablet", "watch", "airpods", "wallet", "bag", "umbrella", "camera", 
                    "book", "glasses"]
        
        # 단어 단위로 정확하게 매칭
        for item in common_items:
            if item in caption_words:
                product_name = item
                print(f"### DEBUG: Found product name: {product_name}")
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
            
        print(f"### DEBUG: Generated title: {title}")
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
    
    # 결과 한국어 번역역
    def _translate_results(self, caption: str, title: str, description: str, 
                     category: str, color: str, material: str, 
                     brand: str, distinctive_features: str) -> Dict[str, str]:
    
        # 카테고리 번역
        translated_category = self.category_translation.get(category.lower(), category)
        
        # 색상 번역
        translated_color = self.color_translation.get(color.lower(), color)
        
        # 재질 번역
        translated_material = self.material_translation.get(material.lower(), material)
        
        # 브랜드 번역
        translated_brand = self.brand_translation.get(brand.lower() if brand else "", brand if brand else "")
        
        # 제목 번역 (색상, 브랜드, 카테고리)
        translated_title = title
        
        # 주요 단어 번역을 통한 제목 번역
        for en_word, ko_word in self.category_translation.items():
            translated_title = translated_title.replace(en_word, ko_word)
            
        for en_word, ko_word in self.color_translation.items():
            translated_title = translated_title.replace(en_word, ko_word)
            
        for en_word, ko_word in self.brand_translation.items():
            translated_title = translated_title.replace(en_word, ko_word)
        
        # 제목 다시 만들기
        if translated_brand:
            translated_title = f"{translated_brand} {translated_color} "
        else:
            translated_title = f"{translated_color} "
            
        # 카테고리 또는 제품명 추가
        common_items_ko = {
            "phone": "휴대폰",
            "umbrella": "우산",
            "wallet": "지갑",
            "bag": "가방",
            "laptop": "노트북",
            "computer": "컴퓨터",
            "watch": "시계",
            "book": "책",
            "headphones": "헤드셋",
            "camera": "카메라",
            "glasses": "안경",
            "tablet": "태블릿",
            "ipad": "아이패드",
            "iphone": "아이폰",
            "airpods": "에어팟"
        }
        
        # 제목에서 제품 찾기 - 정확한 단어 매칭 사용
        product_found = False
        title_words = title.lower().split()
        
        print(f"### DEBUG: Title words: {title_words}")
        
        for en_item, ko_item in common_items_ko.items():
            if en_item in title_words:
                translated_title += ko_item
                product_found = True
                print(f"### DEBUG: Found product '{en_item}' in title, translating to '{ko_item}'")
                break
                
        # 제품이 없으면 카테고리 사용
        if not product_found:
            translated_title += translated_category
            
        # 특이사항 번역 - 영어 특이사항을 간단한 한국어로 번역
        # 기본적인 단어 매핑을 통한 간단한 번역 시도
        translated_features = distinctive_features
        
        # 일반적인 영어 설명 단어들을 한국어로 대체
        common_english_terms = {
            "modern": "현대적인",
            "stylish": "세련된",
            "sleek": "매끈한",
            "shiny": "반짝이는",
            "beautiful": "아름다운",
            "elegant": "우아한",
            "simple": "심플한",
            "complex": "복잡한",
            "big": "큰",
            "small": "작은",
            "thin": "얇은",
            "thick": "두꺼운",
            "lightweight": "가벼운",
            "heavy": "무거운",
            "expensive": "고급스러운",
            "cheap": "저렴한",
            "portable": "휴대용",
            "damaged": "손상된",
            "durable": "내구성 있는",
            "fragile": "깨지기 쉬운",
            "old": "오래된",
            "new": "새로운",
            "it's": "이것은",
            "has": "있는",
            "with": "가진",
            "and": "그리고"
        }
        
        # 단어 변환 적용
        for eng_word, kor_word in common_english_terms.items():
            translated_features = translated_features.replace(eng_word, kor_word)
        
        # 설명 번역 - 간단한 한국어 설명으로 대체
        translated_description = f"이 물건은 {translated_material} 재질의 {translated_title}입니다."
            
        # 결과 반환
        return {
            "title": translated_title,
            "category": translated_category,
            "color": translated_color,
            "material": translated_material,
            "brand": translated_brand,
            "description": translated_description.strip(),
            "distinctive_features": translated_features
        }

    
    # 분실물 이미지 분석 메인 함수
    def analyze_lost_item(self, image_path: str) -> Dict[str, Any]:

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

# FastAPI 앱 생성
app = FastAPI(title="분실물 이미지 분석 API")

# CORS 설정 (Spring Boot 백엔드 서버와 연동을 위해)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시 도메인 제한 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수로 분석기 선언 (앱 시작시 한 번만 로드)
analyzer = None

@app.on_event("startup")
async def startup_event():
    global analyzer
    analyzer = LostItemAnalyzer()
    print("분실물 분석기가 초기화되었습니다.")

@app.get("/")
async def root():
    return {"message": "분실물 이미지 분석 API가 실행 중입니다."}

# 업로드된 이미지 분석 후 정보 반환
@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):

    global analyzer
    
    if not analyzer:
        raise HTTPException(status_code=500, detail="분석기가 초기화되지 않았습니다.")
    
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
            temp_path = temp.name
            content = await file.read()
            temp.write(content)
        
        # 이미지 분석
        result = analyzer.analyze_lost_item(temp_path)
        
        # 임시 파일 삭제
        os.unlink(temp_path)
        
        if result["success"]:
            # 한국어 번역 결과만 반환
            ko_result = {
                "status": "success",
                "data": {
                    "title": result["data"]["translated"]["title"],
                    "category": result["data"]["translated"]["category"],
                    "color": result["data"]["translated"]["color"],
                    "material": result["data"]["translated"]["material"],
                    "brand": result["data"]["translated"]["brand"],
                    "description": result["data"]["translated"]["description"],
                    "distinctive_features": result["data"]["translated"]["distinctive_features"]
                }
            }
            return ko_result
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        # 예외 발생 시 임시 파일 삭제 시도
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"이미지 분석 중 오류 발생: {str(e)}")

# 직접 실행 시 Uvicorn 서버 시작
if __name__ == "__main__":
    # 서버 포트 설정 (기본 8000)
    port = int(os.environ.get("PORT", 8000))
    
    # Uvicorn 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=port)