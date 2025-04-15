# 파파고 API를 사용한 번역 서비스

import requests
import urllib.parse
from typing import Optional, Dict, Any
from config import config

class Translator:

    def __init__(self):
        
        self.client_id = config.NAVER_CLIENT_ID
        self.client_secret = config.NAVER_CLIENT_SECRET
        self.use_papago = bool(self.client_id and self.client_secret)
        
        # 브랜드 매칭이 필요한 카테고리 목록
        self.brand_applicable_categories = [
            "electronics", "phone", "computer", "wallet", "earbuds", "smartwatch", 
            "jewelry", "bag", "accessories", "clothing"
        ]
        
    def translate(self, text: str, source: str = "en", target: str = "ko") -> str:

        if not self.use_papago or not text or text.strip() == "":
            return text
            
        # Papago API 엔드포인트
        url = "https://naveropenapi.apigw.ntruss.com/nmt/v1/translation"
        
        # 헤더 설정
        headers = {
            "X-NCP-APIGW-API-KEY-ID": self.client_id,
            "X-NCP-APIGW-API-KEY": self.client_secret,
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        # 데이터 인코딩
        encoded_text = urllib.parse.quote(text)
        data = f"source={source}&target={target}&text={encoded_text}"
        
        try:
            response = requests.post(url, headers=headers, data=data)
            response.raise_for_status()
            
            result = response.json()
            translated_text = result.get("message", {}).get("result", {}).get("translatedText", "")
            
            return translated_text
        except Exception as e:
            return text  # 오류 발생 시 원본 텍스트 반환
        
    # 분석결과 번역
    def translate_results(self, result_data):
        """분석결과 번역 (개선된 버전)"""
        caption = result_data.get("caption", "")
        title = result_data.get("title", "")
        description = result_data.get("description", "")
        category = result_data.get("category", "")
        color = result_data.get("color", "")
        material = result_data.get("material", "")
        brand = result_data.get("brand", "")
        distinctive_features = result_data.get("distinctive_features", "")
        
        # 카테고리 번역 (매핑 사용)
        translated_category = config.CATEGORY_TRANSLATION.get(category.lower(), category)
        if translated_category == category and self.use_papago:
            # 매핑에 없는 경우 파파고 사용
            translated_category = self.translate(category)
        
        # 색상 번역
        translated_color = self.translate(color) if self.use_papago else color
        
        # 재질 번역
        translated_material = self.translate(material) if self.use_papago else material
        
        # 브랜드 적용 가능 카테고리인지 확인
        is_brand_applicable = any(cat in category.lower() for cat in self.brand_applicable_categories)
        
        # 적용 가능한 경우만 브랜드 번역, 아니면 빈 문자열
        if is_brand_applicable and brand:
            translated_brand = config.BRAND_TRANSLATION.get(brand.lower() if brand else "", brand if brand else "")
        else:
            translated_brand = ""
        
        # 삼성 제품 특별 처리
        is_samsung = False
        product_type = ""
        
        if brand.lower() == "samsung" and is_brand_applicable:
            is_samsung = True
            # 제품 유형 식별
            title_lower = title.lower()
            
            if "buds" in title_lower or "earbuds" in title_lower:
                product_type = "갤럭시 버즈"
            elif "watch" in title_lower or "smartwatch" in title_lower:
                product_type = "갤럭시 워치"
            elif any(phone_word in title_lower for phone_word in ["phone", "smartphone", "s series", "note", "galaxy s", "galaxy note"]):
                product_type = "갤럭시 휴대폰"
        
        # 번역 제목 생성
        if is_samsung and product_type:
            # 삼성 제품 전용 제목
            translated_title = f"삼성 {product_type} {translated_color}"
        else:
            # 일반 제목 생성
            # 색상과 브랜드로 시작
            if translated_brand and is_brand_applicable:
                translated_title = f"{translated_brand} {translated_color} "
            else:
                translated_title = f"{translated_color} "
            
            # 제품명 추가
            title_words = title.lower().split()
            product_found = False
            
            for en_item, ko_item in config.PRODUCT_TRANSLATION.items():
                if en_item in title.lower():
                    translated_title += ko_item
                    product_found = True
                    break
            
            # 제품이 없으면 카테고리 사용
            if not product_found:
                translated_title += translated_category
        
        # 지갑 브랜드 특별 처리
        if "wallet" in category.lower() and translated_brand:
            # 브랜드가 있는 지갑의 경우
            translated_title = f"{translated_brand} {translated_color} 지갑"
        
        # 특이사항 번역
        translated_features = self.translate(distinctive_features) if self.use_papago and distinctive_features else distinctive_features
        
        # 설명 번역 - 간결한 한국어 형식 사용
        if is_samsung and product_type:
            # 삼성 제품 전용 설명
            translated_description = f"이 물건은 삼성 {product_type}입니다. {translated_material} 재질의 {translated_color} 색상입니다."
        elif "wallet" in category.lower() and translated_brand:
            # 브랜드 지갑 설명
            translated_description = f"이 물건은 {translated_brand}의 {translated_color} 지갑입니다. {translated_material} 재질로 만들어졌습니다."
        else:
            # 일반 설명
            translated_description = f"이 물건은 {translated_material} 재질의 {translated_color} {translated_category}입니다."
        
        # 브랜드 정보 추가 (적용 가능한 카테고리인 경우만)
        if translated_brand and is_brand_applicable and not is_samsung and "wallet" not in category.lower():
            translated_description = f"이 물건은 {translated_brand}의 {translated_material} 재질 {translated_color} {translated_category}입니다."
        
        # 특이사항이 있으면 추가
        if translated_features and "unknown" not in translated_features.lower() and "none" not in translated_features.lower():
            translated_description += f" 특징은 {translated_features} 입니다."
        
        # 결과 반환
        return {
            "title": translated_title.strip(),
            "category": translated_category,
            "color": translated_color,
            "material": translated_material,
            "brand": translated_brand,
            "description": translated_description.strip(),
            "distinctive_features": translated_features
        }