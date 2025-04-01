import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from typing import List, Dict, Union, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

from config import Config

class BLIPModel:
    """
    BLIP 모델을 사용하여 이미지 캡셔닝과 텍스트-이미지 유사도 측정
    """
    def __init__(self, model_name: str = None):
        """
        BLIP 모델 초기화
        
        Args:
            model_name: 사용할 BLIP 모델 이름. 기본값은 Config에서 가져옴
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = model_name or Config.BLIP_MODEL_NAME
        
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        # 한국어 처리를 위한 설정
        self.ko_processor = None
        self.ko_model = None
        if Config.PRIMARY_LANGUAGE == 'ko':
            try:
                # 한국어 지원 모델 (예: "kykim/blip-image-captioning-ko")
                ko_model_name = "kykim/blip-image-captioning-ko"
                self.ko_processor = BlipProcessor.from_pretrained(ko_model_name)
                self.ko_model = BlipForConditionalGeneration.from_pretrained(ko_model_name).to(self.device)
            except Exception as e:
                print(f"한국어 모델 로드 실패: {e}. 영어 모델만 사용합니다.")
    
    def generate_captions(self, image_path: str = None, image: Image.Image = None, 
                         num_captions: int = 3, use_korean: bool = True) -> List[str]:
        """
        이미지에 대한 캡션 생성
        
        Args:
            image_path: 분석할 이미지 경로 (image가 None인 경우 사용)
            image: 분석할 PIL 이미지 객체 (image_path가 None인 경우 사용)
            num_captions: 생성할 캡션 수
            use_korean: 한국어 캡션 생성 여부
            
        Returns:
            List[str]: 생성된 캡션 목록
        """
        # 이미지 로드 (경로 또는 이미지 객체 사용)
        if image_path is not None:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"이미지를 불러오는 데 오류가 발생했습니다: {e}")
                return []
        elif image is None:
            print("이미지 경로나 이미지 객체가 필요합니다.")
            return []
        
        captions = []
        
        # 영어 캡션 생성
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        for _ in range(num_captions):
            out = self.model.generate(
                **inputs,
                max_length=Config.BLIP_MAX_LENGTH,
                num_beams=Config.BLIP_NUM_BEAMS,
                min_length=Config.BLIP_MIN_LENGTH,
                top_p=0.9,
                repetition_penalty=1.5,
                do_sample=True,
                temperature=0.7
            )
            caption = self.processor.decode(out[0], skip_special_tokens=True)
            captions.append(caption)
        
        # 한국어 캡션 생성 (한국어 모델이 있고 use_korean이 True인 경우)
        if use_korean and self.ko_model is not None and self.ko_processor is not None:
            ko_inputs = self.ko_processor(image, return_tensors="pt").to(self.device)
            
            for _ in range(num_captions):
                out = self.ko_model.generate(
                    **ko_inputs,
                    max_length=Config.BLIP_MAX_LENGTH,
                    num_beams=Config.BLIP_NUM_BEAMS,
                    min_length=Config.BLIP_MIN_LENGTH,
                    top_p=0.9,
                    repetition_penalty=1.5,
                    do_sample=True,
                    temperature=0.7
                )
                caption = self.ko_processor.decode(out[0], skip_special_tokens=True)
                captions.append(caption)
        
        return captions
    
    def get_caption_features(self, image_path: str = None, image: Image.Image = None) -> torch.Tensor:
        """
        이미지에서 캡션 기반 특성 추출
        
        Args:
            image_path: 분석할 이미지 경로 (image가 None인 경우 사용)
            image: 분석할 PIL 이미지 객체 (image_path가 None인 경우 사용)
            
        Returns:
            torch.Tensor: 이미지 캡션의 특성 벡터
        """
        # 캡션 생성
        captions = self.generate_captions(image_path, image)
        if not captions:
            return None
        
        # 캡션을 모델에 입력하여 임베딩 추출
        caption_embeddings = []
        for caption in captions:
            text_inputs = self.processor(text=caption, return_tensors="pt").to(self.device)
            with torch.no_grad():
                text_features = self.model.text_encoder(**text_inputs)
                # 마지막 히든 스테이트의 평균을 사용
                embedding = text_features.last_hidden_state.mean(dim=1)
                caption_embeddings.append(embedding)
        
        # 모든 캡션 임베딩의 평균
        if caption_embeddings:
            combined_embedding = torch.cat(caption_embeddings).mean(dim=0)
            # 정규화
            combined_embedding = combined_embedding / combined_embedding.norm()
            return combined_embedding
        
        return None
    
    def extract_tags_from_caption(self, caption: str) -> List[str]:
        """
        캡션에서 중요 태그/키워드 추출
        
        Args:
            caption: 분석할 캡션 텍스트
            
        Returns:
            List[str]: 추출된 태그 목록
        """
        # 간단한 명사 추출 (더 정교한 NLP 처리를 위해 확장 가능)
        # 영어와 한국어 구분
        is_korean = any('\uAC00' <= char <= '\uD7A3' for char in caption)
        
        if is_korean:
            try:
                from konlpy.tag import Okt
                okt = Okt()
                nouns = okt.nouns(caption)
                return [noun for noun in nouns if len(noun) > 1]  # 1글자 명사 제외
            except:
                # konlpy가 설치되지 않은 경우 기본 처리
                words = re.findall(r'\w+', caption)
                return [w for w in words if len(w) > 1]
        else:
            # 영어 텍스트 처리
            # 불용어 목록
            stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'of', 'in', 'to', 'for', 'with', 'on', 'at'}
            
            # 단어 추출 및 불용어 제거
            words = re.findall(r'\b\w+\b', caption.lower())
            return [word for word in words if word not in stopwords and len(word) > 2]
    
    def calculate_caption_similarity(self, caption1: str, caption2: str) -> float:
        """
        두 캡션 간의 유사도 계산
        
        Args:
            caption1: 첫 번째 캡션
            caption2: 두 번째 캡션
            
        Returns:
            float: 유사도 점수 (0~1)
        """
        # 태그 추출
        tags1 = set(self.extract_tags_from_caption(caption1))
        tags2 = set(self.extract_tags_from_caption(caption2))
        
        # 자카드 유사도 계산
        if not tags1 or not tags2:
            return 0.0
            
        intersection = len(tags1.intersection(tags2))
        union = len(tags1.union(tags2))
        
        return intersection / union if union > 0 else 0.0
    
    def calculate_blip_similarity(self, image1_path: str, image2_path: str) -> float:
        """
        두 이미지 간의 BLIP 기반 유사도 계산
        
        Args:
            image1_path: 첫 번째 이미지 경로
            image2_path: 두 번째 이미지 경로
            
        Returns:
            float: 유사도 점수 (0~1)
        """
        # 각 이미지의 캡션 생성
        captions1 = self.generate_captions(image1_path)
        captions2 = self.generate_captions(image2_path)
        
        if not captions1 or not captions2:
            return 0.0
        
        # 모든 캡션 쌍에 대한 유사도 계산
        similarities = []
        for cap1 in captions1:
            for cap2 in captions2:
                sim = self.calculate_caption_similarity(cap1, cap2)
                similarities.append(sim)
        
        # 최대 유사도 반환
        return max(similarities) if similarities else 0.0
    
    def calculate_visual_textual_similarity(self, image1_path: str, image2_path: str) -> float:
        """
        이미지와 텍스트 특성을 모두 고려한 유사도 계산
        
        Args:
            image1_path: 첫 번째 이미지 경로
            image2_path: 두 번째 이미지 경로
            
        Returns:
            float: 유사도 점수 (0~1)
        """
        # 각 이미지에서 캡션 특성 추출
        features1 = self.get_caption_features(image1_path)
        features2 = self.get_caption_features(image2_path)
        
        if features1 is None or features2 is None:
            return 0.0
        
        # 코사인 유사도 계산
        cos_sim = torch.nn.functional.cosine_similarity(features1.unsqueeze(0), features2.unsqueeze(0)).item()
        
        # 캡션 텍스트 기반 유사도
        caption_sim = self.calculate_blip_similarity(image1_path, image2_path)
        
        # 두 유사도의 가중 평균
        return 0.7 * cos_sim + 0.3 * caption_sim
    
    def analyze_image_content(self, image_path: str = None, image: Image.Image = None) -> Dict[str, any]:
        """
        이미지 내용에 대한 상세 분석
        
        Args:
            image_path: 분석할 이미지 경로 (image가 None인 경우 사용)
            image: 분석할 PIL 이미지 객체 (image_path가 None인 경우 사용)
            
        Returns:
            Dict[str, any]: 분석 결과를 포함하는 딕셔너리
        """
        # 캡션 생성
        captions = self.generate_captions(image_path, image)
        
        if not captions:
            return {"error": "이미지 분석 실패"}
        
        # 모든 캡션에서 태그 추출
        all_tags = []
        for caption in captions:
            tags = self.extract_tags_from_caption(caption)
            all_tags.extend(tags)
        
        # 태그 빈도수 계산
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # 태그 빈도수로 정렬
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 결과 구성
        return {
            "captions": captions,
            "tags": dict(sorted_tags[:10]),  # 상위 10개 태그만 포함
            "primary_caption": captions[0],  # 첫 번째 캡션을 주요 캡션으로 선택
            "features": self.get_caption_features(image_path, image).tolist() if self.get_caption_features(image_path, image) is not None else None
        }