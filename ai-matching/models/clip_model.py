import torch
import clip
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Union, Optional

from config import Config

class CLIPModel:
    """
    CLIP 모델을 활용한 이미지 분석 및 분류 클래스
    """
    def __init__(self, model_name: str = None):
        """
        CLIP 모델 초기화
        
        Args:
            model_name: 사용할 CLIP 모델 이름. 기본값은 Config에서 가져옴
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = model_name or Config.CLIP_MODEL_NAME
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.custom_classes = self._get_custom_classes()
        self.text_features, self.expanded_classes = self._tokenize_labels(self.custom_classes)
        
    def _get_custom_classes(self) -> List[str]:
        """
        분석에 사용할 커스텀 클래스 목록 생성
        
        Returns:
            List[str]: 모든 커스텀 클래스를 포함하는 리스트
        """
        custom_classes = []
        for category in Config.CATEGORY_GROUPS.values():
            custom_classes.extend(category)
        return list(set(custom_classes))  # 중복 제거
    
    def _tokenize_labels(self, classes: List[str]) -> Tuple[torch.Tensor, List[str]]:
        """
        클래스 레이블을 CLIP 모델에 적합한 형태로 토큰화
        
        Args:
            classes: 토큰화할 클래스 레이블 목록
            
        Returns:
            Tuple[torch.Tensor, List[str]]: 토큰화된 텍스트 특성과 확장된 클래스 목록
        """
        text_inputs = []
        expanded_class_list = []
        
        for c in classes:
            # 한국어/영어 처리
            is_korean = any('\uAC00' <= char <= '\uD7A3' for char in c)
            
            if is_korean:
                # 한국어 프롬프트
                prompts = [
                    f"{c}의 사진",
                    f"{c} 이미지",
                    f"{c}"
                ]
            else:
                # 영어 프롬프트 - 기본 관사 처리 (a/an)
                article = "an" if c[0].lower() in "aeiou" else "a"
                
                # 특수 케이스 처리
                if c in ["airpods", "earpods", "headphones", "glasses", "sunglasses", "keys", "gloves", "galaxy buds", "samsung earbuds", "earbuds"]:
                    # 복수형 단어는 "a pair of"로 시작
                    prompts = [
                        f"a photo of {c}",
                        f"a picture of {c}",
                        f"a pair of {c}",
                        f"the {c}"
                    ]
                else:
                    prompts = [
                        f"a photo of {article} {c}",
                        f"a picture of {article} {c}",
                        f"{article} {c}",
                        f"the {c}"
                    ]
                
                # 한국 돈/화폐 특화 프롬프트
                if c in ["korean won", "10000 won", "50000 won", "현금", "지폐", "동전", "한국 돈", "만원", "오만원", "천원", "오천원", "백원", "오백원"]:
                    prompts.append(f"korean money")
                    prompts.append(f"korean currency")
                    if "won" in c:
                        prompts.append(f"korean {c}")
                
                # 차키 특화 프롬프트
                if c in ["car key", "차키"]:
                    prompts.append(f"automobile key")
                    prompts.append(f"vehicle key")
                    prompts.append(f"key fob")
                    prompts.append(f"remote car key")
                
                # 제조사별 특화된 프롬프트 추가 (애플, 삼성 등)
                # 코드 간결성을 위해 일부 생략됨 - 필요시 원본 코드에서 확장
            
            for p in prompts:
                text_inputs.append(clip.tokenize(p))
                expanded_class_list.append(c)
        
        text_inputs = torch.cat(text_inputs).to(self.device)
        
        # 토큰화된 텍스트의 CLIP 특성 추출
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features, expanded_class_list
    
    def calculate_similarity(self, image: Image.Image, text_features: Optional[torch.Tensor] = None, topk: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        이미지와 텍스트 특성 간의 유사도 계산
        
        Args:
            image: 분석할 PIL 이미지
            text_features: 사용할 텍스트 특성 (없으면 기본값 사용)
            topk: 반환할 상위 결과 수
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 상위 유사도 값과 해당 인덱스
        """
        if text_features is None:
            text_features = self.text_features
            
        # 이미지를 CLIP 전처리기를 통해 전처리
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # 이미지의 CLIP 특성 추출
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # 이미지 특성과 텍스트 특성 간의 코사인 유사도 계산
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Top-K 유사도 값과 해당 인덱스 추출
        values, indices = similarity[0].topk(topk)
        
        # 결과를 원래 클래스와 매핑
        result_values = []
        result_indices = []
        seen_classes = set()
        
        for v, idx in zip(values, indices):
            class_name = self.expanded_classes[idx]
            if class_name not in seen_classes:
                seen_classes.add(class_name)
                original_idx = self.custom_classes.index(class_name) if class_name in self.custom_classes else -1
                result_values.append(v)
                result_indices.append(original_idx)
                
                if len(result_values) >= topk:
                    break
        
        return torch.tensor(result_values), torch.tensor(result_indices)
    
    def classify_image(self, image_path: str = None, image: Image.Image = None, topk: int = 10, 
                       category_filter: List[str] = None) -> Dict[str, float]:
        """
        이미지 분류 및 객체 인식
        
        Args:
            image_path: 분석할 이미지 경로 (image가 None인 경우 사용)
            image: 분석할 PIL 이미지 객체 (image_path가 None인 경우 사용)
            topk: 반환할 상위 결과 수
            category_filter: 필터링할 카테고리 목록 (None이면 모든 카테고리 사용)
            
        Returns:
            Dict[str, float]: 클래스명과 신뢰도를 포함하는 딕셔너리
        """
        # 이미지 로드 (경로 또는 이미지 객체 사용)
        if image_path is not None:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"이미지를 불러오는 데 오류가 발생했습니다: {e}")
                return {}
        elif image is None:
            print("이미지 경로나 이미지 객체가 필요합니다.")
            return {}
        
        # 유사도 계산
        values, indices = self.calculate_similarity(image, self.text_features, topk=topk if category_filter is None else topk*2)
        
        # 카테고리 필터 적용
        if category_filter is not None:
            filtered_values = []
            filtered_indices = []
            
            filter_list = []
            for category in category_filter:
                if category in Config.CATEGORY_GROUPS:
                    filter_list.extend(Config.CATEGORY_GROUPS[category])
                elif category == 'all':
                    filter_list = self.custom_classes
                    break
            
            for v, idx in zip(values, indices):
                class_name = self.custom_classes[idx.item()]
                if class_name in filter_list:
                    filtered_values.append(v)
                    filtered_indices.append(idx)
                    
                    if len(filtered_values) >= topk:
                        break
            
            if filtered_values:
                values = torch.stack(filtered_values)
                indices = torch.stack(filtered_indices)
        
        # 결과를 딕셔너리로 변환
        results = {}
        for v, idx in zip(values, indices):
            class_name = self.custom_classes[idx.item()]
            results[class_name] = v.item()
            
        return results
    
    def get_image_features(self, image_path: str = None, image: Image.Image = None) -> torch.Tensor:
        """
        이미지에서 CLIP 특성 추출
        
        Args:
            image_path: 분석할 이미지 경로 (image가 None인 경우 사용)
            image: 분석할 PIL 이미지 객체 (image_path가 None인 경우 사용)
            
        Returns:
            torch.Tensor: 이미지 특성 벡터
        """
        # 이미지 로드 (경로 또는 이미지 객체 사용)
        if image_path is not None:
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"이미지를 불러오는 데 오류가 발생했습니다: {e}")
                return None
        elif image is None:
            print("이미지 경로나 이미지 객체가 필요합니다.")
            return None
        
        # 이미지 전처리 및 특성 추출
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        return image_features[0]  # 배치 차원 제거