import torch
import clip
import numpy as np
from PIL import Image
import cv2
import logging
from typing import List, Tuple, Dict, Any, Optional, Union

from config.config import CLIP_MODEL_NAME, DEVICE, CUSTOM_CLASSES

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CLIPModel:
    """이미지와 텍스트 임베딩을 위한 CLIP 모델"""
    
    def __init__(self, model_name: str = CLIP_MODEL_NAME, device: str = DEVICE):
        """
        CLIP 모델 초기화
        
        Args:
            model_name (str): 사용할 CLIP 모델 이름
            device (str): 사용할 장치 (cuda 또는 cpu)
        """
        self.device = device
        logger.info(f"{device} 장치에 {model_name} CLIP 모델 로드 중")
        
        try:
            self.model, self.preprocess = clip.load(model_name, device=device)
            logger.info("CLIP 모델 로드 완료")
            
            # 사용자 정의 클래스에 대한 텍스트 특성 생성
            self.text_features, self.expanded_classes = self._tokenize_labels(CUSTOM_CLASSES)
            logger.info(f"{len(CUSTOM_CLASSES)}개 클래스에 대해 {len(self.expanded_classes)}개 변형으로 임베딩 생성 완료")
            
        except Exception as e:
            logger.error(f"CLIP 모델 로드 오류: {str(e)}")
            raise
    
    def _tokenize_labels(self, classes: List[str]) -> Tuple[torch.Tensor, List[str]]:
        """
        분류 레이블에 대한 토큰화된 텍스트 특성 생성
        
        Args:
            classes (List[str]): 클래스 이름 목록
            
        Returns:
            Tuple[torch.Tensor, List[str]]: 토큰화된 특성과 확장된 클래스 목록
        """
        text_inputs = []
        expanded_class_list = []
        
        for c in classes:
            # 클래스 이름이 한국어인지 확인
            is_korean = any('\uAC00' <= char <= '\uD7A3' for char in c)
            
            if is_korean:
                # 한국어 프롬프트
                prompts = [
                    f"{c}의 사진",
                    f"{c} 이미지",
                    f"{c}"
                ]
            else:
                # 영어 프롬프트 (적절한 관사 처리)
                article = "an" if c[0].lower() in "aeiou" else "a"
                
                # 복수형 단어에 대한 특별 처리
                if c in ["airpods", "earpods", "headphones", "glasses", "sunglasses", "keys", 
                         "gloves", "galaxy buds", "samsung earbuds", "earbuds"]:
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
                
                # 특정 카테고리에 대한 특별 프롬프트
                # 한국 화폐
                if c in ["korean won", "10000 won", "50000 won", "현금", "지폐", "동전", "한국 돈", 
                         "만원", "오만원", "천원", "오천원", "백원", "오백원"]:
                    prompts.append(f"korean money")
                    prompts.append(f"korean currency")
                    if "won" in c:
                        prompts.append(f"korean {c}")
                
                # 자동차 키
                if c in ["car key", "차키"]:
                    prompts.append(f"automobile key")
                    prompts.append(f"vehicle key")
                    prompts.append(f"key fob")
                    prompts.append(f"remote car key")
                
                # 애플 제품
                if c in ["iphone", "ipad", "macbook", "airpods", "earpods", "apple watch"]:
                    prompts.append(f"an apple {c}")
                    prompts.append(f"an apple device")
                
                # 삼성 제품
                if c in ["samsung phone", "galaxy phone", "galaxy s", "galaxy note", "galaxy tab", 
                         "galaxy watch", "galaxy buds", "samsung tablet", "samsung earbuds"]:
                    if "galaxy" in c:
                        prompts.append(f"a samsung {c}")
                    prompts.append(f"a samsung device")
                    
                    if c == "galaxy watch":
                        prompts.append(f"a samsung smartwatch")
                        prompts.append(f"a round smartwatch")
                    
                    if c == "galaxy buds" or c == "samsung earbuds":
                        prompts.append(f"samsung wireless earbuds")
                
                # 삼성 폴더블 기기
                if c in ["galaxy z flip", "galaxy z fold", "samsung foldable phone", "foldable phone", 
                        "samsung folding phone", "폴더블 스마트폰", "접이식 휴대폰", "갤럭시 Z 플립", "갤럭시 Z 폴드"]:
                    
                    is_flip = "flip" in c.lower() or "플립" in c
                    is_fold = "fold" in c.lower() or "폴드" in c
                    
                    prompts.append(f"a samsung foldable phone")
                    prompts.append(f"a foldable smartphone")
                    
                    if is_flip:
                        prompts.append(f"a samsung galaxy z flip")
                        prompts.append(f"a flip phone that folds horizontally")
                        prompts.append(f"a clamshell folding phone")
                    
                    if is_fold:
                        prompts.append(f"a samsung galaxy z fold")
                        prompts.append(f"a phone that folds like a book")
                        prompts.append(f"a fold phone that opens into a tablet")
                    
                    prompts.append(f"a samsung device that folds")
                    prompts.append(f"a premium samsung folding phone")
                
                # 일반 전자기기
                if c in ["smartphone", "phone", "tablet", "wireless earbuds", "headphones", "headset", 
                         "laptop", "smartwatch", "smart watch", "earbuds", "smart device", "camera"]:
                    prompts.append(f"a modern {c}")
                    prompts.append(f"an electronic {c}")
                
                # 일반적인 분실물
                if c in ["wallet", "purse", "credit card", "id card", "card", "key", "keys", "backpack", 
                         "bag", "glasses", "sunglasses", "book", "umbrella", "water bottle"]:
                    prompts.append(f"a lost {c}")
                    prompts.append(f"a personal {c}")
            
            # 이 클래스에 대한 모든 프롬프트 토큰화
            for p in prompts:
                text_inputs.append(clip.tokenize(p))
                expanded_class_list.append(c)
        
        # 모든 토큰화된 입력 결합
        text_inputs = torch.cat(text_inputs).to(self.device)
        
        # 텍스트 특성 추출
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features, expanded_class_list
    
    def get_image_embedding(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        이미지에 대한 CLIP 임베딩 생성
        
        Args:
            image: 이미지 경로, PIL 이미지, 또는 numpy 배열
            
        Returns:
            torch.Tensor: 정규화된 이미지 임베딩
        """
        # 다양한 입력 타입 처리
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(np.uint8(image)).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise ValueError(f"지원되지 않는 이미지 타입: {type(image)}")
        
        # 전처리 및 임베딩 생성
        img_input = self.preprocess(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            img_features = self.model.encode_image(img_input)
            img_features /= img_features.norm(dim=-1, keepdim=True)
        
        return img_features
    
    def get_text_embedding(self, text: str) -> torch.Tensor:
        """
        텍스트에 대한 CLIP 임베딩 생성
        
        Args:
            text (str): 임베딩할 텍스트
            
        Returns:
            torch.Tensor: 정규화된 텍스트 임베딩
        """
        text_input = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def calculate_similarity(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        이미지와 텍스트 특성 간의 유사도 계산
        
        Args:
            image_features (torch.Tensor): 이미지 특성
            text_features (torch.Tensor): 텍스트 특성
            
        Returns:
            torch.Tensor: 유사도 점수
        """
        # 코사인 유사도 계산 (값 범위: 0~1)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarity
    
    def classify_image(self, image: Union[str, Image.Image, np.ndarray], topk: int = 5) -> List[Dict[str, Any]]:
        """
        이미지를 분류하고 상위 k개 클래스 반환
        
        Args:
            image: 이미지 경로, PIL 이미지, 또는 numpy 배열
            topk (int): 반환할 상위 클래스 수
            
        Returns:
            List[Dict[str, Any]]: 상위 k개 클래스 및 점수
        """
        # 이미지 임베딩 얻기
        image_features = self.get_image_embedding(image)
        
        # 유사도 계산
        similarity = self.calculate_similarity(image_features, self.text_features)
        
        # 상위 k개 유사도 값과 인덱스 추출
        values, indices = similarity[0].topk(min(len(self.expanded_classes), topk * 2))
        
        # 중복 클래스 제거하면서 결과 구성
        results = []
        seen_classes = set()
        
        for v, idx in zip(values, indices):
            class_name = self.expanded_classes[idx]
            if class_name not in seen_classes:
                seen_classes.add(class_name)
                results.append({
                    'class': class_name,
                    'score': v.item(),
                    'prompt': self.expanded_classes[idx]
                })
                
                if len(results) >= topk:
                    break
        
        return results
    
    def detect_brand_logo(self, image: Union[str, Image.Image, np.ndarray]) -> Tuple[Optional[str], float]:
        """
        이미지에서 애플 또는 삼성 로고 감지
        
        Args:
            image: 이미지 경로, PIL 이미지, 또는 numpy 배열
            
        Returns:
            Tuple[Optional[str], float]: 감지된 브랜드와 신뢰도
        """
        # 이미지 로드
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
            img_np = np.array(img)
        elif isinstance(image, np.ndarray):
            img_np = image
            img = Image.fromarray(np.uint8(img_np)).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
            img_np = np.array(img)
        else:
            raise ValueError(f"지원되지 않는 이미지 타입: {type(image)}")
        
        # 로고 인식을 위한 CLIP 특화 프롬프트
        logo_prompts = [
            "the apple logo, a bitten apple",
            "the iconic apple symbol, bitten apple",
            "the apple brand logo on a device",
            "the samsung logo text",
            "the samsung brand name text",
            "the samsung logo on a device",
            "no visible brand logo",
            "a plain device with no logo"
        ]
        
        # 텍스트 토큰화 및 특성 추출
        text_inputs = torch.cat([clip.tokenize(p) for p in logo_prompts]).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # 이미지 전처리 및 특성 추출
        img_input = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(img_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        # 유사도 계산
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        values, indices = similarity[0].topk(2)
        
        top_index = indices[0].item()
        top_confidence = values[0].item()
        
        # 결과 결정
        # 애플 로고 관련 프롬프트 (0-2 인덱스)
        if top_index < 3:
            # 아이폰 카메라 모듈 확인 (더 확실한 확인)
            if self._detect_iphone_camera(img_np):
                return "apple", max(top_confidence, 0.8)
            return "apple", top_confidence
            
        # 삼성 로고 관련 프롬프트 (3-5 인덱스)
        elif top_index < 6:
            # 삼성 텍스트 로고 확인
            if self._detect_samsung_text(img_np):
                return "samsung", max(top_confidence, 0.8)
            return "samsung", top_confidence
            
        # 특정 장치 특성 기반 판단
        else:
            if self._detect_iphone_camera(img_np):
                return "apple", 0.7
            elif self._detect_samsung_text(img_np):
                return "samsung", 0.7
            else:
                return None, top_confidence
    
    def _detect_iphone_camera(self, image_np: np.ndarray) -> bool:
        """아이폰 카메라 모듈 감지 함수"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # 블러 처리로 노이즈 감소
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 원 감지를 위한 허프 변환 (아이폰 카메라 렌즈는 원형)
        circles = cv2.HoughCircles(
            blurred, 
            cv2.HOUGH_GRADIENT, 
            dp=1, 
            minDist=20, 
            param1=50, 
            param2=30, 
            minRadius=10, 
            maxRadius=50
        )
        
        # 아이폰 특유의 카메라 배열 패턴 확인
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            # 원이 2-3개 있고, 특정 패턴으로 배열되어 있는지 확인
            if 2 <= len(circles[0]) <= 3:
                return True
        
        return False
    
    def _detect_samsung_text(self, image_np: np.ndarray) -> bool:
        """삼성 로고 텍스트 감지 함수"""
        # 텍스트 감지를 위해 이미지 전처리
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # 이진화
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 윤곽선 검출
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # SAMSUNG 로고는 보통 직사각형 형태의 텍스트
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            
            # SAMSUNG 텍스트의 일반적인 가로세로 비율 (가로가 세로보다 약 3-5배 긺)
            if 3 <= aspect_ratio <= 5 and w > 50:
                return True
        
        return False
    
    def detect_fold_hinge(self, image_np: np.ndarray) -> bool:
        """폴더블 폰의 힌지(접히는 부분) 감지 함수"""
        # 그레이스케일 변환
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        
        # 에지 감지
        edges = cv2.Canny(gray, 50, 150)
        
        # 직선 감지 (힌지는 보통 직선으로 나타남)
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=100, 
            minLineLength=max(image_np.shape[0], image_np.shape[1]) // 4,
            maxLineGap=20
        )
        
        if lines is not None:
            height, width = image_np.shape[:2]
            
            # 중앙 부근에 긴 수직/수평선이 있는지 확인
            for line in lines:
                for x1, y1, x2, y2 in line:
                    # 선의 길이 계산
                    line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    
                    # 선의 방향 (수직 또는 수평)
                    is_vertical = abs(x2 - x1) < abs(y2 - y1)
                    is_horizontal = not is_vertical
                    
                    # 이미지 중앙 부근에 있는 긴 선인지 확인
                    is_center_region = (
                        (is_vertical and abs((x1 + x2) / 2 - width / 2) < width * 0.2) or
                        (is_horizontal and abs((y1 + y2) / 2 - height / 2) < height * 0.2)
                    )
                    
                    # 충분히 긴 선이고 중앙 부근에 있으면 힌지로 간주
                    if line_length > min(height, width) * 0.25 and is_center_region:
                        return True
        
        return False
    
    def analyze_image_colors(self, image: Union[str, Image.Image, np.ndarray]) -> Dict[str, float]:
        """
        이미지의 색상 구성을 분석하여 브랜드별 색상 유사성 점수 계산
        
        Args:
            image: 이미지 경로, PIL 이미지, 또는 numpy 배열
            
        Returns:
            Dict[str, float]: 각 브랜드에 대한 색상 유사성 점수
        """
        # 이미지 로드
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
            img_np = np.array(img)
        elif isinstance(image, np.ndarray):
            img_np = image
        elif isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"))
        else:
            raise ValueError(f"지원되지 않는 이미지 타입: {type(image)}")
        
        # 색상 히스토그램 계산
        hist_r = cv2.calcHist([img_np], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img_np], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img_np], [2], None, [256], [0, 256])
        
        # 정규화
        cv2.normalize(hist_r, hist_r, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_g, hist_g, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
        
        # 애플 색상 특성 (흰색, 검정색, 회색, 금색)
        apple_white = np.sum(hist_r[200:256] * hist_g[200:256] * hist_b[200:256])
        apple_black = np.sum(hist_r[0:50] * hist_g[0:50] * hist_b[0:50])
        apple_gray = np.sum(hist_r[100:150] * hist_g[100:150] * hist_b[100:150])
        apple_gold = np.sum(hist_r[180:230] * hist_g[150:200] * hist_b[50:100])
        
        # 삼성 색상 특성 (흰색, 검정색, 파란색)
        samsung_white = apple_white  # 흰색은 동일
        samsung_black = apple_black  # 검정색은 동일
        samsung_blue = np.sum(hist_r[0:100] * hist_g[0:100] * hist_b[150:250])
        
        # 색상 점수 계산
        apple_score = float((apple_white + apple_black + apple_gray + apple_gold) / 4)
        samsung_score = float((samsung_white + samsung_black + samsung_blue) / 3)
        
        return {
            "apple": apple_score,
            "samsung": samsung_score
        }
    
    def advanced_classification(self, image: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        로고 감지와 클래스 분류를 통합한 고급 이미지 분류
        
        Args:
            image: 이미지 경로, PIL 이미지, 또는 numpy 배열
            
        Returns:
            Dict[str, Any]: 분류 결과와 신뢰도 정보
        """
        # 표준 CLIP 분류 수행
        classification_results = self.classify_image(image, topk=3)
        clip_class = classification_results[0]['class'] if classification_results else None
        clip_confidence = classification_results[0]['score'] if classification_results else 0.0
        
        # 이미지를 numpy 배열로 변환 (내부 함수 호출용)
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
            img_np = np.array(img)
        elif isinstance(image, np.ndarray):
            img_np = image
            img = Image.fromarray(np.uint8(img_np)).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
            img_np = np.array(img)
        else:
            raise ValueError(f"지원되지 않는 이미지 타입: {type(image)}")
        
        # 브랜드 로고 감지
        brand, logo_confidence = self.detect_brand_logo(img_np)
        
        # 결과 초기화
        final_class = clip_class
        final_confidence = clip_confidence
        
        # 애플 제품 관련 클래스 목록
        apple_classes = ["iphone", "ipad", "macbook", "airpods", "earpods", "apple watch"]
        
        # 삼성 제품 관련 클래스 목록 (일반 + 폴더블)
        samsung_classes = [
            "samsung phone", "galaxy phone", "galaxy s", "galaxy note", "galaxy tab",
            "galaxy watch", "galaxy buds", "samsung tablet", "samsung earbuds",
            "galaxy z flip", "galaxy z fold", "samsung foldable phone", "foldable phone",
            "samsung folding phone"
        ]
        
        # 로고 감지 결과와 CLIP 결과 비교하여 최종 판단
        if brand == "apple" and logo_confidence > 0.5:
            # CLIP 결과가 애플 제품이 아니면 수정
            if not any(apple_term in str(clip_class).lower() for apple_term in apple_classes):
                logger.info(f"로고 감지: 애플 로고 감지됨 (신뢰도: {logo_confidence:.2f})")
                
                # 아이폰으로 기본 설정 (추가 분석으로 더 정확한 모델 예측 가능)
                final_class = "iphone"
                
                # 애플 로고가 명확하게 감지된 경우 높은 신뢰도 부여
                if logo_confidence > 0.7:
                    final_confidence = logo_confidence
                else:
                    final_confidence = (clip_confidence + logo_confidence) / 2
                    
        elif brand == "samsung" and logo_confidence > 0.5:
            # CLIP 결과가 삼성 제품이 아니면 수정
            if not any(samsung_term in str(clip_class).lower() for samsung_term in samsung_classes):
                logger.info(f"로고 감지: 삼성 로고 감지됨 (신뢰도: {logo_confidence:.2f})")
                
                # 이미지 형태 분석으로 폴더블 여부 판단
                width, height = img.size
                is_foldable = self.detect_fold_hinge(img_np)
                
                if is_foldable:
                    if width > height * 1.2:  # 가로가 세로보다 20% 이상 길면
                        final_class = "galaxy z fold"
                    else:
                        final_class = "galaxy z flip"
                else:
                    # 폴더블 특성이 없으면 일반 삼성폰
                    final_class = "samsung phone"
                
                # 삼성 로고가 명확하게 감지된 경우 높은 신뢰도 부여
                if logo_confidence > 0.7:
                    final_confidence = logo_confidence
                else:
                    final_confidence = (clip_confidence + logo_confidence) / 2
        
        # CLIP과 로고 감지 결과가 충돌하는 경우
        elif clip_class in apple_classes and brand == "samsung":
            # 신뢰도 비교하여 선택
            if logo_confidence > clip_confidence * 1.5:  # 로고 신뢰도가 훨씬 높으면
                logger.info(f"로고 감지(삼성)가 CLIP 결과(애플)보다 더 신뢰할 수 있습니다")
                final_class = "samsung phone"
                final_confidence = logo_confidence
            else:
                logger.info(f"CLIP 결과(애플)를 유지합니다")
                # CLIP 결과 유지
        
        elif clip_class in samsung_classes and brand == "apple":
            # 신뢰도 비교하여 선택
            if logo_confidence > clip_confidence * 1.5:  # 로고 신뢰도가 훨씬 높으면
                logger.info(f"로고 감지(애플)가 CLIP 결과(삼성)보다 더 신뢰할 수 있습니다")
                final_class = "iphone"  # 기본 아이폰으로 설정
                final_confidence = logo_confidence
            else:
                logger.info(f"CLIP 결과(삼성)를 유지합니다")
                # CLIP 결과 유지
        
        # 색상 분석 추가
        color_analysis = self.analyze_image_colors(img_np)
        
        # 결과 구성
        result = {
            'class': final_class,
            'confidence': final_confidence,
            'clip_classification': classification_results,
            'brand_detection': {'brand': brand, 'confidence': logo_confidence},
            'color_analysis': color_analysis,
            'is_foldable': self.detect_fold_hinge(img_np) if 'galaxy' in str(final_class).lower() else False
        }
        
        return result

if __name__ == "__main__":
    # 간단한 테스트
    import matplotlib.pyplot as plt
    
    clip_model = CLIPModel()
    
    # 테스트 이미지 경로 입력
    test_image_path = input("테스트할 이미지 경로를 입력하세요: ").strip()
    
    if test_image_path:
        # 이미지 분류 실행
        result = clip_model.advanced_classification(test_image_path)
        
        # 결과 출력
        print(f"\n최종 분류 결과: {result['class']}, 신뢰도: {result['confidence']*100:.2f}%")
        print("\n상세 CLIP 분류 결과:")
        for item in result['clip_classification']:
            print(f"  {item['class']}: {item['score']*100:.2f}%")
        
        if result['brand_detection']['brand']:
            print(f"\n브랜드 감지: {result['brand_detection']['brand']}, 신뢰도: {result['brand_detection']['confidence']*100:.2f}%")
        else:
            print("\n브랜드가 감지되지 않았습니다.")
        
        print("\n색상 분석:")
        for brand, score in result['color_analysis'].items():
            print(f"  {brand}: {score:.4f}")
        
        if 'galaxy' in str(result['class']).lower():
            print(f"\n폴더블 여부: {'폴더블 기기입니다' if result['is_foldable'] else '일반 기기입니다'}")
        
        # 이미지 표시
        img = Image.open(test_image_path).convert('RGB')
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"분류 결과: {result['class']} ({result['confidence']*100:.2f}%)")
        plt.axis('off')
        plt.show()
    else:
        print("이미지 경로가 입력되지 않았습니다. 프로그램을 종료합니다.")