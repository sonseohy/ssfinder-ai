import torch
import logging
from typing import List, Dict, Any, Union, Optional
from PIL import Image
import numpy as np
from lavis.models import load_model_and_preprocess
import re

from config.config import BLIP_MODEL_NAME, BLIP_MODEL_SIZE, DEVICE, ATTRIBUTE_QUESTIONS

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BLIPModel:
    """이미지 캡셔닝 및 VQA를 위한 BLIP 모델"""
    
    def __init__(self, model_name: str = BLIP_MODEL_NAME, model_size: str = BLIP_MODEL_SIZE, device: str = DEVICE):
        """
        BLIP 모델 초기화
        
        Args:
            model_name (str): 사용할 BLIP 모델 이름
            model_size (str): 모델 크기/버전
            device (str): 사용할 장치 (cuda 또는 cpu)
        """
        self.device = device
        self.model_name = model_name
        self.model_size = model_size
        
        logger.info(f"{device} 장치에 {model_name} BLIP 모델 로드 중")
        
        try:
            # 캡션 생성을 위한 모델 로드
            self.caption_model, self.vis_processors, _ = load_model_and_preprocess(
                name=f"{model_name}_caption", 
                model_type=model_size, 
                is_eval=True, 
                device=device
            )
            
            # VQA를 위한 모델 로드
            self.vqa_model, _, self.text_processors = load_model_and_preprocess(
                name=f"{model_name}_vqa", 
                model_type=model_size, 
                is_eval=True, 
                device=device
            )
            
            logger.info("BLIP 모델 로드 완료")
            
        except Exception as e:
            logger.error(f"BLIP 모델 로드 오류: {str(e)}")
            raise
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        이미지 전처리
        
        Args:
            image: 이미지 경로, PIL 이미지, 또는 numpy 배열
            
        Returns:
            torch.Tensor: 전처리된 이미지 텐서
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
        
        # BLIP 전처리 적용
        image_tensor = self.vis_processors["eval"](img).unsqueeze(0).to(self.device)
        return image_tensor
    
    def generate_caption(self, image: Union[str, Image.Image, np.ndarray], 
                        beam_size: int = 5, max_length: int = 50) -> str:
        """
        이미지에 대한 캡션 생성
        
        Args:
            image: 이미지 경로, PIL 이미지, 또는 numpy 배열
            beam_size (int): 빔 검색 크기
            max_length (int): 생성할 최대 토큰 수
            
        Returns:
            str: 생성된 이미지 캡션
        """
        image_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            caption = self.caption_model.generate({
                "image": image_tensor,
                "prompt": "a photo of"
            }, 
            use_nucleus_sampling=False, 
            num_beams=beam_size, 
            max_length=max_length)
        
        return caption[0]
    
    def answer_question(self, image: Union[str, Image.Image, np.ndarray], 
                        question: str, max_length: int = 50) -> str:
        """
        이미지에 대한 질문에 답변
        
        Args:
            image: 이미지 경로, PIL 이미지, 또는 numpy 배열
            question (str): 이미지에 대한 질문
            max_length (int): 생성할 최대 토큰 수
            
        Returns:
            str: 질문에 대한 답변
        """
        image_tensor = self.preprocess_image(image)
        
        # 텍스트 전처리
        text_input = self.text_processors["eval"](question)
        
        with torch.no_grad():
            answer = self.vqa_model.generate({
                "image": image_tensor,
                "text_input": text_input
            },
            use_nucleus_sampling=False,
            max_length=max_length)
        
        return answer[0]
    
    def extract_attributes(self, image: Union[str, Image.Image, np.ndarray]) -> Dict[str, Any]:
        """
        이미지에서 중요한 속성 추출 (색상, 브랜드, 재질 등)
        
        Args:
            image: 이미지 경로, PIL 이미지, 또는 numpy 배열
            
        Returns:
            Dict[str, Any]: 추출된 속성 정보
        """
        # 캡션 생성
        caption = self.generate_caption(image)
        
        attributes = {
            'caption': caption,
            'attributes': {}
        }
        
        # 각 속성 타입에 대해 질문하고 응답 저장
        for attr_type, questions in ATTRIBUTE_QUESTIONS.items():
            # 더 좋은 결과를 위해 여러 질문 시도
            answers = []
            for question in questions:
                answer = self.answer_question(image, question)
                answers.append(answer)
            
            # 응답 처리 및 정규화
            processed_answer = self._process_attribute_answers(answers, attr_type)
            attributes['attributes'][attr_type] = processed_answer
        
        return attributes
    
    def _process_attribute_answers(self, answers: List[str], attr_type: str) -> str:
        """
        VQA 응답을 처리하고 정규화
        
        Args:
            answers (List[str]): VQA 응답 목록
            attr_type (str): 속성 유형
            
        Returns:
            str: 처리된 속성 값
        """
        # 공통 패턴: "It is X" -> "X"
        processed_answers = []
        for answer in answers:
            # 응답 정규화
            answer = answer.strip().lower()
            
            # "I can't tell" 또는 "I don't know" 패턴 처리
            if any(phrase in answer for phrase in ["can't tell", "don't know", "can't see", "not visible", "unclear"]):
                processed_answers.append("unknown")
                continue
            
            # "It is", "This is", "The item is" 등의 패턴 제거
            patterns = [
                r'^it is (a |an )?(.*?)$',
                r'^this is (a |an )?(.*?)$',
                r'^the .* is (a |an )?(.*?)$',
                r'^i (can )?see (a |an )?(.*?)$',
                r'^looks like (a |an )?(.*?)$'
            ]
            
            replaced = False
            for pattern in patterns:
                match = re.search(pattern, answer)
                if match:
                    clean_answer = match.group(match.lastindex)
                    processed_answers.append(clean_answer)
                    replaced = True
                    break
            
            if not replaced:
                processed_answers.append(answer)
        
        # 속성 유형별 특수 처리
        if attr_type == 'color':
            # 색상 추출을 위한 추가 처리
            color_patterns = [
                r'(black|white|red|orange|yellow|green|blue|purple|pink|brown|gray|grey|silver|gold)',
                r'(검정|흰색|빨간|주황|노란|초록|녹색|파란|남색|보라|분홍|갈색|회색|은색|금색)'
            ]
            
            for answer in processed_answers:
                for pattern in color_patterns:
                    match = re.search(pattern, answer)
                    if match:
                        return match.group(1)
        
        elif attr_type == 'brand':
            # 브랜드 추출을 위한 추가 처리
            brand_patterns = [
                r'(apple|samsung|iphone|galaxy|macbook|lg|sony|airpods|애플|삼성|엘지)',
            ]
            
            for answer in processed_answers:
                for pattern in brand_patterns:
                    match = re.search(pattern, answer)
                    if match:
                        return match.group(1)
        
        # 가장 빈번한 응답 또는 첫 번째 의미 있는 응답 반환
        for answer in processed_answers:
            if answer != "unknown":
                return answer
        
        return "unknown"
    
    def generate_detailed_description(self, image: Union[str, Image.Image, np.ndarray]) -> str:
        """
        이미지에 대한 상세한 설명 생성
        
        Args:
            image: 이미지 경로, PIL 이미지, 또는 numpy 배열
            
        Returns:
            str: 상세 설명
        """
        # 기본 캡션 생성
        caption = self.generate_caption(image)
        
        # 추가 설명을 위한 질문
        detail_questions = [
            "What are the main features of this item?",
            "What is the condition of this item?",
            "Are there any distinctive marks or features on this item?",
            "What is the approximate size of this item?",
        ]
        
        details = []
        for question in detail_questions:
            answer = self.answer_question(image, question)
            
            # 의미 있는 응답만 포함
            if not any(phrase in answer.lower() for phrase in ["can't tell", "don't know", "can't see", "not visible"]):
                # "It is", "This is" 등의 패턴 제거
                cleaned_answer = re.sub(r'^(it is|this is|the item is) ', '', answer.strip(), flags=re.IGNORECASE)
                details.append(cleaned_answer)
        
        # 모든 정보 결합
        if details:
            full_description = f"{caption}. {' '.join(details)}"
        else:
            full_description = caption
            
        return full_description

if __name__ == "__main__":
    # 간단한 테스트
    import matplotlib.pyplot as plt
    
    blip_model = BLIPModel()
    
    # 테스트 이미지 경로 입력
    test_image_path = input("테스트할 이미지 경로를 입력하세요: ").strip()
    
    if test_image_path:
        # 이미지 로드 및 표시
        img = Image.open(test_image_path).convert('RGB')
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        
        # 캡션 생성
        caption = blip_model.generate_caption(img)
        print(f"\n생성된 캡션: {caption}")
        
        # 속성 추출
        attributes = blip_model.extract_attributes(img)
        print("\n추출된 속성:")
        for attr_type, value in attributes['attributes'].items():
            print(f"  {attr_type}: {value}")
        
        # 상세 설명 생성
        description = blip_model.generate_detailed_description(img)
        print(f"\n상세 설명: {description}")
        
        plt.title(caption)
        plt.show()
    else:
        print("이미지 경로가 입력되지 않았습니다. 프로그램을 종료합니다.")