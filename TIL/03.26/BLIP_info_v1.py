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
        
        # 카테고리 정의
        self.categories = [
            "전자기기", "의류", "가방", "지갑", "귀금속", "카드", "증명서", "컴퓨터", "현금", "휴대폰",
            "우산", "화장품", "스포츠용품", "도서용품", "기타", "서류", "산업용품", "쇼핑백", "악기", "자동차", "기타물품"
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
                top_p=0.9
            )
        
        # 토큰을 텍스트로 디코딩
        answer = self.vqa_processor.decode(out[0], skip_special_tokens=True)
        return answer

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
        
        # 다양한 질문으로 특징 추출
        questions = {
            "category": "이 물건은 어떤 종류의 물건인가요? 다음 중에서 선택해주세요: " + ", ".join(self.categories),
            "color": "이 물건의 주요 색상은 무엇인가요?",
            "material": "이 물건은 어떤 재질로 만들어졌나요?",
            "brand": "이 물건의 브랜드나 제조사가 보인다면 무엇인가요?",
            "distinctive_features": "이 물건의 특징적인 부분이나 특이사항은 무엇인가요?",
            "size": "이 물건의 크기는 어느 정도인가요? 작은/중간/큰 중에서 선택해주세요.",
        }
        
        # 각 질문에 대한 답변 수집
        answers = {}
        for key, question in questions.items():
            print(f"{key} 분석 중...")
            answers[key] = self.ask_question(image, question)
        
        # 기본 설명 생성
        description = self._generate_description(caption, answers)
        
        # 제목 생성
        title = self._generate_title(answers)
        
        # 결과 구조화
        result = {
            "caption": caption,
            "title": title,
            "description": description,
            "category": answers["category"],
            "color": answers["color"],
            "material": answers["material"],
            "brand": answers["brand"] if "없음" not in answers["brand"].lower() else "",
            "size": answers["size"],
            "distinctive_features": answers["distinctive_features"],
            "raw_answers": answers  # 디버깅 및 추가 분석용
        }
        
        return result
    
    def _generate_title(self, answers: Dict[str, str]) -> str:
        """
        답변을 기반으로 게시글 제목 생성
        
        Args:
            answers: 질문별 답변 딕셔너리
            
        Returns:
            생성된 제목
        """
        category = answers["category"]
        color = answers["color"]
        
        # 제목 생성
        if category in self.categories:
            title = f"{color} {category}"
        else:
            title = f"{color} 물건"
            
        # 브랜드 추가 (있는 경우)
        if answers["brand"] and "없음" not in answers["brand"].lower():
            title = f"{answers['brand']} {title}"
            
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
        if answers["material"] and "모르" not in answers["material"].lower():
            description += f"재질: {answers['material']}\n"
            
        if answers["size"]:
            description += f"크기: {answers['size']}\n"
            
        if answers["distinctive_features"] and "없" not in answers["distinctive_features"].lower():
            description += f"특이사항: {answers['distinctive_features']}\n"
            
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
    image_path = "mywaller.jpg"
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