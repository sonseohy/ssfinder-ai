import json
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util
from ultralytics import YOLO

class LostItemMatcher:
    def __init__(self, model_path=None):
        """
        분실물-습득물 매칭 모델 초기화
        
        Args:
            model_path: 사전 학습된 모델이 있는 경우 해당 경로
        """
        print("모델 초기화 중...")
        
        # CLIP 모델 로드 (이미지-텍스트 멀티모달 모델)
        print("CLIP 모델 로드 중...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # YOLO 모델 로드 (객체 인식 모델)
        print("YOLO 모델 로드 중...")
        self.yolo_model = YOLO("yolov8n.pt")  # 기본 모델로 시작
        
        # 한국어 텍스트 임베딩 모델
        print("한국어 텍스트 임베딩 모델 로드 중...")
        # 실제 한국어 모델 사용 (KoSBERT, KoBERT 등)
        self.text_model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
        
        # 카테고리 매핑을 위한 임베딩 생성
        self.category_embeddings = {}
        print("카테고리 임베딩 생성 중...")
        self._initialize_category_embeddings()
        
        # 유사도 계산 가중치 (필요에 따라 조정)
        self.weights = {
            "image": 0.4,       # 이미지 유사도
            "category": 0.25,   # 카테고리 유사도
            "color": 0.15,      # 색상 유사도
            "text": 0.2         # 텍스트 설명 유사도
        }
        
        # 임계값 설정 (이 값 이상의 유사도를 가진 항목만 매칭 결과로 반환)
        self.similarity_threshold = 0.6
        
        print("모델 초기화 완료")
    
    def _initialize_category_embeddings(self):
        """카테고리 임베딩 초기화"""
        # 이미지에서 확인된 카테고리 구조
        categories = {
            "가방": ["여성용가방", "남성용가방", "기타가방"],
            "귀금속": ["반지 목걸이", "귀걸이 시계", "기타"],
            "도서용품": ["학습서적 소설", "컴퓨터서적", "만화책 기타서적"],
            "서류": ["서류 기타문류"],
            "산업용품": ["기타용품"],
            "쇼핑백": ["쇼핑백"],
            "스포츠용품": ["스포츠용품"],
            "악기": ["건반악기 관악기", "타악기 현악기", "기타악기"],
            "유가증권": ["여행 상품권", "채권 기타"],
            "의류": ["여성의류", "남성의류", "아기의류 모자", "신발 기타의류"],
            "자동차": ["자동차열쇠", "네비게이션", "자동차번호판", "임시번호판", "기타용품"],
            "전자기기": ["태블릿", "스마트워치", "무선이어폰", "카메라 기타용품"],
            "지갑": ["여성용 지갑", "남성용 지갑", "기타 지갑"],
            "증명서": ["신분증 면허증", "여권 기타"],
            "컴퓨터": ["삼성노트북", "LG노트북", "애플노트북 기타"],
            "카드": ["신용(체크)카드", "일반카드", "교통카드", "기타카드"],
            "현금": ["현금 수표 외화", "기타"],
            "휴대폰": ["삼성휴대폰", "LG휴대폰", "아이폰", "기타휴대폰", "기타통신기기"],
            "기타물품": ["안경 선글라스", "매장문화재 기타"],
            "유류품": ["무연고유류품", "유류품"]
        }
        
        # 메인 카테고리 임베딩
        for main_category in categories.keys():
            self.category_embeddings[main_category] = self.text_model.encode(main_category)
            
            # 서브 카테고리 임베딩
            for sub_category in categories[main_category]:
                embedding_key = f"{main_category}_{sub_category}"
                self.category_embeddings[embedding_key] = self.text_model.encode(sub_category)
        
        # 색상 임베딩
        colors = ["검정색", "흰색", "회색", "빨간색", "파란색", "녹색", "노란색", "갈색", "분홍색", "보라색", "주황색", "기타"]
        for color in colors:
            self.category_embeddings[f"color_{color}"] = self.text_model.encode(color)
    
    def preprocess_image(self, image_path):
        """이미지 전처리 및 특징 추출"""
        if not image_path or not os.path.exists(image_path):
            return None
        
        try:
            # 이미지 로드
            image = Image.open(image_path).convert("RGB")
            
            # CLIP 모델을 사용한 이미지 특징 추출
            clip_inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**clip_inputs)
            
            # YOLO 모델을 사용한 객체 탐지
            yolo_results = self.yolo_model(image)
            # 객체 클래스 ID와 신뢰도 점수 추출
            detected_objects = []
            for result in yolo_results:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    confidence = box.conf.item()
                    detected_objects.append((class_id, confidence))
            
            return {
                "clip_features": image_features.cpu().numpy(),
                "yolo_objects": detected_objects
            }
        except Exception as e:
            print(f"이미지 처리 중 오류 발생: {e}")
            return None
    
    def preprocess_text(self, item_data):
        """텍스트 데이터 전처리 및 특징 추출"""
        # 카테고리 정보
        main_category = item_data.get("main_category", "")
        subcategory = item_data.get("subcategory", "")
        
        # 색상 정보
        color = item_data.get("color", "")
        
        # 설명 텍스트
        description = item_data.get("description", "")
        
        # 텍스트 임베딩 생성
        category_embedding = self.category_embeddings.get(main_category, 
                                                          self.text_model.encode(main_category))
        
        subcategory_key = f"{main_category}_{subcategory}"
        subcategory_embedding = self.category_embeddings.get(subcategory_key, 
                                                            self.text_model.encode(subcategory))
        
        color_key = f"color_{color}"
        color_embedding = self.category_embeddings.get(color_key, 
                                                      self.text_model.encode(color))
        
        description_embedding = self.text_model.encode(description)
        
        return {
            "category_embedding": category_embedding,
            "subcategory_embedding": subcategory_embedding,
            "color_embedding": color_embedding,
            "description_embedding": description_embedding
        }
    
    def extract_features(self, item_data):
        """아이템에서 특징 추출"""
        features = {
            "text_features": self.preprocess_text(item_data),
            "image_features": None
        }
        
        # 이미지 처리 (있는 경우)
        image_path = item_data.get("image_path")
        if image_path:
            features["image_features"] = self.preprocess_image(image_path)
        
        return features
    
    def calculate_similarity(self, lost_features, found_features):
        """두 아이템 간의 유사도 계산"""
        similarity_scores = {}
        
        # 1. 이미지 유사도 (두 항목 모두 이미지가 있는 경우)
        if (lost_features["image_features"] is not None and 
            found_features["image_features"] is not None):
            
            lost_img = lost_features["image_features"]["clip_features"]
            found_img = found_features["image_features"]["clip_features"]
            
            # 코사인 유사도 계산
            img_similarity = np.dot(lost_img, found_img.T) / (
                np.linalg.norm(lost_img) * np.linalg.norm(found_img)
            )
            
            similarity_scores["image"] = float(img_similarity)
        else:
            similarity_scores["image"] = 0.0
        
        # 2. 카테고리 유사도
        category_similarity = np.dot(
            lost_features["text_features"]["category_embedding"],
            found_features["text_features"]["category_embedding"]
        ) / (
            np.linalg.norm(lost_features["text_features"]["category_embedding"]) *
            np.linalg.norm(found_features["text_features"]["category_embedding"])
        )
        
        subcategory_similarity = np.dot(
            lost_features["text_features"]["subcategory_embedding"],
            found_features["text_features"]["subcategory_embedding"]
        ) / (
            np.linalg.norm(lost_features["text_features"]["subcategory_embedding"]) *
            np.linalg.norm(found_features["text_features"]["subcategory_embedding"])
        )
        
        # 카테고리와 서브카테고리 유사도의 평균
        similarity_scores["category"] = float((category_similarity + subcategory_similarity) / 2)
        
        # 3. 색상 유사도
        color_similarity = np.dot(
            lost_features["text_features"]["color_embedding"],
            found_features["text_features"]["color_embedding"]
        ) / (
            np.linalg.norm(lost_features["text_features"]["color_embedding"]) *
            np.linalg.norm(found_features["text_features"]["color_embedding"])
        )
        
        similarity_scores["color"] = float(color_similarity)
        
        # 4. 설명 텍스트 유사도
        description_similarity = np.dot(
            lost_features["text_features"]["description_embedding"],
            found_features["text_features"]["description_embedding"]
        ) / (
            np.linalg.norm(lost_features["text_features"]["description_embedding"]) *
            np.linalg.norm(found_features["text_features"]["description_embedding"])
        )
        
        similarity_scores["text"] = float(description_similarity)
        
        # 가중치 적용 및 최종 유사도 계산
        weighted_sum = 0
        weight_sum = 0
        
        for key, score in similarity_scores.items():
            if key == "image" and score == 0:
                # 이미지가 없는 경우 다른 항목에 가중치 재분배
                continue
            
            weighted_sum += score * self.weights[key]
            weight_sum += self.weights[key]
        
        # 총 가중치가 0이 되는 것 방지
        if weight_sum == 0:
            return 0
        
        final_similarity = weighted_sum / weight_sum
        return final_similarity, similarity_scores
    
    def find_similar_items(self, lost_item, found_items, top_k=5):
        """분실물과 유사한 습득물 찾기"""
        print(f"분실물 '{lost_item.get('main_category')}' 관련 습득물 검색 중...")
        
        # 분실물 특징 추출
        lost_features = self.extract_features(lost_item)
        
        similar_items = []
        
        # 모든 습득물에 대해 유사도 계산
        for found_item in tqdm(found_items):
            found_features = self.extract_features(found_item)
            
            final_similarity, detailed_scores = self.calculate_similarity(
                lost_features, found_features
            )
            
            # 임계값 이상인 경우만 결과에 추가
            if final_similarity >= self.similarity_threshold:
                similar_items.append({
                    "item": found_item,
                    "similarity": final_similarity,
                    "detailed_scores": detailed_scores
                })
        
        # 유사도 내림차순 정렬
        similar_items.sort(key=lambda x: x["similarity"], reverse=True)
        
        # 상위 K개 항목 반환
        return similar_items[:top_k]
    
    def save_model(self, path="saved_model"):
        """모델 저장 (가중치와 설정)"""
        os.makedirs(path, exist_ok=True)
        
        # 가중치와 임계값 저장
        config = {
            "weights": self.weights,
            "similarity_threshold": self.similarity_threshold
        }
        
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f)
        
        print(f"모델 설정이 {path}에 저장되었습니다.")
    
    def load_model(self, path="saved_model"):
        """저장된 모델 로드"""
        config_path = os.path.join(path, "config.json")
        
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            
            self.weights = config["weights"]
            self.similarity_threshold = config["similarity_threshold"]
            
            print(f"{path}에서 모델 설정을 로드했습니다.")
        else:
            print(f"경고: {config_path}에서 모델 설정을 찾을 수 없습니다.")


# 테스트 함수
def test_model():
    """테스트 데이터로 모델 테스트"""
    # 테스트 데이터 로드
    data_dir = Path("test_data")
    
    try:
        with open(data_dir / "found_items.json", "r", encoding="utf-8") as f:
            found_items = json.load(f)
        
        with open(data_dir / "lost_items.json", "r", encoding="utf-8") as f:
            lost_items = json.load(f)
    except FileNotFoundError:
        print("테스트 데이터를 먼저 생성해주세요.")
        return
    
    # 모델 초기화
    matcher = LostItemMatcher()
    
    # 테스트할 분실물 선택
    test_lost_item = lost_items[0]
    print(f"\n테스트 분실물: {test_lost_item['main_category']} - {test_lost_item['description']}")
    
    # 유사한 습득물 찾기
    similar_items = matcher.find_similar_items(test_lost_item, found_items)
    
    # 결과 출력
    print("\n=== 유사 습득물 검색 결과 ===")
    for i, item in enumerate(similar_items, 1):
        print(f"\n{i}. 유사도: {item['similarity']:.4f}")
        print(f"   카테고리: {item['item']['main_category']} ({item['item']['subcategory']})")
        print(f"   색상: {item['item']['color']}")
        print(f"   설명: {item['item']['description']}")
        
        # 상세 점수 출력
        print("   상세 점수:")
        for key, score in item["detailed_scores"].items():
            print(f"   - {key}: {score:.4f}")
    
    return matcher, similar_items

if __name__ == "__main__":
    test_model()