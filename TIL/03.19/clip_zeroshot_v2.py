# 필요한 라이브러리 임포트
import torch
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
from transformers import AutoProcessor, AutoModel

class KoreanLostItemSimilaritySystem:
    def __init__(self, model_name="Bingsu/clip-vit-large-patch14-ko"):
        """
        한국어에 최적화된 분실물 게시글 유사도 판별 시스템 초기화
        
        Args:
            model_name: 사용할 한국어 CLIP 모델 이름 
                      (기본값: KIMGEONUNG/clipse-ko-v1, 한국어 최적화 모델)
                      
        다른 한국어 CLIP 모델 옵션:
        - navervision/tvqa-ko-clip-base: 한국어-비전 멀티모달 모델
        - klue/roberta-large: 한국어 텍스트 모델(이미지만 별도 처리 필요)
        """
        # GPU 사용 가능 여부 확인 및 디바이스 설정
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용 중인 디바이스: {self.device}")
        
        # 한국어 CLIP 모델과 프로세서 로드
        print(f"한국어 최적화 모델 로딩 중: {model_name}")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(model_name)
        print("모델 로딩 완료")
        
    def get_post_embedding(self, post_data):
        """
        게시글 데이터에서 임베딩 추출 - 한국어 최적화 모델 사용
        
        Args:
            post_data: 게시글 데이터 (딕셔너리 형태)
                {
                    "image_path": "이미지 경로 또는 None",
                    "description": "상세 설명 텍스트",
                    "category": "물품분류",
                    "storage_location": "보관장소",
                    "condition": "물품상태",
                    "found_location": "습득장소"
                }
        
        Returns:
            게시글 임베딩 벡터 (numpy array)
        """
        # 게시글의 텍스트 정보를 하나의 문자열로 결합 (한국어)
        text_content = f"분류: {post_data.get('category', '')}. "
        text_content += f"설명: {post_data.get('description', '')}. "
        text_content += f"보관장소: {post_data.get('storage_location', '')}. "
        text_content += f"상태: {post_data.get('condition', '')}. "
        text_content += f"습득장소: {post_data.get('found_location', '')}"
        
        # 이미지가 제공된 경우
        if post_data.get('image_path'):
            try:
                # 이미지 파일 열기
                image = Image.open(post_data['image_path'])
                
                # 이미지와 텍스트를 함께 처리하여 CLIP 입력으로 변환
                # 한국어 최적화 모델에 맞는 프로세서 사용
                inputs = self.processor(
                    text=text_content,
                    images=image,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                # 모델 추론 (그래디언트 계산 없이)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # 한국어 CLIP 모델의 출력 형식에 따라 임베딩 추출
                    # KIMGEONUNG/clipse-ko-v1 모델의 경우 text_embeds, image_embeds 추출
                    text_embedding = outputs.text_embeds.cpu().numpy()[0]
                    image_embedding = outputs.image_embeds.cpu().numpy()[0]
                    
                    # 가중치를 사용하여 결합 (텍스트:이미지 = 3:7)
                    # 한국어 이미지 정보에 더 높은 가중치 부여 (한국어 처리에 최적화)
                    combined_embedding = 0.3 * text_embedding + 0.7 * image_embedding
                    return combined_embedding
            except Exception as e:
                print(f"이미지 처리 중 오류 발생: {e}")
                # 이미지 처리 실패 시 텍스트만 사용
                
        # 이미지가 없거나 처리 실패 시 텍스트만 사용
        inputs = self.processor(
            text=text_content,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # 텍스트 데이터만으로 임베딩 생성
        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)
            text_embedding = outputs.cpu().numpy()[0]
            return text_embedding
    
    def calculate_similarity(self, post1, post2):
        """
        두 게시글 간의 유사도 계산 - 이미지 카테고리 검사 추가
        
        Args:
            post1, post2: 게시글 데이터
            
        Returns:
            유사도 점수 (0~1 사이 값, 1에 가까울수록 유사)
        """
        # 둘 다 이미지가 있는 경우, 이미지 유사도 먼저 계산
        if post1.get('image_path') and post2.get('image_path'):
            try:
                # 이미지만 추출하여 유사도 계산
                image1 = Image.open(post1['image_path'])
                image2 = Image.open(post2['image_path'])
                
                inputs1 = self.processor(images=image1, return_tensors="pt").to(self.device)
                inputs2 = self.processor(images=image2, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    image_embedding1 = self.model.get_image_features(**inputs1).cpu().numpy()[0]
                    image_embedding2 = self.model.get_image_features(**inputs2).cpu().numpy()[0]
                    
                    # 이미지 유사도 계산
                    image_similarity = cosine_similarity([image_embedding1], [image_embedding2])[0][0]
                    
                    # 이미지 유사도가 매우 낮으면 (다른 카테고리로 판단) 낮은 유사도 반환
                    if image_similarity < 0.2:  # 이미지 카테고리 임계값
                        return 0.0
            except Exception as e:
                print(f"이미지 유사도 계산 중 오류: {e}")
        
        # 일반적인 유사도 계산 (텍스트+이미지 결합)
        embedding1 = self.get_post_embedding(post1)
        embedding2 = self.get_post_embedding(post2)
        
        # 카테고리 일치 여부 확인 (텍스트 기반)
        if post1.get('category') and post2.get('category') and post1['category'] != post2['category']:
            return 0.1  # 카테고리가 다르면 낮은 유사도
        
        # 코사인 유사도 계산
        similarity = cosine_similarity([embedding1], [embedding2])[0][0]
        return similarity
    
    def find_similar_posts(self, new_post, existing_posts, threshold=0.65):
        """
        기존 게시글 중 새 게시글과 유사한 게시글 찾기
        
        Args:
            new_post: 새 게시글 데이터
            existing_posts: 기존 게시글 리스트
            threshold: 유사도 임계값 (이 값 이상인 게시글만 반환)
                     한국어 모델은 영어 모델보다 약간 낮은 임계값 사용 권장
            
        Returns:
            유사도 높은 게시글과 유사도 점수 리스트
        """
        # 새 게시글의 임베딩 계산
        new_embedding = self.get_post_embedding(new_post)
        results = []
        
        # 기존 게시글들과 유사도 비교
        for idx, post in enumerate(existing_posts):
            post_embedding = self.get_post_embedding(post)
            similarity = cosine_similarity([new_embedding], [post_embedding])[0][0]
            
            # 임계값 이상인 게시글만 결과에 추가
            if similarity >= threshold:
                results.append({
                    "post_id": idx,
                    "post_data": post,
                    "similarity_score": float(similarity)
                })
        
        # 유사도 내림차순 정렬 (가장 유사한 게시글이 먼저 오도록)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results


# 실행 예시
if __name__ == "__main__":
    # 시작 시간 기록
    start_time = time.time()
    
    # 한국어 최적화 시스템 초기화
    system = KoreanLostItemSimilaritySystem()
    
    # 테스트용 한국어 게시글 데이터
    existing_posts = [
        {
            "image_path": "sample_images/wallet1.jpg",
            "description": "검은색 루이비통 장지갑",
            "category": "지갑",
            "storage_location": "쌍문파출소",
            "condition": "보관중",
            "found_location": "세그루패션고등학교"
        },
        {
            "image_path": None,
            "description": "파란색 우산, 자동 접이식",
            "category": "우산",
            "storage_location": "학생회관",
            "condition": "양호",
            "found_location": "공학관 로비"
        }
    ]
    
    new_post = {
        "image_path": "sample_images/memo.jpg",
        "description": "가죽소재 수첩",
        "category": "수첩",
        "condition": "분실",
        "found_location": "도서관"
    }
    
    # 유사한 게시글 찾기
    similar_posts = system.find_similar_posts(new_post, existing_posts, threshold=0.65) # 0.65 이상일 때 유사 게시글로 판단
    
    # 결과 출력
    print(f"검색 기준 게시글: {new_post['category']} - {new_post['description']}")
    print("-" * 50)
    
    if not similar_posts:
        print("유사한 게시글을 찾을 수 없습니다.")
    else:
        print(f"유사한 게시글 {len(similar_posts)}개 발견:")
        for result in similar_posts:
            print(f"유사도: {result['similarity_score']:.4f}")
            print(f"게시글 ID: {result['post_id']}")
            print(f"카테고리: {result['post_data']['category']}")
            print(f"설명: {result['post_data']['description']}")
            print(f"습득장소: {result['post_data']['found_location']}")
            print("-" * 40)
    
    # 실행 시간 출력
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"실행 시간: {execution_time:.2f}초")
    print(f"사용 디바이스: {system.device}")