"""
경량화된 한국어 CLIP 모델 구현
이 모듈은 HuggingFace의 한국어 CLIP 모델을 ONNX로 변환하여 경량화된 버전 제공
"""
import os
import sys
import logging
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import onnxruntime as ort
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 추가하여 config.py 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLIP_MODEL_NAME, DEVICE, ONNX_MODEL_PATH

class OptimizedKoreanCLIPModel:
    """
    ONNX로 최적화된 한국어 CLIP 모델 클래스
    텍스트와 이미지를 임베딩하고 유사도를 계산하는 기능 제공
    """
    
    def __init__(self, model_path=ONNX_MODEL_PATH, original_model_name=CLIP_MODEL_NAME):
        """
        ONNX CLIP 모델 초기화
        
        Args:
            model_path (str): ONNX 모델 파일 경로 (디렉토리)
            original_model_name (str): 원본 HuggingFace 모델 이름 (토크나이저용)
        """
        self.model_path = model_path
        self.original_model_name = original_model_name
        
        # 모델 파일 경로
        self.text_model_path = os.path.join(model_path, "text_encoder.onnx")
        self.image_model_path = os.path.join(model_path, "image_encoder.onnx")
        
        logger.info(f"ONNX CLIP 모델 로드 중 (모델 경로: {model_path})...")
        
        try:
            # ONNX 런타임 세션 생성
            # 세션 최적화 옵션 설정
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.intra_op_num_threads = 4  # 병렬 처리 스레드 수 설정
            
            # ONNX 런타임 세션 생성
            self.text_session = ort.InferenceSession(
                self.text_model_path, 
                sess_options=session_options,
                providers=['CPUExecutionProvider']
            )
            
            self.image_session = ort.InferenceSession(
                self.image_model_path, 
                sess_options=session_options,
                providers=['CPUExecutionProvider']
            )
            
            # 원본 모델의 프로세서 로드 (토크나이징 및 이미지 전처리용)
            self.processor = CLIPProcessor.from_pretrained(original_model_name)
            
            logger.info("ONNX CLIP 모델 로드 완료")
        except Exception as e:
            logger.error(f"ONNX CLIP 모델 로드 실패: {str(e)}")
            raise
    
    def encode_text(self, text):
        """
        텍스트를 임베딩 벡터로 변환
        
        Args:
            text (str or list): 인코딩할 텍스트 또는 텍스트 리스트
            
        Returns:
            numpy.ndarray: 임베딩 벡터
        """
        if isinstance(text, str):
            text = [text]
            
        try:
            start_time = time.time()
            
            # 텍스트 전처리 (토크나이징)
            inputs = self.processor(text=text, return_tensors="np", padding=True, truncation=True)
            
            # ONNX 세션으로 추론
            text_features = self.text_session.run(
                ['output'], 
                {
                    'input_ids': inputs.input_ids,
                    'attention_mask': inputs.attention_mask
                }
            )[0]
            
            # 임베딩 정규화
            text_embeddings = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
            
            process_time = time.time() - start_time
            logger.debug(f"텍스트 인코딩 시간: {process_time:.4f}초")
            
            return text_embeddings
        except Exception as e:
            logger.error(f"텍스트 인코딩 중 오류 발생: {str(e)}")
            return np.zeros((len(text), 512))  # 적절한 임베딩 차원으로 수정
            
    def encode_image(self, image_source):
        """
        이미지를 임베딩 벡터로 변환
        
        Args:
            image_source: 인코딩할 이미지 (PIL Image, URL 또는 이미지 경로)
            
        Returns:
            numpy.ndarray: 임베딩 벡터
        """
        try:
            start_time = time.time()
            
            # 이미지 로드 (URL, 파일 경로 또는 PIL 이미지 객체)
            if isinstance(image_source, str):
                if image_source.startswith('http'):
                    # URL에서 이미지 로드
                    response = requests.get(image_source)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                else:
                    # 로컬 파일에서 이미지 로드
                    image = Image.open(image_source).convert('RGB')
            else:
                # 이미 PIL 이미지 객체인 경우
                image = image_source.convert('RGB')
            
            # 이미지 전처리
            inputs = self.processor(images=image, return_tensors="np")
            
            # ONNX 세션으로 추론
            image_features = self.image_session.run(
                ['output'], 
                {'pixel_values': inputs.pixel_values}
            )[0]
            
            # 임베딩 정규화
            image_embeddings = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
            
            process_time = time.time() - start_time
            logger.debug(f"이미지 인코딩 시간: {process_time:.4f}초")
            
            return image_embeddings
        except Exception as e:
            logger.error(f"이미지 인코딩 중 오류 발생: {str(e)}")
            return np.zeros((1, 512))  # 적절한 임베딩 차원으로 수정
    
    def calculate_similarity(self, text_embedding, image_embedding=None):
        """
        텍스트와 이미지 임베딩 간의 유사도 계산
        
        Args:
            text_embedding (numpy.ndarray): 텍스트 임베딩
            image_embedding (numpy.ndarray, optional): 이미지 임베딩 (없으면 텍스트만 비교)
            
        Returns:
            float: 유사도 점수 (0~1 사이)
        """
        if image_embedding is None:
            # 텍스트-텍스트 유사도 계산 (코사인 유사도)
            similarity = np.dot(text_embedding, text_embedding.T)[0, 0]
        else:
            # 텍스트-이미지 유사도 계산 (코사인 유사도)
            similarity = np.dot(text_embedding, image_embedding.T)[0, 0]
            
        # 유사도를 0~1 범위로 정규화
        similarity = (similarity + 1) / 2
        return float(similarity)
        
    def encode_batch_texts(self, texts, batch_size=32):
        """
        여러 텍스트를 배치로 임베딩
        
        Args:
            texts (list): 텍스트 목록
            batch_size (int): 배치 크기
            
        Returns:
            numpy.ndarray: 임베딩 벡터 배열
        """
        all_embeddings = []
        
        # 배치 처리
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.encode_text(batch_texts)
            all_embeddings.append(batch_embeddings)
            
        # 모든 배치 결과 병합
        if all_embeddings:
            return np.vstack(all_embeddings)
        else:
            return np.array([])

# ONNX 변환 함수 
def convert_clip_to_onnx(model_name=CLIP_MODEL_NAME, output_dir=ONNX_MODEL_PATH):
    """
    CLIP 모델을 ONNX 형식으로 변환
    
    Args:
        model_name (str): HuggingFace 모델 이름
        output_dir (str): 출력 디렉토리 경로
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logger.info(f"CLIP 모델 '{model_name}'을 ONNX로 변환 중...")
    
    try:
        # 원본 모델 로드
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        
        # 텍스트 인코더 변환
        text_model_path = os.path.join(output_dir, "text_encoder.onnx")
        
        # 더미 입력 생성
        dummy_text = ["텍스트 인코더 변환용 더미 텍스트"]
        dummy_inputs = processor(text=dummy_text, return_tensors="pt", padding=True, truncation=True)
        
        # ONNX 내보내기 (텍스트 인코더)
        dynamic_axes = {
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'output': {0: 'batch_size'}
        }
        
        torch.onnx.export(
            model.text_model,
            (dummy_inputs.input_ids, dummy_inputs.attention_mask),
            text_model_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=12,
            do_constant_folding=True
        )
        
        logger.info(f"텍스트 인코더 ONNX 변환 완료: {text_model_path}")
        
        # 이미지 인코더 변환
        image_model_path = os.path.join(output_dir, "image_encoder.onnx")
        
        # 더미 이미지 입력 생성
        dummy_image = Image.new('RGB', (224, 224))
        dummy_inputs = processor(images=dummy_image, return_tensors="pt")
        
        # ONNX 내보내기 (이미지 인코더)
        dynamic_axes = {
            'pixel_values': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        
        torch.onnx.export(
            model.vision_model,
            dummy_inputs.pixel_values,
            image_model_path,
            input_names=['pixel_values'],
            output_names=['output'],
            dynamic_axes=dynamic_axes,
            opset_version=12,
            do_constant_folding=True
        )
        
        logger.info(f"이미지 인코더 ONNX 변환 완료: {image_model_path}")
        return True
    except Exception as e:
        logger.error(f"ONNX 변환 중 오류 발생: {str(e)}")
        return False

# 모델 성능 테스트 함수
def benchmark_models(original_model, optimized_model, num_iterations=10):
    """
    원본 모델과 최적화된 모델의 성능 비교
    
    Args:
        original_model: 원본 CLIP 모델 인스턴스
        optimized_model: 최적화된 CLIP 모델 인스턴스
        num_iterations: 테스트 반복 횟수
    """
    test_text = "검은색 지갑을 잃어버렸습니다. 현금과 카드가 들어있어요."
    
    # 원본 모델 시간 측정
    original_times = []
    for _ in range(num_iterations):
        start_time = time.time()
        original_model.encode_text(test_text)
        original_times.append(time.time() - start_time)
    
    original_avg_time = sum(original_times) / len(original_times)
    
    # 최적화 모델 시간 측정
    optimized_times = []
    for _ in range(num_iterations):
        start_time = time.time()
        optimized_model.encode_text(test_text)
        optimized_times.append(time.time() - start_time)
    
    optimized_avg_time = sum(optimized_times) / len(optimized_times)
    
    # 결과 출력
    logger.info(f"원본 모델 평균 처리 시간: {original_avg_time:.4f}초")
    logger.info(f"최적화 모델 평균 처리 시간: {optimized_avg_time:.4f}초")
    logger.info(f"속도 향상: {original_avg_time / optimized_avg_time:.2f}배")
    
    return original_avg_time, optimized_avg_time

# 모듈 테스트용 코드
if __name__ == "__main__":
    # ONNX 변환
    success = convert_clip_to_onnx()
    
    if success:
        logger.info("ONNX 변환 성공!")
        
        # 원본 모델과 최적화 모델 로드
        from clip_model import KoreanCLIPModel
        
        original_model = KoreanCLIPModel()
        optimized_model = OptimizedKoreanCLIPModel()
        
        # 성능 비교
        benchmark_models(original_model, optimized_model)
        
        # 샘플 텍스트 인코딩
        sample_text = "검은색 지갑을 잃어버렸습니다. 현금과 카드가 들어있어요."
        text_embedding = optimized_model.encode_text(sample_text)
        
        print(f"텍스트 임베딩 shape: {text_embedding.shape}")
        
        # 유사도 계산 (텍스트만)
        sample_text2 = "검은색 지갑을 찾았습니다. 안에 현금과 카드가 있습니다."
        text_embedding2 = optimized_model.encode_text(sample_text2)
        
        similarity = optimized_model.calculate_similarity(text_embedding, text_embedding2)
        print(f"텍스트 간 유사도: {similarity:.4f}")