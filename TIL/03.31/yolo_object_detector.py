import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
from typing import Dict, List, Optional, Tuple, Union
import urllib.request
import base64
from io import BytesIO

class ObjectDetector:
    """
    YOLOv8를 사용한 객체 탐지 클래스
    """
    def __init__(self, model_path: str = "yolov8m-oiv7.pt", confidence_threshold: float = 0.25):
        """
        객체 탐지 모델 초기화
        
        Args:
            model_path: YOLOv8 모델 파일 경로
            confidence_threshold: 객체 탐지 신뢰도 임계값
        """
        # 모델 파일 확인 및 다운로드
        if not os.path.exists(model_path):
            print(f"모델 파일 {model_path}이 존재하지 않습니다. 기본 모델을 사용합니다.")
            model_path = "yolov8m.pt"  # 기본 모델로 대체

        # 모델 로드
        try:
            self.model = YOLO(model_path)
            self.confidence_threshold = confidence_threshold
            print(f"YOLOv8 모델 로드 완료: {model_path}")
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {str(e)}")
            raise

        # 장치 설정 (CUDA 사용 가능한 경우 GPU 사용)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"사용 장치: {self.device}")

    def _load_image(self, image_path_or_data: Union[str, bytes, Image.Image]) -> Optional[Image.Image]:
        """
        다양한 형태의 이미지 데이터를 로드
        
        Args:
            image_path_or_data: 이미지 파일 경로, URL, base64 인코딩 문자열, 또는 PIL 이미지 객체
            
        Returns:
            PIL.Image.Image 또는 이미지 로드 실패 시 None
        """
        try:
            # 이미 PIL Image 객체인 경우
            if isinstance(image_path_or_data, Image.Image):
                return image_path_or_data

            # 문자열인 경우 (파일 경로 또는 URL 또는 base64)
            if isinstance(image_path_or_data, str):
                # URL인 경우
                if image_path_or_data.startswith(('http://', 'https://')):
                    with urllib.request.urlopen(image_path_or_data) as response:
                        image_data = response.read()
                    return Image.open(BytesIO(image_data)).convert('RGB')
                
                # Base64 인코딩 문자열인 경우
                elif image_path_or_data.startswith('data:image'):
                    # Base64 접두사 제거
                    image_data = image_path_or_data.split(',')[1]
                    image_bytes = base64.b64decode(image_data)
                    return Image.open(BytesIO(image_bytes)).convert('RGB')
                
                # 파일 경로인 경우
                else:
                    if os.path.exists(image_path_or_data):
                        return Image.open(image_path_or_data).convert('RGB')
                    else:
                        print(f"파일이 존재하지 않습니다: {image_path_or_data}")
                        return None
            
            # 바이트 데이터인 경우
            elif isinstance(image_path_or_data, bytes):
                return Image.open(BytesIO(image_path_or_data)).convert('RGB')
            
            # 지원하지 않는 형식
            else:
                print(f"지원하지 않는 이미지 데이터 형식: {type(image_path_or_data)}")
                return None
                
        except Exception as e:
            print(f"이미지 로드 중 오류 발생: {str(e)}")
            return None

    def detect_objects(self, image_path_or_data: Union[str, bytes, Image.Image]) -> Dict:
        """
        이미지에서 객체 탐지 수행
        
        Args:
            image_path_or_data: 이미지 파일 경로, URL, base64 인코딩 문자열, 또는 PIL 이미지 객체
            
        Returns:
            Dict: 탐지 결과 (객체 목록, 바운딩 박스, 신뢰도 점수 등)
        """
        # 이미지가 없는 경우 처리
        if image_path_or_data is None:
            print("이미지가 제공되지 않았습니다.")
            return {
                "objects": [],
                "main_object": None,
                "has_image": False,
                "success": False,
                "error": "이미지가 제공되지 않았습니다."
            }
        
        # 이미지 로드
        image = self._load_image(image_path_or_data)
        if image is None:
            print("이미지 로드에 실패했습니다.")
            return {
                "objects": [],
                "main_object": None,
                "has_image": False,
                "success": False,
                "error": "이미지 로드에 실패했습니다."
            }
        
        try:
            # 객체 탐지 수행
            results = self.model(image, device=self.device, verbose=False)
            
            # 결과 추출
            detected_objects = []
            
            for i, result in enumerate(results):
                boxes = result.boxes  # 바운딩 박스 정보
                
                for j, box in enumerate(boxes):
                    # 좌표 추출
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # 신뢰도 점수
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # 임계값 이상의 결과만 처리
                    if confidence >= self.confidence_threshold:
                        # 클래스 ID 및 이름
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = result.names[class_id]
                        
                        # 객체 정보 저장
                        obj_info = {
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "width": float(x2 - x1),
                            "height": float(y2 - y1),
                            "area": float((x2 - x1) * (y2 - y1)),
                            "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)]
                        }
                        
                        detected_objects.append(obj_info)
            
            # 가장 중요한 객체 식별 (면적 + 신뢰도 기준)
            main_object = None
            if detected_objects:
                # 객체 면적과 신뢰도를 모두 고려하여 점수 계산
                for obj in detected_objects:
                    # 이미지 크기 대비 객체 크기 비율 계산
                    width, height = image.size
                    img_area = width * height
                    relative_area = obj["area"] / img_area
                    
                    # 가중치 점수 계산 (면적 70%, 신뢰도 30%)
                    obj["importance_score"] = (0.7 * relative_area) + (0.3 * obj["confidence"])
                
                # 중요도 점수로 정렬
                detected_objects.sort(key=lambda x: x["importance_score"], reverse=True)
                main_object = detected_objects[0]
            
            return {
                "objects": detected_objects,
                "main_object": main_object,
                "has_image": True,
                "image_size": image.size,
                "success": True,
                "num_objects": len(detected_objects)
            }
            
        except Exception as e:
            print(f"객체 탐지 중 오류 발생: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "objects": [],
                "main_object": None,
                "has_image": True,  # 이미지는 로드되었지만 처리 실패
                "success": False,
                "error": str(e)
            }

    def visualize_detection(self, image_path_or_data: Union[str, bytes, Image.Image], 
                           detection_result: Dict,
                           show_all_objects: bool = True,
                           highlight_main: bool = True) -> Optional[np.ndarray]:
        """
        객체 탐지 결과를 시각화
        
        Args:
            image_path_or_data: 원본 이미지
            detection_result: detect_objects()의 결과
            show_all_objects: 모든 객체를 표시할지 여부
            highlight_main: 주요 객체를 강조 표시할지 여부
            
        Returns:
            np.ndarray: 시각화된 이미지 (OpenCV 형식)
        """
        # 이미지가 없거나 로드 실패한 경우
        if not detection_result.get("has_image", False) or detection_result.get("success", False) is False:
            print("시각화할 이미지 또는 탐지 결과가 없습니다.")
            return None
        
        # 이미지 로드
        image = self._load_image(image_path_or_data)
        if image is None:
            return None
        
        # PIL 이미지를 OpenCV 형식으로 변환
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # 색상 설정
        main_color = (0, 255, 0)  # 주요 객체: 녹색
        other_color = (255, 0, 0)  # 다른 객체: 파란색
        
        # 객체 그리기
        objects = detection_result.get("objects", [])
        main_object = detection_result.get("main_object", None)
        
        for obj in objects:
            # 주요 객체가 아니고 모든 객체를 표시하지 않는 경우 건너뛰기
            if not show_all_objects and main_object and obj != main_object:
                continue
                
            x1, y1, x2, y2 = [int(coord) for coord in obj["bbox"]]
            class_name = obj["class_name"]
            confidence = obj["confidence"]
            
            # 주요 객체인지 확인하여 색상 선택
            color = main_color if highlight_main and obj == main_object else other_color
            
            # 바운딩 박스 그리기
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
            
            # 텍스트 배경 그리기 (가독성 향상)
            text = f"{class_name}: {confidence:.2f}"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image_cv, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            
            # 텍스트 그리기
            cv2.putText(image_cv, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image_cv

    def extract_main_object(self, image_path_or_data: Union[str, bytes, Image.Image]) -> Dict:
        """
        이미지에서 주요 객체 영역을 추출
        
        Args:
            image_path_or_data: 이미지 파일 경로, URL, base64 인코딩 문자열, 또는 PIL 이미지 객체
            
        Returns:
            Dict: 주요 객체 정보, 추출된 이미지
        """
        # 객체 탐지 수행
        detection_result = self.detect_objects(image_path_or_data)
        
        # 이미지가 없거나 탐지 실패한 경우
        if not detection_result.get("has_image", False) or detection_result.get("success", False) is False:
            return {
                "success": False,
                "has_main_object": False,
                "original_image": None,
                "cropped_image": None,
                "main_object": None,
                "error": detection_result.get("error", "알 수 없는 오류")
            }
        
        # 이미지 로드
        image = self._load_image(image_path_or_data)
        
        # 주요 객체가 없는 경우
        main_object = detection_result.get("main_object", None)
        if main_object is None:
            return {
                "success": True,
                "has_main_object": False,
                "original_image": image,
                "cropped_image": None,
                "main_object": None
            }
        
        # 주요 객체 영역 추출
        try:
            x1, y1, x2, y2 = [int(coord) for coord in main_object["bbox"]]
            
            # 이미지 크기 확인
            width, height = image.size
            
            # 경계 확인 및 조정
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # 영역 추출
            cropped_image = image.crop((x1, y1, x2, y2))
            
            return {
                "success": True,
                "has_main_object": True,
                "original_image": image,
                "cropped_image": cropped_image,
                "main_object": main_object
            }
            
        except Exception as e:
            print(f"객체 영역 추출 중 오류 발생: {str(e)}")
            
            return {
                "success": False,
                "has_main_object": False,
                "original_image": image,
                "cropped_image": None,
                "main_object": main_object,
                "error": str(e)
            }

    def process_image_for_matching(self, image_path_or_data: Optional[Union[str, bytes, Image.Image]]) -> Dict:
        """
        분실물-습득물 매칭을 위한 이미지 처리
        이미지가 없는 경우에도 적절히 처리
        
        Args:
            image_path_or_data: 이미지 파일 경로, URL, base64 인코딩 문자열, 또는 PIL 이미지 객체 (None도 가능)
            
        Returns:
            Dict: 처리 결과
        """
        # 이미지가 없는 경우
        if image_path_or_data is None:
            return {
                "success": True,
                "has_image": False,
                "has_main_object": False,
                "original_image": None,
                "processed_image": None,
                "main_object": None,
                "objects": []
            }
        
        # 이미지 로드 시도
        image = self._load_image(image_path_or_data)
        if image is None:
            return {
                "success": False,
                "has_image": False,
                "has_main_object": False,
                "original_image": None,
                "processed_image": None,
                "main_object": None,
                "objects": [],
                "error": "이미지 로드 실패"
            }
        
        # 객체 탐지 수행
        detection_result = self.detect_objects(image)
        
        # 탐지 실패한 경우 전체 이미지 사용
        if not detection_result.get("success", False):
            return {
                "success": True,  # 이미지는 있으므로 전체 이미지 사용 가능
                "has_image": True,
                "has_main_object": False,
                "original_image": image,
                "processed_image": image,  # 전체 이미지 사용
                "main_object": None,
                "objects": [],
                "detection_error": detection_result.get("error", "객체 탐지 실패")
            }
        
        # 주요 객체가 없는 경우 전체 이미지 사용
        main_object = detection_result.get("main_object", None)
        if main_object is None:
            return {
                "success": True,
                "has_image": True,
                "has_main_object": False,
                "original_image": image,
                "processed_image": image,  # 전체 이미지 사용
                "main_object": None,
                "objects": detection_result.get("objects", [])
            }
        
        # 주요 객체 영역 추출
        try:
            x1, y1, x2, y2 = [int(coord) for coord in main_object["bbox"]]
            
            # 이미지 크기 확인
            width, height = image.size
            
            # 경계 확인 및 조정
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # 영역 추출
            cropped_image = image.crop((x1, y1, x2, y2))
            
            return {
                "success": True,
                "has_image": True,
                "has_main_object": True,
                "original_image": image,
                "processed_image": cropped_image,  # 주요 객체 영역
                "main_object": main_object,
                "objects": detection_result.get("objects", [])
            }
            
        except Exception as e:
            print(f"객체 영역 추출 중 오류 발생: {str(e)}")
            
            return {
                "success": True,  # 여전히 전체 이미지 사용 가능
                "has_image": True,
                "has_main_object": False,  # 추출은 실패했지만 객체는 탐지됨
                "original_image": image,
                "processed_image": image,  # 전체 이미지 사용
                "main_object": main_object,
                "objects": detection_result.get("objects", []),
                "extraction_error": str(e)
            }


# 테스트 함수
def test_object_detector(image_path=None):
    """
    객체 탐지 모델 테스트 함수
    
    Args:
        image_path: 테스트할 이미지 경로 (없으면 샘플 이미지 다운로드)
    """
    # 테스트 이미지가 없는 경우 샘플 이미지 다운로드
    if image_path is None or not os.path.exists(image_path):
        print("테스트 이미지가 제공되지 않았습니다. 샘플 이미지를 다운로드합니다.")
        sample_url = "https://ultralytics.com/images/zidane.jpg"
        try:
            with urllib.request.urlopen(sample_url) as response:
                image_data = response.read()
            image_path = "sample_image.jpg"
            with open(image_path, "wb") as f:
                f.write(image_data)
            print(f"샘플 이미지를 {image_path}에 저장했습니다.")
        except Exception as e:
            print(f"샘플 이미지 다운로드 실패: {str(e)}")
            return

    # 객체 탐지 모델 초기화
    try:
        detector = ObjectDetector()
    except Exception as e:
        print(f"객체 탐지 모델 초기화 실패: {str(e)}")
        return

    # 객체 탐지 수행
    print("\n1. 객체 탐지 테스트")
    detection_result = detector.detect_objects(image_path)
    
    print(f"탐지된 객체 수: {len(detection_result.get('objects', []))}")
    if detection_result.get("main_object"):
        main_obj = detection_result["main_object"]
        print(f"주요 객체: {main_obj['class_name']} (신뢰도: {main_obj['confidence']:.2f})")
    
    # 탐지 결과 시각화
    print("\n2. 시각화 테스트")
    vis_image = detector.visualize_detection(image_path, detection_result)
    
    if vis_image is not None:
        plt.figure(figsize=(10, 8))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Object Detection Results")
        plt.show()
    
    # 주요 객체 추출 테스트
    print("\n3. 주요 객체 추출 테스트")
    extraction_result = detector.extract_main_object(image_path)
    
    if extraction_result.get("has_main_object", False):
        main_obj = extraction_result["main_object"]
        cropped = extraction_result["cropped_image"]
        
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(extraction_result["original_image"])
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cropped)
        plt.title(f"Main Object: {main_obj['class_name']}")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # 매칭 처리 테스트
    print("\n4. 매칭용 이미지 처리 테스트")
    processing_result = detector.process_image_for_matching(image_path)
    
    print(f"처리 성공: {processing_result['success']}")
    print(f"이미지 있음: {processing_result['has_image']}")
    print(f"주요 객체 있음: {processing_result['has_main_object']}")
    
    if processing_result.get("processed_image"):
        plt.figure(figsize=(10, 8))
        plt.imshow(processing_result["processed_image"])
        plt.title("Processed Image for Matching")
        plt.axis('off')
        plt.show()

    # 이미지 없는 경우 테스트
    print("\n5. 이미지 없는 경우 테스트")
    no_image_result = detector.process_image_for_matching(None)
    print(f"이미지 없음 처리 성공: {no_image_result['success']}")
    print(f"이미지 있음 플래그: {no_image_result['has_image']}")

    print("\n모든 테스트 완료!")


if __name__ == "__main__":
    # 테스트 실행
    test_object_detector('./blueumb.jpg')