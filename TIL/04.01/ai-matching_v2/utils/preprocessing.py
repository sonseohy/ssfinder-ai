import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List

from config import Config

def resize_image(image_path: str, output_path: Optional[str] = None, 
                size: Tuple[int, int] = None) -> str:
    """
    이미지 리사이즈
    
    Args:
        image_path: 원본 이미지 경로
        output_path: 결과 이미지 저장 경로 (None이면 원본 경로에 _resized 접미사 추가)
        size: 조정할 크기 (None이면 Config에서 기본값 사용)
        
    Returns:
        str: 리사이즈된 이미지 경로
    """
    size = size or Config.IMAGE_SIZE
    
    try:
        img = Image.open(image_path).convert('RGB')
        img_resized = img.resize(size, Image.LANCZOS)
        
        if output_path is None:
            filename, ext = os.path.splitext(image_path)
            output_path = f"{filename}_resized{ext}"
        
        img_resized.save(output_path)
        return output_path
    except Exception as e:
        print(f"이미지 리사이즈 오류: {e}")
        return image_path

def normalize_brightness(image_path: str, output_path: Optional[str] = None, 
                        target_brightness: float = 0.5) -> str:
    """
    이미지 밝기 정규화
    
    Args:
        image_path: 원본 이미지 경로
        output_path: 결과 이미지 저장 경로 (None이면 원본 경로에 _normalized 접미사 추가)
        target_brightness: 목표 밝기 (0~1)
        
    Returns:
        str: 정규화된 이미지 경로
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"이미지를 읽을 수 없습니다: {image_path}")
            return image_path
        
        # 현재 밝기 계산 (HSV 변환 후 V 채널 평균)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        current_brightness = img_hsv[:, :, 2].mean() / 255.0
        
        # 밝기 조정
        alpha = target_brightness / max(current_brightness, 0.01)  # 0 나누기 방지
        
        # 값 제한
        alpha = min(max(alpha, 0.5), 2.0)  # 0.5 ~ 2.0 사이로 제한
        
        # 밝기 조정 적용
        img_adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        
        if output_path is None:
            filename, ext = os.path.splitext(image_path)
            output_path = f"{filename}_normalized{ext}"
        
        cv2.imwrite(output_path, img_adjusted)
        return output_path
    except Exception as e:
        print(f"이미지 밝기 정규화 오류: {e}")
        return image_path

def detect_and_crop_object(image_path: str, output_path: Optional[str] = None) -> str:
    """
    이미지에서 주요 객체 감지 및 크롭
    
    Args:
        image_path: 원본 이미지 경로
        output_path: 결과 이미지 저장 경로 (None이면 원본 경로에 _cropped 접미사 추가)
        
    Returns:
        str: 크롭된 이미지 경로
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"이미지를 읽을 수 없습니다: {image_path}")
            return image_path
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 블러 처리
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 이진화
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 윤곽선 검출
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # 가장 큰 윤곽선 선택
            max_contour = max(contours, key=cv2.contourArea)
            
            # 윤곽선 영역이 너무 작으면 크롭하지 않음
            if cv2.contourArea(max_contour) < 0.1 * img.shape[0] * img.shape[1]:
                print("감지된 객체가 너무 작습니다. 원본 이미지를 유지합니다.")
                return image_path
            
            # 경계 상자 계산
            x, y, w, h = cv2.boundingRect(max_contour)
            
            # 여백 추가 (10%)
            margin_x = int(w * 0.1)
            margin_y = int(h * 0.1)
            
            # 이미지 경계 확인
            x = max(0, x - margin_x)
            y = max(0, y - margin_y)
            w = min(img.shape[1] - x, w + 2 * margin_x)
            h = min(img.shape[0] - y, h + 2 * margin_y)
            
            # 이미지 크롭
            cropped = img[y:y+h, x:x+w]
            
            if output_path is None:
                filename, ext = os.path.splitext(image_path)
                output_path = f"{filename}_cropped{ext}"
            
            cv2.imwrite(output_path, cropped)
            return output_path
        else:
            print("객체를 감지할 수 없습니다. 원본 이미지를 유지합니다.")
            return image_path
    except Exception as e:
        print(f"객체 감지 및 크롭 오류: {e}")
        return image_path

def preprocess_image(image_path: str, output_dir: Optional[str] = None) -> str:
    """
    이미지 전처리 파이프라인 (크롭, 밝기 정규화, 리사이징)
    
    Args:
        image_path: 원본 이미지 경로
        output_dir: 결과 이미지 저장 디렉토리 (None이면 원본 이미지 디렉토리 사용)
        
    Returns:
        str: 전처리된 이미지 경로
    """
    try:
        # 출력 디렉토리 설정
        if output_dir is None:
            output_dir = os.path.dirname(image_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 파일명 추출
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        
        # 전처리 파이프라인
        # 1. 객체 감지 및 크롭
        cropped_path = os.path.join(output_dir, f"{name}_cropped{ext}")
        cropped_path = detect_and_crop_object(image_path, cropped_path)
        
        # 2. 밝기 정규화
        normalized_path = os.path.join(output_dir, f"{name}_normalized{ext}")
        normalized_path = normalize_brightness(cropped_path, normalized_path)
        
        # 3. 리사이징
        resized_path = os.path.join(output_dir, f"{name}_processed{ext}")
        final_path = resize_image(normalized_path, resized_path)
        
        return final_path
    except Exception as e:
        print(f"이미지 전처리 오류: {e}")
        return image_path

def enhance_image_quality(image_path: str, output_path: Optional[str] = None) -> str:
    """
    이미지 품질 향상 (선명도, 대비 개선)
    
    Args:
        image_path: 원본 이미지 경로
        output_path: 결과 이미지 저장 경로 (None이면 원본 경로에 _enhanced 접미사 추가)
        
    Returns:
        str: 향상된 이미지 경로
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"이미지를 읽을 수 없습니다: {image_path}")
            return image_path
        
        # 대비 향상 (CLAHE)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # 병합 및 변환
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # 선명도 향상 (언샤프 마스킹)
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        if output_path is None:
            filename, ext = os.path.splitext(image_path)
            output_path = f"{filename}_enhanced{ext}"
        
        cv2.imwrite(output_path, enhanced)
        return output_path
    except Exception as e:
        print(f"이미지 품질 향상 오류: {e}")
        return image_path

def batch_preprocess_images(image_paths: List[str], output_dir: str) -> List[str]:
    """
    여러 이미지 일괄 전처리
    
    Args:
        image_paths: 원본 이미지 경로 목록
        output_dir: 결과 이미지 저장 디렉토리
        
    Returns:
        List[str]: 전처리된 이미지 경로 목록
    """
    os.makedirs(output_dir, exist_ok=True)
    
    processed_paths = []
    for img_path in image_paths:
        processed_path = preprocess_image(img_path, output_dir)
        processed_paths.append(processed_path)
    
    return processed_paths