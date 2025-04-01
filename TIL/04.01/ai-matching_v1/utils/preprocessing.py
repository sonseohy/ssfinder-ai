import cv2
import numpy as np
from PIL import Image
import torch
from typing import Tuple, List, Dict, Any, Union, Optional
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def resize_image(image: Union[str, np.ndarray, Image.Image], 
                target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """
    이미지 크기 조정
    
    Args:
        image: 이미지 경로, PIL 이미지, 또는 numpy 배열
        target_size: 목표 크기 (너비, 높이)
        
    Returns:
        PIL.Image: 크기가 조정된 이미지
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
    
    return img.resize(target_size, Image.Resampling.LANCZOS)

def normalize_image(image: np.ndarray, 
                   mean: List[float] = [0.485, 0.456, 0.406], 
                   std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    이미지 정규화 (ImageNet 통계 사용)
    
    Args:
        image: 정규화할 이미지 (HxWxC, 0-255 범위)
        mean: 채널별 평균값
        std: 채널별 표준편차
        
    Returns:
        np.ndarray: 정규화된 이미지
    """
    # 0-1 범위로 변환
    norm_image = image.astype(np.float32) / 255.0
    
    # 채널별 정규화
    for i in range(3):
        norm_image[:, :, i] = (norm_image[:, :, i] - mean[i]) / std[i]
    
    return norm_image

def crop_main_object(image: Union[str, np.ndarray, Image.Image]) -> Tuple[np.ndarray, List[int]]:
    """
    이미지에서 주요 객체 영역 잘라내기
    
    Args:
        image: 이미지 경로, PIL 이미지, 또는 numpy 배열
        
    Returns:
        Tuple[np.ndarray, List[int]]: 잘라낸 이미지와 바운딩 박스 [x, y, width, height]
    """
    # 이미지 로드
    if isinstance(image, str):
        img = cv2.imread(image)
    elif isinstance(image, np.ndarray):
        img = image.copy()
        if img.shape[2] == 3 and img[0, 0, 0] <= img[0, 0, 2]:  # RGB 순서 확인
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB에서 BGR로 변환
    elif isinstance(image, Image.Image):
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # RGB에서 BGR로 변환
    else:
        raise ValueError(f"지원되지 않는 이미지 타입: {type(image)}")
    
    # 그레이스케일로 변환
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 가우시안 블러 적용하여 노이즈 감소
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 이진화 (적응형 임계값 처리)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 모폴로지 연산으로 잡음 제거 및 객체 연결
    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # 윤곽선을 찾지 못한 경우 원본 이미지 반환
        return img, [0, 0, img.shape[1], img.shape[0]]
    
    # 가장 큰 윤곽선 찾기 (주 객체로 가정)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 바운딩 박스 구하기
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # 너무 작은 바운딩 박스는 무시하고 원본 이미지 사용
    if w < img.shape[1] * 0.1 or h < img.shape[0] * 0.1:
        return img, [0, 0, img.shape[1], img.shape[0]]
    
    # 바운딩 박스 주변에 여유 공간 추가 (10%)
    margin_x = int(w * 0.1)
    margin_y = int(h * 0.1)
    
    # 이미지 경계를 벗어나지 않도록 좌표 조정
    x_min = max(0, x - margin_x)
    y_min = max(0, y - margin_y)
    x_max = min(img.shape[1], x + w + margin_x)
    y_max = min(img.shape[0], y + h + margin_y)
    
    # 이미지 잘라내기
    cropped_img = img[y_min:y_max, x_min:x_max]
    
    return cropped_img, [x_min, y_min, x_max - x_min, y_max - y_min]

def enhance_image(image: np.ndarray) -> np.ndarray:
    """
    이미지 품질 향상
    
    Args:
        image: 향상시킬 이미지 (BGR 형식)
        
    Returns:
        np.ndarray: 향상된 이미지
    """
    # 대비 향상 (CLAHE)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # 약간의 선명도 향상
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced_img, -1, kernel)
    
    # 가우시안 블러로 노이즈 감소
    final_img = cv2.GaussianBlur(sharpened, (3, 3), 0)
    
    return final_img

def preprocess_for_models(image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
    """
    모델링을 위한 종합 이미지 전처리
    
    Args:
        image: 이미지 경로, PIL 이미지, 또는 numpy 배열
        
    Returns:
        Dict[str, Any]: 다양한 전처리된 이미지 형식
    """
    # 이미지 로드
    if isinstance(image, str):
        original_img = Image.open(image).convert("RGB")
        cv_img = cv2.imread(image)
    elif isinstance(image, np.ndarray):
        if image.shape[2] == 3 and image[0, 0, 0] <= image[0, 0, 2]:  # RGB 순서 확인
            cv_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            cv_img = image.copy()
        original_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    elif isinstance(image, Image.Image):
        original_img = image.convert("RGB")
        cv_img = cv2.cvtColor(np.array(original_img), cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f"지원되지 않는 이미지 타입: {type(image)}")
    
    # 주요 객체 크롭
    cropped_img, bbox = crop_main_object(cv_img)
    
    # 이미지 향상
    enhanced_img = enhance_image(cropped_img)
    
    # PIL 이미지로 변환
    pil_enhanced = Image.fromarray(cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB))
    
    # 다양한 크기로 조정
    resized_224 = resize_image(pil_enhanced, (224, 224))
    resized_384 = resize_image(pil_enhanced, (384, 384))
    
    return {
        'original': original_img,
        'cropped': Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)),
        'enhanced': pil_enhanced,
        'resized_224': resized_224,
        'resized_384': resized_384,
        'bbox': bbox
    }

def batch_preprocess_images(image_paths: List[str]) -> List[Dict[str, Any]]:
    """
    여러 이미지 일괄 전처리
    
    Args:
        image_paths: 이미지 경로 목록
        
    Returns:
        List[Dict[str, Any]]: 전처리된 이미지 목록
    """
    results = []
    for path in image_paths:
        try:
            processed = preprocess_for_models(path)
            processed['path'] = path  # 원본 경로 추가
            results.append(processed)
        except Exception as e:
            logger.error(f"이미지 처리 중 오류 발생 ({path}): {str(e)}")
    
    return results

if __name__ == "__main__":
    # 간단한 테스트
    import matplotlib.pyplot as plt
    
    # 테스트 이미지 경로 입력
    test_image_path = input("테스트할 이미지 경로를 입력하세요: ")
    
    if test_image_path:
        # 전처리 수행
        processed = preprocess_for_models(test_image_path)
        
        # 결과 시각화
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(processed['original'])
        plt.title("원본 이미지")
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(processed['cropped'])
        plt.title(f"객체 감지 (bbox: {processed['bbox']})")
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(processed['enhanced'])
        plt.title("향상된 이미지")
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        plt.imshow(processed['resized_224'])
        plt.title("224x224 크기 조정")
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        plt.imshow(processed['resized_384'])
        plt.title("384x384 크기 조정")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("이미지 경로가 입력되지 않았습니다. 프로그램을 종료합니다.")