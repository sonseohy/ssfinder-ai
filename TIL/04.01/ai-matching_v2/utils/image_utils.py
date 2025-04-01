import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional

from config import Config

def load_image(image_path: str, mode: str = 'RGB') -> Optional[Image.Image]:
    """
    이미지 로드 및 예외 처리
    
    Args:
        image_path: 이미지 경로
        mode: 이미지 모드 ('RGB' 또는 'RGBA')
        
    Returns:
        Optional[Image.Image]: 로드된 PIL 이미지 객체 또는 None (오류 시)
    """
    try:
        return Image.open(image_path).convert(mode)
    except Exception as e:
        print(f"이미지 로드 오류: {e}")
        return None

def get_image_metadata(image_path: str) -> Dict[str, Any]:
    """
    이미지 메타데이터 추출
    
    Args:
        image_path: 이미지 경로
        
    Returns:
        Dict[str, Any]: 이미지 메타데이터
    """
    try:
        img = load_image(image_path)
        if img is None:
            return {}
        
        width, height = img.size
        format_type = img.format
        mode = img.mode
        
        # 기본 정보 수집
        metadata = {
            'width': width,
            'height': height,
            'format': format_type,
            'mode': mode,
            'aspect_ratio': width / height if height > 0 else 0,
            'filesize': os.path.getsize(image_path) / 1024  # KB 단위
        }
        
        # EXIF 데이터 추출 (가능한 경우)
        try:
            exif_data = img._getexif()
            if exif_data:
                # EXIF 태그 ID와 값 매핑
                exif_tags = {
                    271: 'make',  # 제조사
                    272: 'model',  # 모델
                    306: 'datetime',  # 날짜 및 시간
                    36867: 'date_taken',  # 촬영 날짜
                    37385: 'flash',  # 플래시 사용 여부
                    41728: 'source_type'  # 이미지 소스 타입
                }
                
                exif = {}
                for tag_id, tag_name in exif_tags.items():
                    if tag_id in exif_data:
                        exif[tag_name] = exif_data[tag_id]
                
                metadata['exif'] = exif
        except:
            pass
        
        return metadata
    except Exception as e:
        print(f"메타데이터 추출 오류: {e}")
        return {}

def are_images_similar(image1_path: str, image2_path: str, threshold: float = 0.8) -> bool:
    """
    두 이미지가 유사한지 검사 (구조적 유사성 계산)
    
    Args:
        image1_path: 첫 번째 이미지 경로
        image2_path: 두 번째 이미지 경로
        threshold: 유사도 임계값 (0~1)
        
    Returns:
        bool: 유사 여부
    """
    try:
        # OpenCV로 이미지 로드
        img1 = cv2.imread(image1_path)
        img2 = cv2.imread(image2_path)
        
        if img1 is None or img2 is None:
            return False
        
        # 동일한 크기로 리사이즈
        size = (200, 200)
        img1 = cv2.resize(img1, size)
        img2 = cv2.resize(img2, size)
        
        # 그레이스케일 변환
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # SSIM (구조적 유사성) 계산
        try:
            from skimage.metrics import structural_similarity as ssim
            score, _ = ssim(gray1, gray2, full=True)
        except ImportError:
            # skimage가 없으면 히스토그램 비교 사용
            hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
            hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
            cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
            cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
            score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return score >= threshold
    except Exception as e:
        print(f"이미지 유사도 검사 오류: {e}")
        return False

def get_image_hash(image_path: str) -> str:
    """
    이미지의 퍼셉추얼 해시 계산 (유사 이미지 검색용)
    
    Args:
        image_path: 이미지 경로
        
    Returns:
        str: 이미지 해시 값
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return ""
        
        # 이미지 크기를 8x8로 줄임
        img = cv2.resize(img, (8, 8))
        
        # 그레이스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 평균 계산
        avg_val = gray.mean()
        
        # 각 픽셀 값이 평균보다 크면 1, 작으면 0으로 해시 생성
        hash_str = ''
        for i in range(8):
            for j in range(8):
                hash_str += '1' if gray[i, j] >= avg_val else '0'
        
        return hash_str
    except Exception as e:
        print(f"이미지 해시 계산 오류: {e}")
        return ""

def extract_background_color(image_path: str) -> Tuple[Tuple[int, int, int], float]:
    """
    이미지 배경 색상 추출
    
    Args:
        image_path: 이미지 경로
        
    Returns:
        Tuple[Tuple[int, int, int], float]: (배경 색상(RGB), 비율)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return ((0, 0, 0), 0.0)
        
        # 이미지 크기 조정
        img = cv2.resize(img, (100, 100))
        
        # BGR에서 RGB로 변환
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 이미지를 2D 배열로 변환
        pixels = img_rgb.reshape(-1, 3)
        
        # 색상의 양자화 (K-means 클러스터링)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=0)
        kmeans.fit(pixels)
        
        # 각 클러스터의 중심(색상) 및 비율 계산
        counts = np.bincount(kmeans.labels_)
        percentages = counts / len(kmeans.labels_)
        
        # 가장 많은 비율을 차지하는 색상을 배경으로 간주
        bg_idx = np.argmax(percentages)
        bg_color = tuple(map(int, kmeans.cluster_centers_[bg_idx]))
        bg_ratio = percentages[bg_idx]
        
        return (bg_color, bg_ratio)
    except Exception as e:
        print(f"배경 색상 추출 오류: {e}")
        return ((0, 0, 0), 0.0)

def create_composite_image(image_paths: List[str], output_path: str, 
                          rows: int = 2, cols: int = 2) -> bool:
    """
    여러 이미지를 하나의 컴포지트 이미지로 결합
    
    Args:
        image_paths: 이미지 경로 목록
        output_path: 출력 이미지 경로
        rows: 행 수
        cols: 열 수
        
    Returns:
        bool: 성공 여부
    """
    try:
        images = []
        for path in image_paths:
            img = load_image(path)
            if img:
                # 일관된 크기로 리사이징
                img = img.resize((200, 200), Image.LANCZOS)
                images.append(np.array(img))
        
        # 이미지가 충분하지 않을 경우 빈 이미지 추가
        while len(images) < rows * cols:
            blank = np.ones((200, 200, 3), dtype=np.uint8) * 255
            images.append(blank)
        
        # 이미지 그리드 생성
        grid = []
        for i in range(rows):
            row_images = images[i*cols:(i+1)*cols]
            row = np.hstack(row_images)
            grid.append(row)
        
        composite = np.vstack(grid)
        
        # PIL 이미지로 변환 및 저장
        pil_img = Image.fromarray(composite.astype('uint8'))
        pil_img.save(output_path)
        
        return True
    except Exception as e:
        print(f"컴포지트 이미지 생성 오류: {e}")
        return False

def visualize_similarity_comparison(image1_path: str, image2_path: str, 
                                   similarity_scores: Dict[str, float],
                                   output_path: Optional[str] = None) -> None:
    """
    두 이미지와 유사도 점수를 시각화
    
    Args:
        image1_path: 첫 번째 이미지 경로
        image2_path: 두 번째 이미지 경로
        similarity_scores: 유사도 점수 딕셔너리
        output_path: 출력 이미지 경로 (None이면 표시만 함)
    """
    try:
        # 이미지 로드
        img1 = load_image(image1_path)
        img2 = load_image(image2_path)
        
        if img1 is None or img2 is None:
            print("이미지 로드 실패")
            return
        
        # 그림 설정
        plt.figure(figsize=(12, 6))
        
        # 이미지 표시
        plt.subplot(1, 2, 1)
        plt.imshow(img1)
        plt.title('이미지 1')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(img2)
        plt.title('이미지 2')
        plt.axis('off')
        
        # 유사도 점수 텍스트 표시
        plt.figtext(0.5, 0.01, f"유사도 점수:", ha='center', fontsize=12, fontweight='bold')
        
        y_pos = -0.05
        for key, score in similarity_scores.items():
            y_pos -= 0.03
            plt.figtext(0.5, y_pos, f"{key}: {score:.4f}", ha='center')
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.3)
        
        # 이미지 저장 또는 표시
        if output_path:
            plt.savefig(output_path)
        else:
            plt.show()
        
    except Exception as e:
        print(f"유사도 비교 시각화 오류: {e}")

def highlight_object_regions(image_path: str, output_path: Optional[str] = None) -> str:
    """
    이미지에서 주요 객체 영역 강조 표시
    
    Args:
        image_path: 이미지 경로
        output_path: 결과 이미지 저장 경로 (None이면 원본 경로에 _highlighted 접미사 추가)
        
    Returns:
        str: 강조 표시된 이미지 경로
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
        
        # 결과 이미지 (원본 복사)
        result = img.copy()
        
        # 윤곽선 그리기
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        # 각 윤곽선에 대해 바운딩 박스 그리기
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # 작은 노이즈 무시
            if area < 100:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        if output_path is None:
            filename, ext = os.path.splitext(image_path)
            output_path = f"{filename}_highlighted{ext}"
        
        cv2.imwrite(output_path, result)
        return output_path
    except Exception as e:
        print(f"객체 영역 강조 표시 오류: {e}")
        return image_path