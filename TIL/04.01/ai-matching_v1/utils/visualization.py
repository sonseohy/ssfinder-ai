import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import List, Dict, Any, Tuple, Optional, Union
import logging
import os
import io
from matplotlib.colors import LinearSegmentedColormap

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_matching_results(query_item: Dict[str, Any], matched_items: List[Dict[str, Any]], 
                              max_items: int = 5, save_path: Optional[str] = None) -> plt.Figure:
    """
    매칭 결과 시각화
    
    Args:
        query_item: 쿼리 아이템
        matched_items: 매칭된 아이템 목록
        max_items: 표시할 최대 아이템 수
        save_path: 결과를 저장할 경로 (옵션)
        
    Returns:
        plt.Figure: 생성된 플롯 그림
    """
    # 표시할 아이템 수 제한
    items_to_show = min(max_items, len(matched_items))
    
    # 쿼리 이미지 가져오기
    if 'image_bytes' in query_item:
        query_img = Image.open(io.BytesIO(query_item['image_bytes'])).convert('RGB')
    elif 'original' in query_item:
        query_img = query_item['original']
    elif 'fdFilePathImg' in query_item and os.path.exists(query_item['fdFilePathImg']):
        query_img = Image.open(query_item['fdFilePathImg']).convert('RGB')
    else:
        # 이미지가 없으면 빈 이미지 생성
        query_img = Image.new('RGB', (300, 300), color=(240, 240, 240))
        draw = ImageDraw.Draw(query_img)
        draw.text((20, 150), "이미지 없음", fill=(0, 0, 0))
    
    # 그리드 생성
    fig, axes = plt.subplots(1, items_to_show + 1, figsize=(15, 5))
    fig.suptitle("분실물 매칭 결과", fontsize=16)
    
    # 쿼리 아이템 표시
    axes[0].imshow(query_img)
    axes[0].set_title("분실물 (쿼리)")
    
    # 쿼리 아이템 속성 표시
    query_attrs = get_display_attributes(query_item)
    if query_attrs:
        attr_text = "\n".join([f"{k}: {v}" for k, v in query_attrs.items()])
        axes[0].text(0.05, 0.95, attr_text, transform=axes[0].transAxes, 
                     verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.7})
    
    # 매칭된 아이템 표시
    for i in range(items_to_show):
        matched_item = matched_items[i]
        
        # 매칭된 아이템 이미지 가져오기
        if 'image_bytes' in matched_item:
            matched_img = Image.open(io.BytesIO(matched_item['image_bytes'])).convert('RGB')
        elif 'original' in matched_item:
            matched_img = matched_item['original']
        elif 'fdFilePathImg' in matched_item and os.path.exists(matched_item['fdFilePathImg']):
            matched_img = Image.open(matched_item['fdFilePathImg']).convert('RGB')
        else:
            # 이미지가 없으면 빈 이미지 생성
            matched_img = Image.new('RGB', (300, 300), color=(240, 240, 240))
            draw = ImageDraw.Draw(matched_img)
            draw.text((20, 150), "이미지 없음", fill=(0, 0, 0))
        
        # 이미지 표시
        axes[i+1].imshow(matched_img)
        
        # 점수 및 설명 가져오기
        if 'matching' in matched_item:
            score = matched_item['matching'].get('total_score', 0.0)
            explanation = matched_item['matching'].get('match_explanation', "")
            title = f"#{i+1} (유사도: {score:.2f})"
        else:
            title = f"매칭 아이템 #{i+1}"
        
        axes[i+1].set_title(title)
        
        # 속성 및 설명 표시
        matched_attrs = get_display_attributes(matched_item)
        if matched_attrs:
            attr_text = "\n".join([f"{k}: {v}" for k, v in matched_attrs.items()])
            
            # 설명 추가
            if 'matching' in matched_item and 'match_explanation' in matched_item['matching']:
                attr_text += "\n\n" + matched_item['matching']['match_explanation']
                
            axes[i+1].text(0.05, 0.95, attr_text, transform=axes[i+1].transAxes, 
                          verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.7})
    
    # 각 축 조정
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    
    # 결과 저장
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"매칭 결과 이미지 저장됨: {save_path}")
    
    return fig

def get_display_attributes(item: Dict[str, Any]) -> Dict[str, str]:
    """
    표시할 속성 추출
    
    Args:
        item: 대상 아이템
        
    Returns:
        Dict[str, str]: 표시할 속성
    """
    attributes = {}
    
    # BLIP 속성
    if 'blip_attributes' in item and 'attributes' in item['blip_attributes']:
        for k, v in item['blip_attributes']['attributes'].items():
            if k == 'color':
                attributes['색상'] = v
            elif k == 'brand':
                attributes['브랜드'] = v
            elif k == 'type':
                attributes['종류'] = v
    
    # 직접 클래스 속성
    if 'class' in item and not attributes.get('종류'):
        attributes['종류'] = item['class']
    
    # 제품명
    if 'fdPrdtNm' in item and not attributes.get('종류'):
        attributes['종류'] = item['fdPrdtNm']
    
    # 습득 장소
    if 'fdPlaceNm' in item:
        attributes['습득장소'] = item['fdPlaceNm']
    elif 'N_FD_LCT_NM' in item:
        attributes['습득장소'] = item['N_FD_LCT_NM']
    
    # 습득 일자
    if 'fdYmd_datetime' in item:
        attributes['습득일자'] = item['fdYmd_datetime'].strftime('%Y-%m-%d')
    elif 'fdYmd' in item and item['fdYmd']:
        date_str = str(item['fdYmd'])
        if len(date_str) == 8:
            attributes['습득일자'] = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    
    # 분실 장소
    if 'lstPlace' in item:
        attributes['분실장소'] = item['lstPlace']
    
    # 색상 정보
    if 'FD_COL_NM' in item and not attributes.get('색상'):
        attributes['색상'] = item['FD_COL_NM']
    
    # 추출된 색상
    if 'extracted_colors' in item and item['extracted_colors'] and not attributes.get('색상'):
        attributes['색상'] = ', '.join(item['extracted_colors'][:2])
    
    # 추출된 브랜드
    if 'extracted_brands' in item and item['extracted_brands'] and not attributes.get('브랜드'):
        attributes['브랜드'] = ', '.join(item['extracted_brands'][:2])
    
    return attributes

def plot_similarity_heatmap(query_item: Dict[str, Any], matched_items: List[Dict[str, Any]], 
                          attributes: List[str] = ['종류', '색상', '브랜드'], 
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    매칭 결과의 속성별 유사도를 히트맵으로 시각화
    
    Args:
        query_item: 쿼리 아이템
        matched_items: 매칭된 아이템 목록
        attributes: 비교할 속성 목록
        save_path: 결과를 저장할 경로 (옵션)
        
    Returns:
        plt.Figure: 생성된 플롯 그림
    """
    # 최대 10개 아이템으로 제한
    items_to_show = min(10, len(matched_items))
    matched_items = matched_items[:items_to_show]
    
    # 유사도 데이터 준비
    similarity_data = []
    item_labels = []
    
    for i, item in enumerate(matched_items):
        if 'matching' in item and 'detailed_scores' in item['matching']:
            scores = []
            
            # 점수 추출
            if 'object_class_match' in item['matching']['detailed_scores']:
                scores.append(item['matching']['detailed_scores']['object_class_match'])
            else:
                scores.append(0.0)
                
            if 'attribute_match' in item['matching']['detailed_scores']:
                scores.append(item['matching']['detailed_scores']['attribute_match'])
            else:
                scores.append(0.0)
                
            if 'clip_similarity' in item['matching']['detailed_scores']:
                scores.append(item['matching']['detailed_scores']['clip_similarity'])
            else:
                scores.append(0.0)
                
            if 'metadata_match' in item['matching']['detailed_scores']:
                scores.append(item['matching']['detailed_scores']['metadata_match'])
            else:
                scores.append(0.0)
            
            similarity_data.append(scores)
            
            # 라벨 생성 (아이템 번호와 총점)
            total_score = item['matching'].get('total_score', 0.0)
            item_labels.append(f"#{i+1} ({total_score:.2f})")
        else:
            similarity_data.append([0.0, 0.0, 0.0, 0.0])
            item_labels.append(f"#{i+1}")
    
    # 속성 라벨 설정
    score_labels = ['물품 종류', '속성 일치', '시각적 유사도', '메타데이터']
    
    # 데이터 변환
    similarity_array = np.array(similarity_data)
    
    # 히트맵 생성
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#f7fbff', '#4292c6', '#08306b'])
    
    im = ax.imshow(similarity_array, cmap=cmap, vmin=0, vmax=1)
    
    # 색상바 추가
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('유사도 점수', rotation=-90, va='bottom')
    
    # 라벨 추가
    ax.set_xticks(np.arange(len(score_labels)))
    ax.set_yticks(np.arange(len(item_labels)))
    ax.set_xticklabels(score_labels)
    ax.set_yticklabels(item_labels)
    
    # x축 라벨 회전
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 텍스트 주석 추가
    for i in range(len(item_labels)):
        for j in range(len(score_labels)):
            text = ax.text(j, i, f"{similarity_array[i, j]:.2f}",
                          ha="center", va="center", color="white" if similarity_array[i, j] > 0.5 else "black")
    
    ax.set_title("매칭 아이템별 유사도 점수")
    fig.tight_layout()
    
    # 결과 저장
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"히트맵 이미지 저장됨: {save_path}")
    
    return fig

def draw_bounding_box(image: Union[np.ndarray, Image.Image], bbox: List[int], 
                     label: Optional[str] = None, color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2) -> np.ndarray:
    """
    이미지에 바운딩 박스 그리기
    
    Args:
        image: 대상 이미지
        bbox: 바운딩 박스 좌표 [x, y, width, height]
        label: 박스 레이블 (옵션)
        color: 박스 색상 (R, G, B)
        thickness: 선 두께
        
    Returns:
        np.ndarray: 바운딩 박스가 그려진 이미지
    """
    # 이미지 변환
    if isinstance(image, Image.Image):
        img = np.array(image)
        if len(img.shape) == 3 and img.shape[2] == 3 and img[0, 0, 0] <= img[0, 0, 2]:
            # RGB에서 BGR로 변환
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = image.copy()
    
    # 바운딩 박스 그리기
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    
    # 레이블 추가
    if label:
        # 레이블 배경 그리기
        font_scale = 0.7
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(label, font, font_scale, 2)[0]
        text_x = x
        text_y = y - 10 if y - 10 > text_size[1] else y + h + 10
        
        cv2.rectangle(img, (text_x, text_y - text_size[1] - 5), 
                     (text_x + text_size[0] + 5, text_y + 5), color, -1)
        
        # 텍스트 그리기
        cv2.putText(img, label, (text_x + 3, text_y), font, font_scale, (255, 255, 255), 2)
    
    # BGR에서 RGB로 변환
    if isinstance(image, Image.Image):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

def plot_feature_comparison(query_features: np.ndarray, candidate_features: List[np.ndarray], 
                           labels: List[str], title: str = "특성 벡터 비교",
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    쿼리 특성 벡터와 후보 특성 벡터 비교 시각화
    
    Args:
        query_features: 쿼리 특성 벡터
        candidate_features: 후보 특성 벡터 목록
        labels: 후보 아이템 라벨
        title: 그래프 제목
        save_path: 결과를 저장할 경로 (옵션)
        
    Returns:
        plt.Figure: 생성된 플롯 그림
    """
    # 차원 축소 (t-SNE 또는 PCA)
    # 여기서는 간단히 처음 2차원만 사용
    if query_features.shape[0] > 2:
        query_2d = query_features[:2]
    else:
        query_2d = query_features
    
    candidate_2d = []
    for feat in candidate_features:
        if feat.shape[0] > 2:
            candidate_2d.append(feat[:2])
        else:
            candidate_2d.append(feat)
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 쿼리 포인트 표시
    ax.scatter(query_2d[0], query_2d[1], color='red', s=100, marker='*', label='Query')
    
    # 후보 포인트 표시
    colors = plt.cm.viridis(np.linspace(0, 1, len(candidate_2d)))
    
    for i, (feat, label) in enumerate(zip(candidate_2d, labels)):
        ax.scatter(feat[0], feat[1], color=colors[i], s=70, alpha=0.7, label=label)
    
    # 그래프 설정
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 결과 저장
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"특성 비교 그래프 저장됨: {save_path}")
    
    return fig

if __name__ == "__main__":
    # 간단한 테스트
    import numpy as np
    
    # 테스트 데이터 생성
    query_item = {
        'class': 'smartphone',
        'blip_attributes': {
            'attributes': {
                'color': '검정색',
                'brand': '애플',
                'type': '아이폰'
            }
        },
        'fdYmd': '20250215'
    }
    
    matched_items = []
    for i in range(3):
        matched_item = {
            'class': 'smartphone' if i < 2 else 'wallet',
            'blip_attributes': {
                'attributes': {
                    'color': '검정색' if i == 0 else '흰색',
                    'brand': '애플' if i < 2 else '미상',
                    'type': '아이폰' if i == 0 else ('갤럭시' if i == 1 else '지갑')
                }
            },
            'fdYmd': '20250216',
            'fdPlaceNm': '서울특별시 강남구',
            'matching': {
                'total_score': 0.9 - i * 0.2,
                'detailed_scores': {
                    'object_class_match': 1.0 if i < 2 else 0.1,
                    'attribute_match': 0.9 - i * 0.3,
                    'clip_similarity': 0.8 - i * 0.2,
                    'metadata_match': 0.7
                },
                'match_explanation': f"테스트 설명 {i+1}"
            }
        }
        matched_items.append(matched_item)
    
    # 히트맵 테스트
    plot_similarity_heatmap(query_item, matched_items)
    plt.show()
    
    # 특성 비교 테스트
    query_features = np.random.rand(10)
    candidate_features = [
        np.random.rand(10) * 0.2 + query_features * 0.8,  # 유사
        np.random.rand(10) * 0.5 + query_features * 0.5,  # 중간
        np.random.rand(10)  # 무관
    ]
    
    labels = ["유사 항목", "중간 항목", "무관 항목"]
    plot_feature_comparison(query_features, candidate_features, labels)
    plt.show()