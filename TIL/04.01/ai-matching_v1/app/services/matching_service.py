import os
import logging
from typing import List, Dict, Any, Optional, BinaryIO
import time
from datetime import datetime
import base64
from io import BytesIO
from PIL import Image

from app.models.clip_model import CLIPModel
from app.models.blip_model import BLIPModel
from app.models.matching import LostItemMatcher
from data.police_api import fetch_lost_items, search_lost_items
from utils.preprocessing import preprocess_for_models
from config.config import POLICE_CATEGORY_CODES, POLICE_COLOR_CODES

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 모델 인스턴스 (싱글톤 패턴)
_clip_model = None
_blip_model = None

def get_clip_model() -> CLIPModel:
    """CLIP 모델 싱글톤 인스턴스 반환"""
    global _clip_model
    if _clip_model is None:
        logger.info("CLIP 모델 초기화 중...")
        _clip_model = CLIPModel()
    return _clip_model

def get_blip_model() -> BLIPModel:
    """BLIP 모델 싱글톤 인스턴스 반환"""
    global _blip_model
    if _blip_model is None:
        logger.info("BLIP 모델 초기화 중...")
        _blip_model = BLIPModel()
    return _blip_model

def process_image(image_data: BinaryIO) -> Dict[str, Any]:
    """
    이미지 파일 처리 및 특성 추출
    
    Args:
        image_data: 이미지 파일 데이터
        
    Returns:
        Dict[str, Any]: 추출된 특성
    """
    try:
        # 모델 인스턴스 가져오기
        clip_model = get_clip_model()
        blip_model = get_blip_model()
        
        # 이미지 로드 및 전처리
        image_bytes = image_data.read()
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        processed_image = preprocess_for_models(image)
        
        # 결과 초기화
        result = {
            'processed_images': processed_image,
            'original': processed_image['original']
        }
        
        # CLIP 특성 추출
        result['clip_embedding'] = clip_model.get_image_embedding(processed_image['enhanced'])
        result['clip_classification'] = clip_model.classify_image(processed_image['enhanced'])
        
        # 브랜드 감지
        brand_result = clip_model.detect_brand_logo(processed_image['enhanced'])
        result['brand_detection'] = {
            'brand': brand_result[0], 
            'confidence': brand_result[1]
        }
        
        # BLIP 속성 추출
        blip_attributes = blip_model.extract_attributes(processed_image['enhanced'])
        result['blip_attributes'] = blip_attributes
        
        # Base64 인코딩된 썸네일 추가
        thumbnail = processed_image['resized_224'].copy()
        buffered = BytesIO()
        thumbnail.save(buffered, format="JPEG", quality=80)
        result['thumbnail_base64'] = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        logger.info("이미지 처리 완료")
        return result
        
    except Exception as e:
        logger.error(f"이미지 처리 오류: {str(e)}")
        raise

def process_text_query(description: str) -> Dict[str, Any]:
    """
    텍스트 쿼리 처리 및 특성 추출
    
    Args:
        description: 물건 설명
        
    Returns:
        Dict[str, Any]: 추출된 특성
    """
    try:
        # 모델 인스턴스 가져오기
        clip_model = get_clip_model()
        
        # 텍스트 임베딩 생성
        text_embedding = clip_model.get_text_embedding(description)
        
        result = {
            'description': description,
            'text_embedding': text_embedding
        }
        
        logger.info("텍스트 쿼리 처리 완료")
        return result
        
    except Exception as e:
        logger.error(f"텍스트 쿼리 처리 오류: {str(e)}")
        raise

def fetch_database_items(category: Optional[str] = None, 
                        color: Optional[str] = None, 
                        keywords: Optional[str] = None,
                        num_items: int = 30) -> List[Dict[str, Any]]:
    """
    경찰청 API에서 습득물 데이터 가져오기
    
    Args:
        category: 물품 카테고리
        color: 물품 색상
        keywords: 검색 키워드
        num_items: 가져올 아이템 수
        
    Returns:
        List[Dict[str, Any]]: 습득물 목록
    """
    try:
        # 카테고리 및 색상 코드 변환
        category_code = POLICE_CATEGORY_CODES.get(category) if category else None
        color_code = POLICE_COLOR_CODES.get(color) if color else None
        
        # 키워드가 있으면 검색 API 사용
        if keywords:
            items = search_lost_items(
                query_text=keywords,
                category=category,
                color=color,
                max_items=num_items
            )
        else:
            # 필터만 있으면 일반 조회 API 사용
            items = fetch_lost_items(
                num_items=num_items,
                days_back=30,
                category_code=category_code,
                color_code=color_code
            )
        
        logger.info(f"{len(items)}개 습득물 데이터 가져옴")
        return items
        
    except Exception as e:
        logger.error(f"데이터베이스 항목 가져오기 오류: {str(e)}")
        raise

def extract_features_from_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    습득물 아이템에서 특성 추출
    
    Args:
        items: 습득물 아이템 목록
        
    Returns:
        List[Dict[str, Any]]: 특성이 추출된 아이템 목록
    """
    try:
        # 모델 인스턴스 가져오기
        clip_model = get_clip_model()
        blip_model = get_blip_model()
        
        processed_items = []
        start_time = time.time()
        
        for i, item in enumerate(items):
            try:
                logger.info(f"아이템 {i+1}/{len(items)} 처리 중...")
                
                # 이미지 데이터 확인
                if 'image_bytes' not in item:
                    logger.warning(f"아이템 {i+1}에 이미지 데이터가 없습니다. 건너뜁니다.")
                    continue
                
                # 이미지 전처리
                image = Image.open(BytesIO(item['image_bytes'])).convert('RGB')
                processed_image = preprocess_for_models(image)
                
                # 전처리된 이미지 저장
                item_with_features = item.copy()
                
                # CLIP 특성 추출
                item_with_features['clip_embedding'] = clip_model.get_image_embedding(processed_image['enhanced'])
                item_with_features['clip_classification'] = clip_model.classify_image(processed_image['enhanced'])
                
                # 브랜드 감지
                brand_result = clip_model.detect_brand_logo(processed_image['enhanced'])
                item_with_features['brand_detection'] = {
                    'brand': brand_result[0], 
                    'confidence': brand_result[1]
                }
                
                # BLIP 속성 추출
                blip_attributes = blip_model.extract_attributes(processed_image['enhanced'])
                item_with_features['blip_attributes'] = blip_attributes
                
                processed_items.append(item_with_features)
                
            except Exception as e:
                logger.error(f"아이템 {i+1} 처리 중 오류: {str(e)}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"{len(processed_items)}개 아이템 특성 추출 완료 (소요 시간: {elapsed_time:.2f}초)")
        return processed_items
        
    except Exception as e:
        logger.error(f"특성 추출 오류: {str(e)}")
        raise

def match_items(query_item: Dict[str, Any], database_items: List[Dict[str, Any]], 
               weights: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """
    쿼리 아이템과 데이터베이스 아이템 간 매칭 수행
    
    Args:
        query_item: 쿼리 아이템 (분실물)
        database_items: 데이터베이스 아이템 (습득물)
        weights: 매칭 가중치
        
    Returns:
        List[Dict[str, Any]]: 매칭 결과
    """
    try:
        # 매처 초기화
        matcher = LostItemMatcher(weights=weights) if weights else LostItemMatcher()
        
        # 매칭 수행
        start_time = time.time()
        matched_items = matcher.match_items(query_item, database_items)
        elapsed_time = time.time() - start_time
        
        logger.info(f"매칭 완료: {len(matched_items)}개 일치 항목 (소요 시간: {elapsed_time:.2f}초)")
        return matched_items
        
    except Exception as e:
        logger.error(f"매칭 오류: {str(e)}")
        raise

def format_response(matched_items: List[Dict[str, Any]], query_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    매칭 결과를 API 응답 형식으로 변환
    
    Args:
        matched_items: 매칭된 아이템 목록
        query_info: 쿼리 정보
        
    Returns:
        Dict[str, Any]: 포맷된 응답
    """
    results = []
    
    for item in matched_items:
        # 일자 파싱
        found_date = None
        if 'fdYmd_datetime' in item:
            found_date = item['fdYmd_datetime']
        elif 'fdYmd' in item and item['fdYmd']:
            try:
                date_str = str(item['fdYmd'])
                if len(date_str) == 8:
                    found_date = datetime.strptime(date_str, '%Y%m%d')
            except Exception:
                pass
        
        # 매칭 정보 추출
        matching_info = item.get('matching', {})
        detailed_scores = matching_info.get('detailed_scores', {})
        
        # 속성 정보 추출
        attributes = {}
        if 'blip_attributes' in item and 'attributes' in item['blip_attributes']:
            blip_attrs = item['blip_attributes']['attributes']
            attributes = {
                'color': blip_attrs.get('color'),
                'brand': blip_attrs.get('brand'),
                'type': blip_attrs.get('type'),
                'material': blip_attrs.get('material'),
                'condition': blip_attrs.get('condition')
            }
        
        # 결과 아이템 구성
        result_item = {
            'id': item.get('fdSn', str(hash(str(item)))),
            'name': item.get('fdPrdtNm', '알 수 없음'),
            'image_url': item.get('fdFilePathImg'),
            'category': item.get('prdtClNm'),
            'attributes': attributes,
            'found_date': found_date,
            'found_place': item.get('fdPlaceNm') or item.get('N_FD_LCT_NM'),
            'description': item.get('fdSbjt'),
            'matching': {
                'total_score': matching_info.get('total_score', 0.0),
                'object_class_match': detailed_scores.get('object_class_match', 0.0),
                'attribute_match': detailed_scores.get('attribute_match', 0.0),
                'clip_similarity': detailed_scores.get('clip_similarity', 0.0),
                'metadata_match': detailed_scores.get('metadata_match', 0.0),
                'explanation': matching_info.get('match_explanation', '')
            }
        }
        
        results.append(result_item)
    
    # 응답 구성
    response = {
        'total_matches': len(matched_items),
        'query_info': query_info,
        'results': results
    }
    
    return response