import os
import json
import requests
from PIL import Image
from io import BytesIO
import logging
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import csv

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_image_from_path(image_path: str) -> Optional[Image.Image]:
    """
    파일 경로에서 이미지 로드
    
    Args:
        image_path (str): 이미지 파일 경로
        
    Returns:
        Optional[Image.Image]: 로드된 이미지 또는 None
    """
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        logger.error(f"이미지 로드 오류 ({image_path}): {str(e)}")
        return None

def load_image_from_url(image_url: str) -> Optional[Image.Image]:
    """
    URL에서 이미지 로드
    
    Args:
        image_url (str): 이미지 URL
        
    Returns:
        Optional[Image.Image]: 로드된 이미지 또는 None
    """
    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)).convert('RGB')
            return image
        else:
            logger.error(f"이미지 URL 요청 실패: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"이미지 URL 로드 오류: {str(e)}")
        return None

def load_image_from_bytes(image_bytes: bytes) -> Optional[Image.Image]:
    """
    바이트 데이터에서 이미지 로드
    
    Args:
        image_bytes (bytes): 이미지 바이트 데이터
        
    Returns:
        Optional[Image.Image]: 로드된 이미지 또는 None
    """
    try:
        image = Image.open(BytesIO(image_bytes)).convert('RGB')
        return image
    except Exception as e:
        logger.error(f"이미지 바이트 로드 오류: {str(e)}")
        return None

def save_results_to_json(items: List[Dict[str, Any]], output_path: str) -> bool:
    """
    결과를 JSON 파일로 저장
    
    Args:
        items (List[Dict[str, Any]]): 저장할 아이템 목록
        output_path (str): 저장할 파일 경로
        
    Returns:
        bool: 성공 여부
    """
    try:
        # 직렬화할 수 없는 객체 처리
        serializable_items = []
        
        for item in items:
            serializable_item = {}
            
            for key, value in item.items():
                # 직렬화할 수 없는 필드 건너뛰기
                if key in ['image_bytes', 'clip_embedding', 'text_embedding', 
                          'processed_images', 'original']:
                    continue
                
                # 브랜드 감지 정보 처리
                if key == 'brand_detection' and isinstance(value, dict):
                    serializable_item[key] = {
                        'brand': value.get('brand'),
                        'confidence': float(value.get('confidence', 0.0))
                    }
                    continue
                
                # 매칭 정보 처리
                if key == 'matching' and isinstance(value, dict):
                    matching_info = {}
                    for match_key, match_value in value.items():
                        if match_key == 'detailed_scores' and isinstance(match_value, dict):
                            matching_info[match_key] = {k: float(v) for k, v in match_value.items()}
                        elif isinstance(match_value, (int, float, str, bool, dict, list)):
                            matching_info[match_key] = match_value
                        else:
                            matching_info[match_key] = str(match_value)
                    
                    serializable_item[key] = matching_info
                    continue
                
                # 기본 타입 또는 직렬화 가능한 객체 처리
                if isinstance(value, (int, float, str, bool, dict, list, tuple)):
                    serializable_item[key] = value
                else:
                    serializable_item[key] = str(value)
            
            serializable_items.append(serializable_item)
        
        # JSON 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_items, f, ensure_ascii=False, indent=2)
        
        logger.info(f"결과가 JSON 파일로 저장됨: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"JSON 저장 오류: {str(e)}")
        return False

def save_results_to_csv(items: List[Dict[str, Any]], output_path: str, 
                       fields: Optional[List[str]] = None) -> bool:
    """
    결과를 CSV 파일로 저장
    
    Args:
        items (List[Dict[str, Any]]): 저장할 아이템 목록
        output_path (str): 저장할 파일 경로
        fields (Optional[List[str]]): 저장할 필드 목록 (기본값: 모든 기본 필드)
        
    Returns:
        bool: 성공 여부
    """
    try:
        # 기본 필드 정의
        if fields is None:
            fields = [
                'fdSn', 'fdPrdtNm', 'fdYmd', 'fdHor', 'fdPlace', 'fdPlaceSeNm',
                'fdFilePathImg', 'prdtClNm', 'depPlace', 'csteSteNm'
            ]
            
            # 매칭 필드 추가
            fields.extend(['matching_score', 'matching_explanation'])
        
        # CSV 데이터 준비
        rows = []
        
        for item in items:
            row = {}
            
            # 기본 필드 처리
            for field in fields:
                if field in item:
                    row[field] = item[field]
                elif field == 'matching_score' and 'matching' in item:
                    row[field] = item['matching'].get('total_score', 0.0)
                elif field == 'matching_explanation' and 'matching' in item:
                    row[field] = item['matching'].get('match_explanation', '')
                else:
                    row[field] = ''
            
            rows.append(row)
        
        # DataFrame 생성 및 저장
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"결과가 CSV 파일로 저장됨: {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"CSV 저장 오류: {str(e)}")
        return False

def load_database_from_directory(directory_path: str) -> List[Dict[str, Any]]:
    """
    디렉토리에서 이미지 데이터베이스 로드
    
    Args:
        directory_path (str): 이미지가 포함된 디렉토리 경로
        
    Returns:
        List[Dict[str, Any]]: 로드된 아이템 목록
    """
    items = []
    
    # 디렉토리 내 모든 이미지 파일 검색
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    
    try:
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # 이미지 파일만 처리
            if os.path.isfile(file_path) and any(filename.lower().endswith(ext) for ext in image_extensions):
                # 파일에서 메타데이터 추출 (파일명 기반)
                item = {
                    'fdFilePathImg': file_path,
                    'fdPrdtNm': os.path.splitext(filename)[0],  # 확장자 제외한 파일명
                }
                
                # 이미지 데이터 로드
                try:
                    with open(file_path, 'rb') as f:
                        item['image_bytes'] = f.read()
                    items.append(item)
                except Exception as e:
                    logger.error(f"이미지 데이터 로드 오류 ({file_path}): {str(e)}")
        
        logger.info(f"디렉토리에서 {len(items)}개 이미지 로드됨: {directory_path}")
        return items
        
    except Exception as e:
        logger.error(f"디렉토리 로드 오류: {str(e)}")
        return []

def load_database_from_csv(csv_path: str, image_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    CSV 파일에서 데이터베이스 로드
    
    Args:
        csv_path (str): CSV 파일 경로
        image_dir (Optional[str]): 이미지가 저장된 디렉토리 경로
        
    Returns:
        List[Dict[str, Any]]: 로드된 아이템 목록
    """
    items = []
    
    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        
        # 각 행을 아이템으로 변환
        for _, row in df.iterrows():
            item = row.to_dict()
            
            # 이미지 경로 필드 확인
            image_path_field = None
            for field in ['fdFilePathImg', 'image_path', 'filepath']:
                if field in item and item[field]:
                    image_path_field = field
                    break
            
            if image_path_field and image_dir:
                # 이미지 파일 로드
                image_filename = os.path.basename(item[image_path_field])
                image_path = os.path.join(image_dir, image_filename)
                
                if os.path.exists(image_path):
                    try:
                        with open(image_path, 'rb') as f:
                            item['image_bytes'] = f.read()
                    except Exception as e:
                        logger.error(f"이미지 데이터 로드 오류 ({image_path}): {str(e)}")
            
            items.append(item)
        
        logger.info(f"CSV 파일에서 {len(items)}개 아이템 로드됨: {csv_path}")
        return items
        
    except Exception as e:
        logger.error(f"CSV 로드 오류: {str(e)}")
        return []

if __name__ == "__main__":
    # 간단한 테스트
    
    # 테스트 CSV 생성
    test_data = [
        {'id': 1, 'name': '스마트폰', 'description': '검정색 갤럭시'},
        {'id': 2, 'name': '지갑', 'description': '갈색 가죽 지갑'},
        {'id': 3, 'name': '열쇠', 'description': '자동차 키 3개'}
    ]
    
    df = pd.DataFrame(test_data)
    test_csv = 'test_items.csv'
    df.to_csv(test_csv, index=False, encoding='utf-8-sig')
    
    # CSV에서 로드 테스트
    items = load_database_from_csv(test_csv)
    print(f"CSV에서 {len(items)}개 아이템 로드됨")
    
    # JSON으로 저장 테스트
    test_json = 'test_items.json'
    save_results_to_json(items, test_json)
    
    # 파일 정리
    for file in [test_csv, test_json]:
        if os.path.exists(file):
            os.remove(file)