import os
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import logging
from PIL import Image
from io import BytesIO
import base64
import re

from config.config import POLICE_API_URL, POLICE_API_SERVICE_KEY, POLICE_CATEGORY_CODES, POLICE_COLOR_CODES

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_date(date_obj):
    """API 호출을 위해 날짜를 YYYYMMDD 형식으로 포맷팅"""
    return date_obj.strftime('%Y%m%d')

def fetch_lost_items(num_items=10, days_back=30, category_code=None, color_code=None, location_code=None):
    """
    경찰청 API에서 분실물 데이터 가져오기
    
    매개변수:
        num_items (int): 가져올 항목 수
        days_back (int): 과거 몇 일 동안의 항목을 조회할지
        category_code (str, optional): 물품 카테고리 코드
        color_code (str, optional): 색상 코드
        location_code (str, optional): 위치 코드
    
    반환값:
        list: 분실물 데이터 목록
    """
    # 날짜 범위 계산
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # 파라미터 준비
    params = {
        'serviceKey': POLICE_API_SERVICE_KEY,
        'START_YMD': format_date(start_date),
        'END_YMD': format_date(end_date),
        'pageNo': '1',
        'numOfRows': str(num_items),
        'sort': 'DESC',
        'sortField': 'fdYmd'
    }
    
    # 선택적 파라미터 추가
    if category_code:
        params['PRDT_CL_CD_01'] = category_code
    
    if color_code:
        params['FD_COL_CD'] = color_code
        
    if location_code:
        params['N_FD_LCT_CD'] = location_code
    
    try:
        logger.info(f"파라미터와 함께 경찰청 API 호출: {params}")
        response = requests.get(POLICE_API_URL, params=params)
        
        if response.status_code == 200:
            # XML 응답 파싱
            root = ET.fromstring(response.content)
            items = []
            
            # 총 개수가 있다면 추출
            total_count = root.find('.//totalCount')
            if total_count is not None and total_count.text:
                logger.info(f"사용 가능한 총 항목 수: {total_count.text}")
            
            # 항목 추출
            for item in root.findall('.//item'):
                item_data = {}
                
                for child in item:
                    # 이미지 데이터 특별 처리
                    if child.tag == 'fdFilePathImg' and child.text:
                        item_data[child.tag] = child.text
                        try:
                            # 이미지 가져오기 시도
                            img_response = requests.get(child.text)
                            if img_response.status_code == 200:
                                item_data['image_bytes'] = img_response.content
                            else:
                                logger.warning(f"이미지 가져오기 실패: {child.text}")
                        except Exception as e:
                            logger.error(f"이미지 가져오기 오류: {e}")
                    else:
                        item_data[child.tag] = child.text
                
                # 항목 데이터 처리
                process_item_data(item_data)
                items.append(item_data)
            
            logger.info(f"경찰청 API에서 {len(items)}개 항목을 성공적으로 가져옴")
            return items
        else:
            logger.error(f"API 요청 실패, 상태 코드: {response.status_code}")
            logger.error(f"응답 내용: {response.content}")
            return []
            
    except Exception as e:
        logger.error(f"경찰청 API 호출 오류: {str(e)}")
        return []

def process_item_data(item):
    """
    항목 데이터 처리 및 정리
    
    매개변수:
        item (dict): 처리할 항목 데이터 사전
    """
    # 날짜 필드가 있으면 datetime 객체로 변환
    date_fields = ['fdYmd', 'lstYmd', 'lstPrdtYmd']
    for field in date_fields:
        if field in item and item[field]:
            try:
                # YYYYMMDD에서 datetime으로 변환
                item[f'{field}_datetime'] = datetime.strptime(item[field], '%Y%m%d')
            except ValueError:
                logger.warning(f"필드 {field}의 날짜 {item[field]}를 파싱할 수 없음")
    
    # 제품명 추출 및 정리
    if 'fdPrdtNm' in item and item['fdPrdtNm']:
        item['clean_product_name'] = clean_text(item['fdPrdtNm'])
    
    # 제품 상세정보 추출
    if 'fdSbjt' in item and item['fdSbjt']:
        item['clean_subject'] = clean_text(item['fdSbjt'])
        
        # 색상 정보 추출(가능한 경우)
        colors = extract_colors(item['fdSbjt'])
        if colors:
            item['extracted_colors'] = colors
            
        # 브랜드 정보 추출(가능한 경우)
        brands = extract_brands(item['fdSbjt'])
        if brands:
            item['extracted_brands'] = brands

def clean_text(text):
    """텍스트 정리 및 정규화"""
    if not text:
        return ""
    
    # 특수 문자 제거 및 공백 정규화
    cleaned = re.sub(r'[^\w\s가-힣]', ' ', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def extract_colors(text):
    """텍스트에서 색상 정보 추출"""
    if not text:
        return []
    
    # 한국어로 된 일반적인 색상 목록
    korean_colors = ['검정', '검은', '흰', '흰색', '빨간', '빨강', '주황', '노란', '노랑', 
                     '초록', '녹색', '파란', '파랑', '남색', '보라', '분홍', '핑크', 
                     '갈색', '회색', '은색', '금색', '투명']
    
    # 색상 언급 추출
    colors = []
    for color in korean_colors:
        if color in text:
            colors.append(color)
    
    return colors

def extract_brands(text):
    """텍스트에서 브랜드 정보 추출"""
    if not text:
        return []
    
    # 일반적인 전자제품 브랜드 목록
    common_brands = ['삼성', '애플', '엘지', 'LG', '아이폰', '갤럭시', 'iPhone', 'Galaxy', 
                     '애플워치', '에어팟', 'AirPods', '맥북', 'MacBook', '소니', 'Sony']
    
    # 브랜드 언급 추출
    brands = []
    for brand in common_brands:
        if brand in text:
            brands.append(brand)
    
    return brands

def get_image_from_item(item):
    """
    항목 데이터에서 이미지 추출
    
    매개변수:
        item (dict): 항목 데이터 사전
        
    반환값:
        PIL.Image 또는 None: 이미지(사용 가능한 경우)
    """
    if 'image_bytes' in item and item['image_bytes']:
        try:
            return Image.open(BytesIO(item['image_bytes']))
        except Exception as e:
            logger.error(f"이미지 열기 오류: {e}")
    
    return None

def search_lost_items(query_text, category=None, color=None, max_items=20):
    """
    텍스트 쿼리 및 선택적 필터로 분실물 검색
    
    매개변수:
        query_text (str): 검색할 텍스트
        category (str, optional): 카테고리 이름
        color (str, optional): 색상 이름
        max_items (int): 반환할 최대 항목 수
        
    반환값:
        list: 일치하는 분실물 데이터 목록
    """
    # 카테고리와 색상이 제공된 경우 코드로 변환
    category_code = POLICE_CATEGORY_CODES.get(category) if category else None
    color_code = POLICE_COLOR_CODES.get(color) if color else None
    
    # 먼저 제공된 필터로 항목 가져오기
    items = fetch_lost_items(
        num_items=max_items * 2,  # 필터링을 위해 필요한 것보다 더 많이 가져오기
        days_back=60,
        category_code=category_code,
        color_code=color_code
    )
    
    # 쿼리 텍스트가 제공된 경우 결과를 추가로 필터링
    if query_text:
        cleaned_query = clean_text(query_text).lower()
        query_words = set(cleaned_query.split())
        
        filtered_items = []
        for item in items:
            # 제품명 확인
            if 'clean_product_name' in item and any(word in item['clean_product_name'].lower() for word in query_words):
                filtered_items.append(item)
                continue
                
            # 제목 확인
            if 'clean_subject' in item and any(word in item['clean_subject'].lower() for word in query_words):
                filtered_items.append(item)
                continue
        
        items = filtered_items
    
    # 상위 N개 항목 반환
    return items[:max_items]

if __name__ == "__main__":
    # 간단한 테스트
    items = fetch_lost_items(num_items=5)
    for i, item in enumerate(items):
        print(f"\n항목 {i+1}:")
        for key, value in item.items():
            if key != 'image_bytes':  # 바이너리 데이터 건너뛰기
                print(f"  {key}: {value}")