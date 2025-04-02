"""
경찰청 API를 통해 습득물 데이터를 가져오는 모듈
"""
import os
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import sys
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 추가하여 config.py 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import API_BASE_URL, POLICE_API_SERVICE_KEY

def fetch_police_lost_items(service_key=None, num_items=5, start_date=None, end_date=None):
    """
    경찰청 API를 통해 최신 습득물 데이터를 가져옴
    
    Args:
        service_key (str): 경찰청 API 서비스 키, 기본값은 config에서 가져옴
        num_items (int): 가져올 아이템 수
        start_date (str): 시작 날짜 (YYYYMMDD 형식), 기본값은 3개월 전
        end_date (str): 종료 날짜 (YYYYMMDD 형식), 기본값은 현재 날짜
        
    Returns:
        list: 습득물 데이터 리스트
    """
    # 서비스 키가 없으면 환경 변수에서 가져옴
    if not service_key:
        service_key = POLICE_API_SERVICE_KEY
        
        if not service_key:
            logger.error("API 서비스 키가 제공되지 않았습니다.")
            return []
    
    # 날짜 설정 (기본값: 현재부터 3개월 전까지)
    if not end_date:
        end_date = datetime.now().strftime('%Y%m%d')
    
    if not start_date:
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y%m%d')
    
    # API 요청 파라미터 설정
    params = {
        'serviceKey': service_key,
        'START_YMD': start_date,
        'END_YMD': end_date,
        'pageNo': '1',
        'numOfRows': str(num_items),
        'sort': 'DESC',
        'sortField': 'fdYmd'
    }
    
    try:
        logger.info(f"경찰청 API에서 {num_items}개 데이터 요청 중...")
        response = requests.get(API_BASE_URL, params=params)
        
        # XML 응답 파싱
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            items = []
            
            # items 태그 아래의 item 요소들 추출
            for item in root.findall('.//item'):
                item_data = {}
                
                # 각 필드 추출
                for child in item:
                    item_data[child.tag] = child.text
                
                # 이미지가 있는 경우 이미지 URL 추가
                if 'fdFilePathImg' in item_data and item_data['fdFilePathImg']:
                    item_data['image_url'] = item_data['fdFilePathImg']
                else:
                    item_data['image_url'] = None
                
                # 카테고리 정보 추출 및 정리
                if 'prdtClNm' in item_data:
                    item_data['category'] = item_data['prdtClNm']
                
                # 물품명 추출
                if 'fdPrdtNm' in item_data:
                    item_data['item_name'] = item_data['fdPrdtNm']
                
                # 색상 추출
                if 'clrNm' in item_data:
                    item_data['color'] = item_data['clrNm']
                
                # 습득물 내용 추출
                if 'fdSbjt' in item_data:
                    item_data['content'] = item_data['fdSbjt']
                    
                # 습득 장소 추출
                if 'fdPlace' in item_data:
                    item_data['location'] = item_data['fdPlace']
                
                items.append(item_data)
                
            logger.info(f"API에서 {len(items)}개 최신 습득물 데이터를 성공적으로 가져왔습니다.")
            return items
        else:
            logger.error(f"API 호출 실패: {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"API 호출 중 오류 발생: {str(e)}")
        return []

def fetch_item_details(item_id, service_key=None):
    """
    특정 습득물의 상세 정보를 가져옴
    
    Args:
        item_id (str): 습득물 ID
        service_key (str): 경찰청 API 서비스 키
        
    Returns:
        dict: 습득물 상세 정보
    """
    # 여기에 상세 정보를 가져오는 코드를 구현할 수 있음
    # 현재는 기본 프로토타입에 포함되지 않음
    pass

# 모듈 테스트용 코드
if __name__ == "__main__":
    # 환경 변수에서 서비스 키 가져오기
    SERVICE_KEY = os.getenv('POLICE_API_SERVICE_KEY')
    
    # 서비스 키가 없으면 오류 메시지 출력
    if not SERVICE_KEY:
        logger.error("POLICE_API_SERVICE_KEY 환경 변수가 설정되지 않았습니다.")
        sys.exit(1)
    
    # 최신 습득물 데이터 가져오기
    items = fetch_police_lost_items(SERVICE_KEY, 10)
    
    # 결과 확인
    for i, item in enumerate(items):
        print(f"\n아이템 {i+1}:")
        for key, value in item.items():
            print(f"  {key}: {value}")