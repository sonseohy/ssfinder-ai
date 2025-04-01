import os
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import json
from datetime import datetime, timedelta
import tempfile
from PIL import Image
from io import BytesIO
import urllib.request

class PoliceApiClient:
    """
    경찰청 분실물 API 데이터 수집 클라이언트
    """
    def __init__(self, service_key: Optional[str] = None):
        """
        API 클라이언트 초기화
        
        Args:
            service_key: 경찰청 API 서비스 키 (None인 경우 환경 변수에서 가져옴)
        """
        self.service_key = service_key or os.getenv('POLICE_API_SERVICE_KEY')
        if not self.service_key:
            raise ValueError("POLICE_API_SERVICE_KEY 환경 변수가 설정되지 않았습니다.")
            
        self.base_url = 'http://apis.data.go.kr/1320000/LosfundInfoInqireService'
        
    def fetch_lost_items(self, num_items: int = 10, days_ago: int = 30) -> List[Dict[str, Any]]:
        """
        경찰청 API에서 습득물 데이터 가져오기
        
        Args:
            num_items: 가져올 데이터 수
            days_ago: 몇 일 전부터의 데이터를 가져올지 설정
            
        Returns:
            List[Dict[str, Any]]: 습득물 데이터 목록
        """
        # 날짜 범위 계산
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_ago)
        
        start_ymd = start_date.strftime('%Y%m%d')
        end_ymd = end_date.strftime('%Y%m%d')
        
        # API 엔드포인트 및 파라미터 설정
        url = f'{self.base_url}/getLosfundInfoAccToClAreaPd'
        params = {
            'serviceKey': self.service_key,
            'START_YMD': start_ymd,
            'END_YMD': end_ymd,
            'pageNo': '1',
            'numOfRows': str(num_items),
            'sort': 'DESC',
            'sortField': 'fdYmd'
        }
        
        try:
            response = requests.get(url, params=params)
            
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
                    
                    items.append(item_data)
                    
                print(f"API에서 {len(items)}개 습득물 데이터를 성공적으로 가져왔습니다.")
                return items
            else:
                print(f"API 호출 실패: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"API 호출 중 오류 발생: {str(e)}")
            return []
    
    def get_item_image_url(self, item: Dict[str, Any]) -> Optional[str]:
        """
        습득물 데이터에서 이미지 URL 추출
        
        Args:
            item: 습득물 데이터
            
        Returns:
            Optional[str]: 이미지 URL 또는 None
        """
        # 이미지 URL 필드 (atcId 기반으로 URL 생성)
        if 'atcId' in item and item['atcId']:
            # 예제 URL을 생성합니다. 실제 URL 구조는 API 문서를 참고해야 합니다.
            image_url = f"http://www.lost112.go.kr/lostnfs/images/thumb/{item['atcId']}"
            return image_url
        return None
    
    def download_image(self, url: str) -> Optional[str]:
        """
        이미지 URL에서 이미지 다운로드
        
        Args:
            url: 이미지 URL
            
        Returns:
            Optional[str]: 저장된 이미지 경로 또는 None
        """
        try:
            # 임시 파일 생성
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_path = temp_file.name
            temp_file.close()
            
            # 이미지 다운로드
            urllib.request.urlretrieve(url, temp_path)
            
            # 이미지 유효성 확인
            try:
                img = Image.open(temp_path)
                img.verify()  # 유효한 이미지 확인
                return temp_path
            except:
                os.remove(temp_path)
                return None
                
        except Exception as e:
            print(f"이미지 다운로드 오류: {e}")
            return None
    
    def convert_to_post_format(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        경찰청 API 데이터를 게시글 형식으로 변환
        
        Args:
            items: 경찰청 API에서 가져온 습득물 데이터
            
        Returns:
            List[Dict[str, Any]]: 게시글 형식으로 변환된 데이터
        """
        posts = []
        
        for i, item in enumerate(items):
            # 필수 필드 확인
            if 'fdSbjt' not in item or not item['fdSbjt']:
                continue
                
            # 게시글 데이터 생성
            post = {
                "id": f"police_{i+1}",
                "title": item.get('fdSbjt', '제목 없음'),
                "content": f"습득물 종류: {item.get('prdtClNm', '')}\n"
                           f"습득 장소: {item.get('fdPlace', '')}\n"
                           f"습득 일자: {item.get('fdYmd', '')}\n"
                           f"보관 장소: {item.get('depPlace', '경찰서')}\n",
                "category": self._map_category(item.get('prdtClNm', '')),
                "image_path": None,
                "created_at": datetime.now().isoformat()
            }
            
            # 이미지 URL 가져오기
            image_url = self.get_item_image_url(item)
            if image_url:
                image_path = self.download_image(image_url)
                if image_path:
                    post["image_path"] = image_path
            
            posts.append(post)
        
        return posts
    
    def _map_category(self, product_category: str) -> str:
        """
        경찰청 API 분류를 내부 카테고리로 매핑
        
        Args:
            product_category: 경찰청 API의 분류명
            
        Returns:
            str: 내부 카테고리
        """
        # 기본 매핑 테이블
        mapping = {
            '지갑': 'wallet_items',
            '가방': 'lost_items_general',
            '핸드폰': 'electronics_general',
            '스마트폰': 'electronics_general',
            '아이폰': 'electronics_apple',
            '갤럭시': 'electronics_samsung',
            '현금': 'money_items',
            '신분증': 'wallet_items',
            '의류': 'lost_items_general',
            '도서': 'lost_items_general',
            '귀금속': 'lost_items_general',
            '카드': 'wallet_items',
            '안경': 'lost_items_general'
        }
        
        # 분류명에 매핑 키워드가 있는지 확인
        for key, category in mapping.items():
            if key in product_category:
                return category
        
        return 'lost_items_general'  # 기본 카테고리