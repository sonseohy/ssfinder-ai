import os
import requests
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import torch
from transformers import BertTokenizer, BertModel
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import re
from io import BytesIO
import base64
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 설정값
SIMILARITY_THRESHOLD = 0.7  # 유사도 임계값
BATCH_SIZE = 100  # 한 번에 처리할 습득물 데이터 수

# 단계 1: 경찰청 API에서 습득물 데이터 가져오기
def fetch_police_lost_items(service_key, num_items=10):
    """
    경찰청 API를 통해 습득물 데이터를 가져옴
    
    Args:
        service_key (str): 경찰청 API 서비스 키
        num_items (int): 가져올 아이템 수
        
    Returns:
        list: 습득물 데이터 리스트
    """
    url = 'http://apis.data.go.kr/1320000/LosfundInfoInqireService/getLosfundInfoAccToClAreaPd'
    
    # 현재는 지갑(PRH200) 카테고리만 가져오는 예시
    params = {
        'serviceKey': service_key,
        'PRDT_CL_CD_01': 'PRH000',  # 분류 코드
        'PRDT_CL_CD_02': 'PRH200',  # 지갑 분류
        'FD_COL_CD': 'CL1002',      # 분실물 색상 코드
        'START_YMD': '20240101',    # 시작 날짜 (최근 3개월)
        'END_YMD': '20240322',      # 종료 날짜 (현재 날짜)
        'N_FD_LCT_CD': 'LCA000',    # 습득 장소 코드
        'pageNo': '1',
        'numOfRows': str(num_items)
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

# 주 실행 블록
if __name__ == "__main__":
    # 환경 변수에서 서비스 키 가져오기
    SERVICE_KEY = os.getenv('POLICE_API_SERVICE_KEY')
    
    # 서비스 키가 없으면 오류 메시지 출력
    if not SERVICE_KEY:
        print("Error: POLICE_API_SERVICE_KEY 환경 변수가 설정되지 않았습니다.")
        print("다음 방법 중 하나로 서비스 키를 설정하세요:")
        print("1. .env 파일에 POLICE_API_SERVICE_KEY=your_key_here 추가")
        print("2. 시스템 환경 변수로 설정")
        print("3. 스크립트 실행 시 export POLICE_API_SERVICE_KEY=your_key_here")
        exit(1)
    
    # 습득물 데이터 가져오기
    items = fetch_police_lost_items(SERVICE_KEY, 20)
    
    # 결과 확인
    for i, item in enumerate(items[:3]):  # 처음 3개만 출력
        print(f"\n아이템 {i+1}:")
        for key, value in item.items():
            print(f"  {key}: {value}")


