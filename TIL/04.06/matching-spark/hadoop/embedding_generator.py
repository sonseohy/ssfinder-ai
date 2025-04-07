"""
MySQL에서 분실물 게시글 데이터를 가져와 임베딩을 계산하고 하둡에 저장하는 스크립트
"""
import os
import sys
import logging
import pandas as pd
import numpy as np
import pymysql
import json
from datetime import datetime
from dotenv import load_dotenv
from hdfs import InsecureClient
from io import BytesIO
from PIL import Image
import requests
import base64

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 추가하여 모듈 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.clip_model import KoreanCLIPModel
from config import CLIP_MODEL_NAME, DEVICE, BATCH_SIZE

# 환경 변수 로드
load_dotenv()

# MySQL 연결 설정
MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))
MYSQL_USER = os.getenv('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
MYSQL_DB = os.getenv('MYSQL_DB', 'lostfound_db')

# 하둡 설정
HADOOP_HOST = os.getenv('HADOOP_HOST', 'hdfs://nn1:9000')
HADOOP_PORT = os.getenv('HADOOP_PORT', '9870')
HADOOP_USER = os.getenv('HADOOP_USER', 'hadoop')
HADOOP_EMBEDDINGS_DIR = os.getenv('HADOOP_EMBEDDINGS_DIR', '/user/local/embeddings')

# CLIP 모델 초기화
def initialize_clip_model():
    """CLIP 모델 초기화"""
    try:
        logger.info("CLIP 모델을 초기화합니다...")
        model = KoreanCLIPModel(model_name=CLIP_MODEL_NAME, device=DEVICE)
        logger.info("CLIP 모델 초기화 완료")
        return model
    except Exception as e:
        logger.error(f"CLIP 모델 초기화 중 오류 발생: {str(e)}")
        return None

# MySQL에서 데이터 가져오기
def fetch_data_from_mysql(
    query="SELECT * FROM found_item ORDER BY created_at DESC LIMIT %s",
    params=(1000,)
):
    """
    MySQL에서 분실물 게시글 데이터를 가져옴
    
    Args:
        query (str): SQL 쿼리
        params (tuple): 쿼리 파라미터
        
    Returns:
        pandas.DataFrame: 분실물 게시글 데이터
    """
    try:
        logger.info(f"MySQL에서 데이터를 가져옵니다... ({MYSQL_HOST}:{MYSQL_PORT})")
        
        # MySQL 연결
        connection = pymysql.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DB,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(query, params)
                data = cursor.fetchall()
            
            # 데이터프레임으로 변환
            df = pd.DataFrame(data)
            
            logger.info(f"MySQL에서 {len(df)} 개의 레코드를 가져왔습니다.")
            return df
    
    except Exception as e:
        logger.error(f"MySQL 데이터 가져오기 중 오류: {str(e)}")
        return pd.DataFrame()

# 데이터 전처리
def preprocess_data(df):
    """
    분실물 게시글 데이터 전처리
    
    Args:
        df (pandas.DataFrame): 원본 데이터
        
    Returns:
        pandas.DataFrame: 전처리된 데이터
    """
    try:
        logger.info("데이터 전처리 중...")
        
        # null 값 처리
        if 'content' in df.columns:
            df['content'] = df['content'].fillna('')
        if 'category' in df.columns:
            df['category'] = df['category'].fillna('')
        if 'item_name' in df.columns:
            df['item_name'] = df['item_name'].fillna('')
        if 'color' in df.columns:
            df['color'] = df['color'].fillna('')
        
        # 이미지 URL이 상대 경로인 경우 처리
        # 현재는 이미지 URL이 있는지 확인만 함
        if 'image_url' in df.columns:
            df['image'] = df['image_url'].notna() & (df['image_url'] != '')
        
        logger.info("데이터 전처리 완료")
        return df
    
    except Exception as e:
        logger.error(f"데이터 전처리 중 오류: {str(e)}")
        return df

# 이미지 다운로드 및 처리
def process_image(image_url, max_retries=3):
    """
    이미지 URL에서 이미지 다운로드 및 처리
    
    Args:
        image_url (str): 이미지 URL
        max_retries (int): 최대 재시도 횟수
        
    Returns:
        PIL.Image or None: 처리된 이미지 객체 또는 None (실패 시)
    """
    if not image_url:
        return None
    
    retries = 0
    while retries < max_retries:
        try:
            # URL이 상대 경로인 경우 처리
            if not image_url.startswith(('http://', 'https://')):
                # 여기서는 예시로 로컬 경로로 가정
                if os.path.exists(image_url):
                    return Image.open(image_url).convert('RGB')
                else:
                    logger.warning(f"이미지 파일을 찾을 수 없음: {image_url}")
                    return None
            
            # URL에서 이미지 다운로드
            response = requests.get(image_url, stream=True, timeout=10)
            if response.status_code == 200:
                return Image.open(BytesIO(response.content)).convert('RGB')
            else:
                logger.warning(f"이미지 다운로드 실패 (HTTP {response.status_code}): {image_url}")
                retries += 1
        except Exception as e:
            logger.warning(f"이미지 처리 중 오류 ({retries+1}/{max_retries}): {str(e)}")
            retries += 1
    
    return None

# 텍스트 임베딩 생성
def generate_text_embeddings(df, clip_model, batch_size=BATCH_SIZE):
    """
    분실물 게시글의 텍스트 임베딩 생성
    
    Args:
        df (pandas.DataFrame): 분실물 게시글 데이터
        clip_model (KoreanCLIPModel): CLIP 모델 인스턴스
        batch_size (int): 배치 크기
        
    Returns:
        dict: 각 게시글 ID에 대한 텍스트 임베딩
    """
    text_embeddings = {}
    
    try:
        total_items = len(df)
        logger.info(f"{total_items}개 게시글의 텍스트 임베딩을 생성합니다...")
        
        # 배치 처리
        for i in range(0, total_items, batch_size):
            batch_df = df.iloc[i:min(i+batch_size, total_items)]
            logger.info(f"배치 처리 중: {i+1}-{min(i+batch_size, total_items)}/{total_items}")
            
            # 각 게시글에 대한 문장 생성
            batch_texts = []
            batch_ids = []
            
            for _, row in batch_df.iterrows():
                item_id = row['id']
                batch_ids.append(item_id)
                
                # 텍스트 데이터 결합 (카테고리, 물품명, 색상, 내용)
                text_parts = []
                
                if 'category' in row and row['category']:
                    text_parts.append(f"카테고리: {row['category']}")
                
                if 'item_name' in row and row['item_name']:
                    text_parts.append(f"물품명: {row['item_name']}")
                
                if 'color' in row and row['color']:
                    text_parts.append(f"색상: {row['color']}")
                
                if 'content' in row and row['content']:
                    text_parts.append(f"내용: {row['content']}")
                
                combined_text = " ".join(text_parts)
                batch_texts.append(combined_text)
            
            # 배치 임베딩 생성
            if batch_texts:
                embeddings = clip_model.encode_text(batch_texts)
                
                # 결과 저장
                for i, item_id in enumerate(batch_ids):
                    text_embeddings[item_id] = embeddings[i].tolist()
        
        logger.info(f"{len(text_embeddings)}개 텍스트 임베딩 생성 완료")
        return text_embeddings
    
    except Exception as e:
        logger.error(f"텍스트 임베딩 생성 중 오류: {str(e)}")
        return text_embeddings

# 이미지 임베딩 생성
def generate_image_embeddings(df, clip_model):
    """
    분실물 게시글의 이미지 임베딩 생성
    
    Args:
        df (pandas.DataFrame): 분실물 게시글 데이터
        clip_model (KoreanCLIPModel): CLIP 모델 인스턴스
        
    Returns:
        dict: 각 게시글 ID에 대한 이미지 임베딩
    """
    image_embeddings = {}
    
    try:
        # 이미지가 있는 게시글만 필터링
        image_df = df[df['image'] == True].copy()
        total_images = len(image_df)
        
        if total_images == 0:
            logger.info("이미지가 있는 게시글이 없습니다.")
            return image_embeddings
        
        logger.info(f"{total_images}개 게시글의 이미지 임베딩을 생성합니다...")
        
        # 각 이미지 처리
        for idx, row in image_df.iterrows():
            item_id = row['id']
            image_url = row['image_url']
            
            logger.info(f"이미지 처리 중 ({image_df.index.get_loc(idx)+1}/{total_images}): ID={item_id}")
            
            # 이미지 다운로드 및 처리
            image = process_image(image_url)
            
            if image:
                # 이미지 임베딩 생성
                embedding = clip_model.encode_image(image)
                image_embeddings[item_id] = embedding[0].tolist()
            else:
                logger.warning(f"이미지를 처리할 수 없음 (ID={item_id}): {image_url}")
        
        logger.info(f"{len(image_embeddings)}개 이미지 임베딩 생성 완료")
        return image_embeddings
    
    except Exception as e:
        logger.error(f"이미지 임베딩 생성 중 오류: {str(e)}")
        return image_embeddings

# 임베딩 결과 하둡에 저장
def save_embeddings_to_hadoop(text_embeddings, image_embeddings, timestamp=None):
    """
    생성된 임베딩을 EC2 API를 통해 하둡에 저장
    
    Args:
        text_embeddings (dict): 텍스트 임베딩
        image_embeddings (dict): 이미지 임베딩
        timestamp (str, optional): 타임스탬프 (디렉토리명)
        
    Returns:
        bool: 저장 성공 여부
    """
    try:
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # EC2 API URL
        ec2_api_url = os.getenv('EC2_API_URL', 'http://43.201.252.40:5000')
        
        logger.info(f"EC2 API를 통해 임베딩 저장 중... ({ec2_api_url})")
        
        # API 요청 데이터
        request_data = {
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
            "timestamp": timestamp
        }
        
        # API 호출
        response = requests.post(
            f"{ec2_api_url}/api/embeddings/save",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=120  # 큰 데이터에 대비한 긴 타임아웃
        )
        
        # 응답 확인
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                logger.info(f"임베딩 저장 성공: {result.get('path')}")
                return True
            else:
                logger.error(f"API 오류: {result.get('message')}")
                return False
        else:
            logger.error(f"API 응답 오류 (HTTP {response.status_code}): {response.text}")
            return False
        
    except Exception as e:
        logger.error(f"임베딩 저장 중 오류: {str(e)}")
        return False

# 임베딩 생성 및 저장 메인 함수
def generate_and_save_embeddings(batch_size=BATCH_SIZE, limit=None):
    """
    MySQL에서 데이터를 가져와 임베딩을 생성하고 하둡에 저장하는 메인 함수
    
    Args:
        batch_size (int): 배치 크기
        limit (int, optional): 가져올 최대 레코드 수
        
    Returns:
        bool: 성공 여부
    """
    try:
        # 타임스탬프 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"임베딩 생성 및 저장 작업 시작 (timestamp: {timestamp})")
        
        # CLIP 모델 초기화
        clip_model = initialize_clip_model()
        if clip_model is None:
            logger.error("CLIP 모델 초기화 실패. 작업을 종료합니다.")
            return False
        
        # MySQL에서 데이터 가져오기
        query = "SELECT * FROM found_item"
        params = tuple()
        
        if limit:
            query += " LIMIT %s"
            params = (limit,)
        
        df = fetch_data_from_mysql(query, params)
        
        if df.empty:
            logger.error("MySQL에서 데이터를 가져오지 못했습니다. 작업을 종료합니다.")
            return False
        
        # 데이터 전처리
        df = preprocess_data(df)
        
        # 텍스트 임베딩 생성
        text_embeddings = generate_text_embeddings(df, clip_model, batch_size)
        
        # 이미지 임베딩 생성
        image_embeddings = generate_image_embeddings(df, clip_model)
        
        # 하둡에 임베딩 저장
        success = save_embeddings_to_hadoop(text_embeddings, image_embeddings, timestamp)
        
        if success:
            logger.info("임베딩 생성 및 저장 작업이 성공적으로 완료되었습니다.")
        else:
            logger.warning("임베딩은 생성되었지만 하둡 저장에 문제가 발생했습니다.")
        
        return success
    
    except Exception as e:
        logger.error(f"임베딩 생성 및 저장 작업 중 오류: {str(e)}")
        return False

# 메인 실행 코드
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='분실물 게시글 임베딩 생성 및 하둡 저장')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='배치 크기')
    parser.add_argument('--limit', type=int, default=None, help='최대 레코드 수')
    
    args = parser.parse_args()
    
    generate_and_save_embeddings(batch_size=args.batch_size, limit=args.limit)