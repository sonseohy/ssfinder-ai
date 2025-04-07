"""
스파크를 사용하여 하둡에 저장된 임베딩 데이터로 사용자 게시글과 유사한 분실물을 찾는 스크립트
"""
import os
import sys
import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lit, array
from pyspark.sql.types import FloatType, StringType, StructType, StructField, ArrayType, MapType
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import VectorAssembler
import pymysql
import requests
from io import BytesIO
from PIL import Image

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 추가하여 모듈 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.clip_model import KoreanCLIPModel
from config import CLIP_MODEL_NAME, DEVICE, SIMILARITY_THRESHOLD

# 환경 변수 로드
load_dotenv()

# MySQL 연결 설정
MYSQL_HOST = os.getenv('MYSQL_HOST', 'localhost')
MYSQL_PORT = int(os.getenv('MYSQL_PORT', 3306))
MYSQL_USER = os.getenv('MYSQL_USER', 'root')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD', '')
MYSQL_DB = os.getenv('MYSQL_DB', 'lostfound_db')

# 하둡 설정
HADOOP_HOST = os.getenv('HADOOP_HOST', 'http://ec2-x-x-x-x.compute-1.amazonaws.com')
HADOOP_PORT = os.getenv('HADOOP_PORT', '9870')
HADOOP_USER = os.getenv('HADOOP_USER', 'hadoop')
HADOOP_EMBEDDINGS_DIR = os.getenv('HADOOP_EMBEDDINGS_DIR', '/user/hadoop/embeddings')

# 스파크 세션 초기화
def create_spark_session():
    """
    스파크 세션 생성 및 초기화
    
    Returns:
        pyspark.sql.SparkSession: 스파크 세션 객체
    """
    try:
        logger.info("스파크 세션을 초기화합니다...")
        
        # 스파크 세션 생성
        spark = SparkSession.builder \
            .appName("LostFoundSimilarityMatcher") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.cores", "2") \
            .config("spark.hadoop.fs.defaultFS", f"hdfs://{HADOOP_HOST.replace('http://', '')}:{HADOOP_PORT}") \
            .config("spark.hadoop.yarn.resourcemanager.hostname", HADOOP_HOST.replace('http://', '')) \
            .getOrCreate()
            
        logger.info("스파크 세션 초기화 완료")
        return spark
    
    except Exception as e:
        logger.error(f"하둡에서 임베딩 데이터 로드 중 오류: {str(e)}")
        return {}, {}

# 유사도 계산 함수 (스파크 UDF용)
def calculate_cosine_similarity(vec1, vec2):
    """
    두 벡터 간의 코사인 유사도 계산
    
    Args:
        vec1 (list): 첫 번째 벡터
        vec2 (list): 두 번째 벡터
        
    Returns:
        float: 코사인 유사도 (0~1 사이)
    """
    try:
        # numpy 배열로 변환
        a = np.array(vec1)
        b = np.array(vec2)
        
        # 유사도 계산
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        similarity = dot_product / (norm_a * norm_b)
        
        # 유사도를 0~1 범위로 정규화 (코사인 유사도는 -1~1 범위)
        normalized_similarity = (similarity + 1) / 2
        
        return float(normalized_similarity)
    
    except Exception as e:
        logger.error(f"유사도 계산 중 오류: {str(e)}")
        return 0.0

# 사용자 게시글 임베딩 생성
def generate_user_post_embeddings(user_post, clip_model):
    """
    사용자 게시글의 임베딩 생성
    
    Args:
        user_post (dict): 사용자 게시글 정보
        clip_model (KoreanCLIPModel): CLIP 모델 인스턴스
        
    Returns:
        tuple: (텍스트 임베딩, 이미지 임베딩)
    """
    try:
        logger.info("사용자 게시글 임베딩 생성 중...")
        
        # 텍스트 데이터 결합 (카테고리, 물품명, 색상, 내용)
        text_parts = []
        
        if 'category' in user_post and user_post['category']:
            text_parts.append(f"카테고리: {user_post['category']}")
        
        if 'item_name' in user_post and user_post['item_name']:
            text_parts.append(f"물품명: {user_post['item_name']}")
        
        if 'color' in user_post and user_post['color']:
            text_parts.append(f"색상: {user_post['color']}")
        
        if 'content' in user_post and user_post['content']:
            text_parts.append(f"내용: {user_post['content']}")
        
        combined_text = " ".join(text_parts)
        
        # 텍스트 임베딩 생성
        text_embedding = clip_model.encode_text(combined_text)[0].tolist()
        
        # 이미지 임베딩 생성 (이미지가 있는 경우)
        image_embedding = None
        
        if 'image_url' in user_post and user_post['image_url']:
            # 이미지 처리
            try:
                image_url = user_post['image_url']
                
                # URL이 상대 경로인 경우 처리
                if not image_url.startswith(('http://', 'https://')):
                    # 로컬 파일 경로
                    if os.path.exists(image_url):
                        image = Image.open(image_url).convert('RGB')
                    else:
                        logger.warning(f"이미지 파일을 찾을 수 없음: {image_url}")
                        image = None
                else:
                    # URL에서 이미지 다운로드
                    response = requests.get(image_url, timeout=10)
                    if response.status_code == 200:
                        image = Image.open(BytesIO(response.content)).convert('RGB')
                    else:
                        logger.warning(f"이미지 다운로드 실패 (HTTP {response.status_code}): {image_url}")
                        image = None
                
                # 이미지 임베딩 생성
                if image is not None:
                    image_embedding = clip_model.encode_image(image)[0].tolist()
                    logger.info("이미지 임베딩 생성 완료")
                
            except Exception as e:
                logger.warning(f"이미지 처리 중 오류: {str(e)}")
        
        logger.info("사용자 게시글 임베딩 생성 완료")
        return text_embedding, image_embedding
    
    except Exception as e:
        logger.error(f"사용자 게시글 임베딩 생성 중 오류: {str(e)}")
        return None, None

# 임베딩 데이터를 스파크 데이터프레임으로 변환
def create_embeddings_dataframe(spark, text_embeddings, image_embeddings, lost_items_df):
    """
    임베딩 데이터를 스파크 데이터프레임으로 변환
    
    Args:
        spark (pyspark.sql.SparkSession): 스파크 세션 객체
        text_embeddings (dict): 텍스트 임베딩 딕셔너리
        image_embeddings (dict): 이미지 임베딩 딕셔너리
        lost_items_df (pandas.DataFrame): 분실물 데이터
        
    Returns:
        pyspark.sql.DataFrame: 임베딩 데이터가 포함된 스파크 데이터프레임
    """
    try:
        logger.info("임베딩 데이터를 스파크 데이터프레임으로 변환 중...")
        
        # 벡터 변환 함수 정의
        def to_vector(embedding):
            return Vectors.dense(embedding)
        
        # 스파크 UDF 등록
        to_vector_udf = udf(to_vector, VectorUDT())
        
        # 데이터프레임 스키마 정의
        schema = StructType([
            StructField("item_id", StringType(), True),
            StructField("text_embedding_array", ArrayType(FloatType()), True),
            StructField("has_image", StringType(), True),
            StructField("image_embedding_array", ArrayType(FloatType()), True)
        ])
        
        # 임베딩 데이터 생성
        embeddings_data = []
        
        for item_id, text_emb in text_embeddings.items():
            item_id_str = str(item_id)
            has_image = item_id_str in image_embeddings
            image_emb = image_embeddings.get(item_id_str, None)
            
            embeddings_data.append((item_id_str, text_emb, "yes" if has_image else "no", image_emb))
        
        # 스파크 데이터프레임 생성
        embeddings_df = spark.createDataFrame(embeddings_data, schema)
        
        # 벡터 변환
        embeddings_df = embeddings_df.withColumn("text_embedding", to_vector_udf("text_embedding_array"))
        embeddings_df = embeddings_df.withColumn("image_embedding", 
                                                to_vector_udf("image_embedding_array") if "yes" else lit(None))
        
        # 분실물 데이터를 스파크 데이터프레임으로 변환
        lost_items_spark_df = spark.createDataFrame(lost_items_df)
        
        # 임베딩 데이터와 분실물 데이터 조인
        result_df = embeddings_df.join(lost_items_spark_df, 
                                      embeddings_df.item_id == lost_items_spark_df.id, 
                                      "inner")
        
        logger.info(f"스파크 데이터프레임 생성 완료 (레코드 수: {result_df.count()})")
        return result_df
    
    except Exception as e:
        logger.error(f"스파크 데이터프레임 생성 중 오류: {str(e)}")
        raise

# 유사한 분실물 찾기
def find_similar_items(spark, user_post, text_embedding, image_embedding, embeddings_df, threshold=SIMILARITY_THRESHOLD):
    """
    사용자 게시글과 유사한 분실물 찾기
    
    Args:
        spark (pyspark.sql.SparkSession): 스파크 세션 객체
        user_post (dict): 사용자 게시글 정보
        text_embedding (list): 사용자 게시글 텍스트 임베딩
        image_embedding (list): 사용자 게시글 이미지 임베딩 (없으면 None)
        embeddings_df (pyspark.sql.DataFrame): 임베딩 데이터가 포함된 스파크 데이터프레임
        threshold (float): 유사도 임계값
        
    Returns:
        list: 유사한 분실물 목록 (유사도 높은 순)
    """
    try:
        logger.info(f"유사한 분실물 찾는 중... (임계값: {threshold})")
        
        # 텍스트 유사도 계산 UDF 정의
        calculate_text_similarity_udf = udf(
            lambda emb: calculate_cosine_similarity(text_embedding, emb), 
            FloatType()
        )
        
        # 임시 뷰 생성
        embeddings_df.createOrReplaceTempView("embeddings")
        
        # 텍스트 유사도 계산
        result_df = embeddings_df.withColumn("text_similarity", 
                                           calculate_text_similarity_udf("text_embedding_array"))
        
        # 이미지 유사도 계산 (이미지가 있는 경우)
        if image_embedding is not None:
            calculate_image_similarity_udf = udf(
                lambda emb: calculate_cosine_similarity(image_embedding, emb) if emb else 0.0, 
                FloatType()
            )
            
            result_df = result_df.withColumn("image_similarity", 
                                           calculate_image_similarity_udf("image_embedding_array"))
            
            # 종합 유사도 계산 (텍스트 70%, 이미지 30%)
            result_df = result_df.withColumn("total_similarity", 
                                           (col("text_similarity") * 0.7) + (col("image_similarity") * 0.3))
        else:
            # 이미지가 없는 경우 텍스트 유사도만 사용
            result_df = result_df.withColumn("image_similarity", lit(0.0))
            result_df = result_df.withColumn("total_similarity", col("text_similarity"))
        
        # 임계값 이상인 항목만 필터링
        filtered_df = result_df.filter(col("total_similarity") >= threshold)
        
        # 유사도 높은 순으로 정렬
        sorted_df = filtered_df.orderBy(col("total_similarity").desc())
        
        # 결과 데이터 수집
        results = sorted_df.select(
            "id", "category", "item_name", "color", "content", "image_url",
            "text_similarity", "image_similarity", "total_similarity"
        ).collect()
        
        # 결과 목록 변환
        similar_items = []
        
        for row in results:
            item = {
                "id": row["id"],
                "category": row["category"],
                "item_name": row["item_name"],
                "color": row["color"],
                "content": row["content"],
                "image_url": row["image_url"],
                "similarity": {
                    "text": float(row["text_similarity"]),
                    "image": float(row["image_similarity"]),
                    "total": float(row["total_similarity"])
                }
            }
            
            similar_items.append(item)
        
        logger.info(f"유사한 분실물 {len(similar_items)}개 찾음")
        return similar_items
    
    except Exception as e:
        logger.error(f"유사한 분실물 찾기 중 오류: {str(e)}")
        return []

# 메인 함수: 사용자 게시글과 유사한 분실물 찾기
def find_similar_lost_items(user_post, threshold=SIMILARITY_THRESHOLD, limit=10):
    """
    사용자 게시글과 유사한 분실물 찾기
    
    Args:
        user_post (dict): 사용자 게시글 정보
        threshold (float): 유사도 임계값
        limit (int): 최대 결과 수
        
    Returns:
        dict: 매칭 결과
    """
    try:
        logger.info("사용자 게시글과 유사한 분실물 찾기 시작")
        
        # CLIP 모델 초기화
        clip_model = initialize_clip_model()
        if clip_model is None:
            return {
                "success": False,
                "message": "CLIP 모델 초기화 실패",
                "matches": []
            }
        
        # 스파크 세션 생성
        spark = create_spark_session()
        
        # 하둡에서 최신 임베딩 경로 가져오기
        embeddings_path = get_latest_embeddings_path(spark)
        if not embeddings_path:
            return {
                "success": False,
                "message": "하둡에서 임베딩 데이터를 찾을 수 없습니다",
                "matches": []
            }
        
        # 하둡에서 임베딩 데이터 로드
        text_embeddings, image_embeddings = load_embeddings_from_hadoop(spark, embeddings_path)
        if not text_embeddings:
            return {
                "success": False,
                "message": "하둡에서 임베딩 데이터를 불러올 수 없습니다",
                "matches": []
            }
        
        # MySQL에서 분실물 데이터 가져오기
        lost_items_df = fetch_lost_items_from_mysql()
        if lost_items_df.empty:
            return {
                "success": False,
                "message": "MySQL에서 분실물 데이터를 가져올 수 없습니다",
                "matches": []
            }
        
        # 임베딩 데이터를 스파크 데이터프레임으로 변환
        embeddings_df = create_embeddings_dataframe(spark, text_embeddings, image_embeddings, lost_items_df)
        
        # 사용자 게시글 임베딩 생성
        user_text_embedding, user_image_embedding = generate_user_post_embeddings(user_post, clip_model)
        if user_text_embedding is None:
            return {
                "success": False,
                "message": "사용자 게시글 임베딩 생성 실패",
                "matches": []
            }
        
        # 유사한 분실물 찾기
        similar_items = find_similar_items(
            spark, user_post, user_text_embedding, user_image_embedding, embeddings_df, threshold
        )
        
        # 결과 제한
        similar_items = similar_items[:limit]
        
        # 스파크 세션 종료
        spark.stop()
        
        # 결과 반환
        return {
            "success": True,
            "message": f"{len(similar_items)}개의 유사한 분실물을 찾았습니다",
            "threshold": threshold,
            "total_matches": len(similar_items),
            "matches": similar_items
        }
    
    except Exception as e:
        logger.error(f"분실물 매칭 중 오류: {str(e)}")
        return {
            "success": False,
            "message": f"오류 발생: {str(e)}",
            "matches": []
        }

# FastAPI에서 호출하는 매칭 함수
def match_lost_items(user_post, threshold=SIMILARITY_THRESHOLD, limit=10):
    """
    FastAPI에서 호출하는 매칭 함수
    
    Args:
        user_post (dict): 사용자 게시글 정보
        threshold (float): 유사도 임계값
        limit (int): 최대 결과 수
        
    Returns:
        dict: 매칭 결과
    """
    return find_similar_lost_items(user_post, threshold, limit)

# 테스트 코드
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='분실물 유사도 매칭')
    parser.add_argument('--threshold', type=float, default=SIMILARITY_THRESHOLD, help='유사도 임계값')
    parser.add_argument('--limit', type=int, default=10, help='최대 결과 수')
    parser.add_argument('--category', type=str, default='지갑', help='분실물 카테고리')
    parser.add_argument('--item-name', type=str, default='검은색 가죽 지갑', help='물품명')
    parser.add_argument('--color', type=str, default='검정색', help='물품 색상')
    parser.add_argument('--content', type=str, default='지갑을 잃어버렸습니다. 현금과 카드가 들어있어요.', help='게시글 내용')
    parser.add_argument('--image', type=str, default=None, help='이미지 경로 또는 URL')
    
    args = parser.parse_args()
    
    # 테스트용 사용자 게시글
    test_post = {
        "category": args.category,
        "item_name": args.item_name,
        "color": args.color,
        "content": args.content,
        "image_url": args.image
    }
    
    # 유사한 분실물 찾기
    result = find_similar_lost_items(test_post, args.threshold, args.limit)
    
    # 결과 출력
    if result["success"]:
        print(f"🎉 {result['message']}")
        print(f"임계값: {result['threshold']}, 찾은 항목 수: {result['total_matches']}")
        
        for i, item in enumerate(result["matches"]):
            print(f"\n✅ 유사 항목 #{i+1}")
            print(f"ID: {item['id']}")
            print(f"카테고리: {item['category']}")
            print(f"물품명: {item['item_name']}")
            print(f"색상: {item['color']}")
            print(f"내용: {item['content'][:100]}..." if len(item['content']) > 100 else f"내용: {item['content']}")
            print(f"유사도: 텍스트 {item['similarity']['text']:.2f}, 이미지 {item['similarity']['image']:.2f}, 종합 {item['similarity']['total']:.2f}")
    else:
        print(f"❌ {result['message']}")
        logger.error(f"스파크 세션 초기화 중 오류: {str(e)}")
        raise

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

# MySQL에서 분실물 데이터 가져오기
def fetch_lost_items_from_mysql():
    """
    MySQL에서 분실물 게시글 데이터 가져오기
    
    Returns:
        pandas.DataFrame: 분실물 데이터
    """
    try:
        logger.info(f"MySQL에서 분실물 데이터 가져오는 중... ({MYSQL_HOST}:{MYSQL_PORT})")
        
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
                cursor.execute("SELECT * FROM lost_items")
                data = cursor.fetchall()
            
            # 데이터프레임으로 변환
            df = pd.DataFrame(data)
            
            logger.info(f"MySQL에서 {len(df)}개의 분실물 데이터를 가져왔습니다.")
            return df
    
    except Exception as e:
        logger.error(f"MySQL 데이터 가져오기 중 오류: {str(e)}")
        return pd.DataFrame()

# 하둡에서 최신 임베딩 데이터 경로 가져오기
def get_latest_embeddings_path(spark):
    """
    하둡에서 최신 임베딩 데이터 경로 가져오기
    
    Args:
        spark (pyspark.sql.SparkSession): 스파크 세션 객체
        
    Returns:
        str: 최신 임베딩 데이터 디렉토리 경로
    """
    try:
        logger.info(f"하둡에서 최신 임베딩 경로 찾는 중... ({HADOOP_EMBEDDINGS_DIR})")
        
        # 하둡 파일시스템 객체 가져오기
        hadoop_fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
        
        # 임베딩 디렉토리 경로
        path = spark._jvm.org.apache.hadoop.fs.Path(HADOOP_EMBEDDINGS_DIR)
        
        # 디렉토리가 존재하는지 확인
        if not hadoop_fs.exists(path):
            logger.error(f"하둡 경로가 존재하지 않습니다: {HADOOP_EMBEDDINGS_DIR}")
            return None
        
        # 디렉토리 내용 가져오기
        status_list = hadoop_fs.listStatus(path)
        
        # 디렉토리가 비어있는지 확인
        if not status_list or len(status_list) == 0:
            logger.error(f"하둡 디렉토리가 비어있습니다: {HADOOP_EMBEDDINGS_DIR}")
            return None
        
        # 최신 디렉토리 찾기
        latest_dir = None
        latest_time = 0
        
        for status in status_list:
            if status.isDirectory():
                dir_name = status.getPath().getName()
                modify_time = status.getModificationTime()
                
                if modify_time > latest_time:
                    latest_time = modify_time
                    latest_dir = dir_name
        
        if latest_dir:
            latest_path = f"{HADOOP_EMBEDDINGS_DIR}/{latest_dir}"
            logger.info(f"최신 임베딩 경로: {latest_path}")
            return latest_path
        else:
            logger.error("임베딩 디렉토리를 찾을 수 없습니다.")
            return None
    
    except Exception as e:
        logger.error(f"하둡 경로 가져오기 중 오류: {str(e)}")
        return None

# 하둡에서 임베딩 데이터 로드
def load_embeddings_from_hadoop(spark, embeddings_path):
    """
    하둡에서 임베딩 데이터 로드
    
    Args:
        spark (pyspark.sql.SparkSession): 스파크 세션 객체
        embeddings_path (str): 임베딩 데이터 경로
        
    Returns:
        tuple: (텍스트 임베딩 딕셔너리, 이미지 임베딩 딕셔너리)
    """
    try:
        logger.info(f"하둡에서 임베딩 데이터 로드 중... ({embeddings_path})")
        
        # 텍스트 임베딩 파일 경로
        text_embeddings_path = f"{embeddings_path}/text_embeddings.json"
        
        # 이미지 임베딩 파일 경로
        image_embeddings_path = f"{embeddings_path}/image_embeddings.json"
        
        # 텍스트 임베딩 로드
        text_embeddings_df = spark.read.text(text_embeddings_path)
        text_embeddings_json = text_embeddings_df.collect()[0][0]
        text_embeddings = json.loads(text_embeddings_json)
        
        # 이미지 임베딩 로드
        image_embeddings_df = spark.read.text(image_embeddings_path)
        image_embeddings_json = image_embeddings_df.collect()[0][0]
        image_embeddings = json.loads(image_embeddings_json)
        
        logger.info(f"텍스트 임베딩 {len(text_embeddings)}개, 이미지 임베딩 {len(image_embeddings)}개 로드 완료")
        
        return text_embeddings, image_embeddings
    
    except Exception as e:
        logger