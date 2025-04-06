"""
Spark + Ray 통합 모듈 (SparkRayTaskProcessor 없이 직접 구현)
MySQL 데이터를 가져와 임베딩 생성하는 분산 처리 구현
"""
import os
import sys
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, pandas_udf
from pyspark.sql.types import ArrayType, FloatType, StringType
import ray
import numpy as np
import pandas as pd

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 추가하여 모듈 임포트 가능하게 함
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) if current_dir.endswith('/utils') else os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from models.clip_model import KoreanCLIPModel
    from utils.similarity import calculate_text_similarity
    from config import (
        MYSQL_URL, MYSQL_USER, MYSQL_PASSWORD, 
        SPARK_MASTER, SPARK_APP_NAME, SPARK_EXECUTOR_MEMORY, SPARK_DRIVER_MEMORY,
        BATCH_SIZE, NUM_CLIP_SERVICES, CLIP_MODEL_NAME
    )
except ImportError as e:
    logger.warning(f"유틸리티 임포트 실패: {str(e)}")
    raise

# Ray 서비스로 CLIP 모델 구현
@ray.remote
class ClipService:
    def __init__(self, model_name=CLIP_MODEL_NAME):
        """
        CLIP 서비스 초기화
        
        Args:
            model_name (str): 사용할 CLIP 모델 이름
        """
        # 기존 KoreanCLIPModel 활용
        self.clip_model = KoreanCLIPModel(model_name=model_name)
        logger.info(f"Ray CLIP 서비스 초기화 완료 (model: {model_name}, device: {self.clip_model.device})")
    
    def encode_text(self, texts):
        """
        텍스트 배치를 임베딩 벡터로 변환
        
        Args:
            texts (list): 인코딩할 텍스트 리스트
            
        Returns:
            list: 임베딩 벡터 리스트
        """
        try:
            embeddings = self.clip_model.encode_batch_texts(texts)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"텍스트 인코딩 오류: {str(e)}")
            # 오류 발생 시 0 벡터 반환
            return [np.zeros(512).tolist() for _ in range(len(texts))]
    
    def encode_image(self, image_urls):
        """
        이미지 URL 배치를 임베딩 벡터로 변환
        
        Args:
            image_urls (list): 인코딩할 이미지 URL 리스트
            
        Returns:
            list: 임베딩 벡터 리스트
        """
        results = []
        
        for url in image_urls:
            if not url:
                results.append(np.zeros(512).tolist())
                continue
                
            try:
                embedding = self.clip_model.encode_image(url)
                results.append(embedding[0].tolist())
            except Exception as e:
                logger.error(f"이미지 인코딩 오류: {str(e)}")
                results.append(np.zeros(512).tolist())
                
        return results

class MySQLSparkProcessor:
    """
    MySQL 데이터를 Spark로 처리하는 클래스
    (수정: SparkRayTaskProcessor 사용하지 않고 직접 구현)
    """
    
    def __init__(
        self, 
        app_name=SPARK_APP_NAME, 
        master=SPARK_MASTER, 
        executor_memory=SPARK_EXECUTOR_MEMORY,
        driver_memory=SPARK_DRIVER_MEMORY,
        mysql_url=MYSQL_URL,
        mysql_user=MYSQL_USER,
        mysql_password=MYSQL_PASSWORD,
        num_clip_services=NUM_CLIP_SERVICES
    ):
        """
        MySQLSparkProcessor 초기화
        
        Args:
            app_name (str): Spark 앱 이름
            master (str): Spark 마스터 URL
            executor_memory (str): Spark 실행기 메모리
            driver_memory (str): Spark 드라이버 메모리
            mysql_url (str): MySQL JDBC URL
            mysql_user (str): MySQL 사용자 이름
            mysql_password (str): MySQL 암호
            num_clip_services (int): Ray CLIP 서비스 인스턴스 수
        """
        self.app_name = app_name
        self.master = master
        self.executor_memory = executor_memory
        self.driver_memory = driver_memory
        self.mysql_url = mysql_url
        self.mysql_user = mysql_user
        self.mysql_password = mysql_password
        self.num_clip_services = num_clip_services
        
        self.spark = None
        self.clip_services = []
        self.service_index = 0
        self.ray_initialized = False
        
        # MySQL JDBC 드라이버 포함 여부 확인
        self.has_mysql_driver = self._check_mysql_driver()
        
    def _check_mysql_driver(self):
        """
        MySQL JDBC 드라이버 존재 여부 확인
        
        Returns:
            bool: 드라이버 존재 여부
        """
        try:
            # MySQL JDBC 드라이버 확인
            mysql_driver_path = os.path.join(project_root, "lib", "mysql-connector-java-8.0.28.jar")
            
            if os.path.exists(mysql_driver_path):
                logger.info(f"MySQL JDBC 드라이버 발견: {mysql_driver_path}")
                return True
            else:
                # 환경 변수에서 CLASSPATH 확인
                classpath = os.environ.get('CLASSPATH', '')
                if 'mysql-connector' in classpath:
                    logger.info("MySQL JDBC 드라이버가 CLASSPATH에 포함되어 있습니다.")
                    return True
                
                logger.warning("MySQL JDBC 드라이버를 찾을 수 없습니다. lib 디렉토리에 추가하거나 CLASSPATH에 포함시켜주세요.")
                return False
        except Exception as e:
            logger.error(f"MySQL 드라이버 확인 중 오류: {str(e)}")
            return False
        
    def start(self):
        """
        Spark 및 Ray 서비스 시작
        
        Returns:
            bool: 성공 여부
        """
        try:
            # Spark 세션 초기화
            spark_builder = SparkSession.builder \
                .appName(self.app_name) \
                .master(self.master) \
                .config("spark.executor.memory", self.executor_memory) \
                .config("spark.driver.memory", self.driver_memory)
            
            # MySQL JDBC 드라이버 설정
            if self.has_mysql_driver:
                spark_builder = spark_builder.config("spark.jars", "lib/mysql-connector-java-8.0.28.jar")
                
            self.spark = spark_builder.getOrCreate()
                
            logger.info(f"Spark 세션 초기화 완료 (master: {self.master})")
            
            # Ray 초기화 (아직 초기화되지 않은 경우)
            if not ray.is_initialized():
                ray.init(address="auto", ignore_reinit_error=True)
                self.ray_initialized = True
                logger.info("Ray 초기화 완료")
            
            # CLIP 서비스 인스턴스 생성
            self.clip_services = [ClipService.remote() for _ in range(self.num_clip_services)]
            logger.info(f"{self.num_clip_services}개의 CLIP 서비스 인스턴스 생성됨")
            
            return True
            
        except Exception as e:
            logger.error(f"Spark/Ray 초기화 오류: {str(e)}")
            return False
        
    def stop(self):
        """
        Spark 및 Ray 서비스 종료
        """
        if self.spark:
            self.spark.stop()
            logger.info("Spark 세션 종료됨")
            
        # Ray 종료 (이 클래스에서 초기화한 경우에만)
        if self.ray_initialized and ray.is_initialized():
            ray.shutdown()
            logger.info("Ray 종료됨")
    
    def _process_pandas_batch(self, df):
        """
        판다스 DataFrame 배치 처리
        
        Args:
            df (pandas.DataFrame): 처리할 판다스 DataFrame
            
        Returns:
            pandas.DataFrame: 처리된 DataFrame
        """
        # 텍스트 컬럼이 있는지 확인
        if 'content' not in df.columns:
            logger.warning("DataFrame에 'content' 컬럼이 없습니다. 컬럼명: " + str(df.columns.tolist()))
            
        # 텍스트 컬럼 선택 (content 또는 첫 번째 텍스트 컬럼)
        text_column = 'content' if 'content' in df.columns else df.select_dtypes(include=['object']).columns[0]
        
        # 이미지 URL 컬럼 선택 (있는 경우)
        image_column = None
        if 'image_url' in df.columns:
            image_column = 'image_url'
        
        # 다음 CLIP 서비스 선택 (라운드 로빈 방식)
        clip_service = self.clip_services[self.service_index % len(self.clip_services)]
        self.service_index += 1
        
        # 텍스트 임베딩 생성
        texts = df[text_column].fillna("").tolist()
        text_embeddings = ray.get(clip_service.encode_text.remote(texts))
        
        # 결과 데이터프레임에 임베딩 추가
        result_df = df.copy()
        result_df['text_embedding'] = text_embeddings
        
        # 이미지 임베딩 생성 (이미지 URL이 있는 경우)
        if image_column:
            image_urls = df[image_column].fillna("").tolist()
            image_embeddings = ray.get(clip_service.encode_image.remote(image_urls))
            result_df['image_embedding'] = image_embeddings
        
        logger.info(f"{len(df)}개 행 처리 완료")
        return result_df
    
    def process_mysql_table(self, table_name, where_clause=None, batch_size=BATCH_SIZE):
        """
        MySQL 테이블 처리
        
        Args:
            table_name (str): 처리할 MySQL 테이블 이름
            where_clause (str, optional): SQL WHERE 절
            batch_size (int): 배치 크기
            
        Returns:
            pyspark.sql.DataFrame: 처리된 DataFrame
        """
        try:
            # MySQL 연결 옵션
            mysql_options = {
                "url": self.mysql_url,
                "driver": "com.mysql.cj.jdbc.Driver",
                "user": self.mysql_user,
                "password": self.mysql_password
            }
            
            # 쿼리 구성
            query = f"SELECT * FROM {table_name}"
            if where_clause:
                query += f" WHERE {where_clause}"
                
            logger.info(f"MySQL 쿼리 실행: {query}")
            
            # 데이터 로드
            df = self.spark.read.format("jdbc") \
                .options(**mysql_options) \
                .option("dbtable", f"({query}) AS tmp") \
                .load()
                
            total_rows = df.count()
            logger.info(f"MySQL에서 {total_rows}개 행 로드됨")
            
            # 배치 처리를 위한 준비
            # SparkRayTaskProcessor 대신 직접 구현
            
            # 1. 데이터 분할을 위한 인덱스 컬럼 추가
            df = df.withColumn("_partition_id", (col("id") % 10000).cast("int"))
            
            # 2. 텍스트 임베딩 UDF 정의 (Spark DataFrame -> Pandas DataFrame 변환)
            # DataFrame을 판다스로 변환하여 배치 처리
            processed_dfs = []
            
            # 배치 처리를 위해 데이터를 여러 파티션으로 나누어 처리
            num_partitions = max(1, total_rows // batch_size)
            partitioned_dfs = [df.filter(col("_partition_id") % num_partitions == i) for i in range(num_partitions)]
            
            for i, part_df in enumerate(partitioned_dfs):
                logger.info(f"파티션 {i+1}/{num_partitions} 처리 중 ({part_df.count()} 행)")
                
                # Pandas DataFrame으로 변환하여 처리
                pandas_df = part_df.toPandas()
                processed_pandas_df = self._process_pandas_batch(pandas_df)
                
                # 처리된 판다스 DataFrame을 Spark DataFrame으로 변환
                processed_spark_df = self.spark.createDataFrame(processed_pandas_df)
                processed_dfs.append(processed_spark_df)
            
            # 모든 처리된 DataFrame 통합
            if processed_dfs:
                result_df = processed_dfs[0]
                for df in processed_dfs[1:]:
                    result_df = result_df.union(df)
                    
                # 임시 파티션 ID 컬럼 제거
                result_df = result_df.drop("_partition_id")
                
                logger.info(f"처리 완료: {result_df.count()}개 행, 컬럼: {result_df.columns}")
                return result_df
            else:
                logger.warning("처리된 데이터가 없습니다.")
                return df.drop("_partition_id")
            
        except Exception as e:
            logger.error(f"MySQL 테이블 처리 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise
            
    def find_similar_items(self, query_text, result_df, top_n=10):
        """
        쿼리 텍스트와 유사한 항목 찾기
        
        Args:
            query_text (str): 검색 텍스트
            result_df (pyspark.sql.DataFrame): 임베딩이 있는 DataFrame
            top_n (int): 반환할 최대 결과 수
            
        Returns:
            list: 유사한 항목 목록 (유사도 높은 순)
        """
        try:
            # 임베딩 컬럼 확인
            if 'text_embedding' not in result_df.columns:
                raise ValueError("DataFrame에 'text_embedding' 컬럼이 없습니다.")
                
            # CLIP 서비스로 쿼리 텍스트 임베딩 생성
            clip_service = self.clip_services[0]
            query_embedding = ray.get(clip_service.encode_text.remote([query_text]))[0]
            
            # Pandas DataFrame으로 변환 (소규모 데이터에 적합)
            pandas_df = result_df.toPandas()
            
            # 유사도 계산
            similarities = []
            for _, row in pandas_df.iterrows():
                # 텍스트 임베딩 가져오기
                text_embedding = row['text_embedding']
                
                # 코사인 유사도 계산
                similarity = np.dot(query_embedding, text_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)
                )
                # -1~1 범위를 0~1 범위로 변환
                similarity = (similarity + 1) / 2
                
                # ID 컬럼 찾기
                id_column = 'id' if 'id' in row else next(iter(row.index))
                
                similarities.append({
                    'id': row[id_column],
                    'similarity': float(similarity),
                    'data': {k: v for k, v in row.items() if not isinstance(v, list)}
                })
            
            # 유사도 높은 순으로 정렬
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 상위 N개 결과 반환
            return similarities[:top_n]
            
        except Exception as e:
            logger.error(f"유사한 항목 검색 중 오류 발생: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise