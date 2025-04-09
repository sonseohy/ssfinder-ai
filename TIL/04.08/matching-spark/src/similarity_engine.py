"""
PySpark를 이용한 임베딩 유사도 비교 엔진
"""
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf, col, array, lit
from pyspark.sql.types import FloatType, ArrayType, DoubleType, StructType, StructField, StringType
import numpy as np

class SimilarityEngine:
    def __init__(self, hadoop_embeddings_path, threshold=0.7):
        """
        유사도 비교 엔진 초기화
        
        Args:
            hadoop_embeddings_path: 하둡에 저장된 임베딩 경로
            threshold: 유사도 임계값 (기본값: 0.7 또는 70%)
        """
        self.hadoop_embeddings_path = hadoop_embeddings_path
        self.threshold = threshold
        self.spark = SparkSession.builder \
            .appName("LostFoundSimilarityEngine") \
            .config("spark.executor.memory", "4g") \
            .config("spark.driver.memory", "2g") \
            .getOrCreate()
        
        # 임베딩 차원 설정 (CLIP 모델의 경우 일반적으로 512)
        self.embedding_dim = 512
        
    def load_embeddings(self):
        """하둡에서 저장된 임베딩 데이터 로드"""
        # 스키마 정의 (실제 데이터에 맞게 조정 필요)
        embedding_schema = StructType([
            StructField("id", StringType(), False),
            StructField("title", StringType(), True),
            StructField("description", StringType(), True),
            StructField("embedding", ArrayType(DoubleType()), False),
            StructField("image_path", StringType(), True),
            StructField("created_at", StringType(), True)
        ])
        
        # 하둡에서 임베딩 데이터 로드
        try:
            embeddings_df = self.spark.read.schema(embedding_schema).parquet(self.hadoop_embeddings_path)
            print(f"임베딩 데이터 로드 완료: {embeddings_df.count()} 개")
            return embeddings_df
        except Exception as e:
            print(f"임베딩 데이터 로드 실패: {str(e)}")
            return None
        
    def prepare_vector_data(self, embeddings_df):
        """임베딩 데이터를 ML 라이브러리용 벡터로 변환"""
        # 배열 타입 임베딩을 벡터로 변환하는 UDF
        array_to_vector = udf(lambda arr: Vectors.dense(arr), VectorUDT())
        
        # 임베딩 배열을 Spark ML 벡터로 변환
        return embeddings_df.withColumn("features", array_to_vector(col("embedding")))
        
    def calculate_similarity(self, query_embedding, top_k=10):
        """
        주어진 쿼리 임베딩과 저장된 임베딩 간의 유사도 계산
        
        Args:
            query_embedding: 새로운 분실물 이미지의 임베딩 벡터 (numpy array)
            top_k: 반환할 최상위 결과 수
            
        Returns:
            유사도가 임계값 이상인 항목들의 DataFrame
        """
        # 임베딩 데이터 로드
        embeddings_df = self.load_embeddings()
        if embeddings_df is None:
            return None
        
        # 벡터 데이터 준비
        vector_df = self.prepare_vector_data(embeddings_df)
        
        # 쿼리 임베딩을 Spark 벡터로 변환
        query_vector = Vectors.dense(query_embedding)
        
        # 코사인 유사도 계산 UDF
        def cosine_similarity(stored_vector):
            # numpy 배열로 변환
            vec1 = np.array(stored_vector.toArray())
            vec2 = np.array(query_vector.toArray())
            
            # 코사인 유사도 계산
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            # 0으로 나누기 방지
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            return float(dot_product / (norm1 * norm2))
        
        # UDF 등록
        similarity_udf = udf(cosine_similarity, FloatType())
        
        # 모든 저장된 임베딩에 대해 유사도 계산
        results = vector_df.withColumn("similarity", similarity_udf(col("features")))
        
        # 임계값 이상의 결과만 필터링하고 유사도 내림차순으로 정렬
        filtered_results = results.filter(col("similarity") >= self.threshold) \
                                .select("id", "title", "description", "image_path", "similarity") \
                                .orderBy(col("similarity").desc()) \
                                .limit(top_k)
        
        return filtered_results
    
    def find_similar_items(self, query_embedding, threshold=None, top_k=10):
        """
        새로운 분실물 이미지와 유사한 아이템 찾기
        
        Args:
            query_embedding: 새 분실물 이미지의 임베딩 (numpy array)
            threshold: 선택적 임계값 (None인 경우 기본값 사용)
            top_k: 반환할 최상위 결과 수
            
        Returns:
            유사한 아이템의 리스트 (딕셔너리 형태)
        """
        # 임계값 설정
        old_threshold = self.threshold
        if threshold is not None:
            self.threshold = threshold
            
        try:
            # 유사도 계산
            similar_items_df = self.calculate_similarity(query_embedding, top_k)
            
            if similar_items_df is None or similar_items_df.count() == 0:
                return []
            
            # DataFrame을 Python 리스트로 변환
            similar_items = similar_items_df.collect()
            
            # 결과를 딕셔너리 리스트로 변환
            results = []
            for item in similar_items:
                results.append({
                    "id": item["id"],
                    "title": item["title"],
                    "description": item["description"],
                    "image_path": item["image_path"],
                    "similarity": round(item["similarity"] * 100, 2)  # 백분율로 변환
                })
                
            return results
        finally:
            # 임계값 복원
            if threshold is not None:
                self.threshold = old_threshold
                
    def stop(self):
        """Spark 세션 종료"""
        if self.spark:
            self.spark.stop()