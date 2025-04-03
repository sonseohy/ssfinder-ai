"""
Spark 기반 게시글 유사도 계산 시스템
경량화된 CLIP 모델을 사용하여 대량의 게시글에 대한 유사도 계산 처리
"""
import os
import sys
import logging
import time
import json
import requests
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, struct, array, lit
from pyspark.sql.types import FloatType, ArrayType, StringType, StructType, StructField, MapType
from functools import partial
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 상위 디렉토리 추가하여 모듈 임포트 가능하게 함
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SIMILARITY_THRESHOLD, FASTAPI_HOST, FASTAPI_PORT
from db.mysql_connector import save_matching_result
from utils.similarity import calculate_category_similarity, calculate_text_similarity

class SparkSimilarityProcessor:
    """
    Spark를 사용한 유사도 분석 처리기
    """
    
    def __init__(self, app_name="PostSimilarityProcessor"):
        """
        Spark 세션 초기화
        
        Args:
            app_name (str): Spark 애플리케이션 이름
        """
        logger.info(f"Spark 세션 초기화 중: {app_name}")
        
        # Spark 세션 생성
        self.spark = SparkSession.builder \
            .appName(app_name) \
            .config("spark.driver.memory", "4g") \
            .config("spark.executor.memory", "4g") \
            .getOrCreate()
            
        logger.info(f"Spark 버전: {self.spark.version}")
        
        # UDF 등록 (사용자 정의 함수)
        self._register_udfs()
        
    def _register_udfs(self):
        """
        Spark에서 사용할 UDF 함수 등록
        """
        # 텍스트 유사도 계산 UDF
        self.text_similarity_udf = udf(
            lambda text1, text2: float(calculate_text_similarity(text1, text2) or 0.0),
            FloatType()
        )
        
        # 카테고리 유사도 계산 UDF
        self.category_similarity_udf = udf(
            lambda cat1, cat2: float(calculate_category_similarity(cat1, cat2) or 0.0),
            FloatType()
        )
        
        # CLIP 모델 호출 UDF
        self.call_clip_api_udf = udf(self._call_clip_api, MapType(StringType(), FloatType()))
    
    def _call_clip_api(self, post_data):
        """
        FastAPI 서버를 호출하여 CLIP 모델 기반 유사도 계산
        
        Args:
            post_data: 게시글 데이터
            
        Returns:
            dict: 유사도 점수 및 세부 정보
        """
        try:
            # FastAPI 엔드포인트 URL
            api_url = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/api/spark/calculate-similarity"
            
            # API 요청
            response = requests.post(
                api_url, 
                json=post_data,
                timeout=5  # 5초 타임아웃
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('similarity_scores', {})
            else:
                logger.warning(f"API 응답 오류: {response.status_code}, {response.text}")
                return {}
                
        except Exception as e:
            logger.error(f"API 호출 오류: {str(e)}")
            return {}
    
    def load_posts_to_dataframe(self):
        """
        MySQL에서 게시글 데이터를 가져와 Spark DataFrame으로 변환
        
        Returns:
            pyspark.sql.DataFrame: 게시글 DataFrame
        """
        logger.info("MySQL에서 게시글 데이터 로드 중...")
        
        try:
            # pandas DataFrame 가져오기
            pandas_df = fetch_posts_in_batches()
            
            # Spark DataFrame으로 변환
            posts_df = self.spark.createDataFrame(pandas_df)
            
            logger.info(f"게시글 {posts_df.count()}개를 Spark DataFrame으로 로드 완료")
            return posts_df
            
        except Exception as e:
            logger.error(f"데이터 로드 오류: {str(e)}")
            raise
    
    def preprocess_dataframe(self, df):
        """
        DataFrame 전처리 및 최적화
        
        Args:
            df (pyspark.sql.DataFrame): 원본 DataFrame
            
        Returns:
            pyspark.sql.DataFrame: 전처리된 DataFrame
        """
        # 필요한 컬럼만 선택 및 null 값 처리
        processed_df = df.select(
            "id",
            "title",
            col("content").fillna(""),
            col("category").fillna("기타"),
            col("item_name").fillna(""),
            col("color").fillna(""),
            col("location").fillna(""),
            col("image_url").fillna("")
        )
        
        # 데이터 캐싱 (성능 최적화)
        processed_df = processed_df.cache()
        
        return processed_df
    
    def calculate_similarity(self, new_post, posts_df, threshold=SIMILARITY_THRESHOLD, limit=10):
        """
        새 게시글과 기존 게시글 간의 유사도 계산
        
        Args:
            new_post (dict): 새 게시글 데이터
            posts_df (pyspark.sql.DataFrame): 기존 게시글 DataFrame
            threshold (float): 유사도 임계값
            limit (int): 결과 제한 개수
            
        Returns:
            list: 유사도가 높은 게시글 목록
        """
        logger.info(f"새 게시글과 {posts_df.count()}개 게시글 간의 유사도 계산 중...")
        start_time = time.time()
        
        # 새 게시글 정보
        new_post_content = new_post.get('content', '')
        new_post_category = new_post.get('category', '')
        new_post_item_name = new_post.get('item_name', '')
        new_post_color = new_post.get('color', '')
        
        # 1. 로컬 유사도 계산 (Spark UDF 사용)
        with_sim_df = posts_df \
            .withColumn("category_sim", self.category_similarity_udf(lit(new_post_category), col("category"))) \
            .withColumn("content_sim", self.text_similarity_udf(lit(new_post_content), col("content"))) \
            .withColumn("item_name_sim", self.text_similarity_udf(lit(new_post_item_name), col("item_name"))) \
            .withColumn("color_sim", self.text_similarity_udf(lit(new_post_color), col("color")))
        
        # 2. 텍스트 기반 종합 유사도 계산
        text_weighted_df = with_sim_df \
            .withColumn(
                "text_similarity", 
                col("category_sim") * 0.5 + 
                col("item_name_sim") * 0.3 + 
                col("color_sim") * 0.1 + 
                col("content_sim") * 0.1
            )
        
        # 3. CLIP 모델 호출 (새 게시글에 이미지가 있는 경우만)
        # FastAPI 서버에 한 번만 요청을 보내기 위한 사전 필터링
        if 'image_url' in new_post and new_post['image_url']:
            # 텍스트 유사도가 높은 항목만 필터링하여 CLIP 모델 호출 (성능 최적화)
            top_candidates = text_weighted_df \
                .filter(col("text_similarity") >= threshold * 0.7) \
                .select("id", "content", "image_url", "text_similarity")
            
            # CLIP API 호출 준비
            posts_for_api = top_candidates.collect()
            
            # 결과 저장을 위한 구조 생성
            clip_results = []
            
            # 배치 처리 (API 부하 감소)
            batch_size = 20
            for i in range(0, len(posts_for_api), batch_size):
                batch = posts_for_api[i:i+batch_size]
                
                # API 호출 데이터 준비
                batch_data = {
                    "new_post": new_post,
                    "candidates": [
                        {
                            "id": post['id'],
                            "content": post['content'],
                            "image_url": post['image_url'],
                            "text_similarity": float(post['text_similarity'])
                        }
                        for post in batch
                    ]
                }
                
                # FastAPI 호출 (통합 배치 처리)
                try:
                    # FastAPI 엔드포인트 URL
                    api_url = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/api/spark/batch-similarity"
                    
                    response = requests.post(
                        api_url, 
                        json=batch_data,
                        timeout=30  # 배치 처리는 시간이 더 소요됨
                    )
                    
                    if response.status_code == 200:
                        batch_results = response.json().get('results', [])
                        clip_results.extend(batch_results)
                    else:
                        logger.warning(f"배치 API 응답 오류: {response.status_code}")
                        
                except Exception as e:
                    logger.error(f"배치 API 호출 오류: {str(e)}")
            
            # CLIP 결과와 Spark 결과 병합
            final_results = []
            
            # ID를 기준으로 매핑
            clip_results_dict = {item['id']: item for item in clip_results}
            
            # 전체 결과에서 상위 항목 추출
            top_results = text_weighted_df.orderBy(col("text_similarity").desc()).limit(limit * 2).collect()
            
            for post in top_results:
                post_dict = post.asDict()
                post_id = post_dict['id']
                
                # 최종 유사도 계산
                if post_id in clip_results_dict:
                    clip_result = clip_results_dict[post_id]
                    
                    # CLIP 결과와 텍스트 유사도 결합
                    combined_similarity = (
                        post_dict['text_similarity'] * 0.7 + 
                        clip_result.get('final_similarity', 0) * 0.3
                    )
                    
                    post_dict['final_similarity'] = combined_similarity
                    post_dict['clip_similarity'] = clip_result.get('final_similarity', 0)
                else:
                    # CLIP 결과가 없는 경우 텍스트 유사도만 사용
                    post_dict['final_similarity'] = post_dict['text_similarity']
                    post_dict['clip_similarity'] = 0
                
                final_results.append(post_dict)
            
            # 최종 유사도로 정렬
            final_results.sort(key=lambda x: x['final_similarity'], reverse=True)
            final_results = [post for post in final_results if post['final_similarity'] >= threshold]
            
        else:
            # 이미지가 없는 경우 텍스트 유사도만 사용
            final_results = text_weighted_df \
                .filter(col("text_similarity") >= threshold) \
                .orderBy(col("text_similarity").desc()) \
                .limit(limit) \
                .collect()
            
            final_results = [post.asDict() for post in final_results]
            
            # 필드 이름 일관성 유지
            for post in final_results:
                post['final_similarity'] = post['text_similarity']
        
        # 상위 결과 반환
        final_results = final_results[:limit]
        
        # 결과 DB에 저장 (matched_item 테이블)
        if hasattr(self, 'save_matching_results') and callable(self.save_matching_results):
            try:
                # 데이터베이스에 결과 저장
                from db.mysql_connector import save_matching_result
                
                lost_item_id = new_post.get('id')  # 분실물 ID (lost_item_id)
                if lost_item_id:
                    for post in final_results:
                        found_item_id = post['id']  # 습득물 ID (found_item_id)
                        similarity_score = post['final_similarity']
                        
                        # 결과 저장
                        save_matching_result(lost_item_id, found_item_id, similarity_score)
                        logger.info(f"매칭 결과 저장: 분실물 {lost_item_id}, 습득물 {found_item_id}, 유사도 {similarity_score:.4f}")
            except Exception as e:
                logger.error(f"매칭 결과 저장 중 오류 발생: {str(e)}")
        
        process_time = time.time() - start_time
        logger.info(f"유사도 계산 완료: {len(final_results)}개 결과, 처리 시간: {process_time:.2f}초")
        
        return final_results
    
    def close(self):
        """
        Spark 세션 종료
        """
        if self.spark:
            self.spark.stop()
            logger.info("Spark 세션 종료")

# 모듈 실행 함수
def process_similarity(new_post, threshold=SIMILARITY_THRESHOLD, limit=10):
    """
    Spark 기반 유사도 계산 처리 메인 함수
    
    Args:
        new_post (dict): 새 게시글 데이터
        threshold (float): 유사도 임계값
        limit (int): 결과 제한 개수
        
    Returns:
        list: 유사도가 높은 게시글 목록
    """
    processor = None
    
    try:
        # Spark 처리기 초기화
        processor = SparkSimilarityProcessor()
        
        # 게시글 데이터 로드
        posts_df = processor.load_posts_to_dataframe()
        
        # 데이터 전처리
        processed_df = processor.preprocess_dataframe(posts_df)
        
        # 유사도 계산
        similar_posts = processor.calculate_similarity(
            new_post=new_post,
            posts_df=processed_df,
            threshold=threshold,
            limit=limit
        )
        
        return similar_posts
        
    except Exception as e:
        logger.error(f"유사도 계산 처리 중 오류 발생: {str(e)}")
        # 오류 시 빈 결과 반환
        return []
        
    finally:
        # Spark 세션 종료
        if processor:
            processor.close()

# 명령줄 실행 코드
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Spark 기반 게시글 유사도 계산')
    parser.add_argument('--post', type=str, required=True, help='새 게시글 데이터 (JSON 형식)')
    parser.add_argument('--threshold', type=float, default=SIMILARITY_THRESHOLD, help='유사도 임계값')
    parser.add_argument('--limit', type=int, default=10, help='결과 제한 개수')
    
    args = parser.parse_args()
    
    try:
        # 입력 검증
        new_post = json.loads(args.post)
        
        # 유사도 계산 실행
        results = process_similarity(
            new_post=new_post,
            threshold=args.threshold,
            limit=args.limit
        )
        
        # 결과 출력
        print(json.dumps(results, ensure_ascii=False, indent=2))
        
    except json.JSONDecodeError:
        logger.error("JSON 파싱 오류: 올바른 JSON 형식이 아닙니다.")
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}")