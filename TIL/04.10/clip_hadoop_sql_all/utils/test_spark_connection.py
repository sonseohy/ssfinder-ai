#!/usr/bin/env python3
"""
원격 스파크 클러스터 연결 테스트 스크립트
"""
import sys
import argparse
from pyspark.sql import SparkSession

def test_spark_connection(master_url):
    """
    주어진 스파크 마스터 URL에 연결 테스트
    
    Args:
        master_url (str): 스파크 마스터 URL (예: spark://host:port)
        
    Returns:
        bool: 연결 성공 여부
    """
    print(f"스파크 마스터 '{master_url}'에 연결 시도 중...")
    
    try:
        # 스파크 세션 생성
        spark = SparkSession.builder \
            .appName("SparkConnectionTest") \
            .master(master_url) \
            .getOrCreate()
        
        # 간단한 테스트 수행
        test_data = [("Test", 1), ("Connection", 2), ("Success", 3)]
        df = spark.createDataFrame(test_data, ["word", "count"])
        
        # 데이터 출력
        print("테스트 데이터:")
        df.show()
        
        # 스파크 버전 출력
        print(f"스파크 버전: {spark.version}")
        
        # 클러스터 정보 출력
        print("클러스터 정보:")
        print(f"  애플리케이션 ID: {spark.sparkContext.applicationId}")
        print(f"  마스터: {spark.sparkContext.master}")
        print(f"  사용 가능한 코어 수: {spark.sparkContext.defaultParallelism}")
        
        # 스파크 UI URL 출력
        print(f"스파크 UI URL: {spark.sparkContext.uiWebUrl}")
        
        # 세션 종료
        spark.stop()
        
        print("스파크 연결 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"스파크 연결 테스트 실패: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def main():
    """
    메인 함수
    """
    parser = argparse.ArgumentParser(description="스파크 클러스터 연결 테스트")
    parser.add_argument("--master", required=True, help="스파크 마스터 URL (예: spark://host:port)")
    
    args = parser.parse_args()
    
    success = test_spark_connection(args.master)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())