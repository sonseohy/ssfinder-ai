"""
EC2 서버에서 실행할 Spark REST API 서비스 (CLIP 모델 제외)
"""
import os
import sys
import json
import logging
from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lit
from pyspark.sql.types import FloatType, StringType, ArrayType
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("spark_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Spark 세션 초기화
def create_spark_session():
    hadoop_conf = {
        "fs.defaultFS": "hdfs://nn1:9870",
        "hadoop.job.ugi": "hadoop,hadoop"
    }
    
    """Spark 세션 생성"""
    spark = SparkSession.builder \
        .appName("LostFoundSparkAPI") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.ui.port", 4044) \
        .config("spark.hadoop.fs.defaultFS", hadoop_conf["fs.defaultFS"]) \
        .config("spark.hadoop.hadoop.job.ugi", hadoop_conf["hadoop.job.ugi"]) \
        .getOrCreate()
    return spark

# Flask 애플리케이션 생성
app = Flask(__name__)

# 전역 변수로 Spark 세션 초기화
spark = None

def before_first_request_func():
    """첫 요청 전 초기화"""
    global spark
    spark = create_spark_session()
    logger.info("Spark 세션 초기화 완료")

@app.route('/api/health', methods=['GET'])
def health_check():
    """상태 확인 API"""
    return jsonify({"status": "healthy", "spark": spark is not None})

@app.route('/api/embeddings/latest', methods=['GET'])
def get_latest_embeddings_path():
    """최신 임베딩 경로 조회 API"""
    try:
        global spark
        if not spark:
            spark = create_spark_session()
        
        # 하둡 디렉토리 경로
        embeddings_dir = "/user/hadoop/embeddings"
        
        # 하둡 파일시스템 객체 가져오기
        hadoop_fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
        path = spark._jvm.org.apache.hadoop.fs.Path(embeddings_dir)
        
        # 디렉토리가 존재하는지 확인
        if not hadoop_fs.exists(path):
            return jsonify({"success": False, "message": "임베딩 디렉토리가 존재하지 않습니다"})
        
        # 디렉토리 내용 가져오기
        status_list = hadoop_fs.listStatus(path)
        
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
            latest_path = f"{embeddings_dir}/{latest_dir}"
            return jsonify({"success": True, "path": latest_path})
        else:
            return jsonify({"success": False, "message": "임베딩 디렉토리를 찾을 수 없습니다"})
            
    except Exception as e:
        logger.error(f"최신 임베딩 경로 조회 중 오류: {str(e)}")
        return jsonify({"success": False, "message": f"오류 발생: {str(e)}"})

# 임베딩 저장 API
@app.route('/api/embeddings/save', methods=['POST'])
def save_embeddings():
    """클라이언트에서 생성된 임베딩을 하둡에 저장"""
    try:
        # 요청 데이터 확인
        data = request.json
        if not data:
            return jsonify({"success": False, "message": "요청 데이터가 없습니다"})
        
        text_embeddings = data.get('text_embeddings', {})
        image_embeddings = data.get('image_embeddings', {})
        timestamp = data.get('timestamp', datetime.now().strftime("%Y%m%d_%H%M%S"))
        
        if not text_embeddings:
            return jsonify({"success": False, "message": "텍스트 임베딩 데이터가 없습니다"})
        
        # Spark 세션 초기화 확인
        global spark
        if not spark:
            spark = create_spark_session()
        
        # 하둡에 저장할 경로
        embeddings_dir = f"/user/hadoop/embeddings/{timestamp}"
        
        # 하둡 파일시스템 객체 가져오기
        hadoop_fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
        
        # 디렉토리 생성
        path = spark._jvm.org.apache.hadoop.fs.Path(embeddings_dir)
        hadoop_fs.mkdirs(path)
        
        # 텍스트 임베딩 저장
        text_path = spark._jvm.org.apache.hadoop.fs.Path(f"{embeddings_dir}/text_embeddings.json")
        text_out = hadoop_fs.create(text_path)
        text_writer = spark._jvm.java.io.OutputStreamWriter(text_out)
        text_writer.write(json.dumps(text_embeddings))
        text_writer.close()
        
        # 이미지 임베딩 저장
        image_path = spark._jvm.org.apache.hadoop.fs.Path(f"{embeddings_dir}/image_embeddings.json")
        image_out = hadoop_fs.create(image_path)
        image_writer = spark._jvm.java.io.OutputStreamWriter(image_out)
        image_writer.write(json.dumps(image_embeddings))
        image_writer.close()
        
        # 메타데이터 저장
        metadata = {
            "timestamp": timestamp,
            "text_embeddings_count": len(text_embeddings),
            "image_embeddings_count": len(image_embeddings),
            "created_at": datetime.now().isoformat()
        }
        
        meta_path = spark._jvm.org.apache.hadoop.fs.Path(f"{embeddings_dir}/metadata.json")
        meta_out = hadoop_fs.create(meta_path)
        meta_writer = spark._jvm.java.io.OutputStreamWriter(meta_out)
        meta_writer.write(json.dumps(metadata))
        meta_writer.close()
        
        logger.info(f"임베딩이 하둡에 저장됨: {embeddings_dir}") 
        
        return jsonify({
            "success": True,
            "message": "임베딩이 하둡에 성공적으로 저장되었습니다",
            "path": embeddings_dir,
            "timestamp": timestamp,
            "text_count": len(text_embeddings),
            "image_count": len(image_embeddings)
        })
    
    except Exception as e:
        logger.error(f"임베딩 저장 중 오류: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"임베딩 저장 중 오류 발생: {str(e)}"
        })

# 임베딩 목록 조회 API
@app.route('/api/embeddings/list', methods=['GET'])
def list_embeddings():
    """하둡에 저장된 임베딩 목록 조회"""
    try:
        # Spark 세션 초기화 확인
        global spark
        if not spark:
            spark = create_spark_session()
        
        # 하둡 임베딩 디렉토리
        embeddings_dir = "/user/hadoop/embeddings"
        
        # 하둡 파일시스템 객체 가져오기
        hadoop_fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
        path = spark._jvm.org.apache.hadoop.fs.Path(embeddings_dir)
        
        # 디렉토리가 존재하는지 확인
        if not hadoop_fs.exists(path):
            return jsonify({
                "success": True,
                "message": "임베딩 디렉토리가 존재하지 않습니다",
                "embeddings": []
            })
        
        # 디렉토리 내용 가져오기
        status_list = hadoop_fs.listStatus(path)
        
        embeddings = []
        for status in status_list:
            if status.isDirectory():
                dir_name = status.getPath().getName()
                dir_path = f"{embeddings_dir}/{dir_name}"
                
                # 메타데이터 파일 확인
                meta_path = spark._jvm.org.apache.hadoop.fs.Path(f"{dir_path}/metadata.json")
                metadata = {}
                
                if hadoop_fs.exists(meta_path):
                    # 메타데이터 읽기
                    meta_in = hadoop_fs.open(meta_path)
                    meta_reader = spark._jvm.java.io.InputStreamReader(meta_in)
                    meta_buffer = spark._jvm.java.io.BufferedReader(meta_reader)
                    
                    meta_content = ""
                    line = meta_buffer.readLine()
                    while line is not None:
                        meta_content += line
                        line = meta_buffer.readLine()
                    
                    meta_reader.close()
                    metadata = json.loads(meta_content)
                
                # 디렉토리 정보 추가
                embeddings.append({
                    "id": dir_name,
                    "path": dir_path,
                    "created_at": metadata.get("created_at", ""),
                    "text_count": metadata.get("text_embeddings_count", 0),
                    "image_count": metadata.get("image_embeddings_count", 0)
                })
        
        # 생성 시간 기준 내림차순 정렬
        embeddings.sort(key=lambda x: x["id"], reverse=True)
        
        return jsonify({
            "success": True,
            "message": f"{len(embeddings)}개의 임베딩 세트를 찾았습니다",
            "embeddings": embeddings
        })
    
    except Exception as e:
        logger.error(f"임베딩 목록 조회 중 오류: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"임베딩 목록 조회 중 오류 발생: {str(e)}",
            "embeddings": []
        })

# 임베딩 파일 직접 조회 API (테스트용)
@app.route('/api/embeddings/read', methods=['GET'])
def read_embedding_file():
    """하둡에 저장된 임베딩 파일 직접 조회 (디버깅용)"""
    try:
        path = request.args.get('path')
        if not path:
            return jsonify({"success": False, "message": "파일 경로를 지정해주세요"})
        
        # Spark 세션 초기화 확인
        global spark
        if not spark:
            spark = create_spark_session()
        
        # 하둡 파일시스템 객체 가져오기
        hadoop_fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration())
        file_path = spark._jvm.org.apache.hadoop.fs.Path(path)
        
        # 파일 존재 확인
        if not hadoop_fs.exists(file_path):
            return jsonify({"success": False, "message": f"파일이 존재하지 않습니다: {path}"})
        
        # 파일 크기 확인
        file_status = hadoop_fs.getFileStatus(file_path)
        file_size = file_status.getLen()
        
        # 파일 내용 읽기 (파일이 너무 크면 일부만 읽음)
        file_in = hadoop_fs.open(file_path)
        file_reader = spark._jvm.java.io.InputStreamReader(file_in)
        file_buffer = spark._jvm.java.io.BufferedReader(file_reader)
        
        # 첫 10,000자만 읽기
        max_chars = 10000
        content = ""
        line = file_buffer.readLine()
        char_count = 0
        
        while line is not None and char_count < max_chars:
            content += line + "\n"
            char_count += len(line)
            line = file_buffer.readLine()
        
        file_reader.close()
        
        # 파일이 더 큰 경우 잘렸음을 표시
        truncated = file_size > char_count
        
        return jsonify({
            "success": True,
            "path": path,
            "size": file_size,
            "truncated": truncated,
            "content": content
        })
    
    except Exception as e:
        logger.error(f"임베딩 파일 읽기 중 오류: {str(e)}")
        return jsonify({
            "success": False,
            "message": f"오류 발생: {str(e)}"
        })

if __name__ == '__main__':
    # 환경 변수에서 포트 가져오기 (기본값: 5000)
    port = int(os.environ.get('PORT', 5000))
    before_first_request_func()
    app.run(host='0.0.0.0', port=port)