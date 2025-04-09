"""
분실물 매칭 서비스 설정 파일
"""

# Hadoop/HDFS 설정
HADOOP_CONF = {
    "fs.defaultFS": "hdfs://nn1:9000",
    "dfs.replication": 3
}

# 임베딩 관련 설정
EMBEDDING_CONFIG = {
    "model_name": "openai/clip-vit-base-patch32",
    "embedding_dim": 512,
    "hdfs_embeddings_path": "/user/hadoop/clip_embeddings",
    "batch_size": 32
}

# Spark 관련 설정
SPARK_CONFIG = {
    "app_name": "LostFoundSimilarityEngine",
    "master": "yarn",
    "executor_memory": "4g",
    "executor_cores": 2,
    "driver_memory": "2g",
    "shuffle_partitions": 100,
    "default_parallelism": 200
}

# 유사도 계산 설정
SIMILARITY_CONFIG = {
    "default_threshold": 0.7,  # 70% 유사도
    "default_top_k": 10,
    "metrics": {
        "cosine": {
            "enabled": True,
            "weight": 1.0
        },
        "euclidean": {
            "enabled": False,
            "weight": 0.0
        },
        "dot_product": {
            "enabled": False,
            "weight": 0.0
        }
    }
}

# 캐싱 설정
CACHE_CONFIG = {
    "enabled": True,
    "ttl_seconds": 3600,
    "max_size": 1000
}

# 로깅 설정
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_dir": "/var/log/lost_found/",
    "rotate_logs": True,
    "max_log_size_mb": 100,
    "backup_count": 5
}