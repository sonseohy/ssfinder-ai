#!/bin/bash
# Spark 작업 실행 스크립트
# 사용법: ./run_spark_job.sh <새 게시글 JSON 파일>

# 오류 발생 시 종료
set -e

# 환경 설정
SPARK_HOME=${SPARK_HOME:-"/opt/spark"}
HADOOP_CONF_DIR=${HADOOP_CONF_DIR:-"/etc/hadoop/conf"}
PYTHON_PATH="$(pwd)"

# 스크립트 경로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 인수 확인
if [ $# -lt 1 ]; then
  echo "사용법: $0 <새 게시글 JSON 파일>"
  exit 1
fi

POST_JSON_FILE="$1"
if [ ! -f "$POST_JSON_FILE" ]; then
  echo "오류: 게시글 파일을 찾을 수 없습니다: $POST_JSON_FILE"
  exit 1
fi

# JSON 파일 내용 로드
POST_JSON=$(cat "$POST_JSON_FILE")

# 환경 변수 로드
if [ -f .env ]; then
  source .env
fi

echo "Spark 작업 실행 준비 중..."

# YARN 모드 또는 로컬 모드 확인
if [ "${SPARK_MASTER:-}" == "yarn" ]; then
  # YARN 모드 설정
  DEPLOY_MODE="cluster"
  MASTER="yarn"
  echo "YARN 모드로 실행..."
else
  # 로컬 모드 설정
  DEPLOY_MODE="client"
  MASTER="local[*]"
  echo "로컬 모드로 실행..."
fi

# 필요한 디렉토리 생성
mkdir -p logs

# 의존성 패키징 (필요한 경우)
echo "의존성 패키징 중..."
pip install -r requirements.txt -t ./packages
cd packages
zip -r ../dependencies.zip .
cd ..

# Spark 작업 실행
echo "Spark 작업 실행 중..."
$SPARK_HOME/bin/spark-submit \
  --master $MASTER \
  --deploy-mode $DEPLOY_MODE \
  --conf "spark.executor.memory=${SPARK_EXECUTOR_MEMORY:-4g}" \
  --conf "spark.driver.memory=${SPARK_DRIVER_MEMORY:-4g}" \
  --conf "spark.executor.cores=${SPARK_EXECUTOR_CORES:-2}" \
  --conf "spark.python.worker.reuse=true" \
  --conf "spark.dynamicAllocation.enabled=true" \
  --py-files dependencies.zip \
  --files .env,config.py \
  spark/similarity_processor.py \
  --post "$POST_JSON" \
  --threshold "${SIMILARITY_THRESHOLD:-0.5}" \
  --limit "${LIMIT:-10}" \
  2>&1 | tee logs/spark_job_$(date +%Y%m%d_%H%M%S).log

echo "Spark 작업 완료"