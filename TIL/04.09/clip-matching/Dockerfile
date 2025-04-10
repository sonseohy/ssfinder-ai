FROM python:3.9

# 캐시 디렉토리 환경변수 설정
ENV TRANSFORMERS_CACHE=/tmp/huggingface_cache
ENV HF_HOME=/tmp/huggingface_cache 

WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 캐시 디렉토리 생성 및 권한 설정
RUN mkdir -p $TRANSFORMERS_CACHE && chmod -R 777 $TRANSFORMERS_CACHE

# 임시 디렉토리 권한 설정
RUN mkdir -p /tmp/uploads && chmod 777 /tmp/uploads
ENV TMPDIR=/tmp/uploads

# 데이터 디렉토리 생성
RUN mkdir -p /app/data/embeddings /app/data/uploads
RUN chmod -R 777 /app/data

# 요구사항 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 파일 복사
COPY . .

# 환경 변수 설정
ENV PYTHONPATH=/app

# 포트 노출
EXPOSE 5000

# 애플리케이션 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5000"]