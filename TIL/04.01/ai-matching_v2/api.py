from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import json
import os
import shutil
from datetime import datetime
import asyncio
import tempfile

from config import Config
from similarity import SimilarityCalculator
from utils.preprocessing import preprocess_image

# FastAPI 앱 생성
app = FastAPI(
    title="분실물 유사도 비교 API",
    description="CLIP과 BLIP 모델을 활용한 분실물 게시글 유사도 비교 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 명시적인 출처 지정 필요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 임시 파일 저장 경로
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 유사도 계산기 초기화
similarity_calculator = SimilarityCalculator()

# 요청 모델
class PostRequest(BaseModel):
    title: str
    content: Optional[str] = ""
    category: Optional[str] = None
    
class SimilarityRequest(BaseModel):
    query_post_id: str
    db_post_ids: List[str] = []
    threshold: Optional[float] = None
    max_results: Optional[int] = None

# 응답 모델
class SimilarityResponse(BaseModel):
    query_post: Dict[str, Any]
    similar_posts: List[Dict[str, Any]]
    processing_time: float

# 업로드된 이미지 처리 함수
async def save_upload_file(upload_file: UploadFile) -> str:
    """
    업로드된 파일을 저장하고 경로 반환
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{upload_file.filename}"
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    with open(file_path, "wb") as f:
        shutil.copyfileobj(upload_file.file, f)
    
    return file_path

# 임시 데이터베이스 (실제 시스템에서는 DB로 대체)
post_database = {}

@app.post("/api/posts", response_model=Dict[str, Any])
async def create_post(
    title: str = Form(...),
    content: Optional[str] = Form(""),
    category: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    새 게시글 생성
    """
    post_id = f"post_{len(post_database) + 1}"
    
    post = {
        "id": post_id,
        "title": title,
        "content": content,
        "category": category,
        "image_path": None,
        "created_at": datetime.now().isoformat()
    }
    
    # 이미지 저장 (있는 경우)
    if image:
        image_path = await save_upload_file(image)
        post["image_path"] = image_path
        
        # 이미지 전처리
        try:
            processed_path = preprocess_image(image_path)
            post["processed_image_path"] = processed_path
        except Exception as e:
            print(f"이미지 전처리 오류: {e}")
            post["processed_image_path"] = image_path
    
    # 게시글 특성 추출
    if post["image_path"]:
        try:
            features = similarity_calculator.extract_post_features(post)
            post.update(features)
        except Exception as e:
            print(f"특성 추출 오류: {e}")
    
    # 데이터베이스에 저장
    post_database[post_id] = post
    
    return {"post_id": post_id, "post": post}

@app.get("/api/posts/{post_id}", response_model=Dict[str, Any])
async def get_post(post_id: str):
    """
    게시글 조회
    """
    if post_id not in post_database:
        raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다")
    
    return {"post": post_database[post_id]}

@app.get("/api/posts", response_model=Dict[str, Any])
async def list_posts():
    """
    모든 게시글 목록 조회
    """
    return {"posts": list(post_database.values())}

@app.post("/api/similarity/compare", response_model=SimilarityResponse)
async def compare_similarity(request: SimilarityRequest, background_tasks: BackgroundTasks):
    """
    게시글 유사도 비교
    """
    start_time = datetime.now()
    
    # 질의 게시글 확인
    if request.query_post_id not in post_database:
        raise HTTPException(status_code=404, detail="질의 게시글을 찾을 수 없습니다")
    
    query_post = post_database[request.query_post_id]
    
    # 비교 대상 게시글 목록 생성
    db_posts = []
    if request.db_post_ids:
        for post_id in request.db_post_ids:
            if post_id in post_database:
                db_posts.append(post_database[post_id])
    else:
        # 모든 게시글 사용 (질의 게시글 제외)
        db_posts = [p for pid, p in post_database.items() if pid != request.query_post_id]
    
    # 임계값 및 최대 결과 수 설정
    threshold = request.threshold or Config.SIMILARITY_THRESHOLD
    max_results = request.max_results or Config.MAX_RECOMMENDATIONS
    
    # 유사한 게시글 찾기
    similar_posts = similarity_calculator.find_similar_posts(
        query_post, db_posts, threshold, max_results
    )
    
    # 결과 포맷팅
    result = {
        "query_post": query_post,
        "similar_posts": [
            {
                "post": post,
                "similarity": similarity
            }
            for post, similarity in similar_posts
        ],
        "processing_time": (datetime.now() - start_time).total_seconds()
    }
    
    return result

@app.post("/api/similarity/analyze", response_model=Dict[str, Any])
async def analyze_image(image: UploadFile = File(...)):
    """
    이미지 분석
    """
    # 이미지 저장
    image_path = await save_upload_file(image)
    
    # 이미지 전처리
    try:
        processed_path = preprocess_image(image_path)
    except Exception as e:
        print(f"이미지 전처리 오류: {e}")
        processed_path = image_path
    
    # 이미지 비교기 초기화
    from similarity import ImageComparator
    image_comparator = ImageComparator()
    
    # 이미지 분석
    analysis = image_comparator.classify_image(processed_path)
    
    # 임시 파일 정리를 위한 백그라운드 태스크
    async def cleanup_temp_files():
        await asyncio.sleep(300)  # 5분 후 정리
        try:
            if os.path.exists(image_path):
                os.remove(image_path)
            if os.path.exists(processed_path) and processed_path != image_path:
                os.remove(processed_path)
        except Exception as e:
            print(f"임시 파일 정리 오류: {e}")
    
    background_tasks.add_task(cleanup_temp_files)
    
    return {
        "image_path": processed_path,
        "analysis": analysis
    }

@app.delete("/api/posts/{post_id}", response_model=Dict[str, Any])
async def delete_post(post_id: str):
    """
    게시글 삭제
    """
    if post_id not in post_database:
        raise HTTPException(status_code=404, detail="게시글을 찾을 수 없습니다")
    
    post = post_database[post_id]
    
    # 이미지 파일 삭제
    if "image_path" in post and post["image_path"] and os.path.exists(post["image_path"]):
        try:
            os.remove(post["image_path"])
        except Exception as e:
            print(f"이미지 파일 삭제 오류: {e}")
    
    if "processed_image_path" in post and post["processed_image_path"] and os.path.exists(post["processed_image_path"]):
        try:
            os.remove(post["processed_image_path"])
        except Exception as e:
            print(f"처리된 이미지 파일 삭제 오류: {e}")
    
    # 게시글 삭제
    del post_database[post_id]
    
    return {"message": "게시글이 삭제되었습니다", "post_id": post_id}

# 서버 실행
if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)