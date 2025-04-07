"""
FastAPI 임베딩 관리 라우터
"""
import os
import sys
import logging
import requests
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# EC2 API 설정
EC2_API_URL = os.getenv('EC2_API_URL', 'http://43.201.252.40:5000')

# 라우터 생성
router = APIRouter(
    prefix="/embeddings",
    tags=["embeddings"],
    responses={404: {"description": "Not found"}},
)

# Pydantic 모델
class EmbeddingSet(BaseModel):
    id: str
    path: str
    created_at: Optional[str] = None
    text_count: int = 0
    image_count: int = 0

class EmbeddingListResponse(BaseModel):
    success: bool
    message: str
    embeddings: List[EmbeddingSet] = []

# 임베딩 목록 조회 API
@router.get("/list", response_model=EmbeddingListResponse)
async def list_embeddings():
    """하둡에 저장된 임베딩 목록 조회"""
    try:
        # EC2 API 호출
        response = requests.get(f"{EC2_API_URL}/api/embeddings/list")
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"API 응답 오류 (HTTP {response.status_code}): {response.text}")
            raise HTTPException(status_code=500, detail="임베딩 목록을 가져올 수 없습니다")
    
    except Exception as e:
        logger.error(f"임베딩 목록 조회 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"오류 발생: {str(e)}")

# 임베딩 관리 UI
@router.get("/", response_class=HTMLResponse)
async def embeddings_ui():
    """임베딩 관리 UI"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>임베딩 관리</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            h1 { color: #333; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { border: 1px solid #ddd; border-radius: 8px; padding: 16px; margin-bottom: 16px; }
            .card-header { display: flex; justify-content: space-between; margin-bottom: 8px; }
            .card-title { font-size: 18px; font-weight: bold; }
            .card-content { margin-top: 8px; }
            .stats { display: flex; gap: 16px; }
            .stat-item { background-color: #f5f5f5; border-radius: 4px; padding: 8px 12px; }
            .status { padding: 4px 8px; border-radius: 4px; font-size: 14px; }
            .status-success { background-color: #e6f7e6; color: #28a745; }
            .status-loading { background-color: #e6e6ff; color: #0066cc; }
            .status-error { background-color: #ffe6e6; color: #dc3545; }
            .button { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; }
            .primary-button { background-color: #007bff; color: white; }
            .error-message { color: #dc3545; margin-top: 16px; }
            #loading { display: none; margin-top: 16px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>임베딩 관리</h1>
            
            <div>
                <button id="btnGenerate" class="button primary-button">새 임베딩 생성</button>
                <button id="btnRefresh" class="button">목록 새로고침</button>
            </div>
            
            <div id="loading">임베딩 생성 중... 이 과정은 데이터 크기에 따라 몇 분 정도 걸릴 수 있습니다.</div>
            
            <div id="errorMessage" class="error-message"></div>
            
            <h2>임베딩 목록</h2>
            <div id="embeddingsList"></div>
        </div>
        
        <script>
            // 임베딩 목록 로드
            async function loadEmbeddings() {
                try {
                    const response = await fetch('/embeddings/list');
                    const data = await response.json();
                    
                    const listElement = document.getElementById('embeddingsList');
                    listElement.innerHTML = '';
                    
                    if (data.success && data.embeddings.length > 0) {
                        data.embeddings.forEach(embedding => {
                            const card = document.createElement('div');
                            card.className = 'card';
                            
                            const createdAt = embedding.created_at ? new Date(embedding.created_at).toLocaleString() : '날짜 정보 없음';
                            
                            card.innerHTML = `
                                <div class="card-header">
                                    <div class="card-title">${embedding.id}</div>
                                    <div>${createdAt}</div>
                                </div>
                                <div class="card-content">
                                    <div>경로: ${embedding.path}</div>
                                    <div class="stats">
                                        <div class="stat-item">텍스트 임베딩: ${embedding.text_count}개</div>
                                        <div class="stat-item">이미지 임베딩: ${embedding.image_count}개</div>
                                    </div>
                                </div>
                            `;
                            
                            listElement.appendChild(card);
                        });
                    } else {
                        listElement.innerHTML = '<p>저장된 임베딩이 없습니다.</p>';
                    }
                    
                    document.getElementById('errorMessage').textContent = '';
                } catch (error) {
                    console.error('임베딩 목록 로드 중 오류:', error);
                    document.getElementById('errorMessage').textContent = `임베딩 목록을 로드할 수 없습니다: ${error.message}`;
                }
            }
            
            // 임베딩 생성 요청
            async function generateEmbeddings() {
                try {
                    document.getElementById('btnGenerate').disabled = true;
                    document.getElementById('loading').style.display = 'block';
                    document.getElementById('errorMessage').textContent = '';
                    
                    // 비동기 작업이므로 생성 작업 시작만 요청하고 결과는 폴링으로 확인
                    // 실제 구현에서는 백그라운드 작업 API 추가 필요
                    const response = await fetch('/api/embedding/generate', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            limit: 50  // 테스트용으로 50개만 처리
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        // 작업이 시작되면 5초 후 목록 자동 새로고침
                        setTimeout(() => {
                            loadEmbeddings();
                            document.getElementById('loading').style.display = 'none';
                            document.getElementById('btnGenerate').disabled = false;
                        }, 5000);
                    } else {
                        document.getElementById('errorMessage').textContent = data.message;
                        document.getElementById('loading').style.display = 'none';
                        document.getElementById('btnGenerate').disabled = false;
                    }
                } catch (error) {
                    console.error('임베딩 생성 중 오류:', error);
                    document.getElementById('errorMessage').textContent = `임베딩 생성 중 오류 발생: ${error.message}`;
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('btnGenerate').disabled = false;
                }
            }
            
            // 이벤트 리스너
            document.addEventListener('DOMContentLoaded', () => {
                loadEmbeddings();
                
                document.getElementById('btnRefresh').addEventListener('click', loadEmbeddings);
                document.getElementById('btnGenerate').addEventListener('click', generateEmbeddings);
            });
        </script>
    </body>
    </html>
    """
    return html_content

# 임베딩 생성 API
@router.post("/generate")
async def generate_embeddings(limit: int = 50):
    """임베딩 생성 비동기 작업 시작"""
    try:
        # 백그라운드 작업으로 임베딩 생성 시작
        # 실제 구현에서는 Celery 등의 백그라운드 작업 큐 사용 권장
        from concurrent.futures import ThreadPoolExecutor
        import subprocess
        
        def run_embedding_generator():
            try:
                # 현재 디렉토리에서 상대 경로로 실행
                cmd = f"python -m hadoop.embedding_generator --limit {limit}"
                subprocess.run(cmd, shell=True, check=True)
                return True
            except Exception as e:
                logger.error(f"임베딩 생성 작업 중 오류: {str(e)}")
                return False
        
        # 별도 스레드에서 실행
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(run_embedding_generator)
        
        return {
            "success": True,
            "message": f"임베딩 생성 작업이 시작되었습니다 (처리 항목 수: {limit}개)"
        }
    
    except Exception as e:
        logger.error(f"임베딩 생성 요청 중 오류: {str(e)}")
        return {
            "success": False,
            "message": f"임베딩 생성 요청 중 오류 발생: {str(e)}"
        }