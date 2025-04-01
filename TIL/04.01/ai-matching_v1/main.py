import os
import uvicorn
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

if __name__ == "__main__":
    # 환경 변수에서 포트 가져오기 (기본값: 8000)
    port = int(os.getenv("PORT", 8000))
    
    # Uvicorn으로 FastAPI 서버 실행
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        reload=bool(os.getenv("DEBUG", "False") == "True")
    )