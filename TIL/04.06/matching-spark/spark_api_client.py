"""
윈도우 환경에서 실행할 EC2 Spark API 클라이언트 (형식 수정 버전)
"""
import os
import requests
import json
import base64
from dotenv import load_dotenv
import logging
from PIL import Image
import io

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# EC2 API 설정
EC2_API_URL = os.getenv('EC2_API_URL', 'http://43.201.252.40:5000')

class SparkAPIClient:
    """EC2 Spark API 클라이언트 클래스"""
    
    def __init__(self, base_url=EC2_API_URL):
        """
        초기화
        
        Args:
            base_url (str): EC2 API 기본 URL
        """
        self.base_url = base_url
        
    def health_check(self):
        """
        API 상태 확인
        
        Returns:
            dict: API 상태 정보
        """
        try:
            response = requests.get(f"{self.base_url}/")
            return response.json()
        except Exception as e:
            logger.error(f"API 상태 확인 중 오류: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def match_lost_items(user_post, threshold=0.5, limit=10):
        """
        EC2 Spark API를 호출하여 분실물 매칭 결과를 얻는 함수
        
        Args:
            user_post (dict): 사용자 게시글 정보
            threshold (float): 유사도 임계값
            limit (int): 최대 결과 수
            
        Returns:
            dict: 매칭 결과
        """
        try:
            # EC2 API 설정 - 환경변수에서만 가져옴
            ec2_api_url = os.getenv('EC2_API_URL')
            if not ec2_api_url:
                logger.error("EC2_API_URL 환경변수가 설정되지 않았습니다.")
                return {
                    "success": False,
                    "message": "API 서버 URL이 설정되지 않았습니다.",
                    "matches": []
                }
            
            logger.info(f"EC2 API 호출 준비: {ec2_api_url}/api/match")
            
            # 요청 데이터 준비 - API 형식에 맞게 수정
            request_data = {
                "user_post": user_post,  # API는 user_post 키를 사용함
                "threshold": threshold,
                "limit": limit
            }
            
            # 이미지가 로컬 파일 경로인 경우 base64 인코딩
            if 'image_url' in user_post and user_post['image_url'] and os.path.exists(user_post['image_url']):
                with open(user_post['image_url'], 'rb') as img_file:
                    img_data = img_file.read()
                    user_post['image_data'] = base64.b64encode(img_data).decode('utf-8')
                    # image_url을 제거하고 image_data만 사용
                    del user_post['image_url']
            
            # API 호출
            response = requests.post(
                f"{ec2_api_url}/api/match",
                json=request_data,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API 오류 응답 (HTTP {response.status_code}): {response.text}")
                return {
                    "success": False,
                    "message": f"API 서버 오류 발생 (HTTP {response.status_code})",
                    "matches": []
                }
            
        except Exception as e:
            logger.error(f"API 호출 중 오류: {str(e)}")
            return {
                "success": False,
                "message": f"API 호출 중 오류 발생: {str(e)}",
                "matches": []
            }

# FastAPI API 라우터에서 호출할 함수
def match_lost_items_via_api(user_post, threshold=0.5, limit=10):
    """
    FastAPI 라우터에서 호출할 API 매칭 함수
    
    Args:
        user_post (dict): 사용자 게시글 정보
        threshold (float): 유사도 임계값
        limit (int): 최대 결과 수
        
    Returns:
        dict: 매칭 결과
    """
    client = SparkAPIClient()
    result = client.match_lost_items(user_post, threshold, limit)
    
    # 결과가 None인 경우 처리
    if result is None:
        return {
            "success": False,
            "message": "API 서버에서 응답을 받지 못했습니다",
            "matches": []
        }
    
    # success 키가 없는 경우 기본값 추가
    if "success" not in result:
        result["success"] = "matches" in result and len(result.get("matches", [])) > 0
    if "message" not in result:
        matches_count = len(result.get("matches", []))
        result["message"] = f"{matches_count}개의 유사한 분실물을 찾았습니다" if matches_count > 0 else "일치하는 분실물이 없습니다"
    if "matches" not in result:
        result["matches"] = []
    
    return result

# 직접 테스트용 코드
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='EC2 Spark API 클라이언트 테스트')
    parser.add_argument('--threshold', type=float, default=0.5, help='유사도 임계값')
    parser.add_argument('--limit', type=int, default=10, help='최대 결과 수')
    parser.add_argument('--category', type=str, default='지갑', help='분실물 카테고리')
    parser.add_argument('--item-name', type=str, default='검은색 가죽 지갑', help='물품명')
    parser.add_argument('--color', type=str, default='검정색', help='물품 색상')
    parser.add_argument('--content', type=str, default='지갑을 잃어버렸습니다. 현금과 카드가 들어있어요.', help='게시글 내용')
    parser.add_argument('--location', type=str, default='강남역', help='분실 장소')
    parser.add_argument('--image', type=str, default=None, help='이미지 경로')
    
    args = parser.parse_args()
    
    # 테스트용 사용자 게시글
    test_post = {
        "category": args.category,
        "item_name": args.item_name,
        "color": args.color,
        "content": args.content,
        "location": args.location,
        "image_url": args.image
    }
    
    # API 클라이언트 초기화
    client = SparkAPIClient()
    
    # 상태 확인
    print("API 상태 확인 중...")
    health = client.health_check()
    print(f"API 상태: {health}")
    
    # 분실물 매칭 테스트
    print("\n분실물 매칭 테스트 중...")
    result = client.match_lost_items(test_post, args.threshold, args.limit)
    
    # 결과 출력
    if result.get("success", False):
        print(f"🎉 {result.get('message', '매칭 성공')}")
        print(f"임계값: {result.get('threshold', args.threshold)}, 찾은 항목 수: {result.get('total_matches', 0)}")
        
        for i, item in enumerate(result.get('matches', [])):
            print(f"\n✅ 유사 항목 #{i+1}")
            print(f"ID: {item.get('id', 'N/A')}")
            print(f"카테고리: {item.get('category', 'N/A')}")
            print(f"물품명: {item.get('title', item.get('item_name', 'N/A'))}")
            print(f"색상: {item.get('color', 'N/A')}")
            content = item.get('content', 'N/A')
            print(f"내용: {content[:100]}..." if len(content) > 100 else f"내용: {content}")
            
            similarity = item.get('similarity', {})
            if isinstance(similarity, dict):
                print(f"유사도: 텍스트 {similarity.get('text', 0):.2f}, " +
                      f"이미지 {similarity.get('image', 0):.2f}, " +
                      f"종합 {similarity.get('total', 0):.2f}")
            else:
                print(f"유사도: {similarity}")
    else:
        print(f"❌ {result.get('message', '매칭 실패')}")