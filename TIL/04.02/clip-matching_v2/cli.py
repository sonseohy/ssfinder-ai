"""
CLI 테스트 모듈 - 커맨드 라인에서 습득물 유사도 시스템 테스트
"""
import os
import sys
import logging
import argparse
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

# 경로 설정 및 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import SIMILARITY_THRESHOLD, POLICE_API_SERVICE_KEY
from api.police_api import fetch_police_lost_items
from models.clip_model import KoreanCLIPModel
from utils.similarity import find_similar_items

def create_test_post():
    """
    테스트용 사용자 게시글 생성
    
    Returns:
        dict: 테스트용 사용자 게시글
    """
    return {
        'category': '지갑',
        'item_name': '검은색 가죽 지갑',
        'color': '검정색',
        'content': '지난주 토요일 강남역 근처에서 검정색 가죽 지갑을 잃어버렸습니다. 현금과 카드가 들어있어요.',
        'location': '강남역',
        'image_url': None  # 테스트에서는 이미지 없음
    }

def test_with_different_post():
    """
    다른 종류의 테스트용 사용자 게시글 생성
    
    Returns:
        dict: 테스트용 사용자 게시글
    """
    return {
        'category': '전자기기',
        'item_name': '갤럭시 핸드폰',
        'color': '검정색',
        'content': '어제 저녁 시청역 근처 카페에서 갤럭시 S23 핸드폰을 잃어버렸습니다. 케이스는 파란색입니다.',
        'location': '시청역',
        'image_url': None
    }

def test_with_other_category():
    """
    기타 카테고리 테스트용 사용자 게시글 생성
    
    Returns:
        dict: 테스트용 사용자 게시글
    """
    return {
        'category': '기타',
        'item_name': '에어팟',
        'color': '화이트',
        'content': '에어팟 프로를 분당선 전철 안에서 잃어버렸습니다. 케이스에 스티커가 붙어있습니다.',
        'location': '분당선',
        'image_url': None
    }

def main():
    """
    메인 함수: 사용자 게시글과 습득물 데이터 비교
    """
    parser = argparse.ArgumentParser(description='습득물 게시글 유사도 비교 시스템')
    parser.add_argument('--threshold', type=float, default=SIMILARITY_THRESHOLD,
                      help='유사도 임계값 (기본값: config 설정)')
    parser.add_argument('--items', type=int, default=20,
                      help='가져올 습득물 데이터 수 (기본값: 20)')
    parser.add_argument('--test-type', type=int, choices=[1, 2, 3], default=1,
                      help='테스트 유형 (1: 기본, 2: 전자기기, 3: 기타 카테고리)')
    
    args = parser.parse_args()
    
    # 서비스 키 확인
    service_key = POLICE_API_SERVICE_KEY
    if not service_key:
        logger.error("API 서비스 키가 설정되지 않았습니다. .env 파일에 POLICE_API_SERVICE_KEY를 설정하세요.")
        return
    
    # 테스트 유형에 따른 사용자 게시글 생성
    if args.test_type == 1:
        user_post = create_test_post()
        logger.info("테스트 유형: 지갑 분실 게시글")
    elif args.test_type == 2:
        user_post = test_with_different_post()
        logger.info("테스트 유형: 전자기기 분실 게시글")
    else:
        user_post = test_with_other_category()
        logger.info("테스트 유형: 기타 카테고리 게시글")
    
    logger.info("사용자 게시글:")
    for key, value in user_post.items():
        logger.info(f"  {key}: {value}")
    
    # 경찰청 API에서 습득물 데이터 가져오기
    logger.info(f"경찰청 API에서 {args.items}개 습득물 데이터 가져오는 중...")
    lost_items = fetch_police_lost_items(service_key, args.items)
    
    if not lost_items:
        logger.error("습득물 데이터를 가져오지 못했습니다.")
        return
    
    logger.info(f"{len(lost_items)}개 습득물 데이터를 가져왔습니다.")
    
    # CLIP 모델 초기화
    try:
        logger.info("한국어 CLIP 모델 초기화 중...")
        clip_model = KoreanCLIPModel()
    except Exception as e:
        logger.error(f"CLIP 모델 초기화 실패: {str(e)}")
        # 이미지 비교 없이 텍스트만 비교
        clip_model = None
        logger.warning("이미지 비교 없이 텍스트만 비교합니다.")
    
    # 유사한 습득물 찾기
    logger.info(f"유사도 임계값: {args.threshold}")
    similar_items = find_similar_items(user_post, lost_items, args.threshold, clip_model)
    
    # 결과 출력
    if similar_items:
        logger.info(f"\n유사도 {args.threshold} 이상인 습득물 {len(similar_items)}개를 찾았습니다:")
        
        for i, item_data in enumerate(similar_items):
            item = item_data['item']
            similarity = item_data['similarity']
            details = item_data['details']
            
            logger.info(f"\n{i+1}. 유사도: {similarity:.4f}")
            logger.info(f"  카테고리: {item.get('category', '정보 없음')}")
            logger.info(f"  물품명: {item.get('item_name', '정보 없음')}")
            logger.info(f"  색상: {item.get('color', '정보 없음')}")
            logger.info(f"  습득장소: {item.get('location', '정보 없음')}")
            
            # 세부 유사도 정보 출력
            logger.info("  세부 유사도:")
            for key, value in details['details'].items():
                logger.info(f"    {key}: {value:.4f}")
    else:
        logger.info(f"유사도 {args.threshold} 이상인 습득물을 찾지 못했습니다.")

if __name__ == "__main__":
    main()