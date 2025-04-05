import os
import argparse
from typing import List, Dict, Any
import json
import time
from dotenv import load_dotenv

from config import Config
from utils.police_api import PoliceApiClient
from similarity import SimilarityCalculator
from utils.preprocessing import preprocess_image
from utils.image_utils import visualize_similarity_comparison

def parse_arguments():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='경찰청 API 분실물 데이터와 유사도 비교 테스트')
    
    parser.add_argument('--query', type=str, help='질의 게시글 정보 (JSON 파일 경로)')
    parser.add_argument('--query-image', type=str, help='질의 이미지 경로')
    parser.add_argument('--query-title', type=str, help='질의 제목')
    parser.add_argument('--query-content', type=str, default='', help='질의 내용')
    parser.add_argument('--query-category', type=str, default=None, help='질의 카테고리')
    
    parser.add_argument('--num-items', type=int, default=10, help='가져올 경찰청 데이터 수')
    parser.add_argument('--days-ago', type=int, default=30, help='몇 일 전부터의 데이터를 가져올지 설정')
    
    parser.add_argument('--output', type=str, default='results.json', help='결과 출력 경로 (JSON 파일)')
    parser.add_argument('--threshold', type=float, default=None, help='유사도 임계값 (기본값: Config에서 설정)')
    parser.add_argument('--max-results', type=int, default=None, help='최대 결과 수 (기본값: Config에서 설정)')
    parser.add_argument('--visualize', action='store_true', help='결과 시각화 저장 여부')
    parser.add_argument('--preprocess', action='store_true', help='이미지 전처리 수행 여부')
    
    return parser.parse_args()

def create_query_post(args) -> Dict[str, Any]:
    """명령줄 인자로부터 질의 게시글 생성"""
    if args.query:
        # JSON 파일에서 질의 게시글 로드
        try:
            with open(args.query, 'r', encoding='utf-8') as f:
                query_post = json.load(f)
            return query_post
        except Exception as e:
            print(f"질의 게시글 로드 오류: {e}")
            return {}
    else:
        # 명령줄 인자로 질의 게시글 생성
        query_post = {
            "id": "query_post",
            "title": args.query_title or "테스트 질의",
            "content": args.query_content or "",
            "category": args.query_category,
            "image_path": args.query_image
        }
        return query_post

def save_results(results: Dict[str, Any], file_path: str) -> bool:
    """결과 저장"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"결과 저장 오류: {e}")
        return False

def main():
    """메인 함수"""
    # .env 파일 로드
    load_dotenv()
    
    # 인자 파싱
    args = parse_arguments()
    
    print("경찰청 API에서 분실물 데이터 가져오는 중...")
    # 경찰청 API 클라이언트 초기화
    try:
        police_client = PoliceApiClient()
    except ValueError as e:
        print(f"오류: {e}")
        print("환경 변수 POLICE_API_SERVICE_KEY를 설정해주세요.")
        return
    
    # 경찰청 API에서 데이터 가져오기
    police_items = police_client.fetch_lost_items(num_items=args.num_items, days_ago=args.days_ago)
    if not police_items:
        print("데이터를 가져오지 못했습니다.")
        return
    
    # 게시글 형식으로 변환
    db_posts = police_client.convert_to_post_format(police_items)
    print(f"경찰청 API에서 {len(db_posts)}개 게시글을 가져왔습니다.")
    
    # 임시로 JSON 파일에 저장 (확인용)
    with open('police_posts.json', 'w', encoding='utf-8') as f:
        json.dump(db_posts, f, ensure_ascii=False, indent=2)
    print("경찰청 데이터를 police_posts.json에 저장했습니다.")
    
    # 질의 게시글 생성
    query_post = create_query_post(args)
    if not query_post:
        print("질의 게시글을 생성할 수 없습니다.")
        return
    
    # 이미지 전처리 (필요한 경우)
    if args.preprocess:
        print("이미지 전처리 수행 중...")
        # 질의 게시글 이미지 전처리
        if 'image_path' in query_post and query_post['image_path']:
            processed_path = preprocess_image(query_post['image_path'])
            query_post['original_image_path'] = query_post['image_path']
            query_post['image_path'] = processed_path
        
        # DB 게시글 이미지 전처리
        for post in db_posts:
            if 'image_path' in post and post['image_path']:
                processed_path = preprocess_image(post['image_path'])
                post['original_image_path'] = post['image_path']
                post['image_path'] = processed_path
    
    # 유사도 계산기 초기화
    similarity_calculator = SimilarityCalculator()
    
    # 임계값 및 최대 결과 수 설정
    threshold = args.threshold or Config.SIMILARITY_THRESHOLD
    max_results = args.max_results or Config.MAX_RECOMMENDATIONS
    
    print(f"\n질의 게시글 정보:")
    print(f"제목: {query_post.get('title', '제목 없음')}")
    print(f"내용: {query_post.get('content', '내용 없음')}")
    print(f"카테고리: {query_post.get('category', '카테고리 없음')}")
    print(f"이미지 경로: {query_post.get('image_path', '이미지 없음')}")
    print("\n유사도 비교 중...")
    
    # 유사도 비교 시작 시간
    start_time = time.time()
    
    # 유사한 게시글 찾기
    similar_posts = similarity_calculator.find_similar_posts(
        query_post, db_posts, threshold, max_results
    )
    
    # 유사도 비교 종료 시간 및 소요 시간 계산
    end_time = time.time()
    processing_time = end_time - start_time
    
    # 결과 포맷팅
    result = {
        'query_post': query_post,
        'similar_posts': [
            {
                'post': post,
                'similarity': similarity
            }
            for post, similarity in similar_posts
        ],
        'processing_time': processing_time
    }
    
    # 결과 출력
    print(f"\n유사한 게시글 {len(similar_posts)}개 찾음:")
    for i, (post, similarity) in enumerate(similar_posts):
        print(f"\n[{i+1}] {post.get('title', '제목 없음')} - 유사도: {similarity:.4f}")
        print(f"  내용: {post.get('content', '내용 없음')[:100]}...")
        print(f"  카테고리: {post.get('category', '카테고리 없음')}")
        print(f"  이미지: {post.get('image_path', '이미지 없음')}")
    
    print(f"\n처리 시간: {processing_time:.2f}초")
    
    # 결과 시각화 (필요한 경우)
    if args.visualize and similar_posts and 'image_path' in query_post and query_post['image_path']:
        print("\n결과 시각화 중...")
        os.makedirs('visualizations', exist_ok=True)
        
        from similarity import ImageComparator
        image_comparator = ImageComparator()
        
        for i, (post, similarity) in enumerate(similar_posts[:3]):  # 상위 3개만 시각화
            if 'image_path' in post and post['image_path']:
                # 유사도 점수 계산
                similarity_scores = image_comparator.calculate_combined_similarity(
                    query_post['image_path'],
                    post['image_path']
                )
                
                # 각 유사도 점수에 최종 유사도 추가
                similarity_scores['final_similarity'] = similarity
                
                # 시각화 저장
                output_path = f"visualizations/comparison_{i+1}.png"
                visualize_similarity_comparison(
                    query_post['image_path'],
                    post['image_path'],
                    similarity_scores,
                    output_path
                )
                print(f"  시각화 저장됨: {output_path}")
    
    # 결과 저장
    if save_results(result, args.output):
        print(f"\n결과가 {args.output}에 저장되었습니다.")
    else:
        print("\n결과 저장에 실패했습니다.")

if __name__ == "__main__":
    main()