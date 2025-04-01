import os
import argparse
from typing import List, Dict, Any
import json

from config import Config
from models import CLIPModel, BLIPModel, ColorAnalyzer
from similarity import CategoryMatcher, ImageComparator, SimilarityCalculator
from utils.preprocessing import preprocess_image
from utils.text_utils import normalize_text, extract_keywords
from utils.image_utils import visualize_similarity_comparison

def parse_arguments():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='분실물 게시글 유사도 비교 시스템')
    
    parser.add_argument('--query', type=str, required=True,
                        help='질의 게시글 정보 (JSON 파일 경로)')
    
    parser.add_argument('--db', type=str, required=True,
                        help='데이터베이스 게시글 정보 (JSON 파일 경로)')
    
    parser.add_argument('--output', type=str, default='results.json',
                        help='결과 출력 경로 (JSON 파일)')
    
    parser.add_argument('--threshold', type=float, default=None,
                        help='유사도 임계값 (기본값: Config에서 설정)')
    
    parser.add_argument('--max-results', type=int, default=None,
                        help='최대 결과 수 (기본값: Config에서 설정)')
    
    parser.add_argument('--visualize', action='store_true',
                        help='결과 시각화 저장 여부')
    
    parser.add_argument('--preprocess', action='store_true',
                        help='이미지 전처리 수행 여부')
    
    return parser.parse_args()

def load_posts(file_path: str) -> List[Dict[str, Any]]:
    """게시글 정보 로드"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"게시글 로드 오류: {e}")
        return []

def save_results(results: List[Dict[str, Any]], file_path: str) -> bool:
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
    # 인자 파싱
    args = parse_arguments()
    
    # 질의 게시글 로드
    query_posts = load_posts(args.query)
    if not query_posts:
        print("질의 게시글을 로드할 수 없습니다.")
        return
    
    # 단일 질의 게시글 모드 또는 배치 모드 확인
    is_single_query = isinstance(query_posts, dict)
    if is_single_query:
        query_posts = [query_posts]
    
    # 데이터베이스 게시글 로드
    db_posts = load_posts(args.db)
    if not db_posts:
        print("데이터베이스 게시글을 로드할 수 없습니다.")
        return
    
    # 이미지 전처리 (필요한 경우)
    if args.preprocess:
        print("이미지 전처리 수행 중...")
        for post in query_posts + db_posts:
            if 'image_path' in post and post['image_path']:
                processed_path = preprocess_image(post['image_path'])
                post['original_image_path'] = post['image_path']
                post['image_path'] = processed_path
    
    # 유사도 계산기 초기화
    similarity_calculator = SimilarityCalculator()
    
    # 임계값 및 최대 결과 수 설정
    threshold = args.threshold or Config.SIMILARITY_THRESHOLD
    max_results = args.max_results or Config.MAX_RECOMMENDATIONS
    
    # 결과 저장용 리스트
    all_results = []
    
    # 각 질의 게시글에 대해 유사한 게시글 찾기
    for i, query_post in enumerate(query_posts):
        print(f"\n[{i+1}/{len(query_posts)}] 질의 게시글 처리 중: {query_post.get('title', '제목 없음')}")
        
        # 유사한 게시글 찾기
        similar_posts = similarity_calculator.find_similar_posts(
            query_post, db_posts, threshold, max_results
        )
        
        # 결과 포맷팅
        result = {
            'query_post': query_post,
            'similar_posts': [
                {
                    'post': post,
                    'similarity': similarity
                }
                for post, similarity in similar_posts
            ]
        }
        
        all_results.append(result)
        
        # 결과 출력
        print(f"유사한 게시글 {len(similar_posts)}개 찾음:")
        for j, (post, similarity) in enumerate(similar_posts):
            print(f"  {j+1}. {post.get('title', '제목 없음')} - 유사도: {similarity:.4f}")
        
        # 결과 시각화 (필요한 경우)
        if args.visualize and similar_posts:
            os.makedirs('visualizations', exist_ok=True)
            
            for j, (post, similarity) in enumerate(similar_posts[:3]):  # 상위 3개만 시각화
                if 'image_path' in query_post and 'image_path' in post:
                    # 이미지 비교기 초기화
                    image_comparator = ImageComparator()
                    
                    # 유사도 점수 계산
                    similarity_scores = image_comparator.calculate_combined_similarity(
                        query_post['image_path'],
                        post['image_path']
                    )
                    
                    # 시각화 저장
                    output_path = f"visualizations/comparison_{i+1}_{j+1}.png"
                    visualize_similarity_comparison(
                        query_post['image_path'],
                        post['image_path'],
                        similarity_scores,
                        output_path
                    )
                    print(f"  시각화 저장됨: {output_path}")
    
    # 단일 질의인 경우 결과 포맷 조정
    if is_single_query:
        all_results = all_results[0]
    
    # 결과 저장
    if save_results(all_results, args.output):
        print(f"\n결과가 {args.output}에 저장되었습니다.")
    else:
        print("\n결과 저장에 실패했습니다.")

if __name__ == "__main__":
    main()