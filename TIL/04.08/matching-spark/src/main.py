"""
분실물 매칭 서비스의 메인 실행 파일
"""
import argparse
import numpy as np
from similarity_engine import SimilarityEngine
import json
import os
from datetime import datetime

def load_query_embedding(embedding_path):
    """
    새로운 분실물 이미지의 임베딩 로드
    
    실제 구현에서는 CLIP 모델을 사용하여 임베딩을 생성할 수 있음
    여기서는 미리 계산된 임베딩을 로드한다고 가정
    """
    try:
        # numpy 배열로 저장된 임베딩 로드
        embedding = np.load(embedding_path)
        return embedding
    except Exception as e:
        print(f"임베딩 로드 오류: {str(e)}")
        return None

def save_results(results, output_path):
    """결과를 JSON 파일로 저장"""
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"결과가 {output_path}에 저장되었습니다.")

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='분실물 매칭 서비스')
    parser.add_argument('--embedding', required=True, help='쿼리 임베딩 파일 경로 (.npy)')
    parser.add_argument('--hdfs-path', required=True, help='하둡에 저장된 임베딩 경로')
    parser.add_argument('--threshold', type=float, default=0.7, help='유사도 임계값 (0.0-1.0)')
    parser.add_argument('--top-k', type=int, default=10, help='반환할 최상위 결과 수')
    parser.add_argument('--output', default='./results.json', help='결과 저장 경로')
    
    args = parser.parse_args()
    
    # 쿼리 임베딩 로드
    query_embedding = load_query_embedding(args.embedding)
    if query_embedding is None:
        print("임베딩을 로드할 수 없습니다. 종료합니다.")
        return
    
    # 유사도 엔진 초기화
    similarity_engine = SimilarityEngine(
        hadoop_embeddings_path=args.hdfs_path,
        threshold=args.threshold
    )
    
    try:
        # 유사한 아이템 찾기
        print(f"유사도 {args.threshold * 100}% 이상인 아이템을 검색 중...")
        start_time = datetime.now()
        
        similar_items = similarity_engine.find_similar_items(
            query_embedding=query_embedding,
            top_k=args.top_k
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"검색 완료! ({duration:.2f}초 소요)")
        print(f"유사한 아이템 수: {len(similar_items)}")
        
        if similar_items:
            # 결과 저장
            save_results(similar_items, args.output)
            
            # 결과 요약 출력
            print("\n상위 유사 아이템:")
            for i, item in enumerate(similar_items[:5], 1):
                print(f"{i}. {item['title']} - 유사도: {item['similarity']}%")
        else:
            print("임계값을 충족하는 유사한 아이템이 없습니다.")
    
    finally:
        # Spark 세션 종료
        similarity_engine.stop()

if __name__ == "__main__":
    main()