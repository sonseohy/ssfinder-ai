import logging
import sys
import argparse
from typing import List, Dict, Any
from .embedding_processor import EmbeddingProcessor
from .simple_db import init_connection_pool

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description='습득물 항목의 임베딩을 생성하고 하둡에 저장합니다.')
    parser.add_argument('--host', default='localhost', help='MySQL 호스트')
    parser.add_argument('--port', type=int, default=3306, help='MySQL 포트')
    parser.add_argument('--user', required=True, help='MySQL 사용자')
    parser.add_argument('--password', required=True, help='MySQL 비밀번호')
    parser.add_argument('--database', required=True, help='데이터베이스 이름')
    parser.add_argument('--limit', type=int, default=10, help='처리할 최대 항목 수')
    parser.add_argument('--api-url', default='http://localhost:5000', help='API 서버 URL')
    parser.add_argument('--item-id', type=int, help='특정 항목 ID (지정시 단일 항목만 처리)')
    return parser.parse_args()

def main():
    args = get_args()
    
    try:
        # 연결 풀 초기화 (선택적)
        init_connection_pool(args.host, args.port, args.user, args.password, args.database)
        
        processor = EmbeddingProcessor(api_base_url=args.api_url)
        
        if args.item_id:
            # 단일 항목 처리
            logger.info(f"항목 ID {args.item_id} 처리 중...")
            result = processor.process_by_id(
                args.item_id,
                args.host, args.port, args.user, args.password, args.database
            )
            
            if result.get('success'):
                logger.info(f"항목 ID {args.item_id} 처리 성공: {result.get('message')}")
            else:
                logger.error(f"항목 ID {args.item_id} 처리 실패: {result.get('message')}")
        else:
            # 여러 항목 처리
            logger.info(f"최대 {args.limit}개 항목 처리 중...")
            results = processor.process_multiple_items(
                args.host, args.port, args.user, args.password, args.database, args.limit
            )
            
            success_count = sum(1 for r in results if r.get('result', {}).get('success', False))
            logger.info(f"처리 완료: {success_count}/{len(results)} 항목 성공")
            
            # 실패한 항목 로깅
            for result in results:
                if not result.get('result', {}).get('success', False):
                    logger.warning(f"항목 ID {result.get('item_id')} 처리 실패: {result.get('result', {}).get('message')}")
        
        return 0
    except Exception as e:
        logger.error(f"실행 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())