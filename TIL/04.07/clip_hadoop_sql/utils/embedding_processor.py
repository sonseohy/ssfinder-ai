import os
import logging
import requests
import json
from typing import List, Dict, Any, Optional
from .simple_db import fetch_found_items, fetch_found_item_by_id

# 로깅 설정
logger = logging.getLogger(__name__)

class EmbeddingProcessor:
    def __init__(self, api_base_url: str = "http://localhost:5000"):
        """
        임베딩 생성 및 하둡 저장 처리기 초기화
        
        Args:
            api_base_url (str): API 서버 기본 URL
        """
        self.api_base_url = api_base_url
        self.embedding_endpoint = f"{api_base_url}/api/hadoop/save-embedding"
    
    def process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        단일 항목을 처리하여 임베딩 생성 및 하둡 저장
        
        Args:
            item (Dict): 처리할 항목 데이터
            
        Returns:
            Dict: API 응답 결과
        """
        if not item:
            logger.warning("처리할 항목이 비어 있습니다.")
            return {"success": False, "message": "항목이 비어 있습니다."}
        
        # 필요한 필드 추출
        item_id = str(item.get('id', ''))
        
        # 텍스트 데이터 구성 (이름과 상세 내용 결합)
        name = item.get('name', '')
        detail = item.get('detail', '')
        location = item.get('location', '')
        
        # 제목, 설명, 위치를 결합하여 텍스트 생성
        text_parts = []
        if name:
            text_parts.append(f"이름: {name}")
        if detail:
            text_parts.append(f"상세: {detail}")
        if location:
            text_parts.append(f"위치: {location}")
            
        text = " ".join(text_parts)
        
        # 이미지 URL 가져오기
        image_url = item.get('image', None)
        
        # 메타데이터 구성 (날짜, 상태 등 추가 정보)
        metadata = {
            "found_at": item.get('found_at', ''),
            "status": item.get('status', ''),
            "phone": item.get('phone', ''),
            "management_id": item.get('management_id', '')
        }
        
        # API 요청 데이터 구성
        payload = {
            "item_id": item_id,
            "text": text,
            "image_url": image_url,
            "metadata": metadata
        }
        
        logger.info(f"항목 ID {item_id}에 대한 임베딩 생성 및 저장 요청")
        
        try:
            # API 호출
            response = requests.post(
                self.embedding_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            # 응답 처리
            if response.status_code == 200:
                result = response.json()
                logger.info(f"항목 ID {item_id}의 임베딩이 성공적으로 저장되었습니다: {result.get('file_path')}")
                return result
            else:
                error_msg = f"API 오류 (코드: {response.status_code}): {response.text}"
                logger.error(error_msg)
                return {"success": False, "message": error_msg}
                
        except Exception as e:
            error_msg = f"임베딩 생성 중 오류: {str(e)}"
            logger.error(error_msg)
            return {"success": False, "message": error_msg}
    
    def process_multiple_items(self, 
                              host: str = None, 
                              port: int = None, 
                              user: str = None,
                              password: str = None, 
                              database: str = None, 
                              limit: int = 10) -> List[Dict[str, Any]]:
        """
        여러 항목을 처리하여 임베딩 생성 및 하둡 저장
        
        Args:
            host (str): DB 호스트
            port (int): DB 포트
            user (str): DB 사용자
            password (str): DB 비밀번호
            database (str): DB 이름
            limit (int): 처리할 최대 항목 수
            
        Returns:
            List[Dict]: 각 항목의 처리 결과
        """
        # DB에서 항목 가져오기
        items = fetch_found_items(host, port, user, password, database, limit)
        
        if not items:
            logger.warning("처리할 항목이 없습니다.")
            return []
        
        logger.info(f"{len(items)}개 항목에 대한 임베딩 생성 및 저장 시작")
        
        results = []
        for item in items:
            result = self.process_item(item)
            results.append({
                "item_id": item.get('id'),
                "result": result
            })
        
        logger.info(f"{len(items)}개 항목 처리 완료")
        return results
    
    def process_by_id(self, 
                     item_id: int,
                     host: str = None, 
                     port: int = None, 
                     user: str = None,
                     password: str = None, 
                     database: str = None) -> Dict[str, Any]:
        """
        특정 ID의 항목을 처리하여 임베딩 생성 및 하둡 저장
        
        Args:
            item_id (int): 처리할 항목 ID
            host (str): DB 호스트
            port (int): DB 포트
            user (str): DB 사용자
            password (str): DB 비밀번호
            database (str): DB 이름
            
        Returns:
            Dict: 처리 결과
        """
        # ID로 항목 가져오기
        item = fetch_found_item_by_id(item_id)
        
        if not item:
            error_msg = f"ID {item_id}인 항목을 찾을 수 없습니다."
            logger.warning(error_msg)
            return {"success": False, "message": error_msg}
        
        return self.process_item(item)