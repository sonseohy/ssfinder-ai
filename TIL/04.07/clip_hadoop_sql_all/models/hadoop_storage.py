"""
CLIP 임베딩 데이터를 하둡에 저장하는 모듈
"""
import os
import sys
import logging
import json
import numpy as np
from pyhdfs import HdfsClient
from typing import Dict, List, Optional, Union, Any

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HadoopStorage:
    """
    CLIP 임베딩 데이터를 Hadoop에 저장하고 불러오는 클래스
    """
    
    def __init__(self, hdfs_host: str, hdfs_port: int = 9870, hdfs_user: str = "nn1"):
        """
        HadoopStorage 초기화
        
        Args:
            hdfs_host (str): Hadoop 네임노드 호스트 주소
            hdfs_port (int): WebHDFS 포트 (기본값: 9870)
            hdfs_user (str): HDFS 사용자 이름 (기본값: 'nn1')
        """
        self.base_dir = "/user/hadoop/clip_embeddings"
        self.host = hdfs_host
        self.port = hdfs_port
        self.user = hdfs_user
        
        try:
            # 호스트 문자열에서 "http://" 또는 "https://" 제거 (있는 경우)
            clean_host = hdfs_host
            if clean_host.startswith("http://"):
                clean_host = clean_host[7:]
            elif clean_host.startswith("https://"):
                clean_host = clean_host[8:]
            
            # HDFS 클라이언트 초기화 - 단순한 형식으로 호스트와 포트 지정
            self.client = HdfsClient(hosts=f"{clean_host}:{hdfs_port}", user_name=hdfs_user)
            
            # 기본 디렉토리 생성 (존재하지 않는 경우)
            if not self.client.exists(self.base_dir):
                self.client.mkdirs(self.base_dir)
                logger.info(f"기본 디렉토리 생성됨: {self.base_dir}")
                
            logger.info(f"HDFS 클라이언트 초기화 완료 (host: {clean_host}:{hdfs_port}, user: {hdfs_user})")
        except Exception as e:
            logger.error(f"HDFS 클라이언트 초기화 실패: {str(e)}")
            raise
    
    def _numpy_to_list(self, arr: np.ndarray) -> List:
        """
        NumPy 배열을 리스트로 변환 (JSON 직렬화용)
        
        Args:
            arr (numpy.ndarray): 변환할 NumPy 배열
            
        Returns:
            list: 변환된 리스트
        """
        return arr.tolist()
    
    def _list_to_numpy(self, data: List) -> np.ndarray:
        """
        리스트를 NumPy 배열로 변환
        
        Args:
            data (list): 변환할 리스트
            
        Returns:
            numpy.ndarray: 변환된 NumPy 배열
        """
        return np.array(data)
    
    def save_embedding(self, 
                  item_id: str, 
                  text_embedding: np.ndarray, 
                  image_embedding: Optional[np.ndarray] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        CLIP 임베딩을 HDFS에 저장
        """
        # 세이프 파일명으로 변환 (특수문자 제거)
        safe_id = ''.join(c if c.isalnum() else '_' for c in item_id)
        
        # 저장할 디렉토리 및 파일 경로
        directory = f"{self.base_dir}"
        file_path = f"{directory}/{safe_id}.json"
        
        # 임베딩 데이터 구성
        embedding_data = {
            "id": item_id,
            "text_embedding": self._numpy_to_list(text_embedding)
        }
        
        # 이미지 임베딩이 있는 경우 추가
        if image_embedding is not None:
            embedding_data["image_embedding"] = self._numpy_to_list(image_embedding)
        
        # 메타데이터가 있는 경우 추가
        if metadata:
            embedding_data["metadata"] = metadata
        
        try:
            # JSON으로 직렬화
            json_data = json.dumps(embedding_data)
            
            # WebHDFS REST API 직접 사용
            import requests
            
            # 파일 생성 URL
            # 처음 임베딩 생성 시작할때는 아래 코드로 했었음음
            # create_url = f"http://{self.host}:{self.port}/webhdfs/v1{file_path}?op=CREATE&user.name={self.user}&overwrite=true"
            # 중간부터 시작할때는 아래 코드로 하니까 됨
            create_url = f"{self.host}:{self.port}/webhdfs/v1{file_path}?op=CREATE&user.name={self.user}&overwrite=true"
            
            # PUT 요청 보내기 (리다이렉트 없이)
            response = requests.put(create_url, allow_redirects=False)
            
            if response.status_code == 307:
                # 리다이렉트 URL 가져오기
                location = response.headers['Location']
                
                # 리다이렉트 URL을 변경하지 않고 그대로 사용
                # 이전 코드에서 문제가 되었던 부분 제거
                
                # 데이터 업로드
                upload_response = requests.put(location, data=json_data.encode('utf-8'))
                
                if upload_response.status_code == 201:
                    logger.info(f"임베딩이 HDFS에 저장됨: {file_path}")
                    return file_path
                else:
                    raise Exception(f"데이터 업로드 실패: {upload_response.status_code}, {upload_response.text}")
            else:
                raise Exception(f"파일 생성 요청 실패: {response.status_code}, {response.text}")
                
        except Exception as e:
            logger.error(f"HDFS에 임베딩 저장 실패: {str(e)}")
            raise
    
    def load_embedding(self, item_id: str) -> Dict[str, Any]:
        """
        HDFS에서 임베딩 데이터 로드
        
        Args:
            item_id (str): 아이템 ID (파일명으로 사용됨)
            
        Returns:
            dict: 로드된 임베딩 데이터 (NumPy 배열로 변환된 임베딩 포함)
        """
        # 세이프 파일명으로 변환
        safe_id = ''.join(c if c.isalnum() else '_' for c in item_id)
        file_path = f"{self.base_dir}/{safe_id}.json"
        
        try:
            # 파일 존재 확인
            if not self.client.exists(file_path):
                raise FileNotFoundError(f"임베딩 파일이 없습니다: {file_path}")
            
            # 파일 읽기
            content = self.client.open(file_path).read().decode('utf-8')
            embedding_data = json.loads(content)
            
            # 리스트를 NumPy 배열로 변환
            if "text_embedding" in embedding_data:
                embedding_data["text_embedding"] = self._list_to_numpy(embedding_data["text_embedding"])
                
            if "image_embedding" in embedding_data:
                embedding_data["image_embedding"] = self._list_to_numpy(embedding_data["image_embedding"])
            
            return embedding_data
        except Exception as e:
            logger.error(f"HDFS에서 임베딩 로드 실패: {str(e)}")
            raise
    
    def search_similar_embeddings(self, 
                                query_embedding: np.ndarray, 
                                threshold: float = 0.5, 
                                limit: int = 10,
                                embedding_type: str = "text_embedding") -> List[Dict[str, Any]]:
        """
        유사한 임베딩을 검색
        
        Args:
            query_embedding (numpy.ndarray): 검색할 쿼리 임베딩
            threshold (float): 유사도 임계값 (0~1 사이)
            limit (int): 반환할 최대 결과 수
            embedding_type (str): 비교할 임베딩 유형 ('text_embedding' 또는 'image_embedding')
            
        Returns:
            list: 유사도 기준으로 정렬된 항목 리스트
        """
        results = []
        
        try:
            # 임베딩 파일 목록 가져오기
            file_list = self.client.listdir(self.base_dir)
            
            for file_name in file_list:
                if not file_name.endswith('.json'):
                    continue
                
                file_path = f"{self.base_dir}/{file_name}"
                
                # 파일 읽기
                content = self.client.open(file_path).read().decode('utf-8')
                item_data = json.loads(content)
                
                # 비교할 임베딩이 있는지 확인
                if embedding_type not in item_data:
                    continue
                
                # 임베딩 가져오기
                item_embedding = self._list_to_numpy(item_data[embedding_type])
                
                # 유사도 계산 (코사인 유사도)
                similarity = np.dot(query_embedding, item_embedding.T)[0, 0]
                similarity = (similarity + 1) / 2  # -1~1 범위를 0~1 범위로 변환
                
                if similarity >= threshold:
                    results.append({
                        "id": item_data["id"],
                        "similarity": float(similarity),
                        "metadata": item_data.get("metadata", {})
                    })
            
            # 유사도 기준으로 정렬
            results.sort(key=lambda x: x["similarity"], reverse=True)
            
            # 결과 제한
            return results[:limit]
        
        except Exception as e:
            logger.error(f"유사 임베딩 검색 실패: {str(e)}")
            raise
    
    def delete_embedding(self, item_id: str) -> bool:
        """
        임베딩 파일 삭제
        
        Args:
            item_id (str): 삭제할 아이템 ID
            
        Returns:
            bool: 삭제 성공 여부
        """
        safe_id = ''.join(c if c.isalnum() else '_' for c in item_id)
        file_path = f"{self.base_dir}/{safe_id}.json"
        
        try:
            if not self.client.exists(file_path):
                logger.warning(f"삭제할 파일이 없습니다: {file_path}")
                return False
            
            self.client.delete(file_path)
            logger.info(f"임베딩 파일 삭제됨: {file_path}")
            return True
        except Exception as e:
            logger.error(f"임베딩 파일 삭제 실패: {str(e)}")
            return False

# 모듈 테스트용 코드
if __name__ == "__main__":
    # 테스트 코드
    import numpy as np
    
    # 테스트용 환경 변수 설정
    os.environ["HADOOP_HOST"] = "localhost"  # EC2 인스턴스의 IP로 변경 필요
    
    # 테스트 임베딩 생성
    test_text_embedding = np.random.rand(1, 512)  # 임의의 512차원 임베딩
    test_image_embedding = np.random.rand(1, 512)
    test_metadata = {
        "category": "지갑",
        "item_name": "검은색 가죽 지갑",
        "description": "테스트 항목입니다."
    }
    
    # Hadoop 스토리지 초기화
    try:
        hadoop_storage = HadoopStorage(
            hdfs_host=os.environ.get("HADOOP_HOST", "localhost"),
            hdfs_user="nn1"  # 사용자 이름을 명시적으로 지정
        )
        
        # 임베딩 저장 테스트
        file_path = hadoop_storage.save_embedding(
            "test_item_001",
            test_text_embedding,
            test_image_embedding,
            test_metadata
        )
        print(f"임베딩이 저장됨: {file_path}")
        
        # 임베딩 로드 테스트
        loaded_data = hadoop_storage.load_embedding("test_item_001")
        print("로드된 임베딩:", loaded_data["id"])
        print("텍스트 임베딩 shape:", loaded_data["text_embedding"].shape)
        print("이미지 임베딩 shape:", loaded_data["image_embedding"].shape)
        
        # 유사 임베딩 검색 테스트
        similar_items = hadoop_storage.search_similar_embeddings(
            test_text_embedding,
            threshold=0.5,
            limit=5
        )
        print(f"유사한 항목 {len(similar_items)}개 찾음")
        for item in similar_items:
            print(f"  - {item['id']}: 유사도 {item['similarity']:.4f}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")