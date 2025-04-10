# 한국어 CLIP 기반 임베딩 생성 및 유사도 비교 API

이 서비스는 한국어 CLIP 모델을 사용하여 텍스트와 이미지의 임베딩을 생성하고 유사도를 계산하는 API를 제공합니다. 습득물 게시글과 분실물 게시글 간의 유사도를 계산하여 자동 매칭 서비스에 활용할 수 있습니다.

## 주요 기능

- 텍스트 임베딩 생성: 한국어 텍스트를 벡터로 변환
- 이미지 임베딩 생성: 이미지를 벡터로 변환
- 텍스트-이미지 교차 유사도 계산: 텍스트와 이미지 간의 유사도 측정
- 유사한 아이템 검색: 주어진 아이템과 유사한 아이템 목록 반환
- 임베딩 캐싱: 효율적인 처리를 위한 임베딩 결과 캐싱

## 기술 스택

- FastAPI: 고성능 API 프레임워크
- HuggingFace Transformers: CLIP 모델 활용
- PyTorch: 딥러닝 프레임워크
- Kiwi: 한국어 형태소 분석
- Docker: 컨테이너화 및 배포

## 모델 설명

이 서비스는 한국어에 특화된 'Bingsu/clip-vit-large-patch14-ko' CLIP 모델을 사용합니다. 이 모델은 한국어 텍스트와 이미지 간의 관계를 학습한 멀티모달 모델로, 텍스트와 이미지를 같은 벡터 공간에 임베딩하여 유사도 계산이 가능하게 합니다.

## API 엔드포인트

### 기본 엔드포인트

- `GET /`: API 서비스 정보
- `GET /health`: 서비스 상태 체크
- `GET /docs`: API 문서 (Swagger UI)

### 임베딩 및 유사도 관련 엔드포인트

- `POST /api/embedding/encode-text`: 텍스트 임베딩 생성
- `POST /api/embedding/upload-image`: 이미지 업로드
- `POST /api/embedding/find-similar`: 유사한 아이템 검색
- `POST /api/embedding/calculate-similarity`: 두 아이템 간의 유사도 계산
- `POST /api/embedding/batch-encode`: 여러 아이템을 배치로 인코딩

## 설치 및 실행 방법

### 도커 사용 (권장)

1. 도커 이미지 빌드:
```bash
docker build -t embedding-service .
```

2. 도커 컨테이너 실행:
```bash
docker run -d -p 5000:5000 --name embedding-service embedding-service
```

### 직접 설치

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 서버 실행:
```bash
cd app
python main.py
```

## 환경 변수 설정

다음 환경 변수로 서비스를 구성할 수 있습니다:

- `MODEL_NAME`: 사용할 CLIP 모델 이름 (기본값: 'Bingsu/clip-vit-large-patch14-ko')
- `CUDA_VISIBLE_DEVICES`: GPU 사용 설정
- `SIMILARITY_THRESHOLD`: 유사도 임계값 (기본값: 0.7)
- `TEXT_WEIGHT`: 텍스트 유사도 가중치 (기본값: 0.7)
- `IMAGE_WEIGHT`: 이미지 유사도 가중치 (기본값: 0.3)

## API 사용 예시

### 유사한 아이템 검색

```python
import requests
import json

url = "http://localhost:5000/api/embedding/find-similar"
payload = {
    "category": "지갑",
    "item_name": "검은색 가죽 지갑",
    "color": "검정색",
    "content": "강남역 근처에서 검정색 가죽 지갑을 잃어버렸습니다. 현금과 카드가 들어있어요.",
    "location": "강남역"
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)
result = response.json()
print(json.dumps(result, indent=2, ensure_ascii=False))
```

## 스프링 부트 연동

스프링 부트 애플리케이션에서 다음과 같이 이 서비스를 호출할 수 있습니다:

```java
@Service
public class AiMatchingService {
    private final RestTemplate restTemplate;
    
    @Value("${fastapi.base-url}")
    private String fastApiBaseUrl;
    
    // 유사도 검색 요청
    public SimilaritySearchResponse findSimilarItems(SimilaritySearchRequest request) {
        // FastAPI 요청 객체 생성
        FastApiSimilarityRequest fastApiRequest = FastApiSimilarityRequest.builder()
                .category(targetItem.getCategory())
                .item_name(targetItem.getItemName())
                .color(targetItem.getColor())
                .content(targetItem.getContent())
                .location(targetItem.getLocation())
                .image_url(targetItem.getImageUrl())
                .build();

        // API URL 구성
        String url = UriComponentsBuilder.fromHttpUrl(fastApiBaseUrl + "/api/embedding/find-similar")
                .queryParam("threshold", threshold)
                .queryParam("limit", limit)
                .toUriString();

        // FastAPI 호출
        ResponseEntity<SimilaritySearchResponse> response = restTemplate.postForEntity(
                url, fastApiRequest, SimilaritySearchResponse.class);
                
        return response.getBody();
    }
}
```

## 주의사항

- 실제 운영 환경에서는 인증 및 권한 설정을 추가하세요.
- 대용량 데이터 처리 시 메모리 사용량에 주의하세요.
- GPU 환경에서 실행하면 성능이 크게 향상됩니다.
