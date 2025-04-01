from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime

class MatchingWeight(BaseModel):
    """매칭 가중치 모델"""
    object_class_match: float = Field(0.3, ge=0, le=1, description="물품 종류 일치 가중치")
    clip_similarity: float = Field(0.25, ge=0, le=1, description="CLIP 시각적 유사도 가중치")
    attribute_match: float = Field(0.35, ge=0, le=1, description="속성 일치 가중치")
    metadata_match: float = Field(0.1, ge=0, le=1, description="메타데이터 일치 가중치")

class CategoryEnum(str, Enum):
    """분류 카테고리 열거형"""
    PHONE = "휴대폰"
    WALLET = "지갑"
    BAG = "가방"
    CARD = "카드"
    ACCESSORY = "귀금속"
    BOOK = "도서"
    DOCUMENT = "서류"
    TOOL = "산업용품"
    SPORTS = "스포츠용품"
    MUSIC = "악기"
    CLOTHING = "의류"
    CAR = "자동차"
    ELECTRONICS = "전자기기"
    COMPUTER = "컴퓨터"
    MONEY = "현금"
    OTHER = "기타물품"

class ColorEnum(str, Enum):
    """색상 열거형"""
    BLACK = "검정색"
    WHITE = "흰색"
    RED = "빨간색"
    ORANGE = "주황색"
    YELLOW = "노란색"
    GREEN = "녹색"
    BLUE = "파란색"
    NAVY = "남색"
    PURPLE = "보라색"
    PINK = "분홍색"
    BROWN = "갈색"
    GRAY = "회색"
    SILVER = "은색"
    GOLD = "금색"
    TRANSPARENT = "투명"
    OTHER = "기타"

class MatchingRequest(BaseModel):
    """매칭 요청 모델"""
    description: Optional[str] = Field(None, description="분실물 설명")
    category: Optional[CategoryEnum] = Field(None, description="물품 카테고리")
    color: Optional[ColorEnum] = Field(None, description="물품 색상")
    max_results: Optional[int] = Field(10, ge=1, le=50, description="반환할 최대 결과 수")
    weights: Optional[MatchingWeight] = Field(None, description="매칭 알고리즘 가중치")

class AttributeInfo(BaseModel):
    """속성 정보 모델"""
    color: Optional[str] = Field(None, description="색상")
    brand: Optional[str] = Field(None, description="브랜드")
    type: Optional[str] = Field(None, description="종류")
    material: Optional[str] = Field(None, description="재질")
    condition: Optional[str] = Field(None, description="상태")

class MatchingScore(BaseModel):
    """매칭 점수 모델"""
    total_score: float = Field(..., description="총 매칭 점수")
    object_class_match: float = Field(..., description="물품 종류 일치 점수")
    attribute_match: float = Field(..., description="속성 일치 점수")
    clip_similarity: float = Field(..., description="CLIP 시각적 유사도 점수")
    metadata_match: float = Field(..., description="메타데이터 일치 점수")
    explanation: str = Field(..., description="매칭 설명")

class MatchedItem(BaseModel):
    """매칭된 아이템 모델"""
    id: str = Field(..., description="습득물 ID")
    name: str = Field(..., description="습득물 이름")
    image_url: Optional[str] = Field(None, description="습득물 이미지 URL")
    category: Optional[str] = Field(None, description="습득물 카테고리")
    attributes: Optional[AttributeInfo] = Field(None, description="습득물 속성")
    found_date: Optional[datetime] = Field(None, description="습득 일자")
    found_place: Optional[str] = Field(None, description="습득 장소")
    description: Optional[str] = Field(None, description="습득물 설명")
    matching: MatchingScore = Field(..., description="매칭 점수 정보")

class MatchingResponse(BaseModel):
    """매칭 응답 모델"""
    total_matches: int = Field(..., description="총 매칭 결과 수")
    query_info: Dict[str, Any] = Field(..., description="검색 쿼리 정보")
    results: List[MatchedItem] = Field(..., description="매칭 결과 목록")