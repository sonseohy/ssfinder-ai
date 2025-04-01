from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from typing import Optional, List

from app.schemas.matching import (
    MatchingRequest, MatchingResponse, MatchedItem, 
    MatchingWeight, CategoryEnum, ColorEnum
)
from app.services.matching_service import (
    process_image, process_text_query, fetch_database_items, 
    extract_features_from_items, match_items, format_response
)

# 라우터 설정
router = APIRouter(
    prefix="/api/match",
    tags=["matching"],
    responses={404: {"description": "Not found"}},
)

@router.post("/image", response_model=MatchingResponse)
async def match_with_image(
    image: UploadFile = File(...),
    description: Optional[str] = Form(None),
    category: Optional[CategoryEnum] = Form(None),
    color: Optional[ColorEnum] = Form(None),
    max_results: int = Form(10)
):
    """이미지와 설명으로 분실물 매칭 수행"""
    try:
        # 쿼리 정보 로깅
        print(f"이미지 매칭 요청: description={description}, category={category}, color={color}")
        
        # 이미지 처리
        query_item = process_image(image.file)
        
        # 설명이 제공된 경우 처리
        if description:
            text_features = process_text_query(description)
            query_item.update(text_features)
        
        # 쿼리 정보 수집
        query_info = {
            'description': description,
            'category': category.value if category else None,
            'color': color.value if color else None,
            'has_image': True,
            'query_type': 'image'
        }
        
        # 데이터베이스 아이템 가져오기
        database_items = fetch_database_items(
            category=category.value if category else None,
            color=color.value if color else None,
            keywords=description,
            num_items=max_results * 3  # 필터링을 고려해 더 많이 가져옴
        )
        
        # 특성 추출
        processed_items = extract_features_from_items(database_items)
        
        # 아이템 매칭
        matched_items = match_items(query_item, processed_items)
        
        # 응답 포맷팅
        response = format_response(matched_items[:max_results], query_info)
        
        return response
        
    except Exception as e:
        print(f"이미지 매칭 처리 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/text", response_model=MatchingResponse)
async def match_with_text(request: MatchingRequest):
    """텍스트 설명만으로 분실물 매칭 수행"""
    try:
        # 요청 유효성 검증
        if not request.description:
            raise HTTPException(status_code=400, detail="설명 텍스트가 필요합니다")
        
        # 쿼리 정보 로깅
        print(f"텍스트 매칭 요청: description={request.description}, category={request.category}, color={request.color}")
        
        # 텍스트 처리
        query_item = process_text_query(request.description)
        
        # 쿼리 정보 수집
        query_info = {
            'description': request.description,
            'category': request.category.value if request.category else None,
            'color': request.color.value if request.color else None,
            'has_image': False,
            'query_type': 'text'
        }
        
        # 매칭 가중치 설정
        weights = None
        if request.weights:
            weights = {
                'object_class_match': request.weights.object_class_match,
                'clip_similarity': request.weights.clip_similarity,
                'attribute_match': request.weights.attribute_match,
                'metadata_match': request.weights.metadata_match
            }
        
        # 데이터베이스 아이템 가져오기
        database_items = fetch_database_items(
            category=request.category.value if request.category else None,
            color=request.color.value if request.color else None,
            keywords=request.description,
            num_items=request.max_results * 3  # 필터링을 고려해 더 많이 가져옴
        )
        
        # 특성 추출
        processed_items = extract_features_from_items(database_items)
        
        # 아이템 매칭
        matched_items = match_items(query_item, processed_items, weights)
        
        # 응답 포맷팅
        response = format_response(matched_items[:request.max_results], query_info)
        
        return response
        
    except Exception as e:
        print(f"텍스트 매칭 처리 중 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/categories")
async def get_categories():
    """카테고리 목록 반환"""
    return {category.name: category.value for category in CategoryEnum}

@router.get("/colors")
async def get_colors():
    """색상 목록 반환"""
    return {color.name: color.value for color in ColorEnum}