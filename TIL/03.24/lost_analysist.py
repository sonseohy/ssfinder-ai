import os
import requests
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from dotenv import load_dotenv
import time
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
from tensorflow.keras.preprocessing import image

# Load environment variables
load_dotenv()

# 설정값
SIMILARITY_THRESHOLD = 0.7  # 유사도 임계값
BATCH_SIZE = 100  # 한 번에 처리할 습득물 데이터 수

# 단계 1: 경찰청 API에서 습득물 데이터 가져오기
def fetch_police_lost_items(service_key, num_items=5):
    """
    경찰청 API를 통해 최신 습득물 데이터를 가져옴
    
    Args:
        service_key (str): 경찰청 API 서비스 키
        num_items (int): 가져올 아이템 수
        
    Returns:
        list: 습득물 데이터 리스트
    """
    url = 'http://apis.data.go.kr/1320000/LosfundInfoInqireService/getLosfundInfoAccToClAreaPd'
    
    # 현재 날짜 기준으로 최신 데이터를 가져오기 위한 파라미터 설정
    params = {
        'serviceKey': service_key,
        # 'PRDT_CL_CD_01': 'PRH000',  # 분류 코드
        # 'PRDT_CL_CD_02': 'PRH200',  # 지갑 분류
        # 'FD_COL_CD': 'CL1002',      # 분실물 색상 코드
        'START_YMD': '20250101',    # 시작 날짜 (최근 3개월)
        'END_YMD': '20250322',      # 종료 날짜 (현재 날짜)
        # 'N_FD_LCT_CD': 'LCA000',    # 습득 장소 코드
        'pageNo': '1',              # 첫 페이지부터 조회
        'numOfRows': str(num_items),# 가져올 항목 수 (5개)
        'sort': 'DESC',             # 내림차순 정렬 (최신순)
        'sortField': 'fdYmd'        # 습득일자 기준으로 정렬
    }
    
    try:
        response = requests.get(url, params=params)
        
        # XML 응답 파싱
        if response.status_code == 200:
            root = ET.fromstring(response.content)
            items = []
            
            # items 태그 아래의 item 요소들 추출
            for item in root.findall('.//item'):
                item_data = {}
                
                # 각 필드 추출
                for child in item:
                    item_data[child.tag] = child.text
                
                items.append(item_data)
                
            print(f"API에서 {len(items)}개 최신 습득물 데이터를 성공적으로 가져왔습니다.")
            return items
        else:
            print(f"API 호출 실패: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"API 호출 중 오류 발생: {str(e)}")
        return []

# 이미지 URL에서 이미지 다운로드
def download_image(image_url):
    """
    URL에서 이미지를 다운로드하여 OpenCV 형식으로 반환
    
    Args:
        image_url (str): 이미지 URL
        
    Returns:
        numpy.ndarray: OpenCV 이미지 또는 다운로드 실패 시 None
    """
    try:
        response = requests.get(image_url, stream=True)
        if response.status_code == 200:
            # 바이트 스트림을 이미지로 변환
            image_bytes = BytesIO(response.content)
            
            # PIL 이미지로 변환
            pil_image = Image.open(image_bytes)
            
            # OpenCV 형식으로 변환
            cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            
            return cv_image
        else:
            print(f"이미지 다운로드 실패: {response.status_code}")
            return None
    except Exception as e:
        print(f"이미지 다운로드 중 오류 발생: {str(e)}")
        return None

# 모델 로드
def load_models():
    """
    YOLO 및 EfficientNet 모델 로드
    
    Returns:
        tuple: (yolo_model, efficientnet_model)
    """
    try:
        # YOLO 모델 로드
        yolo_model = YOLO('yolov8n.pt')  # 사전학습된 YOLOv8 nano 모델 사용
        
        # EfficientNet 모델 로드
        efficientnet_model = EfficientNetB4(
            include_top=True,
            weights='imagenet',
            input_shape=(380, 380, 3),
            classes=1000
        )
        
        print("YOLO와 EfficientNet 모델이 성공적으로 로드되었습니다.")
        return yolo_model, efficientnet_model
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {str(e)}")
        return None, None

# 이미지 분석 (YOLO + EfficientNet)
def analyze_image(image, yolo_model, efficientnet_model):
    """
    이미지를 YOLO로 객체 감지하고 EfficientNet으로 분류
    
    Args:
        image (numpy.ndarray): 분석할 OpenCV 이미지
        yolo_model: 로드된 YOLO 모델
        efficientnet_model: 로드된 EfficientNet 모델
        
    Returns:
        dict: 분석 결과
    """
    try:
        # YOLO로 객체 감지
        yolo_results = yolo_model(image)
        
        # 결과 정보 추출
        yolo_detections = []
        
        if len(yolo_results) > 0:
            # 첫 번째 이미지의 결과 (배치 처리 아님)
            result = yolo_results[0]
            
            # 객체마다 정보 추출
            for box in result.boxes:
                cls_id = int(box.cls.item())
                class_name = result.names[cls_id]
                confidence = float(box.conf.item())
                bbox = box.xyxy.cpu().numpy()[0]  # [x1, y1, x2, y2] 형식
                
                # 신뢰도 0.3 이상만 유효 결과로 간주
                if confidence >= 0.3:
                    yolo_detections.append({
                        "class": class_name,
                        "confidence": confidence,
                        "bbox": bbox.tolist()
                    })
        
        # 가장 신뢰도 높은 객체 선택
        main_object = None
        if yolo_detections:
            main_object = max(yolo_detections, key=lambda x: x["confidence"])
            
            # 객체 영역 추출
            bbox = main_object["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            object_image = image[y1:y2, x1:x2]
            
            # EfficientNet 분류 수행
            try:
                # 이미지 크기 조정 및 전처리
                resized_img = cv2.resize(object_image, (380, 380))
                rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
                img_array = np.expand_dims(rgb_img, axis=0)
                img_array = preprocess_input(img_array)
                
                # 예측 수행
                preds = efficientnet_model.predict(img_array)
                
                # 상위 3개 예측 결과 가져오기
                top_indices = np.argsort(preds[0])[-3:][::-1]
                
                # ImageNet 클래스 이름 불러오기 (미리 다운로드 필요)
                labels_path = 'imagenet_classes.txt'  # ImageNet 클래스 파일 경로
                try:
                    with open(labels_path, 'r') as f:
                        class_labels = [line.strip() for line in f.readlines()]
                except:
                    # 파일이 없는 경우 처리
                    print("ImageNet 클래스 파일을 찾을 수 없습니다. 임시 인덱스를 사용합니다.")
                    class_labels = [f"class_{i}" for i in range(1000)]
                
                # 상위 예측 결과 추출
                top_predictions = []
                for i in top_indices:
                    top_predictions.append({
                        "class": class_labels[i],
                        "score": float(preds[0][i])
                    })
                    
                # EfficientNet 결과 추가
                main_object["efficientnet_results"] = top_predictions
                
            except Exception as e:
                print(f"EfficientNet 분류 중 오류 발생: {str(e)}")
        
        # 분석 요약 결과 생성
        if main_object:
            # YOLO와 EfficientNet 결과를 종합하여 설명 생성
            yolo_class = main_object["class"]
            yolo_conf = main_object["confidence"]
            
            efficientnet_class = "불명"
            efficientnet_conf = 0
            if "efficientnet_results" in main_object and main_object["efficientnet_results"]:
                efficientnet_class = main_object["efficientnet_results"][0]["class"]
                efficientnet_conf = main_object["efficientnet_results"][0]["score"]
            
            # 기본 요약 설명
            summary = f"이 물건은 {yolo_conf:.1%} 확률로 '{yolo_class}'로 감지되었으며, "
            summary += f"{efficientnet_conf:.1%} 확률로 '{efficientnet_class}'로 분류되었습니다."
            
            # 추가 감지 객체가 있는 경우
            if len(yolo_detections) > 1:
                other_objects = [d["class"] for d in yolo_detections if d != main_object]
                summary += f" 추가로 이미지에서 {', '.join(other_objects)} 객체가 감지되었습니다."
            
            return {
                "main_object": main_object,
                "all_detections": yolo_detections,
                "summary": summary
            }
        else:
            return {
                "main_object": None,
                "all_detections": [],
                "summary": "이미지에서 객체를 감지하지 못했습니다."
            }
        
    except Exception as e:
        print(f"이미지 분석 중 오류 발생: {str(e)}")
        return {
            "main_object": None,
            "all_detections": [],
            "summary": f"분석 중 오류 발생: {str(e)}"
        }

# 이미지에 검출 결과 시각화
def visualize_detections(image, detections):
    """
    이미지에 검출 결과를 시각화
    
    Args:
        image (numpy.ndarray): 원본 OpenCV 이미지
        detections (list): 검출 결과 리스트
        
    Returns:
        numpy.ndarray: 시각화된 이미지
    """
    # 이미지 복사본 생성
    vis_image = image.copy()
    
    # 각 검출 객체에 대해 바운딩 박스와 라벨 표시
    for det in detections:
        # 바운딩 박스 좌표
        bbox = det["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        
        # 클래스 및 신뢰도
        class_name = det["class"]
        confidence = det["confidence"]
        
        # 바운딩 박스 그리기
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 라벨 텍스트
        label = f"{class_name}: {confidence:.2f}"
        
        # 텍스트 배경 크기 계산
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # 텍스트 배경 그리기
        cv2.rectangle(
            vis_image, 
            (x1, y1 - text_height - 5), 
            (x1 + text_width, y1), 
            (0, 255, 0), 
            -1
        )
        
        # 텍스트 그리기
        cv2.putText(
            vis_image, 
            label, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 0, 0), 
            1
        )
    
    return vis_image

# 주 실행 블록
if __name__ == "__main__":
    # 환경 변수에서 서비스 키 가져오기
    SERVICE_KEY = os.getenv('POLICE_API_SERVICE_KEY')
    
    # 서비스 키가 없으면 오류 메시지 출력
    if not SERVICE_KEY:
        print("Error: POLICE_API_SERVICE_KEY 환경 변수가 설정되지 않았습니다.")
        print("다음 방법 중 하나로 서비스 키를 설정하세요:")
        print("1. .env 파일에 POLICE_API_SERVICE_KEY=your_key_here 추가")
        print("2. 시스템 환경 변수로 설정")
        print("3. 스크립트 실행 시 export POLICE_API_SERVICE_KEY=your_key_here")
        exit(1)
    
    # 모델 로드
    yolo_model, efficientnet_model = load_models()
    if yolo_model is None or efficientnet_model is None:
        print("모델 로드에 실패했습니다. 프로그램을 종료합니다.")
        exit(1)
    
    # 최신 습득물 데이터 가져오기
    num_items = 20  # 데이터 개수 (이미지가 있는 항목을 찾기 위해 여러 개 가져옴)
    items = fetch_police_lost_items(SERVICE_KEY, num_items)
    
    # 각 아이템에 대해 이미지 가져오기 및 분석
    for i, item in enumerate(items):
        print(f"\n[아이템 {i+1}/{len(items)}]")
        print(f"  ID: {item.get('atcId', 'N/A')}")
        print(f"  습득물명: {item.get('fdPrdtNm', 'N/A')}")
        print(f"  습득일자: {item.get('fdYmd', 'N/A')}")
        print(f"  습득장소: {item.get('fdPlace', 'N/A')}")
        
        # 이미지 URL 확인
        image_url = item.get('fdFilePathImg')
        if not image_url or image_url == "":
            print("  이미지 없음: 이 항목에는 이미지가 없습니다.")
            continue
        
        print(f"  이미지 URL: {image_url}")
        
        # 이미지 다운로드
        print("  이미지 다운로드 중...")
        image = download_image(image_url)
        if image is None:
            print("  이미지 다운로드 실패: 다음 항목으로 넘어갑니다.")
            continue
        
        # 이미지 크기 확인
        height, width, channels = image.shape
        print(f"  이미지 크기: {width}x{height}, 채널: {channels}")
        
        # 이미지 분석
        print("  AI 분석 중...")
        start_time = time.time()
        analysis_result = analyze_image(image, yolo_model, efficientnet_model)
        end_time = time.time()
        
        # 분석 결과 출력
        print(f"  분석 완료 (소요시간: {end_time - start_time:.2f}초)")
        print("  분석 요약:")
        print(f"    {analysis_result['summary']}")
        
        # 주요 객체 상세 정보
        if analysis_result["main_object"]:
            main_obj = analysis_result["main_object"]
            print("  주요 객체 정보:")
            print(f"    YOLO 클래스: {main_obj['class']} ({main_obj['confidence']:.2f})")
            
            if "efficientnet_results" in main_obj:
                print("    EfficientNet 상위 분류 결과:")
                for idx, pred in enumerate(main_obj["efficientnet_results"]):
                    print(f"      {idx+1}. {pred['class']} ({pred['score']:.2f})")
        
        # 검출 결과 시각화
        if analysis_result["all_detections"]:
            vis_image = visualize_detections(image, analysis_result["all_detections"])
            
            # 시각화 이미지 표시 (GUI 환경에서 실행 시)
            try:
                # 이미지 크기 조정 (너무 큰 경우)
                max_width = 800
                if vis_image.shape[1] > max_width:
                    scale = max_width / vis_image.shape[1]
                    vis_image = cv2.resize(vis_image, None, fx=scale, fy=scale)
                
                # OpenCV를 사용하여 이미지 표시
                window_name = f"아이템 {i+1} - {item.get('fdPrdtNm', 'N/A')}"
                cv2.imshow(window_name, vis_image)
                cv2.waitKey(0)  # 키 입력 대기
                cv2.destroyWindow(window_name)
            except Exception as e:
                print(f"  시각화 표시 오류: {str(e)} (GUI 환경이 아닐 수 있음)")
        
        print("-" * 80)
    
    # 모든 윈도우 닫기
    cv2.destroyAllWindows()
    print("\n모든 항목 분석 완료!")