import os
import gradio as gr
from models.models import LostItemAnalyzer

# 분석기 초기화
analyzer = LostItemAnalyzer()

def analyze_image(image):
    try:
        # 임시 이미지 파일 저장
        temp_path = "temp_uploaded_image.jpg"
        image.save(temp_path)
        
        # 분석 수행
        result = analyzer.analyze_lost_item(temp_path)
        
        if result['success']:
            # 번역된 결과 반환
            translated = result['data']['translated']
            return (
                f"제목: {translated['title']}\n"
                f"설명: {translated['description']}\n"
                f"카테고리: {translated['category']}\n"
                f"색상: {translated['color']}\n"
                f"재질: {translated['material']}\n"
                f"브랜드: {translated['brand']}"
            )
        else:
            return f"분석 중 오류 발생: {result['error']}"
    
    except Exception as e:
        return f"오류 발생: {str(e)}"

# Gradio 인터페이스 생성
iface = gr.Interface(
    fn=analyze_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Textbox(label="분석 결과"),
    title="분실물 이미지 분석 AI",
    description="이미지를 업로드하면 분실물의 특징을 분석해드립니다."
)

# 앱 실행
if __name__ == "__main__":
    iface.launch()