import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio
import io
import base64
from datetime import datetime

def apply_custom_theme():
    """깔끔한 하얀색/회색 테마를 적용합니다."""
    st.markdown("""
    <style>
    /* 메인 앱 배경 - 깔끔한 하얀색 */
    .stApp {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
    }
    
    /* 사이드바 - 연한 회색 */
    .stSidebar {
        background-color: #f8f9fa !important;
    }
    
    /* 메트릭 카드 - 연한 회색 배경 */
    div[data-testid="metric-container"] {
        background-color: #f1f3f4 !important;
        color: #2c3e50 !important;
        border: 1px solid #e1e5e9 !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* 메트릭 카드 호버 효과 */
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
        transform: translateY(-1px) !important;
        transition: all 0.2s ease !important;
    }
    
    /* 버튼 스타일 개선 */
    .stButton > button {
        background-color: #ffffff !important;
        color: #2c3e50 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 0.375rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    .stButton > button:hover {
        background-color: #f8f9fa !important;
        border-color: #9ca3af !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* 다운로드 버튼 */
    .stDownloadButton > button {
        background-color: #3b82f6 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0.375rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    .stDownloadButton > button:hover {
        background-color: #2563eb !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* 입력 필드들 */
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 0.375rem !important;
    }
    
    .stTextInput > div > div > input {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 0.375rem !important;
        color: #2c3e50 !important;
    }
    
    .stMultiSelect > div > div {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
        border-radius: 0.375rem !important;
    }
    
    /* 파일 업로더 */
    .stFileUploader > div {
        background-color: #f8f9fa !important;
        border: 2px dashed #d1d5db !important;
        border-radius: 0.5rem !important;
        padding: 2rem !important;
    }
    
    /* 데이터프레임 */
    .stDataFrame {
        background-color: #ffffff !important;
        border: 1px solid #e1e5e9 !important;
        border-radius: 0.375rem !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* 확장자 (Expander) */
    .stExpander {
        background-color: #f8f9fa !important;
        border: 1px solid #e1e5e9 !important;
        border-radius: 0.375rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }
    
    /* 성공 메시지 */
    .stSuccess {
        background-color: #f0f9ff !important;
        color: #065f46 !important;
        border: 1px solid #a7f3d0 !important;
        border-radius: 0.375rem !important;
    }
    
    /* 경고 메시지 */
    .stWarning {
        background-color: #fffbeb !important;
        color: #92400e !important;
        border: 1px solid #fde68a !important;
        border-radius: 0.375rem !important;
    }
    
    /* 에러 메시지 */
    .stError {
        background-color: #fef2f2 !important;
        color: #991b1b !important;
        border: 1px solid #fecaca !important;
        border-radius: 0.375rem !important;
    }
    
    /* 정보 메시지 */
    .stInfo {
        background-color: #eff6ff !important;
        color: #1e40af !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 0.375rem !important;
    }
    
    /* 제목들 */
    h1, h2, h3, h4, h5, h6 {
        color: #1f2937 !important;
        font-weight: 600 !important;
    }
    
    /* 구분선 */
    hr {
        border-color: #e5e7eb !important;
        margin: 2rem 0 !important;
    }
    
    /* 슬라이더 */
    .stSlider > div > div > div > div {
        background-color: #3b82f6 !important;
    }
    
    /* 체크박스 */
    .stCheckbox > label > div[data-testid="stCheckbox"] > div {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
    }
    
    /* 라디오 버튼 */
    .stRadio > label > div[data-testid="stRadio"] > div {
        background-color: #ffffff !important;
        border: 1px solid #d1d5db !important;
    }
    </style>
    """, unsafe_allow_html=True)

def add_download_button(fig, filename_prefix="chart"):
    """
    Plotly 차트를 이미지로 다운로드할 수 있는 버튼을 추가합니다.
    
    Parameters:
    fig: Plotly figure object
    filename_prefix: 파일명 접두사
    """
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("📸 PNG로 저장", key=f"png_{filename_prefix}_{id(fig)}"):
            try:
                # PNG로 저장
                img_bytes = pio.to_image(fig, format="png", width=1200, height=800, scale=2)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{filename_prefix}_{timestamp}.png"
                
                st.download_button(
                    label="📥 PNG 다운로드",
                    data=img_bytes,
                    file_name=filename,
                    mime="image/png",
                    key=f"download_png_{filename_prefix}_{id(fig)}"
                )
                st.success("PNG 이미지가 준비되었습니다!")
                
            except Exception as e:
                st.error(f"이미지 생성 중 오류가 발생했습니다: {str(e)}")
    
    with col2:
        if st.button("🖼️ SVG로 저장", key=f"svg_{filename_prefix}_{id(fig)}"):
            try:
                # SVG로 저장
                img_svg = pio.to_image(fig, format="svg", width=1200, height=800)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{filename_prefix}_{timestamp}.svg"
                
                st.download_button(
                    label="📥 SVG 다운로드",
                    data=img_svg,
                    file_name=filename,
                    mime="image/svg+xml",
                    key=f"download_svg_{filename_prefix}_{id(fig)}"
                )
                st.success("SVG 이미지가 준비되었습니다!")
                
            except Exception as e:
                st.error(f"이미지 생성 중 오류가 발생했습니다: {str(e)}")
    
    with col3:
        if st.button("📄 HTML로 저장", key=f"html_{filename_prefix}_{id(fig)}"):
            try:
                # HTML로 저장
                html_str = pio.to_html(fig, include_plotlyjs='cdn')
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{filename_prefix}_{timestamp}.html"
                
                st.download_button(
                    label="📥 HTML 다운로드",
                    data=html_str.encode('utf-8'),
                    file_name=filename,
                    mime="text/html",
                    key=f"download_html_{filename_prefix}_{id(fig)}"
                )
                st.success("인터랙티브 HTML이 준비되었습니다!")
                
            except Exception as e:
                st.error(f"HTML 생성 중 오류가 발생했습니다: {str(e)}")

def add_chart_export_section(fig, chart_name="chart"):
    """
    차트 내보내기 섹션을 추가합니다.
    
    Parameters:
    fig: Plotly figure object
    chart_name: 차트 이름
    """
    st.markdown("---")
    st.markdown("### 📥 차트 내보내기")
    st.markdown("차트를 다양한 형식으로 저장할 수 있습니다.")
    
    add_download_button(fig, chart_name)
    
    # 이미지 설정 옵션
    with st.expander("⚙️ 이미지 설정"):
        col1, col2 = st.columns(2)
        with col1:
            width = st.slider("이미지 너비", 600, 2000, 1200, 100, key=f"width_{chart_name}_{id(fig)}")
        with col2:
            height = st.slider("이미지 높이", 400, 1500, 800, 100, key=f"height_{chart_name}_{id(fig)}")
        
        dpi = st.slider("해상도 (DPI)", 72, 300, 150, 10, key=f"dpi_{chart_name}_{id(fig)}")
        
        if st.button(f"🎨 고해상도 PNG 생성", key=f"custom_png_{chart_name}_{id(fig)}"):
            try:
                img_bytes = pio.to_image(
                    fig, 
                    format="png", 
                    width=width, 
                    height=height, 
                    scale=dpi/72
                )
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{chart_name}_custom_{timestamp}.png"
                
                st.download_button(
                    label="📥 고해상도 PNG 다운로드",
                    data=img_bytes,
                    file_name=filename,
                    mime="image/png",
                    key=f"download_custom_png_{chart_name}_{id(fig)}"
                )
                st.success(f"고해상도 이미지가 준비되었습니다! ({width}x{height}, {dpi}DPI)")
                
            except Exception as e:
                st.error(f"이미지 생성 중 오류가 발생했습니다: {str(e)}")

def style_metric_cards():
    """메트릭 카드의 스타일을 개선합니다."""
    st.markdown("""
    <style>
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #d1d5db;
        padding: 0.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    div[data-testid="metric-container"]:hover {
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }
    </style>
    """, unsafe_allow_html=True) 