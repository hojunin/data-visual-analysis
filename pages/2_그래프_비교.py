import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# utils.py 모듈 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import add_dark_mode_toggle, add_chart_export_section, style_metric_cards

st.title("📊 두 개 그래프 비교 시각화")

# 다크모드 토글 및 스타일 추가
add_dark_mode_toggle()
style_metric_cards()

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("데이터 미리보기", df.head())

    all_cols = df.columns.tolist()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 좌측 그래프")
        chart_type_1 = st.selectbox("그래프 1 종류", ["scatter", "line", "bar", "box"], key="g1")
        x1 = st.selectbox("X축 1", all_cols, key="x1")
        y1 = st.selectbox("Y축 1", all_cols, key="y1")
        color1 = st.selectbox("색상 1", [None] + all_cols, key="c1")

    with col2:
        st.markdown("### 우측 그래프")
        chart_type_2 = st.selectbox("그래프 2 종류", ["scatter", "line", "bar", "box"], key="g2")
        x2 = st.selectbox("X축 2", all_cols, key="x2")
        y2 = st.selectbox("Y축 2", all_cols, key="y2")
        color2 = st.selectbox("색상 2", [None] + all_cols, key="c2")

    col1_, col2_ = st.columns(2)
    
    with col1_:
        st.markdown("#### 📈 그래프 1")
        if chart_type_1 == "scatter":
            fig1 = px.scatter(df, x=x1, y=y1, color=color1, title=f"{y1} vs {x1}")
        elif chart_type_1 == "line":
            fig1 = px.line(df, x=x1, y=y1, color=color1, title=f"{y1} vs {x1}")
        elif chart_type_1 == "bar":
            fig1 = px.bar(df, x=x1, y=y1, color=color1, title=f"{y1} vs {x1}")
        elif chart_type_1 == "box":
            fig1 = px.box(df, x=x1, y=y1, color=color1, title=f"{y1} vs {x1}")
        st.plotly_chart(fig1, use_container_width=True)
        
        # 차트 1 내보내기 기능
        add_chart_export_section(fig1, f"compare_chart1_{chart_type_1}")

    with col2_:
        st.markdown("#### 📈 그래프 2")
        if chart_type_2 == "scatter":
            fig2 = px.scatter(df, x=x2, y=y2, color=color2, title=f"{y2} vs {x2}")
        elif chart_type_2 == "line":
            fig2 = px.line(df, x=x2, y=y2, color=color2, title=f"{y2} vs {x2}")
        elif chart_type_2 == "bar":
            fig2 = px.bar(df, x=x2, y=y2, color=color2, title=f"{y2} vs {x2}")
        elif chart_type_2 == "box":
            fig2 = px.box(df, x=x2, y=y2, color=color2, title=f"{y2} vs {x2}")
        st.plotly_chart(fig2, use_container_width=True)
        
        # 차트 2 내보내기 기능
        add_chart_export_section(fig2, f"compare_chart2_{chart_type_2}")

else:
    st.info("CSV 파일을 업로드하여 그래프 비교를 시작하세요!")
    st.markdown("""
    ### 기능 소개
    - 두 개의 서로 다른 그래프를 나란히 비교
    - 다양한 차트 유형 지원 (산점도, 선 그래프, 막대 그래프, 박스 플롯)
    - 각 그래프의 변수와 색상을 독립적으로 설정
    - 고해상도 이미지 다운로드 지원
    """)
