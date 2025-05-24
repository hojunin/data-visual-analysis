import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# utils.py 모듈 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import apply_custom_theme, add_chart_export_section, style_metric_cards

st.title("📈 기본 분석 기능")

# 커스텀 테마 및 스타일 추가
apply_custom_theme()
style_metric_cards()

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("데이터 미리보기", df.head())

    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    chart_type = st.selectbox("그래프 종류", ["scatter", "line", "bar", "box"])
    x_col = st.selectbox("X축", all_cols)
    y_col = st.selectbox("Y축", all_cols)
    color_col = st.selectbox("색상 기준", [None] + all_cols)

    if chart_type == "scatter":
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
    elif chart_type == "line":
        fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
    elif chart_type == "bar":
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
    elif chart_type == "box":
        fig = px.box(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")

    st.plotly_chart(fig, use_container_width=True)
    
    # 차트 내보내기 기능 추가
    add_chart_export_section(fig, f"basic_analysis_{chart_type}")
