import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")
st.title("📊 두 개 그래프 비교 시각화 도구")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("데이터 미리보기", df.head())

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    all_cols = df.columns.tolist()

    st.markdown("### 🔧 좌측 그래프 설정")
    with st.sidebar:
        st.subheader("📍 좌측 그래프")
        chart_type_1 = st.selectbox("그래프 1 종류", ["scatter", "line", "bar", "box"])
        x1 = st.selectbox("X축 (그래프 1)", all_cols, key="x1")
        y1 = st.selectbox("Y축 (그래프 1)", all_cols, key="y1")
        color1 = st.selectbox("색상 기준 (선택)", [None] + all_cols, key="color1")

        st.markdown("---")
        st.subheader("📍 우측 그래프")
        chart_type_2 = st.selectbox("그래프 2 종류", ["scatter", "line", "bar", "box"])
        x2 = st.selectbox("X축 (그래프 2)", all_cols, key="x2")
        y2 = st.selectbox("Y축 (그래프 2)", all_cols, key="y2")
        color2 = st.selectbox("색상 기준 (선택)", [None] + all_cols, key="color2")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📈 그래프 1")
        if chart_type_1 == "scatter":
            fig1 = px.scatter(df, x=x1, y=y1, color=color1)
        elif chart_type_1 == "line":
            fig1 = px.line(df, x=x1, y=y1, color=color1)
        elif chart_type_1 == "bar":
            fig1 = px.bar(df, x=x1, y=y1, color=color1)
        elif chart_type_1 == "box":
            fig1 = px.box(df, x=x1, y=y1, color=color1)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown("#### 📈 그래프 2")
        if chart_type_2 == "scatter":
            fig2 = px.scatter(df, x=x2, y=y2, color=color2)
        elif chart_type_2 == "line":
            fig2 = px.line(df, x=x2, y=y2, color=color2)
        elif chart_type_2 == "bar":
            fig2 = px.bar(df, x=x2, y=y2, color=color2)
        elif chart_type_2 == "box":
            fig2 = px.box(df, x=x2, y=y2, color=color2)
        st.plotly_chart(fig2, use_container_width=True)
