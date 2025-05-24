import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# utils.py 모듈 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import add_dark_mode_toggle, add_chart_export_section, style_metric_cards

st.title("🔄 그래프 겹쳐보기")

# 다크모드 토글 및 스타일 추가
add_dark_mode_toggle()
style_metric_cards()
st.markdown("두 개의 그래프를 겹쳐서 비교 분석할 수 있습니다.")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("데이터 미리보기", df.head())

    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    st.markdown("### 겹쳐볼 그래프 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**첫 번째 그래프**")
        chart_type_1 = st.selectbox("그래프 1 종류", ["scatter", "line", "bar"], key="overlay_g1")
        x1 = st.selectbox("X축 1", all_cols, key="overlay_x1")
        y1 = st.selectbox("Y축 1", numeric_cols, key="overlay_y1")
        color1 = st.selectbox("색상 1", [None] + all_cols, key="overlay_c1")
        opacity1 = st.slider("투명도 1", 0.1, 1.0, 0.7, key="opacity1")
        
    with col2:
        st.markdown("**두 번째 그래프**")
        chart_type_2 = st.selectbox("그래프 2 종류", ["scatter", "line", "bar"], key="overlay_g2")
        x2 = st.selectbox("X축 2", all_cols, key="overlay_x2")
        y2 = st.selectbox("Y축 2", numeric_cols, key="overlay_y2")
        color2 = st.selectbox("색상 2", [None] + all_cols, key="overlay_c2")
        opacity2 = st.slider("투명도 2", 0.1, 1.0, 0.7, key="opacity2")

    # NOTE: 두 축의 스케일을 독립적으로 설정할 수 있는 옵션 추가
    use_secondary_y = st.checkbox("두 번째 그래프에 별도 Y축 사용", value=False)
    
    if st.button("그래프 겹쳐보기"):
        # 첫 번째 그래프 생성
        if chart_type_1 == "scatter":
            fig1 = px.scatter(df, x=x1, y=y1, color=color1, title="겹쳐진 그래프")
        elif chart_type_1 == "line":
            fig1 = px.line(df, x=x1, y=y1, color=color1, title="겹쳐진 그래프")
        elif chart_type_1 == "bar":
            fig1 = px.bar(df, x=x1, y=y1, color=color1, title="겹쳐진 그래프")
        
        # 두 번째 그래프 생성
        if chart_type_2 == "scatter":
            fig2 = px.scatter(df, x=x2, y=y2, color=color2)
        elif chart_type_2 == "line":
            fig2 = px.line(df, x=x2, y=y2, color=color2)
        elif chart_type_2 == "bar":
            fig2 = px.bar(df, x=x2, y=y2, color=color2)
        
        # 첫 번째 그래프의 투명도 설정
        for trace in fig1.data:
            trace.opacity = opacity1
            if hasattr(trace, 'marker'):
                trace.marker.opacity = opacity1
        
        # 두 번째 그래프의 투명도 및 색상 설정
        for trace in fig2.data:
            trace.opacity = opacity2
            if hasattr(trace, 'marker'):
                trace.marker.opacity = opacity2
                # NOTE: 색상 구분을 위해 다른 색상 팔레트 사용
                trace.marker.color = 'red' if not color2 else trace.marker.color
            # 선 그래프의 경우 다른 색상 적용
            if hasattr(trace, 'line'):
                trace.line.color = 'red' if not color2 else trace.line.color
        
        # 두 그래프 합치기
        if use_secondary_y:
            # 서브플롯으로 이중 Y축 구현
            from plotly.subplots import make_subplots
            combined_fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 첫 번째 그래프 추가 (primary y-axis)
            for trace in fig1.data:
                combined_fig.add_trace(trace, secondary_y=False)
            
            # 두 번째 그래프 추가 (secondary y-axis)
            for trace in fig2.data:
                combined_fig.add_trace(trace, secondary_y=True)
            
            # 축 라벨 설정
            combined_fig.update_xaxes(title_text=f"{x1} / {x2}")
            combined_fig.update_yaxes(title_text=y1, secondary_y=False)
            combined_fig.update_yaxes(title_text=y2, secondary_y=True)
            combined_fig.update_layout(title="겹쳐진 그래프 (이중 Y축)")
            
        else:
            # 단일 Y축으로 그래프 합치기
            combined_fig = fig1
            for trace in fig2.data:
                combined_fig.add_trace(trace)
            
            # 범례 개선
            combined_fig.update_layout(
                title="겹쳐진 그래프",
                xaxis_title=f"{x1} / {x2}",
                yaxis_title=f"{y1} / {y2}",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
        
        st.plotly_chart(combined_fig, use_container_width=True)
        
        # 차트 내보내기 기능 추가
        add_chart_export_section(combined_fig, "overlay_graph")
        
        # 분석 인사이트 제공
        st.markdown("### 📊 분석 인사이트")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("첫 번째 데이터 포인트 수", len(df))
            if y1 in numeric_cols:
                st.metric(f"{y1} 평균", f"{df[y1].mean():.2f}")
        
        with col2:
            if y2 in numeric_cols:
                st.metric(f"{y2} 평균", f"{df[y2].mean():.2f}")
                correlation = df[y1].corr(df[y2]) if y1 in numeric_cols and y2 in numeric_cols else None
                if correlation is not None:
                    st.metric("상관계수", f"{correlation:.3f}")
        
        with col3:
            if y1 in numeric_cols and y2 in numeric_cols:
                diff = abs(df[y1].mean() - df[y2].mean())
                st.metric("평균 차이", f"{diff:.2f}")
                
        # TODO: 추가 통계 분석 기능 구현
        # 예: 회귀분석, 클러스터링 등 