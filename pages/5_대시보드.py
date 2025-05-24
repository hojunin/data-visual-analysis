import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import sys
import os

# utils.py 모듈 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import apply_custom_theme, add_chart_export_section, style_metric_cards

st.title("📊 통합 대시보드")

# 다크모드 토글 및 스타일 추가
apply_custom_theme()
style_metric_cards()
st.markdown("데이터의 전반적인 인사이트를 한 눈에 확인할 수 있는 대시보드입니다.")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # 사이드바에서 대시보드 설정
    st.sidebar.header("대시보드 설정")
    
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # 대시보드 레이아웃 선택
    layout_option = st.sidebar.selectbox(
        "레이아웃 선택",
        ["자동 대시보드", "커스텀 대시보드"]
    )
    
    if layout_option == "자동 대시보드":
        st.markdown("### 🤖 자동 생성 대시보드")
        st.markdown("데이터 특성에 따라 자동으로 최적의 차트들을 생성합니다.")
        
        # 데이터 개요
        st.markdown("#### 📋 데이터 개요")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("총 데이터 수", len(df))
        with col2:
            st.metric("수치형 변수", len(numeric_cols))
        with col3:
            st.metric("범주형 변수", len(categorical_cols))
        with col4:
            missing_count = df.isnull().sum().sum()
            st.metric("결측치 수", missing_count)
        
        # 결측치 시각화 (있는 경우)
        if missing_count > 0:
            st.markdown("#### 🚨 결측치 분포")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            fig_missing = px.bar(
                x=missing_data.values, 
                y=missing_data.index,
                orientation='h',
                title="변수별 결측치 개수"
            )
            fig_missing.update_xaxes(title="결측치 개수")
            fig_missing.update_yaxes(title="변수명")
            st.plotly_chart(fig_missing, use_container_width=True)
        
        # 수치형 변수 분석
        if len(numeric_cols) >= 2:
            st.markdown("#### 📈 수치형 변수 분석")
            
            # 상관관계 히트맵
            col1, col2 = st.columns(2)
            
            with col1:
                corr_matrix = df[numeric_cols].corr()
                fig_corr = px.imshow(
                    corr_matrix,
                    title="상관관계 매트릭스",
                    color_continuous_scale='RdBu_r',
                    aspect="auto"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with col2:
                # 수치형 변수들의 분포 (박스플롯)
                # 데이터 정규화 후 박스플롯
                df_normalized = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())
                
                fig_box = go.Figure()
                for col in numeric_cols:
                    fig_box.add_trace(go.Box(
                        y=df_normalized[col],
                        name=col,
                        boxpoints='outliers'
                    ))
                
                fig_box.update_layout(
                    title="정규화된 수치형 변수 분포",
                    yaxis_title="정규화된 값"
                )
                st.plotly_chart(fig_box, use_container_width=True)
            
            # 주요 관계 시각화
            st.markdown("#### 🔍 주요 변수 간 관계")
            
            # 가장 강한 상관관계 찾기
            corr_matrix_abs = corr_matrix.abs()
            np.fill_diagonal(corr_matrix_abs.values, 0)  # 대각선 제거
            
            # 상위 3개 상관관계 찾기
            top_correlations = []
            for i in range(len(corr_matrix_abs.columns)):
                for j in range(i+1, len(corr_matrix_abs.columns)):
                    top_correlations.append({
                        'var1': corr_matrix_abs.columns[i],
                        'var2': corr_matrix_abs.columns[j],
                        'corr': corr_matrix_abs.iloc[i, j]
                    })
            
            top_correlations = sorted(top_correlations, key=lambda x: x['corr'], reverse=True)[:3]
            
            for idx, corr_info in enumerate(top_correlations):
                if corr_info['corr'] > 0.1:  # NOTE: 의미있는 상관관계만 표시
                    var1, var2 = corr_info['var1'], corr_info['var2']
                    
                    fig_scatter = px.scatter(
                        df, x=var1, y=var2,
                        title=f"{var1} vs {var2} (상관계수: {corr_matrix.loc[var1, var2]:.3f})",
                        trendline="ols"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
        
        # 범주형 변수 분석
        if categorical_cols:
            st.markdown("#### 📊 범주형 변수 분석")
            
            # 각 범주형 변수의 분포
            n_cat_cols = min(len(categorical_cols), 3)  # 최대 3개까지만 표시
            cols = st.columns(n_cat_cols)
            
            for idx, cat_col in enumerate(categorical_cols[:n_cat_cols]):
                with cols[idx]:
                    value_counts = df[cat_col].value_counts()
                    fig_pie = px.pie(
                        values=value_counts.values,
                        names=value_counts.index,
                        title=f"{cat_col} 분포"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
        
        # 범주형 vs 수치형 분석
        if categorical_cols and numeric_cols:
            st.markdown("#### 🔗 범주형 vs 수치형 변수 분석")
            
            # 첫 번째 범주형 변수와 첫 번째 수치형 변수로 분석
            cat_var = categorical_cols[0]
            num_var = numeric_cols[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_box = px.box(df, x=cat_var, y=num_var, 
                               title=f"{cat_var}별 {num_var} 분포")
                st.plotly_chart(fig_box, use_container_width=True)
            
            with col2:
                fig_violin = px.violin(df, x=cat_var, y=num_var,
                                     title=f"{cat_var}별 {num_var} 바이올린 플롯")
                st.plotly_chart(fig_violin, use_container_width=True)
    
    else:  # 커스텀 대시보드
        st.markdown("### 🎨 커스텀 대시보드")
        st.markdown("원하는 차트들을 선택하여 개인화된 대시보드를 만들어보세요.")
        
        # 사이드바에서 차트 선택
        chart_options = st.sidebar.multiselect(
            "표시할 차트 선택",
            ["데이터 요약", "상관관계 히트맵", "분포 히스토그램", "산점도", "박스플롯", "바 차트", "라인 차트"],
            default=["데이터 요약", "상관관계 히트맵"]
        )
        
        if "데이터 요약" in chart_options:
            st.markdown("#### 📋 데이터 요약")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**기본 정보**")
                st.dataframe(df.describe().round(2) if numeric_cols else pd.DataFrame({"메시지": ["수치형 데이터가 없습니다."]}))
            
            with col2:
                st.write("**데이터 타입**")
                type_info = pd.DataFrame({
                    '변수명': df.columns,
                    '데이터 타입': df.dtypes.astype(str),
                    '결측치 수': df.isnull().sum()
                })
                st.dataframe(type_info)
        
        if "상관관계 히트맵" in chart_options and len(numeric_cols) >= 2:
            st.markdown("#### 🔗 상관관계 히트맵")
            corr_matrix = df[numeric_cols].corr()
            fig_corr = px.imshow(
                corr_matrix,
                title="변수 간 상관관계",
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        if "분포 히스토그램" in chart_options and numeric_cols:
            st.markdown("#### 📊 분포 히스토그램")
            selected_vars = st.sidebar.multiselect(
                "히스토그램으로 볼 변수 선택",
                numeric_cols,
                default=numeric_cols[:2]
            )
            
            if selected_vars:
                n_cols = min(len(selected_vars), 2)
                cols = st.columns(n_cols)
                
                for idx, var in enumerate(selected_vars[:n_cols]):
                    with cols[idx % n_cols]:
                        fig_hist = px.histogram(df, x=var, title=f"{var} 분포")
                        st.plotly_chart(fig_hist, use_container_width=True)
        
        if "산점도" in chart_options and len(numeric_cols) >= 2:
            st.markdown("#### 🔍 산점도")
            col1, col2 = st.columns(2)
            
            with col1:
                x_var = st.selectbox("X축 변수", numeric_cols, key="custom_scatter_x")
            with col2:
                y_var = st.selectbox("Y축 변수", numeric_cols, key="custom_scatter_y")
            
            color_var = st.selectbox("색상 기준 (선택사항)", [None] + all_cols, key="custom_scatter_color")
            
            if x_var != y_var:
                fig_scatter = px.scatter(df, x=x_var, y=y_var, color=color_var,
                                       title=f"{x_var} vs {y_var}")
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        if "박스플롯" in chart_options:
            st.markdown("#### 📦 박스플롯")
            
            if categorical_cols and numeric_cols:
                col1, col2 = st.columns(2)
                with col1:
                    cat_var = st.selectbox("범주형 변수", categorical_cols, key="custom_box_cat")
                with col2:
                    num_var = st.selectbox("수치형 변수", numeric_cols, key="custom_box_num")
                
                fig_box = px.box(df, x=cat_var, y=num_var,
                               title=f"{cat_var}별 {num_var} 박스플롯")
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("박스플롯을 그리려면 범주형 변수와 수치형 변수가 모두 필요합니다.")
        
        if "바 차트" in chart_options:
            st.markdown("#### 📊 바 차트")
            
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("X축 변수", all_cols, key="custom_bar_x")
            with col2:
                y_var = st.selectbox("Y축 변수", numeric_cols, key="custom_bar_y")
            
            fig_bar = px.bar(df, x=x_var, y=y_var, title=f"{x_var}별 {y_var}")
            st.plotly_chart(fig_bar, use_container_width=True)
        
        if "라인 차트" in chart_options and len(numeric_cols) >= 2:
            st.markdown("#### 📈 라인 차트")
            
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("X축 변수", all_cols, key="custom_line_x")
            with col2:
                y_var = st.selectbox("Y축 변수", numeric_cols, key="custom_line_y")
            
            fig_line = px.line(df, x=x_var, y=y_var, title=f"{x_var}에 따른 {y_var} 변화")
            st.plotly_chart(fig_line, use_container_width=True)

    # 데이터 인사이트 요약
    st.markdown("### 💡 데이터 인사이트 요약")
    
    insights = []
    
    # 데이터 크기 인사이트
    if len(df) < 100:
        insights.append("⚠️ 데이터 샘플 수가 적습니다. 분석 결과의 신뢰성을 고려해주세요.")
    elif len(df) > 10000:
        insights.append("✅ 충분한 데이터 샘플을 보유하고 있어 안정적인 분석이 가능합니다.")
    
    # 결측치 인사이트
    missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
    if missing_ratio > 0.1:
        insights.append(f"🚨 결측치 비율이 {missing_ratio:.1%}로 높습니다. 데이터 전처리를 고려해보세요.")
    elif missing_ratio > 0:
        insights.append(f"⚠️ 결측치가 {missing_ratio:.1%} 존재합니다.")
    else:
        insights.append("✅ 결측치가 없는 깨끗한 데이터입니다.")
    
    # 상관관계 인사이트
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        strong_corr_count = (corr_matrix.abs() > 0.7).sum().sum() - len(numeric_cols)  # 대각선 제외
        if strong_corr_count > 0:
            insights.append(f"🔗 강한 상관관계(|r| > 0.7)를 보이는 변수 쌍이 {strong_corr_count//2}개 있습니다.")
    
    # 범주형 변수 인사이트
    for col in categorical_cols:
        unique_ratio = df[col].nunique() / len(df)
        if unique_ratio > 0.9:
            insights.append(f"📊 '{col}' 변수는 고유값 비율이 {unique_ratio:.1%}로 높아 식별자에 가까운 특성을 보입니다.")
    
    for insight in insights:
        st.write(insight)

else:
    st.info("CSV 파일을 업로드하여 대시보드를 생성하세요!")
    st.markdown("""
    ### 대시보드 기능 미리보기
    
    **자동 대시보드**
    - 데이터 개요 및 요약 통계
    - 자동 결측치 분석
    - 상관관계 매트릭스 및 주요 관계 시각화
    - 범주형/수치형 변수 분포 분석
    
    **커스텀 대시보드**
    - 원하는 차트 조합 선택
    - 개인화된 분석 뷰
    - 인터랙티브한 변수 선택
    """)

# TODO: 실시간 필터링, 드릴다운 기능, 대시보드 저장/로드 기능 추가 예정 