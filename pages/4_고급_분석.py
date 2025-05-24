import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import sys
import os

# utils.py 모듈 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import apply_custom_theme, add_chart_export_section, style_metric_cards

st.title("📈 고급 분석 기능")

# 다크모드 토글 및 스타일 추가
apply_custom_theme()
style_metric_cards()
st.markdown("데이터의 통계적 특성과 패턴을 심층 분석합니다.")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("데이터 미리보기", df.head())

    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    analysis_type = st.selectbox(
        "분석 유형 선택",
        ["기술통계", "상관관계 분석", "회귀분석", "분포 분석", "그룹별 분석"]
    )

    if analysis_type == "기술통계":
        st.markdown("### 📊 기술통계 요약")
        
        if numeric_cols:
            # 전체 데이터 기술통계
            st.write("**수치형 변수 기술통계**")
            desc_stats = df[numeric_cols].describe()
            st.dataframe(desc_stats.round(2))
            
            # 개별 변수 상세 분석
            selected_col = st.selectbox("상세 분석할 변수 선택", numeric_cols)
            if selected_col:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("평균", f"{df[selected_col].mean():.2f}")
                    st.metric("중앙값", f"{df[selected_col].median():.2f}")
                
                with col2:
                    st.metric("표준편차", f"{df[selected_col].std():.2f}")
                    st.metric("분산", f"{df[selected_col].var():.2f}")
                
                with col3:
                    st.metric("최솟값", f"{df[selected_col].min():.2f}")
                    st.metric("최댓값", f"{df[selected_col].max():.2f}")
                
                with col4:
                    skewness = stats.skew(df[selected_col].dropna())
                    kurtosis = stats.kurtosis(df[selected_col].dropna())
                    st.metric("왜도", f"{skewness:.3f}")
                    st.metric("첨도", f"{kurtosis:.3f}")
                
                # 히스토그램과 박스플롯
                col1, col2 = st.columns(2)
                with col1:
                    fig_hist = px.histogram(df, x=selected_col, nbins=20, title=f"{selected_col} 분포")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    fig_box = px.box(df, y=selected_col, title=f"{selected_col} 박스플롯")
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # 차트 내보내기 기능
                col1_exp, col2_exp = st.columns(2)
                with col1_exp:
                    add_chart_export_section(fig_hist, f"histogram_{selected_col}")
                with col2_exp:
                    add_chart_export_section(fig_box, f"boxplot_{selected_col}")

    elif analysis_type == "상관관계 분석":
        st.markdown("### 🔗 상관관계 분석")
        
        if len(numeric_cols) >= 2:
            # 상관관계 매트릭스
            corr_matrix = df[numeric_cols].corr()
            
            # Plotly 히트맵
            fig_corr = px.imshow(
                corr_matrix, 
                title="상관관계 매트릭스",
                color_continuous_scale='RdBu_r',
                aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # 차트 내보내기 기능
            add_chart_export_section(fig_corr, "correlation_matrix")
            
            # 상관관계 수치 표시
            st.write("**상관계수 매트릭스**")
            st.dataframe(corr_matrix.round(3))
            
            # 강한 상관관계 찾기
            st.markdown("### 🎯 주요 상관관계")
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:  # NOTE: 0.5 이상의 상관관계만 표시
                        strong_corr.append({
                            '변수1': corr_matrix.columns[i],
                            '변수2': corr_matrix.columns[j],
                            '상관계수': corr_val,
                            '강도': '강한 양의 상관관계' if corr_val > 0 else '강한 음의 상관관계'
                        })
            
            if strong_corr:
                strong_corr_df = pd.DataFrame(strong_corr)
                st.dataframe(strong_corr_df)
            else:
                st.info("강한 상관관계(|r| > 0.5)가 발견되지 않았습니다.")

    elif analysis_type == "회귀분석":
        st.markdown("### 📈 회귀분석")
        
        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("독립변수 (X)", numeric_cols, key="reg_x")
            with col2:
                y_var = st.selectbox("종속변수 (Y)", numeric_cols, key="reg_y")
            
            if x_var != y_var:
                # 데이터 준비
                x_data = df[x_var].dropna()
                y_data = df[y_var].dropna()
                
                # 공통 인덱스 찾기
                common_idx = x_data.index.intersection(y_data.index)
                x_clean = x_data.loc[common_idx].values.reshape(-1, 1)
                y_clean = y_data.loc[common_idx].values
                
                # 선형 회귀 모델 생성
                model = LinearRegression()
                model.fit(x_clean, y_clean)
                y_pred = model.predict(x_clean)
                
                # 결과 메트릭
                r2 = r2_score(y_clean, y_pred)
                correlation = np.corrcoef(x_clean.flatten(), y_clean)[0, 1]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² (결정계수)", f"{r2:.3f}")
                with col2:
                    st.metric("상관계수", f"{correlation:.3f}")
                with col3:
                    st.metric("기울기", f"{model.coef_[0]:.3f}")
                
                # 회귀 직선과 산점도
                fig_reg = px.scatter(x=x_clean.flatten(), y=y_clean, 
                                   title=f"{y_var} vs {x_var} 회귀분석")
                
                # 회귀선 추가
                fig_reg.add_trace(go.Scatter(
                    x=x_clean.flatten(),
                    y=y_pred,
                    mode='lines',
                    name='회귀선',
                    line=dict(color='red', width=2)
                ))
                
                fig_reg.update_xaxes(title=x_var)
                fig_reg.update_yaxes(title=y_var)
                st.plotly_chart(fig_reg, use_container_width=True)
                
                # 차트 내보내기 기능
                add_chart_export_section(fig_reg, f"regression_{x_var}_{y_var}")
                
                # 회귀 방정식
                st.markdown(f"**회귀 방정식**: {y_var} = {model.coef_[0]:.3f} × {x_var} + {model.intercept_:.3f}")
                
                # 잔차 분석
                residuals = y_clean - y_pred
                fig_residual = px.scatter(x=y_pred, y=residuals, title="잔차 플롯")
                fig_residual.update_xaxes(title="예측값")
                fig_residual.update_yaxes(title="잔차")
                st.plotly_chart(fig_residual, use_container_width=True)

    elif analysis_type == "분포 분석":
        st.markdown("### 📊 분포 분석")
        
        if numeric_cols:
            selected_var = st.selectbox("분석할 변수 선택", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # 히스토그램
                fig_hist = px.histogram(df, x=selected_var, nbins=30, 
                                      title=f"{selected_var} 히스토그램")
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with col2:
                # Q-Q 플롯 (정규분포와 비교)
                data = df[selected_var].dropna()
                fig = go.Figure()
                
                # 이론적 분위수 vs 실제 분위수
                sorted_data = np.sort(data)
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_data)))
                
                fig.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=sorted_data,
                    mode='markers',
                    name='데이터 포인트'
                ))
                
                # 이상적인 정규분포 선
                fig.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=theoretical_quantiles * data.std() + data.mean(),
                    mode='lines',
                    name='정규분포 기준선',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(title=f"{selected_var} Q-Q 플롯")
                fig.update_xaxes(title="이론적 분위수")
                fig.update_yaxes(title="실제 분위수")
                st.plotly_chart(fig, use_container_width=True)
            
            # 정규성 검정
            st.markdown("### 🧪 정규성 검정")
            shapiro_stat, shapiro_p = stats.shapiro(df[selected_var].dropna())
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Shapiro-Wilk 통계량", f"{shapiro_stat:.4f}")
            with col2:
                st.metric("p-value", f"{shapiro_p:.4f}")
            
            if shapiro_p > 0.05:
                st.success("정규분포를 따를 가능성이 높습니다. (p > 0.05)")
            else:
                st.warning("정규분포를 따르지 않을 가능성이 높습니다. (p ≤ 0.05)")

    elif analysis_type == "그룹별 분석":
        st.markdown("### 👥 그룹별 분석")
        
        if categorical_cols and numeric_cols:
            group_var = st.selectbox("그룹 변수 선택", categorical_cols)
            numeric_var = st.selectbox("분석할 수치 변수", numeric_cols)
            
            # 그룹별 기술통계
            group_stats = df.groupby(group_var)[numeric_var].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).round(2)
            
            st.write(f"**{group_var}별 {numeric_var} 통계**")
            st.dataframe(group_stats)
            
            # 박스플롯
            fig_box = px.box(df, x=group_var, y=numeric_var, 
                           title=f"{group_var}별 {numeric_var} 분포")
            st.plotly_chart(fig_box, use_container_width=True)
            
            # 바이올린 플롯
            fig_violin = px.violin(df, x=group_var, y=numeric_var,
                                 title=f"{group_var}별 {numeric_var} 바이올린 플롯")
            st.plotly_chart(fig_violin, use_container_width=True)
            
            # ANOVA 검정 (그룹이 3개 이상인 경우)
            groups = [group[numeric_var].dropna() for name, group in df.groupby(group_var)]
            if len(groups) >= 2:
                st.markdown("### 🧪 통계적 유의성 검정")
                
                if len(groups) == 2:
                    # 두 그룹: t-검정
                    t_stat, t_p = stats.ttest_ind(groups[0], groups[1])
                    st.write(f"**독립표본 t-검정**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("t-통계량", f"{t_stat:.4f}")
                    with col2:
                        st.metric("p-value", f"{t_p:.4f}")
                    
                    if t_p < 0.05:
                        st.success("그룹 간 평균에 유의한 차이가 있습니다. (p < 0.05)")
                    else:
                        st.info("그룹 간 평균에 유의한 차이가 없습니다. (p ≥ 0.05)")
                        
                else:
                    # 세 그룹 이상: ANOVA
                    f_stat, f_p = stats.f_oneway(*groups)
                    st.write(f"**일원분산분석 (ANOVA)**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("F-통계량", f"{f_stat:.4f}")
                    with col2:
                        st.metric("p-value", f"{f_p:.4f}")
                    
                    if f_p < 0.05:
                        st.success("그룹 간 평균에 유의한 차이가 있습니다. (p < 0.05)")
                    else:
                        st.info("그룹 간 평균에 유의한 차이가 없습니다. (p ≥ 0.05)")

else:
    st.info("CSV 파일을 업로드하여 고급 분석을 시작하세요!")

# TODO: 다중회귀분석, 클러스터링 분석 추가 예정 