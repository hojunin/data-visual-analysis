import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import sys
import os

# utils.py 모듈 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import add_dark_mode_toggle, add_chart_export_section, style_metric_cards

st.title("📈 회귀분석 전문 도구")

# 다크모드 토글 및 스타일 추가
add_dark_mode_toggle()
style_metric_cards()
st.markdown("다양한 회귀분석 모델을 비교하고 성능을 평가할 수 있는 전문 도구입니다.")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("데이터 미리보기", df.head())

    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    if len(numeric_cols) >= 2:
        # 사이드바에서 회귀분석 설정
        st.sidebar.header("회귀분석 설정")
        
        regression_type = st.sidebar.selectbox(
            "회귀분석 유형",
            ["단순 선형회귀", "다중 선형회귀", "다항 회귀", "정규화 회귀", "모델 비교"]
        )
        
        # 종속변수 선택
        target_var = st.sidebar.selectbox("종속변수 (Y)", numeric_cols, key="target")
        
        # 독립변수 선택
        available_features = [col for col in numeric_cols if col != target_var]
        
        if regression_type == "단순 선형회귀":
            st.markdown("## 📊 단순 선형회귀분석")
            st.markdown("하나의 독립변수와 종속변수 간의 선형 관계를 분석합니다.")
            
            feature_var = st.selectbox("독립변수 (X)", available_features)
            
            # 데이터 준비
            X = df[[feature_var]].dropna()
            y = df[target_var].dropna()
            
            # 공통 인덱스
            common_idx = X.index.intersection(y.index)
            X_clean = X.loc[common_idx]
            y_clean = y.loc[common_idx]
            
            # 훈련/테스트 분할
            test_size = st.slider("테스트 데이터 비율", 0.1, 0.5, 0.2, 0.05)
            X_train, X_test, y_train, y_test = train_test_split(
                X_clean, y_clean, test_size=test_size, random_state=42
            )
            
            # 모델 학습
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # 예측
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # 성능 메트릭
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                train_r2 = r2_score(y_train, y_train_pred)
                st.metric("훈련 R²", f"{train_r2:.4f}")
            
            with col2:
                test_r2 = r2_score(y_test, y_test_pred)
                st.metric("테스트 R²", f"{test_r2:.4f}")
            
            with col3:
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                st.metric("훈련 RMSE", f"{train_rmse:.4f}")
            
            with col4:
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                st.metric("테스트 RMSE", f"{test_rmse:.4f}")
            
            # 회귀 방정식
            coef = model.coef_[0]
            intercept = model.intercept_
            st.markdown(f"**회귀 방정식**: {target_var} = {coef:.4f} × {feature_var} + {intercept:.4f}")
            
            # 시각화
            col1, col2 = st.columns(2)
            
            with col1:
                # 산점도와 회귀선
                fig_scatter = px.scatter(
                    x=X_clean[feature_var], y=y_clean, 
                    title=f"{target_var} vs {feature_var}",
                    labels={'x': feature_var, 'y': target_var}
                )
                
                # 회귀선 추가
                x_range = np.linspace(X_clean[feature_var].min(), X_clean[feature_var].max(), 100)
                y_pred_line = model.predict(x_range.reshape(-1, 1))
                
                fig_scatter.add_trace(go.Scatter(
                    x=x_range, y=y_pred_line,
                    mode='lines', name='회귀선',
                    line=dict(color='red', width=2)
                ))
                
                st.plotly_chart(fig_scatter, use_container_width=True)
                add_chart_export_section(fig_scatter, f"simple_regression_{feature_var}_{target_var}")
            
            with col2:
                # 잔차 플롯
                residuals = y_test - y_test_pred
                fig_residual = px.scatter(
                    x=y_test_pred, y=residuals,
                    title="잔차 플롯 (테스트 데이터)",
                    labels={'x': '예측값', 'y': '잔차'}
                )
                fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_residual, use_container_width=True)
                add_chart_export_section(fig_residual, f"residuals_{feature_var}_{target_var}")
            
            # 회귀 진단
            st.markdown("### 🔍 회귀 진단")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 정규성 검정 (잔차)
                shapiro_stat, shapiro_p = stats.shapiro(residuals)
                st.metric("Shapiro-Wilk p-value", f"{shapiro_p:.4f}")
                if shapiro_p > 0.05:
                    st.success("잔차가 정규분포를 따릅니다")
                else:
                    st.warning("잔차가 정규분포를 따르지 않습니다")
            
            with col2:
                # Durbin-Watson 검정 (자기상관)
                from statsmodels.stats.diagnostic import durbin_watson
                dw_stat = durbin_watson(residuals)
                st.metric("Durbin-Watson", f"{dw_stat:.4f}")
                if 1.5 <= dw_stat <= 2.5:
                    st.success("자기상관이 없습니다")
                else:
                    st.warning("자기상관이 있을 수 있습니다")
            
            with col3:
                # 상관계수
                correlation = np.corrcoef(X_clean[feature_var], y_clean)[0, 1]
                st.metric("상관계수", f"{correlation:.4f}")
        
        elif regression_type == "다중 선형회귀":
            st.markdown("## 📊 다중 선형회귀분석")
            st.markdown("여러 독립변수와 종속변수 간의 관계를 분석합니다.")
            
            # 독립변수 선택
            selected_features = st.multiselect(
                "독립변수들 선택",
                available_features,
                default=available_features[:min(3, len(available_features))]
            )
            
            if len(selected_features) >= 1:
                # 데이터 준비
                X = df[selected_features].dropna()
                y = df[target_var].dropna()
                
                # 공통 인덱스
                common_idx = X.index.intersection(y.index)
                X_clean = X.loc[common_idx]
                y_clean = y.loc[common_idx]
                
                # 훈련/테스트 분할
                test_size = st.slider("테스트 데이터 비율", 0.1, 0.5, 0.2, 0.05, key="multi_test_size")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_clean, y_clean, test_size=test_size, random_state=42
                )
                
                # 모델 학습
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # 예측
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                # 성능 메트릭
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    train_r2 = r2_score(y_train, y_train_pred)
                    st.metric("훈련 R²", f"{train_r2:.4f}")
                
                with col2:
                    test_r2 = r2_score(y_test, y_test_pred)
                    st.metric("테스트 R²", f"{test_r2:.4f}")
                
                with col3:
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                    st.metric("훈련 RMSE", f"{train_rmse:.4f}")
                
                with col4:
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                    st.metric("테스트 RMSE", f"{test_rmse:.4f}")
                
                # 회귀 계수
                st.markdown("### 📊 회귀 계수")
                coef_df = pd.DataFrame({
                    '변수': selected_features,
                    '계수': model.coef_,
                    '절댓값': np.abs(model.coef_)
                }).sort_values('절댓값', ascending=False)
                
                st.dataframe(coef_df.round(4))
                
                # 회귀 방정식
                equation = f"{target_var} = "
                for i, feature in enumerate(selected_features):
                    if i == 0:
                        equation += f"{model.coef_[i]:.4f} × {feature}"
                    else:
                        sign = "+" if model.coef_[i] >= 0 else ""
                        equation += f" {sign} {model.coef_[i]:.4f} × {feature}"
                equation += f" + {model.intercept_:.4f}"
                
                st.markdown(f"**회귀 방정식**: {equation}")
                
                # 계수 중요도 시각화
                fig_coef = px.bar(
                    coef_df, x='절댓값', y='변수', orientation='h',
                    title="변수 중요도 (계수 절댓값)"
                )
                st.plotly_chart(fig_coef, use_container_width=True)
                add_chart_export_section(fig_coef, f"coefficients_{target_var}")
                
                # 예측값 vs 실제값
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_pred = px.scatter(
                        x=y_test, y=y_test_pred,
                        title="예측값 vs 실제값 (테스트 데이터)"
                    )
                    # 대각선 추가 (완벽한 예측)
                    min_val = min(y_test.min(), y_test_pred.min())
                    max_val = max(y_test.max(), y_test_pred.max())
                    fig_pred.add_trace(go.Scatter(
                        x=[min_val, max_val], y=[min_val, max_val],
                        mode='lines', name='완벽한 예측',
                        line=dict(color='red', dash='dash')
                    ))
                    st.plotly_chart(fig_pred, use_container_width=True)
                
                with col2:
                    # 잔차 플롯
                    residuals = y_test - y_test_pred
                    fig_residual = px.scatter(
                        x=y_test_pred, y=residuals,
                        title="잔차 플롯"
                    )
                    fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig_residual, use_container_width=True)
                
                # 교차검증
                st.markdown("### 🔄 교차검증")
                cv_scores = cross_val_score(model, X_clean, y_clean, cv=5, scoring='r2')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("CV 평균 R²", f"{cv_scores.mean():.4f}")
                with col2:
                    st.metric("CV 표준편차", f"{cv_scores.std():.4f}")
                with col3:
                    st.metric("CV 점수 범위", f"{cv_scores.min():.3f} ~ {cv_scores.max():.3f}")
        
        elif regression_type == "다항 회귀":
            st.markdown("## 📊 다항 회귀분석")
            st.markdown("비선형 관계를 모델링하기 위한 다항 회귀분석입니다.")
            
            feature_var = st.selectbox("독립변수 (X)", available_features, key="poly_feature")
            degree = st.slider("다항식 차수", 1, 5, 2)
            
            # 데이터 준비
            X = df[[feature_var]].dropna()
            y = df[target_var].dropna()
            
            common_idx = X.index.intersection(y.index)
            X_clean = X.loc[common_idx]
            y_clean = y.loc[common_idx]
            
            # 다항 특성 생성
            poly_features = PolynomialFeatures(degree=degree, include_bias=False)
            X_poly = poly_features.fit_transform(X_clean)
            
            # 훈련/테스트 분할
            test_size = st.slider("테스트 데이터 비율", 0.1, 0.5, 0.2, 0.05, key="poly_test_size")
            X_train, X_test, y_train, y_test = train_test_split(
                X_poly, y_clean, test_size=test_size, random_state=42
            )
            
            # 모델 학습
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # 예측
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # 성능 메트릭
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                train_r2 = r2_score(y_train, y_train_pred)
                st.metric("훈련 R²", f"{train_r2:.4f}")
            
            with col2:
                test_r2 = r2_score(y_test, y_test_pred)
                st.metric("테스트 R²", f"{test_r2:.4f}")
            
            with col3:
                train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
                st.metric("훈련 RMSE", f"{train_rmse:.4f}")
            
            with col4:
                test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                st.metric("테스트 RMSE", f"{test_rmse:.4f}")
            
            # 과적합 경고
            if train_r2 - test_r2 > 0.1:
                st.warning("⚠️ 과적합이 의심됩니다. 차수를 낮춰보세요.")
            
            # 시각화
            fig = px.scatter(
                x=X_clean[feature_var], y=y_clean,
                title=f"{degree}차 다항 회귀: {target_var} vs {feature_var}"
            )
            
            # 다항 회귀 곡선 그리기
            x_range = np.linspace(X_clean[feature_var].min(), X_clean[feature_var].max(), 300)
            X_range_poly = poly_features.transform(x_range.reshape(-1, 1))
            y_range_pred = model.predict(X_range_poly)
            
            fig.add_trace(go.Scatter(
                x=x_range, y=y_range_pred,
                mode='lines', name=f'{degree}차 다항 회귀',
                line=dict(color='red', width=2)
            ))
            
            st.plotly_chart(fig, use_container_width=True)
            add_chart_export_section(fig, f"polynomial_regression_degree_{degree}")
            
            # 차수별 성능 비교
            st.markdown("### 📊 차수별 성능 비교")
            
            degrees = range(1, 6)
            train_scores = []
            test_scores = []
            
            X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
                X_clean, y_clean, test_size=test_size, random_state=42
            )
            
            for d in degrees:
                poly = PolynomialFeatures(degree=d, include_bias=False)
                X_train_poly = poly.fit_transform(X_train_orig)
                X_test_poly = poly.transform(X_test_orig)
                
                temp_model = LinearRegression()
                temp_model.fit(X_train_poly, y_train_orig)
                
                train_pred = temp_model.predict(X_train_poly)
                test_pred = temp_model.predict(X_test_poly)
                
                train_scores.append(r2_score(y_train_orig, train_pred))
                test_scores.append(r2_score(y_test_orig, test_pred))
            
            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Scatter(
                x=list(degrees), y=train_scores,
                mode='lines+markers', name='훈련 R²',
                line=dict(color='blue')
            ))
            fig_comparison.add_trace(go.Scatter(
                x=list(degrees), y=test_scores,
                mode='lines+markers', name='테스트 R²',
                line=dict(color='red')
            ))
            
            fig_comparison.update_layout(
                title="다항식 차수별 성능 비교",
                xaxis_title="다항식 차수",
                yaxis_title="R² 점수"
            )
            
            st.plotly_chart(fig_comparison, use_container_width=True)
            add_chart_export_section(fig_comparison, "polynomial_degree_comparison")
        
        elif regression_type == "정규화 회귀":
            st.markdown("## 📊 정규화 회귀분석")
            st.markdown("과적합을 방지하는 Ridge, Lasso, ElasticNet 회귀분석입니다.")
            
            # 독립변수 선택
            selected_features = st.multiselect(
                "독립변수들 선택",
                available_features,
                default=available_features[:min(5, len(available_features))],
                key="regularization_features"
            )
            
            if len(selected_features) >= 1:
                # 데이터 준비
                X = df[selected_features].dropna()
                y = df[target_var].dropna()
                
                common_idx = X.index.intersection(y.index)
                X_clean = X.loc[common_idx]
                y_clean = y.loc[common_idx]
                
                # 데이터 표준화
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_clean)
                
                # 훈련/테스트 분할
                test_size = st.slider("테스트 데이터 비율", 0.1, 0.5, 0.2, 0.05, key="reg_test_size")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_clean, test_size=test_size, random_state=42
                )
                
                # 정규화 파라미터
                alpha = st.slider("정규화 강도 (alpha)", 0.01, 10.0, 1.0, 0.01)
                
                # 모델들 학습
                models = {
                    'Linear': LinearRegression(),
                    'Ridge': Ridge(alpha=alpha),
                    'Lasso': Lasso(alpha=alpha),
                    'ElasticNet': ElasticNet(alpha=alpha, l1_ratio=0.5)
                }
                
                results = {}
                
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)
                    
                    results[name] = {
                        'train_r2': r2_score(y_train, y_train_pred),
                        'test_r2': r2_score(y_test, y_test_pred),
                        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                        'model': model
                    }
                
                # 결과 비교 테이블
                st.markdown("### 📊 모델 성능 비교")
                
                comparison_df = pd.DataFrame({
                    '모델': list(results.keys()),
                    '훈련 R²': [results[name]['train_r2'] for name in results.keys()],
                    '테스트 R²': [results[name]['test_r2'] for name in results.keys()],
                    '훈련 RMSE': [results[name]['train_rmse'] for name in results.keys()],
                    '테스트 RMSE': [results[name]['test_rmse'] for name in results.keys()]
                }).round(4)
                
                st.dataframe(comparison_df)
                
                # 성능 시각화
                fig_comparison = go.Figure()
                
                fig_comparison.add_trace(go.Bar(
                    name='훈련 R²',
                    x=list(results.keys()),
                    y=[results[name]['train_r2'] for name in results.keys()],
                    marker_color='lightblue'
                ))
                
                fig_comparison.add_trace(go.Bar(
                    name='테스트 R²',
                    x=list(results.keys()),
                    y=[results[name]['test_r2'] for name in results.keys()],
                    marker_color='lightcoral'
                ))
                
                fig_comparison.update_layout(
                    title="모델별 R² 성능 비교",
                    xaxis_title="모델",
                    yaxis_title="R² 점수",
                    barmode='group'
                )
                
                st.plotly_chart(fig_comparison, use_container_width=True)
                add_chart_export_section(fig_comparison, f"regularization_comparison_alpha_{alpha}")
                
                # 계수 비교 (Ridge vs Lasso)
                st.markdown("### 📊 회귀 계수 비교")
                
                coef_comparison = pd.DataFrame({
                    '변수': selected_features,
                    'Linear': results['Linear']['model'].coef_,
                    'Ridge': results['Ridge']['model'].coef_,
                    'Lasso': results['Lasso']['model'].coef_,
                    'ElasticNet': results['ElasticNet']['model'].coef_
                })
                
                st.dataframe(coef_comparison.round(4))
                
                # 계수 시각화
                fig_coef = go.Figure()
                
                for model_name in ['Linear', 'Ridge', 'Lasso', 'ElasticNet']:
                    fig_coef.add_trace(go.Bar(
                        name=model_name,
                        x=selected_features,
                        y=coef_comparison[model_name]
                    ))
                
                fig_coef.update_layout(
                    title="모델별 회귀 계수 비교",
                    xaxis_title="변수",
                    yaxis_title="계수 값",
                    barmode='group'
                )
                
                st.plotly_chart(fig_coef, use_container_width=True)
                add_chart_export_section(fig_coef, "coefficient_comparison")
                
                # 알파 값에 따른 성능 변화
                st.markdown("### 📈 정규화 강도에 따른 성능 변화")
                
                alphas = np.logspace(-3, 2, 20)  # 0.001 ~ 100
                ridge_scores = []
                lasso_scores = []
                
                for a in alphas:
                    ridge_model = Ridge(alpha=a)
                    lasso_model = Lasso(alpha=a)
                    
                    ridge_model.fit(X_train, y_train)
                    lasso_model.fit(X_train, y_train)
                    
                    ridge_scores.append(r2_score(y_test, ridge_model.predict(X_test)))
                    lasso_scores.append(r2_score(y_test, lasso_model.predict(X_test)))
                
                fig_alpha = go.Figure()
                fig_alpha.add_trace(go.Scatter(
                    x=alphas, y=ridge_scores,
                    mode='lines+markers', name='Ridge',
                    line=dict(color='blue')
                ))
                fig_alpha.add_trace(go.Scatter(
                    x=alphas, y=lasso_scores,
                    mode='lines+markers', name='Lasso',
                    line=dict(color='red')
                ))
                
                fig_alpha.update_layout(
                    title="정규화 강도에 따른 테스트 R² 변화",
                    xaxis_title="Alpha (로그 스케일)",
                    yaxis_title="테스트 R²",
                    xaxis_type="log"
                )
                
                st.plotly_chart(fig_alpha, use_container_width=True)
                add_chart_export_section(fig_alpha, "regularization_path")
        
        elif regression_type == "모델 비교":
            st.markdown("## 📊 회귀 모델 종합 비교")
            st.markdown("다양한 회귀 모델의 성능을 한 번에 비교합니다.")
            
            # 독립변수 선택
            selected_features = st.multiselect(
                "독립변수들 선택",
                available_features,
                default=available_features[:min(4, len(available_features))],
                key="comparison_features"
            )
            
            if len(selected_features) >= 1:
                # 데이터 준비
                X = df[selected_features].dropna()
                y = df[target_var].dropna()
                
                common_idx = X.index.intersection(y.index)
                X_clean = X.loc[common_idx]
                y_clean = y.loc[common_idx]
                
                # 표준화
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_clean)
                
                # 훈련/테스트 분할
                test_size = st.slider("테스트 데이터 비율", 0.1, 0.5, 0.2, 0.05, key="comp_test_size")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y_clean, test_size=test_size, random_state=42
                )
                
                # 다항 특성 생성 (2차)
                poly_features = PolynomialFeatures(degree=2, include_bias=False)
                X_train_poly = poly_features.fit_transform(X_train)
                X_test_poly = poly_features.transform(X_test)
                
                # 모델들 정의
                models = {
                    '선형 회귀': LinearRegression(),
                    'Ridge 회귀': Ridge(alpha=1.0),
                    'Lasso 회귀': Lasso(alpha=1.0),
                    'ElasticNet': ElasticNet(alpha=1.0, l1_ratio=0.5),
                    '2차 다항 회귀': LinearRegression()
                }
                
                # 결과 저장
                results = {}
                
                for name, model in models.items():
                    if name == '2차 다항 회귀':
                        model.fit(X_train_poly, y_train)
                        y_train_pred = model.predict(X_train_poly)
                        y_test_pred = model.predict(X_test_poly)
                    else:
                        model.fit(X_train, y_train)
                        y_train_pred = model.predict(X_train)
                        y_test_pred = model.predict(X_test)
                    
                    # 교차검증 점수
                    if name == '2차 다항 회귀':
                        X_for_cv = poly_features.fit_transform(X_scaled)
                        cv_scores = cross_val_score(LinearRegression(), X_for_cv, y_clean, cv=5, scoring='r2')
                    else:
                        cv_scores = cross_val_score(model, X_scaled, y_clean, cv=5, scoring='r2')
                    
                    results[name] = {
                        'train_r2': r2_score(y_train, y_train_pred),
                        'test_r2': r2_score(y_test, y_test_pred),
                        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
                        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
                        'train_mae': mean_absolute_error(y_train, y_train_pred),
                        'test_mae': mean_absolute_error(y_test, y_test_pred),
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                
                # 종합 결과 테이블
                st.markdown("### 📊 모델 성능 종합 비교")
                
                comparison_df = pd.DataFrame(results).T
                comparison_df = comparison_df.round(4)
                
                # 최고 성능 모델 표시
                best_test_r2 = comparison_df['test_r2'].max()
                best_model = comparison_df['test_r2'].idxmax()
                
                st.success(f"🏆 **최고 성능 모델**: {best_model} (테스트 R² = {best_test_r2:.4f})")
                
                st.dataframe(comparison_df)
                
                # 성능 시각화 - R² 비교
                fig_r2 = go.Figure()
                
                fig_r2.add_trace(go.Bar(
                    name='훈련 R²',
                    x=list(results.keys()),
                    y=[results[name]['train_r2'] for name in results.keys()],
                    marker_color='lightblue'
                ))
                
                fig_r2.add_trace(go.Bar(
                    name='테스트 R²',
                    x=list(results.keys()),
                    y=[results[name]['test_r2'] for name in results.keys()],
                    marker_color='lightcoral'
                ))
                
                fig_r2.update_layout(
                    title="모델별 R² 성능 비교",
                    xaxis_title="모델",
                    yaxis_title="R² 점수",
                    barmode='group'
                )
                
                st.plotly_chart(fig_r2, use_container_width=True)
                add_chart_export_section(fig_r2, "comprehensive_model_comparison")
                
                # RMSE 비교
                fig_rmse = px.bar(
                    x=list(results.keys()),
                    y=[results[name]['test_rmse'] for name in results.keys()],
                    title="모델별 테스트 RMSE 비교"
                )
                st.plotly_chart(fig_rmse, use_container_width=True)
                
                # 교차검증 결과
                st.markdown("### 🔄 교차검증 결과")
                
                cv_df = pd.DataFrame({
                    '모델': list(results.keys()),
                    'CV 평균 R²': [results[name]['cv_mean'] for name in results.keys()],
                    'CV 표준편차': [results[name]['cv_std'] for name in results.keys()]
                }).round(4)
                
                st.dataframe(cv_df)
                
                # 과적합 분석
                st.markdown("### 🔍 과적합 분석")
                
                overfitting_df = pd.DataFrame({
                    '모델': list(results.keys()),
                    '훈련 R²': [results[name]['train_r2'] for name in results.keys()],
                    '테스트 R²': [results[name]['test_r2'] for name in results.keys()],
                    '차이': [results[name]['train_r2'] - results[name]['test_r2'] for name in results.keys()]
                }).round(4)
                
                # 과적합 경고
                for idx, row in overfitting_df.iterrows():
                    if row['차이'] > 0.1:
                        st.warning(f"⚠️ {row['모델']}: 과적합 가능성 (차이 = {row['차이']:.3f})")
                    elif row['차이'] < 0.05:
                        st.success(f"✅ {row['모델']}: 안정적인 성능")
                
                st.dataframe(overfitting_df)
    
    else:
        st.warning("회귀분석을 위해서는 최소 2개 이상의 수치형 변수가 필요합니다.")

else:
    st.info("CSV 파일을 업로드하여 회귀분석을 시작하세요!")
    st.markdown("""
    ### 회귀분석 기능 미리보기
    
    **단순 선형회귀**
    - 하나의 독립변수와 종속변수 간의 관계 분석
    - 회귀선 시각화 및 잔차 분석
    - 회귀 진단 (정규성, 자기상관 검정)
    
    **다중 선형회귀**
    - 여러 독립변수를 사용한 모델링
    - 변수 중요도 분석
    - 교차검증을 통한 모델 안정성 평가
    
    **다항 회귀**
    - 비선형 관계 모델링
    - 차수별 성능 비교
    - 과적합 방지 권장사항
    
    **정규화 회귀**
    - Ridge, Lasso, ElasticNet 회귀
    - 과적합 방지 및 특성 선택
    - 정규화 강도에 따른 성능 변화 분석
    
    **모델 비교**
    - 모든 회귀 모델의 종합 성능 비교
    - 교차검증 및 과적합 분석
    - 최적 모델 추천
    """)

# TODO: 로지스틱 회귀, 시계열 회귀, 베이지안 회귀 추가 예정 