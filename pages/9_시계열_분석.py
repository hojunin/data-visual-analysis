import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import acf, pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

import sys
sys.path.append('..')
from utils import add_chart_export_section

def create_sample_timeseries():
    """샘플 시계열 데이터 생성"""
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # 트렌드 + 계절성 + 노이즈
    trend = np.linspace(100, 200, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 5, len(dates))
    
    values = trend + seasonal + noise
    
    df = pd.DataFrame({
        'date': dates,
        'value': values,
        'category': np.random.choice(['A', 'B', 'C'], len(dates))
    })
    
    return df

def perform_stationarity_test(series):
    """정상성 검정"""
    results = {}
    
    # ADF 검정
    adf_result = adfuller(series.dropna())
    results['ADF'] = {
        'statistic': adf_result[0],
        'p_value': adf_result[1],
        'critical_values': adf_result[4],
        'is_stationary': adf_result[1] < 0.05
    }
    
    # KPSS 검정
    kpss_result = kpss(series.dropna())
    results['KPSS'] = {
        'statistic': kpss_result[0],
        'p_value': kpss_result[1],
        'critical_values': kpss_result[3],
        'is_stationary': kpss_result[1] > 0.05
    }
    
    return results

def plot_acf_pacf(series, lags=40):
    """ACF/PACF 플롯 생성"""
    series_clean = series.dropna()
    
    # ACF 계산
    acf_values, acf_confint = acf(series_clean, nlags=lags, alpha=0.05)
    
    # PACF 계산
    pacf_values, pacf_confint = pacf(series_clean, nlags=lags, alpha=0.05)
    
    # 플롯 생성
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['자기상관함수 (ACF)', '편자기상관함수 (PACF)'],
        vertical_spacing=0.1
    )
    
    # ACF
    lags_range = list(range(len(acf_values)))
    fig.add_trace(
        go.Bar(x=lags_range, y=acf_values, name='ACF', marker_color='lightblue'),
        row=1, col=1
    )
    
    # 신뢰구간
    upper_bound = acf_confint[:, 1] - acf_values
    lower_bound = acf_values - acf_confint[:, 0]
    
    fig.add_trace(
        go.Scatter(
            x=lags_range + lags_range[::-1],
            y=list(acf_confint[:, 1]) + list(acf_confint[:, 0])[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% 신뢰구간',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # PACF
    fig.add_trace(
        go.Bar(x=lags_range, y=pacf_values, name='PACF', marker_color='lightcoral'),
        row=2, col=1
    )
    
    # PACF 신뢰구간
    upper_bound_pacf = pacf_confint[:, 1] - pacf_values
    lower_bound_pacf = pacf_values - pacf_confint[:, 0]
    
    fig.add_trace(
        go.Scatter(
            x=lags_range + lags_range[::-1],
            y=list(pacf_confint[:, 1]) + list(pacf_confint[:, 0])[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='95% 신뢰구간',
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(height=600, title_text="자기상관 분석")
    fig.update_xaxes(title_text="Lag")
    fig.update_yaxes(title_text="상관계수")
    
    return fig

def main():
    st.title("📈 시계열 분석")
    st.markdown("시간에 따른 데이터의 패턴과 트렌드를 분석해보세요.")
    
    if not STATSMODELS_AVAILABLE:
        st.error("statsmodels 라이브러리가 필요합니다. 'pip install statsmodels'로 설치해주세요.")
        return
    
    # 사이드바
    st.sidebar.header("📂 데이터 설정")
    
    data_source = st.sidebar.radio("데이터 소스", ["샘플 데이터", "CSV 업로드"])
    
    if data_source == "CSV 업로드":
        uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # 날짜 컬럼 선택
            date_columns = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col].head())
                        date_columns.append(col)
                    except:
                        pass
            
            if not date_columns:
                st.error("날짜 컬럼을 찾을 수 없습니다. 날짜 형식이 올바른지 확인해주세요.")
                return
            
            date_col = st.sidebar.selectbox("날짜 컬럼 선택", date_columns)
            
            # 수치형 컬럼 선택
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_columns:
                st.error("수치형 컬럼이 없습니다.")
                return
            
            value_col = st.sidebar.selectbox("분석할 수치 컬럼 선택", numeric_columns)
            
            # 날짜 파싱
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.sort_values(date_col)
            
        else:
            st.info("CSV 파일을 업로드하거나 샘플 데이터를 사용하세요.")
            return
    else:
        # 샘플 데이터 사용
        df = create_sample_timeseries()
        date_col = 'date'
        value_col = 'value'
        st.info("샘플 시계열 데이터를 사용합니다.")
    
    # 데이터 미리보기
    if st.sidebar.checkbox("데이터 미리보기"):
        st.subheader("📊 데이터 미리보기")
        st.dataframe(df.head())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 데이터 포인트", len(df))
        with col2:
            st.metric("기간", f"{df[date_col].min().strftime('%Y-%m-%d')} ~ {df[date_col].max().strftime('%Y-%m-%d')}")
        with col3:
            st.metric("결측값", df[value_col].isnull().sum())
    
    # 시계열 데이터 준비
    ts_data = df.set_index(date_col)[value_col].sort_index()
    
    # 기본 시계열 플롯
    st.header("📊 시계열 데이터 시각화")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts_data.index,
        y=ts_data.values,
        mode='lines',
        name='시계열 데이터',
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title="원본 시계열 데이터",
        xaxis_title="날짜",
        yaxis_title="값",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    add_chart_export_section(fig, "timeseries_original")
    
    # 시계열 분해
    st.header("🔍 시계열 분해")
    
    decompose_freq = st.sidebar.selectbox(
        "분해 주기 선택",
        ["자동 감지", "일간 (7)", "월간 (30)", "연간 (365)", "커스텀"]
    )
    
    if decompose_freq == "커스텀":
        custom_freq = st.sidebar.number_input("커스텀 주기", min_value=2, max_value=len(ts_data)//2, value=7)
        period = custom_freq
    elif decompose_freq == "일간 (7)":
        period = 7
    elif decompose_freq == "월간 (30)":
        period = 30
    elif decompose_freq == "연간 (365)":
        period = 365
    else:
        # 자동 감지 (데이터 길이에 따라)
        if len(ts_data) > 730:  # 2년 이상
            period = 365
        elif len(ts_data) > 60:  # 2개월 이상
            period = 30
        else:
            period = 7
    
    try:
        decomposition = seasonal_decompose(ts_data.dropna(), model='additive', period=period)
        
        # 분해 결과 플롯
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['원본 데이터', '트렌드', '계절성', '잔차'],
            vertical_spacing=0.05
        )
        
        # 원본 데이터
        fig.add_trace(
            go.Scatter(x=decomposition.observed.index, y=decomposition.observed.values, 
                      name='원본', line=dict(color='blue')),
            row=1, col=1
        )
        
        # 트렌드
        fig.add_trace(
            go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, 
                      name='트렌드', line=dict(color='red')),
            row=2, col=1
        )
        
        # 계절성
        fig.add_trace(
            go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, 
                      name='계절성', line=dict(color='green')),
            row=3, col=1
        )
        
        # 잔차
        fig.add_trace(
            go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, 
                      name='잔차', line=dict(color='orange')),
            row=4, col=1
        )
        
        fig.update_layout(height=800, showlegend=False, title_text=f"시계열 분해 (주기: {period})")
        st.plotly_chart(fig, use_container_width=True)
        add_chart_export_section(fig, "timeseries_decomposition")
        
        # 분해 통계
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("트렌드 변화율", f"{((decomposition.trend.dropna().iloc[-1] - decomposition.trend.dropna().iloc[0]) / decomposition.trend.dropna().iloc[0] * 100):.2f}%")
        with col2:
            st.metric("계절성 강도", f"{decomposition.seasonal.std():.4f}")
        with col3:
            st.metric("잔차 표준편차", f"{decomposition.resid.std():.4f}")
        with col4:
            explained_variance = 1 - (decomposition.resid.var() / decomposition.observed.var())
            st.metric("설명된 분산", f"{explained_variance:.4f}")
            
    except Exception as e:
        st.error(f"시계열 분해 중 오류가 발생했습니다: {str(e)}")
    
    # 정상성 검정
    st.header("📏 정상성 검정")
    
    stationarity_results = perform_stationarity_test(ts_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ADF 검정 (Augmented Dickey-Fuller)")
        st.metric("검정 통계량", f"{stationarity_results['ADF']['statistic']:.4f}")
        st.metric("p-value", f"{stationarity_results['ADF']['p_value']:.4f}")
        
        if stationarity_results['ADF']['is_stationary']:
            st.success("✅ ADF 검정: 정상 시계열")
        else:
            st.warning("⚠️ ADF 검정: 비정상 시계열")
    
    with col2:
        st.subheader("KPSS 검정")
        st.metric("검정 통계량", f"{stationarity_results['KPSS']['statistic']:.4f}")
        st.metric("p-value", f"{stationarity_results['KPSS']['p_value']:.4f}")
        
        if stationarity_results['KPSS']['is_stationary']:
            st.success("✅ KPSS 검정: 정상 시계열")
        else:
            st.warning("⚠️ KPSS 검정: 비정상 시계열")
    
    # 차분 옵션
    st.subheader("🔄 차분을 통한 정상화")
    
    diff_order = st.selectbox("차분 차수", [0, 1, 2])
    
    if diff_order > 0:
        ts_diff = ts_data.diff(diff_order).dropna()
        
        # 차분된 데이터 플롯
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ts_diff.index,
            y=ts_diff.values,
            mode='lines',
            name=f'{diff_order}차 차분',
            line=dict(color='purple')
        ))
        
        fig.update_layout(
            title=f"{diff_order}차 차분된 시계열",
            xaxis_title="날짜",
            yaxis_title="값",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        add_chart_export_section(fig, f"timeseries_diff_{diff_order}")
        
        # 차분된 데이터의 정상성 검정
        diff_stationarity = perform_stationarity_test(ts_diff)
        
        col1, col2 = st.columns(2)
        with col1:
            if diff_stationarity['ADF']['is_stationary']:
                st.success(f"✅ {diff_order}차 차분 후 ADF: 정상 시계열")
            else:
                st.warning(f"⚠️ {diff_order}차 차분 후 ADF: 비정상 시계열")
        
        with col2:
            if diff_stationarity['KPSS']['is_stationary']:
                st.success(f"✅ {diff_order}차 차분 후 KPSS: 정상 시계열")
            else:
                st.warning(f"⚠️ {diff_order}차 차분 후 KPSS: 비정상 시계열")
    
    # ACF/PACF 분석
    st.header("📊 자기상관 분석")
    
    acf_lags = st.slider("ACF/PACF 지연(lag) 수", 10, min(100, len(ts_data)//4), 40, key="timeseries_acf_lags")
    
    # ACF/PACF 사용할 데이터 선택
    if diff_order > 0:
        analysis_data = ts_diff
        data_name = f"{diff_order}차 차분 데이터"
    else:
        analysis_data = ts_data
        data_name = "원본 데이터"
    
    acf_pacf_fig = plot_acf_pacf(analysis_data, lags=acf_lags)
    acf_pacf_fig.update_layout(title=f"{data_name}의 자기상관 분석")
    
    st.plotly_chart(acf_pacf_fig, use_container_width=True)
    add_chart_export_section(acf_pacf_fig, f"acf_pacf_{data_name}")
    
    # 예측 모델
    st.header("🔮 시계열 예측")
    
    prediction_method = st.selectbox(
        "예측 방법 선택",
        ["ARIMA", "지수평활법 (Exponential Smoothing)", "단순 이동평균"]
    )
    
    forecast_periods = st.number_input("예측 기간 (포인트)", min_value=1, max_value=100, value=30)
    
    if prediction_method == "ARIMA":
        st.subheader("ARIMA 모델")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.number_input("AR 차수 (p)", min_value=0, max_value=5, value=1)
        with col2:
            d = st.number_input("차분 차수 (d)", min_value=0, max_value=2, value=1)
        with col3:
            q = st.number_input("MA 차수 (q)", min_value=0, max_value=5, value=1)
        
        if st.button("ARIMA 모델 실행"):
            try:
                # ARIMA 모델 학습
                model = ARIMA(ts_data.dropna(), order=(p, d, q))
                fitted_model = model.fit()
                
                # 예측
                forecast = fitted_model.forecast(steps=forecast_periods)
                conf_int = fitted_model.get_forecast(steps=forecast_periods).conf_int()
                
                # 예측 날짜 생성
                last_date = ts_data.index[-1]
                freq = pd.infer_freq(ts_data.index)
                if freq is None:
                    freq = 'D'  # 기본값으로 일간 주기 사용
                
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=forecast_periods,
                    freq=freq
                )
                
                # 플롯
                fig = go.Figure()
                
                # 원본 데이터
                fig.add_trace(go.Scatter(
                    x=ts_data.index,
                    y=ts_data.values,
                    mode='lines',
                    name='원본 데이터',
                    line=dict(color='blue')
                ))
                
                # 예측값
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast.values,
                    mode='lines',
                    name='예측값',
                    line=dict(color='red', dash='dash')
                ))
                
                # 신뢰구간
                fig.add_trace(go.Scatter(
                    x=list(forecast_dates) + list(forecast_dates[::-1]),
                    y=list(conf_int.iloc[:, 1]) + list(conf_int.iloc[:, 0][::-1]),
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='95% 신뢰구간'
                ))
                
                fig.update_layout(
                    title=f"ARIMA({p},{d},{q}) 예측",
                    xaxis_title="날짜",
                    yaxis_title="값",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                add_chart_export_section(fig, f"arima_forecast_{p}_{d}_{q}")
                
                # 모델 통계
                st.subheader("모델 통계")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("AIC", f"{fitted_model.aic:.2f}")
                with col2:
                    st.metric("BIC", f"{fitted_model.bic:.2f}")
                with col3:
                    st.metric("Log-Likelihood", f"{fitted_model.llf:.2f}")
                
                # 예측 결과 테이블
                forecast_df = pd.DataFrame({
                    '날짜': forecast_dates,
                    '예측값': forecast.values,
                    '하한': conf_int.iloc[:, 0].values,
                    '상한': conf_int.iloc[:, 1].values
                })
                
                with st.expander("예측 결과 상세"):
                    st.dataframe(forecast_df)
                
            except Exception as e:
                st.error(f"ARIMA 모델 실행 중 오류가 발생했습니다: {str(e)}")
    
    elif prediction_method == "지수평활법 (Exponential Smoothing)":
        st.subheader("지수평활법")
        
        trend_type = st.selectbox("트렌드 유형", ["add", "mul", None])
        seasonal_type = st.selectbox("계절성 유형", ["add", "mul", None])
        seasonal_periods = st.number_input("계절 주기", min_value=2, max_value=365, value=12)
        
        if st.button("지수평활법 실행"):
            try:
                # 지수평활법 모델
                model = ExponentialSmoothing(
                    ts_data.dropna(),
                    trend=trend_type,
                    seasonal=seasonal_type,
                    seasonal_periods=seasonal_periods if seasonal_type else None
                )
                fitted_model = model.fit()
                
                # 예측
                forecast = fitted_model.forecast(steps=forecast_periods)
                
                # 예측 날짜 생성
                last_date = ts_data.index[-1]
                freq = pd.infer_freq(ts_data.index)
                if freq is None:
                    freq = 'D'
                
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=forecast_periods,
                    freq=freq
                )
                
                # 플롯
                fig = go.Figure()
                
                # 원본 데이터
                fig.add_trace(go.Scatter(
                    x=ts_data.index,
                    y=ts_data.values,
                    mode='lines',
                    name='원본 데이터',
                    line=dict(color='blue')
                ))
                
                # 예측값
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=forecast.values,
                    mode='lines',
                    name='예측값',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title="지수평활법 예측",
                    xaxis_title="날짜",
                    yaxis_title="값",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                add_chart_export_section(fig, "exponential_smoothing_forecast")
                
                # 모델 통계
                st.subheader("모델 통계")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("AIC", f"{fitted_model.aic:.2f}")
                with col2:
                    st.metric("SSE", f"{fitted_model.sse:.2f}")
                
            except Exception as e:
                st.error(f"지수평활법 실행 중 오류가 발생했습니다: {str(e)}")
    
    else:  # 단순 이동평균
        st.subheader("단순 이동평균")
        
        window_size = st.number_input("이동평균 윈도우 크기", min_value=2, max_value=len(ts_data)//2, value=30)
        
        if st.button("단순 이동평균 실행"):
            # 이동평균 계산
            ma = ts_data.rolling(window=window_size).mean()
            
            # 단순 예측 (마지막 이동평균값을 사용)
            last_ma = ma.iloc[-1]
            forecast = [last_ma] * forecast_periods
            
            # 예측 날짜 생성
            last_date = ts_data.index[-1]
            freq = pd.infer_freq(ts_data.index)
            if freq is None:
                freq = 'D'
            
            forecast_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1),
                periods=forecast_periods,
                freq=freq
            )
            
            # 플롯
            fig = go.Figure()
            
            # 원본 데이터
            fig.add_trace(go.Scatter(
                x=ts_data.index,
                y=ts_data.values,
                mode='lines',
                name='원본 데이터',
                line=dict(color='blue')
            ))
            
            # 이동평균
            fig.add_trace(go.Scatter(
                x=ma.index,
                y=ma.values,
                mode='lines',
                name=f'{window_size}일 이동평균',
                line=dict(color='green')
            ))
            
            # 예측값
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast,
                mode='lines',
                name='예측값',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f"{window_size}일 이동평균 예측",
                xaxis_title="날짜",
                yaxis_title="값",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            add_chart_export_section(fig, f"moving_average_forecast_{window_size}")

if __name__ == "__main__":
    main() 