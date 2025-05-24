import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import sys
import os

# utils.py 모듈 import를 위한 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import add_chart_export_section, style_metric_cards

st.title("🛠️ 데이터 전처리")

# 메트릭 카드 스타일 적용
style_metric_cards()
st.markdown("데이터 품질을 개선하고 분석에 적합하게 변환하는 다양한 전처리 기능을 제공합니다.")

uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])
if uploaded_file:
    # 원본 데이터 로드
    if 'original_df' not in st.session_state:
        st.session_state.original_df = pd.read_csv(uploaded_file)
    
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = st.session_state.original_df.copy()
    
    df_original = st.session_state.original_df
    df_processed = st.session_state.processed_df
    
    # 사이드바에서 전처리 옵션 선택
    st.sidebar.header("전처리 옵션")
    preprocessing_options = st.sidebar.multiselect(
        "적용할 전처리 선택",
        ["결측치 처리", "이상치 처리", "데이터 변환", "피처 엔지니어링", "데이터 필터링"],
        default=["결측치 처리"]
    )
    
    # 원본과 처리된 데이터 비교
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 원본 데이터")
        st.write(f"**크기**: {df_original.shape[0]}행 × {df_original.shape[1]}열")
        st.write(f"**결측치**: {df_original.isnull().sum().sum()}개")
        st.dataframe(df_original.head())
    
    with col2:
        st.markdown("### 🔧 처리된 데이터")
        st.write(f"**크기**: {df_processed.shape[0]}행 × {df_processed.shape[1]}열")
        st.write(f"**결측치**: {df_processed.isnull().sum().sum()}개")
        st.dataframe(df_processed.head())
    
    # 전처리 섹션들
    if "결측치 처리" in preprocessing_options:
        st.markdown("## 🚨 결측치 처리")
        
        # 결측치 현황 분석
        missing_summary = df_processed.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]
        
        if len(missing_summary) > 0:
            st.markdown("### 결측치 현황")
            
            # 결측치 시각화
            fig_missing = px.bar(
                x=missing_summary.values,
                y=missing_summary.index,
                orientation='h',
                title="변수별 결측치 개수",
                labels={'x': '결측치 개수', 'y': '변수명'}
            )
            st.plotly_chart(fig_missing, use_container_width=True)
            
            # 결측치 처리 방법 선택
            st.markdown("### 결측치 처리 방법")
            
            col1, col2 = st.columns(2)
            
            with col1:
                missing_strategy = st.selectbox(
                    "수치형 변수 결측치 처리",
                    ["삭제", "평균값", "중앙값", "최빈값", "앞값으로 채우기", "뒤값으로 채우기"]
                )
            
            with col2:
                categorical_strategy = st.selectbox(
                    "범주형 변수 결측치 처리",
                    ["삭제", "최빈값", "새로운 카테고리('Unknown')"]
                )
            
            if st.button("결측치 처리 적용"):
                df_temp = df_processed.copy()
                
                numeric_cols = df_temp.select_dtypes(include='number').columns
                categorical_cols = df_temp.select_dtypes(include=['object', 'category']).columns
                
                # 수치형 변수 처리
                for col in numeric_cols:
                    if df_temp[col].isnull().sum() > 0:
                        if missing_strategy == "삭제":
                            df_temp = df_temp.dropna(subset=[col])
                        elif missing_strategy == "평균값":
                            df_temp[col].fillna(df_temp[col].mean(), inplace=True)
                        elif missing_strategy == "중앙값":
                            df_temp[col].fillna(df_temp[col].median(), inplace=True)
                        elif missing_strategy == "최빈값":
                            df_temp[col].fillna(df_temp[col].mode().iloc[0] if not df_temp[col].mode().empty else 0, inplace=True)
                        elif missing_strategy == "앞값으로 채우기":
                            df_temp[col].fillna(method='ffill', inplace=True)
                        elif missing_strategy == "뒤값으로 채우기":
                            df_temp[col].fillna(method='bfill', inplace=True)
                
                # 범주형 변수 처리
                for col in categorical_cols:
                    if df_temp[col].isnull().sum() > 0:
                        if categorical_strategy == "삭제":
                            df_temp = df_temp.dropna(subset=[col])
                        elif categorical_strategy == "최빈값":
                            mode_val = df_temp[col].mode().iloc[0] if not df_temp[col].mode().empty else "Unknown"
                            df_temp[col].fillna(mode_val, inplace=True)
                        elif categorical_strategy == "새로운 카테고리('Unknown')":
                            df_temp[col].fillna("Unknown", inplace=True)
                
                st.session_state.processed_df = df_temp
                st.success("결측치 처리가 완료되었습니다!")
                st.rerun()
        
        else:
            st.success("✅ 결측치가 없습니다!")
    
    if "이상치 처리" in preprocessing_options:
        st.markdown("## 📊 이상치 처리")
        
        numeric_cols = df_processed.select_dtypes(include='number').columns.tolist()
        
        if numeric_cols:
            # 이상치 탐지 방법 선택
            outlier_method = st.selectbox(
                "이상치 탐지 방법",
                ["IQR 방법", "Z-Score 방법", "수정된 Z-Score 방법"]
            )
            
            selected_col = st.selectbox("분석할 변수 선택", numeric_cols)
            
            if selected_col:
                data = df_processed[selected_col].dropna()
                
                # 이상치 탐지
                if outlier_method == "IQR 방법":
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = data[(data < lower_bound) | (data > upper_bound)]
                    
                elif outlier_method == "Z-Score 방법":
                    z_scores = np.abs(stats.zscore(data))
                    threshold = st.slider("Z-Score 임계값", 2.0, 4.0, 3.0, 0.1, key="zscore_threshold")
                    outliers = data[z_scores > threshold]
                    
                elif outlier_method == "수정된 Z-Score 방법":
                    median = data.median()
                    mad = np.median(np.abs(data - median))
                    modified_z_scores = 0.6745 * (data - median) / mad
                    threshold = st.slider("수정된 Z-Score 임계값", 2.0, 4.0, 3.5, 0.1, key="modified_zscore_threshold")
                    outliers = data[np.abs(modified_z_scores) > threshold]
                
                # 이상치 시각화
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_box = px.box(y=data, title=f"{selected_col} 박스플롯")
                    st.plotly_chart(fig_box, use_container_width=True)
                
                with col2:
                    fig_hist = px.histogram(x=data, nbins=30, title=f"{selected_col} 분포")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # 이상치 정보
                st.write(f"**탐지된 이상치 개수**: {len(outliers)}개 ({len(outliers)/len(data)*100:.1f}%)")
                
                if len(outliers) > 0:
                    st.write(f"**이상치 값**: {sorted(outliers.values)}")
                    
                    # 이상치 처리 방법
                    outlier_action = st.selectbox(
                        "이상치 처리 방법",
                        ["제거", "평균값으로 대체", "중앙값으로 대체", "경계값으로 대체"]
                    )
                    
                    if st.button("이상치 처리 적용"):
                        df_temp = df_processed.copy()
                        
                        if outlier_method == "IQR 방법":
                            outlier_mask = (df_temp[selected_col] < lower_bound) | (df_temp[selected_col] > upper_bound)
                        elif outlier_method == "Z-Score 방법":
                            z_scores = np.abs(stats.zscore(df_temp[selected_col].dropna()))
                            outlier_mask = z_scores > threshold
                        else:  # 수정된 Z-Score
                            median = df_temp[selected_col].median()
                            mad = np.median(np.abs(df_temp[selected_col] - median))
                            modified_z_scores = 0.6745 * (df_temp[selected_col] - median) / mad
                            outlier_mask = np.abs(modified_z_scores) > threshold
                        
                        if outlier_action == "제거":
                            df_temp = df_temp[~outlier_mask]
                        elif outlier_action == "평균값으로 대체":
                            df_temp.loc[outlier_mask, selected_col] = df_temp[selected_col].mean()
                        elif outlier_action == "중앙값으로 대체":
                            df_temp.loc[outlier_mask, selected_col] = df_temp[selected_col].median()
                        elif outlier_action == "경계값으로 대체":
                            if outlier_method == "IQR 방법":
                                df_temp.loc[df_temp[selected_col] < lower_bound, selected_col] = lower_bound
                                df_temp.loc[df_temp[selected_col] > upper_bound, selected_col] = upper_bound
                        
                        st.session_state.processed_df = df_temp
                        st.success("이상치 처리가 완료되었습니다!")
                        st.rerun()
                
                else:
                    st.success("✅ 이상치가 탐지되지 않았습니다!")
        
        else:
            st.info("수치형 변수가 없어서 이상치 분석을 수행할 수 없습니다.")
    
    if "데이터 변환" in preprocessing_options:
        st.markdown("## 🔄 데이터 변환")
        
        numeric_cols = df_processed.select_dtypes(include='number').columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 수치형 데이터 스케일링
        if numeric_cols:
            st.markdown("### 수치형 데이터 스케일링")
            
            scaling_method = st.selectbox(
                "스케일링 방법",
                ["없음", "표준화 (StandardScaler)", "정규화 (MinMaxScaler)", "로그 변환"]
            )
            
            scaling_cols = st.multiselect(
                "스케일링할 변수 선택",
                numeric_cols,
                default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
            )
            
            if scaling_cols and scaling_method != "없음" and st.button("스케일링 적용"):
                df_temp = df_processed.copy()
                
                if scaling_method == "표준화 (StandardScaler)":
                    scaler = StandardScaler()
                    df_temp[scaling_cols] = scaler.fit_transform(df_temp[scaling_cols])
                    
                elif scaling_method == "정규화 (MinMaxScaler)":
                    scaler = MinMaxScaler()
                    df_temp[scaling_cols] = scaler.fit_transform(df_temp[scaling_cols])
                    
                elif scaling_method == "로그 변환":
                    for col in scaling_cols:
                        # 양수값만 로그 변환 (0이나 음수가 있으면 경고)
                        if (df_temp[col] <= 0).any():
                            st.warning(f"{col}에 0 이하의 값이 있어 로그 변환을 건너뜁니다.")
                        else:
                            df_temp[col] = np.log(df_temp[col])
                
                st.session_state.processed_df = df_temp
                st.success(f"{scaling_method} 변환이 완료되었습니다!")
                st.rerun()
        
        # 범주형 데이터 인코딩
        if categorical_cols:
            st.markdown("### 범주형 데이터 인코딩")
            
            encoding_method = st.selectbox(
                "인코딩 방법",
                ["없음", "라벨 인코딩", "원-핫 인코딩"]
            )
            
            encoding_cols = st.multiselect(
                "인코딩할 변수 선택",
                categorical_cols,
                default=categorical_cols[:2] if len(categorical_cols) >= 2 else categorical_cols
            )
            
            if encoding_cols and encoding_method != "없음" and st.button("인코딩 적용"):
                df_temp = df_processed.copy()
                
                if encoding_method == "라벨 인코딩":
                    for col in encoding_cols:
                        le = LabelEncoder()
                        df_temp[col] = le.fit_transform(df_temp[col].astype(str))
                        
                elif encoding_method == "원-핫 인코딩":
                    df_temp = pd.get_dummies(df_temp, columns=encoding_cols, prefix=encoding_cols)
                
                st.session_state.processed_df = df_temp
                st.success(f"{encoding_method}이 완료되었습니다!")
                st.rerun()
    
    if "피처 엔지니어링" in preprocessing_options:
        st.markdown("## ⚙️ 피처 엔지니어링")
        
        numeric_cols = df_processed.select_dtypes(include='number').columns.tolist()
        
        if len(numeric_cols) >= 2:
            st.markdown("### 새로운 변수 생성")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                var1 = st.selectbox("첫 번째 변수", numeric_cols, key="fe_var1")
            with col2:
                operation = st.selectbox("연산", ["+", "-", "*", "/", "평균"])
            with col3:
                var2 = st.selectbox("두 번째 변수", numeric_cols, key="fe_var2")
            
            new_var_name = st.text_input("새 변수명", value=f"{var1}_{operation}_{var2}")
            
            if st.button("새 변수 생성") and var1 != var2:
                df_temp = df_processed.copy()
                
                if operation == "+":
                    df_temp[new_var_name] = df_temp[var1] + df_temp[var2]
                elif operation == "-":
                    df_temp[new_var_name] = df_temp[var1] - df_temp[var2]
                elif operation == "*":
                    df_temp[new_var_name] = df_temp[var1] * df_temp[var2]
                elif operation == "/":
                    df_temp[new_var_name] = df_temp[var1] / (df_temp[var2] + 1e-8)  # NOTE: 0으로 나누기 방지
                elif operation == "평균":
                    df_temp[new_var_name] = (df_temp[var1] + df_temp[var2]) / 2
                
                st.session_state.processed_df = df_temp
                st.success(f"새 변수 '{new_var_name}'이 생성되었습니다!")
                st.rerun()
        
        # 범주형 변수의 빈도 기반 새 변수
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            st.markdown("### 범주형 변수 빈도 기반 변수")
            
            freq_col = st.selectbox("빈도를 계산할 범주형 변수", categorical_cols)
            
            if st.button("빈도 변수 생성"):
                df_temp = df_processed.copy()
                freq_map = df_temp[freq_col].value_counts().to_dict()
                df_temp[f"{freq_col}_frequency"] = df_temp[freq_col].map(freq_map)
                
                st.session_state.processed_df = df_temp
                st.success(f"'{freq_col}_frequency' 변수가 생성되었습니다!")
                st.rerun()
    
    if "데이터 필터링" in preprocessing_options:
        st.markdown("## 🔍 데이터 필터링")
        
        # 조건부 필터링
        st.markdown("### 조건부 데이터 필터링")
        
        all_cols = df_processed.columns.tolist()
        numeric_cols = df_processed.select_dtypes(include='number').columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        filter_col = st.selectbox("필터링할 변수", all_cols)
        
        if filter_col in numeric_cols:
            col_min = float(df_processed[filter_col].min())
            col_max = float(df_processed[filter_col].max())
            
            filter_range = st.slider(
                f"{filter_col} 범위 선택",
                min_value=col_min,
                max_value=col_max,
                value=(col_min, col_max),
                key="data_filter_range"
            )
            
            if st.button("수치형 필터 적용"):
                mask = (df_processed[filter_col] >= filter_range[0]) & (df_processed[filter_col] <= filter_range[1])
                df_temp = df_processed[mask]
                
                st.session_state.processed_df = df_temp
                st.success(f"필터링 완료: {len(df_temp)}개 행이 선택되었습니다.")
                st.rerun()
        
        elif filter_col in categorical_cols:
            unique_values = df_processed[filter_col].unique().tolist()
            selected_values = st.multiselect(
                f"{filter_col} 값 선택",
                unique_values,
                default=unique_values
            )
            
            if st.button("범주형 필터 적용"):
                mask = df_processed[filter_col].isin(selected_values)
                df_temp = df_processed[mask]
                
                st.session_state.processed_df = df_temp
                st.success(f"필터링 완료: {len(df_temp)}개 행이 선택되었습니다.")
                st.rerun()
    
    # 전처리 결과 다운로드
    st.markdown("## 💾 전처리 결과 저장")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("원본 데이터로 되돌리기"):
            st.session_state.processed_df = st.session_state.original_df.copy()
            st.success("원본 데이터로 복원되었습니다!")
            st.rerun()
    
    with col2:
        csv = df_processed.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="처리된 데이터 다운로드",
            data=csv,
            file_name='processed_data.csv',
            mime='text/csv'
        )
    
    with col3:
        if st.button("전처리 요약 보기"):
            st.markdown("### 📊 전처리 요약")
            
            summary_data = {
                "항목": ["원본 행 수", "처리된 행 수", "원본 열 수", "처리된 열 수", "원본 결측치", "처리된 결측치"],
                "값": [
                    df_original.shape[0],
                    df_processed.shape[0],
                    df_original.shape[1],
                    df_processed.shape[1],
                    df_original.isnull().sum().sum(),
                    df_processed.isnull().sum().sum()
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df)

else:
    st.info("CSV 파일을 업로드하여 데이터 전처리를 시작하세요!")
    st.markdown("""
    ### 전처리 기능 미리보기
    
    **결측치 처리**
    - 다양한 대체 방법 (평균, 중앙값, 최빈값 등)
    - 수치형/범주형 변수별 맞춤 처리
    
    **이상치 처리**
    - IQR, Z-Score, 수정된 Z-Score 방법
    - 시각적 이상치 탐지 및 처리
    
    **데이터 변환**
    - 스케일링 (표준화, 정규화)
    - 로그 변환
    - 범주형 인코딩 (라벨, 원-핫)
    
    **피처 엔지니어링**
    - 변수 간 연산으로 새 변수 생성
    - 빈도 기반 변수 생성
    
    **데이터 필터링**
    - 조건부 데이터 선택
    - 범위 기반 필터링
    """)

# TODO: 자동 전처리 파이프라인, 전처리 이력 추적 기능 추가 예정 