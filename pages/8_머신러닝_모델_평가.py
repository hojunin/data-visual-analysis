import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, validation_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from utils import add_chart_export_section

def main():
    st.title("🤖 머신러닝 모델 평가")
    st.markdown("다양한 머신러닝 모델의 성능을 평가하고 비교해보세요.")
    
    # 사이드바
    st.sidebar.header("📂 데이터 업로드")
    uploaded_file = st.sidebar.file_uploader("CSV 파일을 업로드하세요", type=['csv'])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        # 기본 테스트 데이터 사용
        st.info("샘플 데이터를 사용합니다. CSV 파일을 업로드하여 자신의 데이터를 분석해보세요.")
        df = pd.read_csv('test.csv')
    
    st.sidebar.markdown("---")
    
    # 데이터 미리보기
    if st.sidebar.checkbox("데이터 미리보기"):
        st.subheader("📊 데이터 미리보기")
        st.dataframe(df.head())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("행 수", len(df))
        with col2:
            st.metric("열 수", len(df.columns))
        with col3:
            st.metric("결측값", df.isnull().sum().sum())
    
    # 모델링 설정
    st.sidebar.header("🎯 모델링 설정")
    
    # 수치형 컬럼만 선택
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        st.error("최소 2개 이상의 수치형 컬럼이 필요합니다.")
        return
    
    # 타겟 변수 선택
    target_col = st.sidebar.selectbox("타겟 변수 선택", numeric_columns)
    
    # 피처 변수 선택
    feature_cols = st.sidebar.multiselect(
        "피처 변수 선택", 
        [col for col in numeric_columns if col != target_col],
        default=[col for col in numeric_columns if col != target_col][:5]
    )
    
    if not feature_cols:
        st.warning("최소 하나의 피처 변수를 선택해주세요.")
        return
    
    # 모델 유형 선택
    model_type = st.sidebar.radio("모델 유형", ["회귀", "분류"])
    
    # 데이터 전처리
    X = df[feature_cols].dropna()
    y = df[target_col].dropna()
    
    # 결측값이 있는 행 제거
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    
    if len(X) == 0:
        st.error("유효한 데이터가 없습니다. 결측값을 확인해주세요.")
        return
    
    # 분류 문제의 경우 타겟 변수 이진화
    if model_type == "분류":
        if len(y.unique()) > 10:
            # 연속형 변수를 이진 분류로 변환 (중앙값 기준)
            threshold = y.median()
            y = (y > threshold).astype(int)
            st.info(f"타겟 변수를 중앙값({threshold:.2f}) 기준으로 이진 분류로 변환했습니다.")
        else:
            # 카테고리형 변수 인코딩
            le = LabelEncoder()
            y = le.fit_transform(y)
    
    # 데이터 분할
    test_size = st.sidebar.slider("테스트 데이터 비율", 0.1, 0.5, 0.2, 0.05, key="ml_test_size")
    random_state = st.sidebar.number_input("랜덤 시드", 0, 1000, 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # 피처 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델 선택
    if model_type == "회귀":
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=random_state),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
            'SVM': SVR(kernel='rbf'),
            'Decision Tree': DecisionTreeRegressor(random_state=random_state),
            'KNN': KNeighborsRegressor(n_neighbors=5)
        }
        scoring = 'neg_mean_squared_error'
    else:
        models = {
            'Logistic Regression': LogisticRegression(random_state=random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
            'SVM': SVC(kernel='rbf', probability=True, random_state=random_state),
            'Decision Tree': DecisionTreeClassifier(random_state=random_state),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB()
        }
        scoring = 'accuracy'
    
    # 모델 선택
    selected_models = st.sidebar.multiselect(
        "비교할 모델 선택",
        list(models.keys()),
        default=list(models.keys())[:4]
    )
    
    if not selected_models:
        st.warning("최소 하나의 모델을 선택해주세요.")
        return
    
    # 분석 시작
    st.header("🔍 모델 성능 비교")
    
    # 모델 학습 및 평가
    results = {}
    trained_models = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, model_name in enumerate(selected_models):
        status_text.text(f"모델 학습 중: {model_name}")
        
        model = models[model_name]
        
        # 스케일링이 필요한 모델들
        if model_name in ['SVM', 'Logistic Regression', 'KNN']:
            X_train_model = X_train_scaled
            X_test_model = X_test_scaled
        else:
            X_train_model = X_train
            X_test_model = X_test
        
        # 모델 학습
        model.fit(X_train_model, y_train)
        trained_models[model_name] = model
        
        # 예측
        y_pred = model.predict(X_test_model)
        
        # 교차 검증
        cv_scores = cross_val_score(
            model, X_train_model, y_train, 
            cv=5, scoring=scoring
        )
        
        if model_type == "회귀":
            # 회귀 모델 평가
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[model_name] = {
                'MSE': mse,
                'MAE': mae,
                'R²': r2,
                'RMSE': np.sqrt(mse),
                'CV Score (avg)': -cv_scores.mean(),
                'CV Score (std)': cv_scores.std(),
                'Predictions': y_pred
            }
        else:
            # 분류 모델 평가
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            results[model_name] = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'CV Score (avg)': cv_scores.mean(),
                'CV Score (std)': cv_scores.std(),
                'Predictions': y_pred
            }
        
        progress_bar.progress((i + 1) / len(selected_models))
    
    status_text.text("모델 학습 완료!")
    
    # 결과 표시
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 모델 성능 비교표")
        results_df = pd.DataFrame(results).T
        st.dataframe(results_df.round(4))
    
    with col2:
        st.subheader("🏆 최고 성능 모델")
        if model_type == "회귀":
            best_model = results_df['R²'].idxmax()
            st.metric("최고 R² 모델", best_model, f"{results_df.loc[best_model, 'R²']:.4f}")
        else:
            best_model = results_df['Accuracy'].idxmax()
            st.metric("최고 정확도 모델", best_model, f"{results_df.loc[best_model, 'Accuracy']:.4f}")
    
    # 성능 비교 차트
    st.subheader("📈 성능 비교 시각화")
    
    if model_type == "회귀":
        # 회귀 모델 성능 비교
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['R² Score', 'Mean Squared Error', 'Mean Absolute Error', 'Cross-Validation Scores'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        models_list = list(results.keys())
        
        # R² Score
        r2_scores = [results[m]['R²'] for m in models_list]
        fig.add_trace(go.Bar(x=models_list, y=r2_scores, name='R²', marker_color='lightblue'), row=1, col=1)
        
        # MSE
        mse_scores = [results[m]['MSE'] for m in models_list]
        fig.add_trace(go.Bar(x=models_list, y=mse_scores, name='MSE', marker_color='lightcoral'), row=1, col=2)
        
        # MAE
        mae_scores = [results[m]['MAE'] for m in models_list]
        fig.add_trace(go.Bar(x=models_list, y=mae_scores, name='MAE', marker_color='lightgreen'), row=2, col=1)
        
        # CV Scores
        cv_means = [results[m]['CV Score (avg)'] for m in models_list]
        cv_stds = [results[m]['CV Score (std)'] for m in models_list]
        fig.add_trace(go.Bar(x=models_list, y=cv_means, error_y=dict(type='data', array=cv_stds), 
                            name='CV RMSE', marker_color='lightyellow'), row=2, col=2)
        
    else:
        # 분류 모델 성능 비교
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        models_list = list(results.keys())
        
        # Accuracy
        accuracy_scores = [results[m]['Accuracy'] for m in models_list]
        fig.add_trace(go.Bar(x=models_list, y=accuracy_scores, name='Accuracy', marker_color='lightblue'), row=1, col=1)
        
        # Precision
        precision_scores = [results[m]['Precision'] for m in models_list]
        fig.add_trace(go.Bar(x=models_list, y=precision_scores, name='Precision', marker_color='lightcoral'), row=1, col=2)
        
        # Recall
        recall_scores = [results[m]['Recall'] for m in models_list]
        fig.add_trace(go.Bar(x=models_list, y=recall_scores, name='Recall', marker_color='lightgreen'), row=2, col=1)
        
        # F1-Score
        f1_scores = [results[m]['F1-Score'] for m in models_list]
        fig.add_trace(go.Bar(x=models_list, y=f1_scores, name='F1-Score', marker_color='lightyellow'), row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False, title_text="모델 성능 종합 비교")
    st.plotly_chart(fig, use_container_width=True)
    add_chart_export_section(fig, "model_performance_comparison")
    
    # 예측 vs 실제값 비교
    st.subheader("🎯 예측 vs 실제값 비교")
    
    comparison_model = st.selectbox("비교할 모델 선택", selected_models)
    
    if model_type == "회귀":
        # 산점도
        fig = go.Figure()
        
        y_pred_comparison = results[comparison_model]['Predictions']
        
        fig.add_trace(go.Scatter(
            x=y_test, y=y_pred_comparison,
            mode='markers',
            name='예측값',
            marker=dict(color='blue', opacity=0.6)
        ))
        
        # 완벽한 예측선 (y=x)
        min_val = min(y_test.min(), y_pred_comparison.min())
        max_val = max(y_test.max(), y_pred_comparison.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines',
            name='완벽한 예측',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f"{comparison_model}: 예측값 vs 실제값",
            xaxis_title="실제값",
            yaxis_title="예측값",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        add_chart_export_section(fig, f"prediction_vs_actual_{comparison_model}")
        
    else:
        # 혼동 행렬
        y_pred_comparison = results[comparison_model]['Predictions']
        cm = confusion_matrix(y_test, y_pred_comparison)
        
        fig = px.imshow(
            cm,
            labels=dict(x="예측값", y="실제값", color="개수"),
            x=['클래스 0', '클래스 1'],
            y=['클래스 0', '클래스 1'],
            color_continuous_scale='Blues',
            text_auto=True
        )
        fig.update_layout(title=f"{comparison_model}: 혼동 행렬")
        
        st.plotly_chart(fig, use_container_width=True)
        add_chart_export_section(fig, f"confusion_matrix_{comparison_model}")
        
        # ROC 곡선 (이진 분류인 경우)
        if len(np.unique(y)) == 2:
            st.subheader("📈 ROC 곡선")
            
            fig = go.Figure()
            
            for model_name in selected_models:
                model = trained_models[model_name]
                
                # 스케일링이 필요한 모델들
                if model_name in ['SVM', 'Logistic Regression', 'KNN']:
                    X_test_model = X_test_scaled
                else:
                    X_test_model = X_test
                
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test_model)[:, 1]
                elif hasattr(model, "decision_function"):
                    y_prob = model.decision_function(X_test_model)
                else:
                    continue
                
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_score = auc(fpr, tpr)
                
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC = {auc_score:.3f})'
                ))
            
            # 대각선 (랜덤 분류기)
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='gray', dash='dash')
            ))
            
            fig.update_layout(
                title="ROC 곡선 비교",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            add_chart_export_section(fig, "roc_curves_comparison")
    
    # 학습 곡선
    st.subheader("📚 학습 곡선")
    
    learning_model = st.selectbox("학습 곡선을 볼 모델 선택", selected_models, key="learning_curve")
    
    model = models[learning_model]
    
    # 스케일링이 필요한 모델들
    if learning_model in ['SVM', 'Logistic Regression', 'KNN']:
        X_learning = X_train_scaled
    else:
        X_learning = X_train
    
    # 학습 곡선 계산
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_learning, y_train,
        cv=5, scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=random_state
    )
    
    # 점수가 음수인 경우 (MSE) 양수로 변환
    if scoring == 'neg_mean_squared_error':
        train_scores = -train_scores
        val_scores = -val_scores
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig = go.Figure()
    
    # 훈련 점수
    fig.add_trace(go.Scatter(
        x=train_sizes, y=train_mean,
        mode='lines+markers',
        name='Training Score',
        line=dict(color='blue'),
        error_y=dict(type='data', array=train_std, visible=True)
    ))
    
    # 검증 점수
    fig.add_trace(go.Scatter(
        x=train_sizes, y=val_mean,
        mode='lines+markers',
        name='Validation Score',
        line=dict(color='red'),
        error_y=dict(type='data', array=val_std, visible=True)
    ))
    
    score_name = 'Accuracy' if model_type == "분류" else 'RMSE' if scoring == 'neg_mean_squared_error' else 'Score'
    
    fig.update_layout(
        title=f"{learning_model}: 학습 곡선",
        xaxis_title="훈련 샘플 수",
        yaxis_title=score_name,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    add_chart_export_section(fig, f"learning_curve_{learning_model}")
    
    # 모델별 상세 결과
    with st.expander("📋 모델별 상세 결과"):
        for model_name in selected_models:
            st.subheader(f"🔍 {model_name} 상세 결과")
            
            result = results[model_name]
            
            if model_type == "회귀":
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("R² Score", f"{result['R²']:.4f}")
                with col2:
                    st.metric("RMSE", f"{result['RMSE']:.4f}")
                with col3:
                    st.metric("MAE", f"{result['MAE']:.4f}")
                with col4:
                    st.metric("CV RMSE", f"{result['CV Score (avg)']:.4f} ± {result['CV Score (std)']:.4f}")
            else:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Accuracy", f"{result['Accuracy']:.4f}")
                with col2:
                    st.metric("Precision", f"{result['Precision']:.4f}")
                with col3:
                    st.metric("Recall", f"{result['Recall']:.4f}")
                with col4:
                    st.metric("F1-Score", f"{result['F1-Score']:.4f}")
                
                # 분류 리포트
                model = trained_models[model_name]
                if model_name in ['SVM', 'Logistic Regression', 'KNN']:
                    X_test_model = X_test_scaled
                else:
                    X_test_model = X_test
                
                y_pred = model.predict(X_test_model)
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.round(4))
            
            st.markdown("---")

if __name__ == "__main__":
    main() 