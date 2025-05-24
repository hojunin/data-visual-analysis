import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from utils import add_chart_export_section

def find_optimal_clusters(X, max_k=10, method='kmeans'):
    """최적의 클러스터 수를 찾기 위한 분석"""
    if method == 'kmeans':
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(X)))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, cluster_labels))
        
        return k_range, inertias, silhouette_scores
    
    return None, None, None

def perform_pca_analysis(X, n_components=2):
    """PCA 분석 수행"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    return X_pca, explained_variance, cumulative_variance, pca

def perform_tsne_analysis(X, n_components=2, perplexity=30):
    """t-SNE 분석 수행"""
    # 데이터가 너무 크면 샘플링
    if len(X) > 1000:
        sample_idx = np.random.choice(len(X), 1000, replace=False)
        X_sample = X[sample_idx]
    else:
        X_sample = X
        sample_idx = np.arange(len(X))
    
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    X_tsne = tsne.fit_transform(X_sample)
    
    return X_tsne, sample_idx

def plot_cluster_results(X_reduced, labels, title, feature_names=None, method_name=""):
    """클러스터링 결과를 시각화"""
    if X_reduced.shape[1] == 2:
        fig = px.scatter(
            x=X_reduced[:, 0], y=X_reduced[:, 1],
            color=labels.astype(str),
            title=title,
            labels={'x': f'{method_name} Component 1', 'y': f'{method_name} Component 2'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
    else:
        # 3D 플롯
        fig = px.scatter_3d(
            x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2],
            color=labels.astype(str),
            title=title,
            labels={'x': f'{method_name} Component 1', 'y': f'{method_name} Component 2', 'z': f'{method_name} Component 3'},
            color_discrete_sequence=px.colors.qualitative.Set3
        )
    
    fig.update_layout(height=600)
    return fig

def main():
    st.title("🧩 클러스터링 분석")
    st.markdown("데이터의 숨겨진 패턴을 발견하고 유사한 데이터 포인트들을 그룹화해보세요.")
    
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
    
    # 클러스터링 설정
    st.sidebar.header("🎯 클러스터링 설정")
    
    # 수치형 컬럼만 선택
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_columns) < 2:
        st.error("최소 2개 이상의 수치형 컬럼이 필요합니다.")
        return
    
    # 피처 선택
    selected_features = st.sidebar.multiselect(
        "클러스터링에 사용할 피처 선택",
        numeric_columns,
        default=numeric_columns[:min(5, len(numeric_columns))]
    )
    
    if len(selected_features) < 2:
        st.warning("최소 2개 이상의 피처를 선택해주세요.")
        return
    
    # 데이터 전처리
    X = df[selected_features].dropna()
    
    if len(X) == 0:
        st.error("유효한 데이터가 없습니다. 결측값을 확인해주세요.")
        return
    
    # 스케일링 방법 선택
    scaling_method = st.sidebar.selectbox(
        "데이터 스케일링 방법",
        ["StandardScaler", "MinMaxScaler", "스케일링 없음"]
    )
    
    if scaling_method == "StandardScaler":
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    elif scaling_method == "MinMaxScaler":
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X.values
    
    # 차원 축소 방법 선택
    st.sidebar.header("📐 차원 축소")
    dim_reduction = st.sidebar.selectbox(
        "차원 축소 방법",
        ["없음", "PCA", "t-SNE"]
    )
    
    n_components = st.sidebar.slider(
        "축소할 차원 수",
        2, min(10, len(selected_features)),
        2 if dim_reduction != "없음" else len(selected_features),
        key="clustering_n_components"
    )
    
    # 차원 축소 수행
    if dim_reduction == "PCA":
        X_reduced, explained_var, cumulative_var, pca_model = perform_pca_analysis(X_scaled, n_components)
        reduction_method_name = "PCA"
    elif dim_reduction == "t-SNE":
        perplexity = st.sidebar.slider("t-SNE Perplexity", 5, 50, 30, key="clustering_perplexity")
        X_reduced, sample_idx = perform_tsne_analysis(X_scaled, n_components, perplexity)
        reduction_method_name = "t-SNE"
        # t-SNE의 경우 원본 데이터도 샘플링된 것을 사용
        X_scaled = X_scaled[sample_idx]
        X = X.iloc[sample_idx]
    else:
        X_reduced = X_scaled
        reduction_method_name = "Original"
    
    # 차원 축소 결과 시각화
    if dim_reduction == "PCA":
        st.header("📊 PCA 분석 결과")
        
        # 설명된 분산 비율
        fig_var = go.Figure()
        fig_var.add_trace(go.Bar(
            x=[f'PC{i+1}' for i in range(len(explained_var))],
            y=explained_var,
            name='개별 설명된 분산',
            marker_color='lightblue'
        ))
        fig_var.add_trace(go.Scatter(
            x=[f'PC{i+1}' for i in range(len(cumulative_var))],
            y=cumulative_var,
            mode='lines+markers',
            name='누적 설명된 분산',
            yaxis='y2',
            line=dict(color='red')
        ))
        
        fig_var.update_layout(
            title="PCA 설명된 분산 비율",
            xaxis_title="주성분",
            yaxis_title="설명된 분산 비율",
            yaxis2=dict(
                title="누적 설명된 분산 비율",
                overlaying='y',
                side='right'
            ),
            height=400
        )
        
        st.plotly_chart(fig_var, use_container_width=True)
        add_chart_export_section(fig_var, "pca_explained_variance")
        
        # PCA 로딩 행렬
        if n_components <= 3:
            st.subheader("PCA 로딩 행렬")
            loadings = pca_model.components_.T
            loading_df = pd.DataFrame(
                loadings,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=selected_features
            )
            st.dataframe(loading_df.round(4))
    
    # 클러스터링 방법 선택
    st.sidebar.header("🎲 클러스터링 알고리즘")
    clustering_method = st.sidebar.selectbox(
        "클러스터링 방법",
        ["K-Means", "DBSCAN", "Gaussian Mixture", "Agglomerative Clustering"]
    )
    
    # 클러스터링 수행
    st.header("🔍 클러스터링 분석")
    
    if clustering_method == "K-Means":
        st.subheader("K-Means 클러스터링")
        
        # 최적 클러스터 수 찾기
        if st.checkbox("최적 클러스터 수 분석"):
            max_k = st.slider("최대 클러스터 수", 3, 15, 10, key="clustering_max_k")
            
            with st.spinner("최적 클러스터 수를 분석 중..."):
                k_range, inertias, silhouette_scores = find_optimal_clusters(X_reduced, max_k, 'kmeans')
            
            if k_range is not None:
                # 엘보우 방법과 실루엣 점수
                fig_elbow = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=['엘보우 방법 (Inertia)', '실루엣 점수'],
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # 엘보우 방법
                fig_elbow.add_trace(
                    go.Scatter(x=list(k_range), y=inertias, mode='lines+markers',
                              name='Inertia', line=dict(color='blue')),
                    row=1, col=1
                )
                
                # 실루엣 점수
                fig_elbow.add_trace(
                    go.Scatter(x=list(k_range), y=silhouette_scores, mode='lines+markers',
                              name='Silhouette Score', line=dict(color='red')),
                    row=1, col=2
                )
                
                fig_elbow.update_layout(height=400, showlegend=False)
                fig_elbow.update_xaxes(title_text="클러스터 수 (k)")
                
                st.plotly_chart(fig_elbow, use_container_width=True)
                add_chart_export_section(fig_elbow, "kmeans_optimal_clusters")
                
                # 추천 클러스터 수
                best_k_silhouette = k_range[np.argmax(silhouette_scores)]
                st.info(f"실루엣 점수 기준 추천 클러스터 수: {best_k_silhouette}")
        
        # K-Means 파라미터
        n_clusters = st.slider("클러스터 수", 2, 10, 3, key="kmeans_n_clusters")
        random_state = st.number_input("랜덤 시드", 0, 1000, 42)
        
        # K-Means 실행
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(X_reduced)
        
        # 성능 지표
        silhouette_avg = silhouette_score(X_reduced, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X_reduced, cluster_labels)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("클러스터 수", n_clusters)
        with col2:
            st.metric("실루엣 점수", f"{silhouette_avg:.4f}")
        with col3:
            st.metric("Calinski-Harabasz 지수", f"{calinski_harabasz:.2f}")
    
    elif clustering_method == "DBSCAN":
        st.subheader("DBSCAN 클러스터링")
        
        # DBSCAN 파라미터
        eps = st.slider("Epsilon (eps)", 0.1, 2.0, 0.5, 0.1, key="dbscan_eps")
        min_samples = st.slider("최소 샘플 수", 2, 20, 5, key="dbscan_min_samples")
        
        # DBSCAN 실행
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(X_reduced)
        
        # 노이즈 포인트 수
        n_noise = list(cluster_labels).count(-1)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("발견된 클러스터 수", n_clusters)
        with col2:
            st.metric("노이즈 포인트", n_noise)
        with col3:
            if n_clusters > 1:
                # 노이즈 포인트 제외하고 실루엣 점수 계산
                mask = cluster_labels != -1
                if np.sum(mask) > 1 and len(set(cluster_labels[mask])) > 1:
                    silhouette_avg = silhouette_score(X_reduced[mask], cluster_labels[mask])
                    st.metric("실루엣 점수", f"{silhouette_avg:.4f}")
                else:
                    st.metric("실루엣 점수", "계산 불가")
            else:
                st.metric("실루엣 점수", "계산 불가")
    
    elif clustering_method == "Gaussian Mixture":
        st.subheader("Gaussian Mixture 모델")
        
        # GMM 파라미터
        n_components_gmm = st.slider("컴포넌트 수", 2, 10, 3, key="gmm_n_components")
        covariance_type = st.selectbox("공분산 타입", ["full", "tied", "diag", "spherical"])
        
        # GMM 실행
        gmm = GaussianMixture(n_components=n_components_gmm, covariance_type=covariance_type, random_state=42)
        cluster_labels = gmm.fit_predict(X_reduced)
        
        # 성능 지표
        silhouette_avg = silhouette_score(X_reduced, cluster_labels)
        aic = gmm.aic(X_reduced)
        bic = gmm.bic(X_reduced)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("컴포넌트 수", n_components_gmm)
        with col2:
            st.metric("실루엣 점수", f"{silhouette_avg:.4f}")
        with col3:
            st.metric("AIC", f"{aic:.2f}")
        with col4:
            st.metric("BIC", f"{bic:.2f}")
    
    else:  # Agglomerative Clustering
        st.subheader("계층적 클러스터링")
        
        # 계층적 클러스터링 파라미터
        n_clusters_agg = st.slider("클러스터 수", 2, 10, 3, key="agg_n_clusters")
        linkage = st.selectbox("연결 방법", ["ward", "complete", "average", "single"])
        
        # 계층적 클러스터링 실행
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters_agg, linkage=linkage)
        cluster_labels = agg_clustering.fit_predict(X_reduced)
        
        # 성능 지표
        silhouette_avg = silhouette_score(X_reduced, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X_reduced, cluster_labels)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("클러스터 수", n_clusters_agg)
        with col2:
            st.metric("실루엣 점수", f"{silhouette_avg:.4f}")
        with col3:
            st.metric("Calinski-Harabasz 지수", f"{calinski_harabasz:.2f}")
    
    # 클러스터링 결과 시각화
    st.subheader("📊 클러스터링 결과 시각화")
    
    if X_reduced.shape[1] >= 2:
        title = f"{clustering_method} 클러스터링 결과"
        if dim_reduction != "없음":
            title += f" ({reduction_method_name})"
        
        cluster_fig = plot_cluster_results(
            X_reduced[:, :min(3, X_reduced.shape[1])], 
            cluster_labels, 
            title,
            selected_features,
            reduction_method_name
        )
        
        st.plotly_chart(cluster_fig, use_container_width=True)
        add_chart_export_section(cluster_fig, f"{clustering_method.lower()}_clustering_result")
    
    # 클러스터별 특성 분석
    st.subheader("📈 클러스터별 특성 분석")
    
    # 클러스터 라벨을 원본 데이터에 추가
    cluster_df = X.copy()
    cluster_df['Cluster'] = cluster_labels
    
    # 클러스터별 통계
    cluster_stats = cluster_df.groupby('Cluster')[selected_features].agg(['mean', 'std', 'count'])
    
    st.markdown("**클러스터별 평균값**")
    mean_stats = cluster_stats.xs('mean', level=1, axis=1)
    st.dataframe(mean_stats.round(4))
    
    # 클러스터별 특성 히트맵
    if len(selected_features) > 1:
        fig_heatmap = px.imshow(
            mean_stats.T,
            labels=dict(x="클러스터", y="피처", color="평균값"),
            x=[f"클러스터 {i}" for i in mean_stats.index],
            y=selected_features,
            color_continuous_scale='RdYlBu_r',
            title="클러스터별 피처 평균값 히트맵"
        )
        fig_heatmap.update_layout(height=400)
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        add_chart_export_section(fig_heatmap, "cluster_characteristics_heatmap")
    
    # 클러스터별 분포 시각화
    st.subheader("📊 클러스터별 피처 분포")
    
    # 피처 선택
    feature_for_dist = st.selectbox("분포를 볼 피처 선택", selected_features)
    
    # 박스플롯
    fig_box = px.box(
        cluster_df, 
        x='Cluster', 
        y=feature_for_dist,
        title=f"{feature_for_dist}의 클러스터별 분포",
        color='Cluster',
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_box.update_layout(height=400)
    
    st.plotly_chart(fig_box, use_container_width=True)
    add_chart_export_section(fig_box, f"cluster_distribution_{feature_for_dist}")
    
    # 클러스터 크기 분석
    st.subheader("📏 클러스터 크기 분석")
    
    cluster_sizes = cluster_df['Cluster'].value_counts().sort_index()
    
    # 파이 차트
    fig_pie = px.pie(
        values=cluster_sizes.values,
        names=[f"클러스터 {i}" for i in cluster_sizes.index],
        title="클러스터별 데이터 포인트 분포"
    )
    fig_pie.update_layout(height=400)
    
    st.plotly_chart(fig_pie, use_container_width=True)
    add_chart_export_section(fig_pie, "cluster_size_distribution")
    
    # 클러스터별 요약 통계
    col1, col2 = st.columns(2)
    with col1:
        st.metric("총 클러스터 수", len(cluster_sizes))
    with col2:
        st.metric("가장 큰 클러스터 크기", cluster_sizes.max())
    
    # 클러스터링 결과 다운로드
    st.subheader("💾 결과 내보내기")
    
    # 결과 데이터프레임 생성
    result_df = df.copy()
    if dim_reduction == "t-SNE":
        result_df = result_df.iloc[sample_idx].copy()
    
    result_df['Cluster'] = cluster_labels
    
    if X_reduced.shape[1] <= 3:
        for i in range(X_reduced.shape[1]):
            result_df[f'{reduction_method_name}_Component_{i+1}'] = X_reduced[:, i]
    
    # CSV 다운로드
    csv = result_df.to_csv(index=False)
    st.download_button(
        label="📥 클러스터링 결과 CSV 다운로드",
        data=csv,
        file_name=f"clustering_results_{clustering_method.lower()}.csv",
        mime="text/csv"
    )
    
    # 클러스터별 상세 분석
    with st.expander("🔍 클러스터별 상세 분석"):
        selected_cluster = st.selectbox(
            "분석할 클러스터 선택",
            sorted(cluster_df['Cluster'].unique())
        )
        
        cluster_data = cluster_df[cluster_df['Cluster'] == selected_cluster]
        
        st.subheader(f"클러스터 {selected_cluster} 상세 정보")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("데이터 포인트 수", len(cluster_data))
        with col2:
            st.metric("전체 데이터 비율", f"{len(cluster_data)/len(cluster_df)*100:.1f}%")
        with col3:
            if clustering_method == "DBSCAN" and selected_cluster == -1:
                st.metric("클러스터 타입", "노이즈")
            else:
                st.metric("클러스터 타입", "정상")
        
        # 클러스터 내 통계
        st.markdown("**클러스터 내 피처 통계**")
        cluster_stats_detail = cluster_data[selected_features].describe()
        st.dataframe(cluster_stats_detail.round(4))
        
        # 클러스터 내 데이터 샘플
        st.markdown("**클러스터 내 데이터 샘플**")
        st.dataframe(cluster_data.head(10))

if __name__ == "__main__":
    main() 