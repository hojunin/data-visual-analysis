import streamlit as st

st.set_page_config(page_title="데이터 시각화 도구", layout="wide")

# 전역 스타일 설정
st.markdown("""
<style>
div[data-testid="metric-container"] {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    padding: 1rem;
    border-radius: 0.375rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
}
</style>
""", unsafe_allow_html=True)

st.title("📊 데이터 시각화 웹앱")
st.markdown("""
왼쪽 메뉴에서 기능을 선택하세요:

- **기본 분석**: 단일 그래프 탐색
- **그래프 비교**: 그래프 2개 비교 시각화
- **그래프 겹치기**: 두 그래프 겹쳐보기 기능
- **고급 분석**: 통계 분석 및 상관관계 분석
- **대시보드**: 통합 대시보드 및 인사이트
- **데이터 전처리**: 데이터 전처리 및 변환
- **회귀분석**: 전문 회귀분석 도구
""")
