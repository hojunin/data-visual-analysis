import streamlit as st

st.set_page_config(page_title="데이터 시각화 도구", layout="wide")
st.title("📊 데이터 시각화 웹앱")
st.markdown("""
왼쪽 메뉴에서 기능을 선택하세요:

- **Basic Analysis**: 단일 그래프 탐색
- **Multiple Compare**: 그래프 2개 비교 시각화
- **Graph Overlay**: 두 그래프 겹쳐보기 기능
- **Advanced Analysis**: 통계 분석 및 상관관계 분석
- **Dashboard**: 통합 대시보드 및 인사이트
- **Data Preprocessing**: 데이터 전처리 및 변환
""")
