# 실시간 약국 찿기
import streamlit as st
import pandas as pd

# 1. 페이지 설정 (초기화 단계)
st.set_page_config(page_title="AI Pill Scanner", layout="wide")

st.title("💊 내 근처 의약품 판매점 찾기")

# 2. 사이드바 (입력 제어부)
st.sidebar.header("검색 필터")
location = st.sidebar.text_input("현재 위치를 입력하세요 (예: 강남역):", "강남역")
drug_name = st.sidebar.text_input("필요한 상비약 이름:")

# 3. 데이터 샘플 (추후 공공데이터 API 연동 필요)
data = {
    "이름": ["강남약국", "GS25 강남점", "CU 역삼점", "행복약국"],
    "유형": ["약국", "편의점", "편의점", "약국"],
    "주소": ["서울 강남구 강남대로 100", "서울 강남구 강남대로 110", "서울 강남구 역삼로 20", "서울 강남구 테헤란로 50"],
    "상태": ["영업 중", "영업 중", "영업 중", "종료"]
}
df = pd.DataFrame(data)

# 4. 지도 시각화 (네이버 지도 연동부)
st.subheader("📍 위치 지도")

# 네이버 지도 검색 URL 생성 패턴
base_url = "https://map.naver.com/p/search/"
map_url = f"{base_url}{location} {drug_name if drug_name else '약국'}"

# Streamlit의 iframe 컴포넌트로 네이버 지도 렌더링
st.components.v1.iframe(map_url, width=1200, height=400, scrolling=True)

# 5. 결과 리스트 출력 (데이터 시각화)
st.subheader("📋 검색 결과")
st.dataframe(df, use_container_width=True)

# 6. 알림 인터럽트
if st.sidebar.button("검색 실행"):
    st.sidebar.success(f"'{location}' 주변 '{drug_name}' 검색 완료")