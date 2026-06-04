# 공공데이터 활용 실시간 근처 약국 대시보드
import streamlit as st
import pandas as pd
import requests
from datetime import datetime

st.set_page_config(page_title="AI Pill Scanner Pro", layout="wide")

# 1. 영업 상태 컬러 칩 로직 (Color Chip Engine)
def get_status_chip(status):
    if "영업 중" in status:
        return ":green[● 영업 중]"
    return ":red[● 영업 종료]"

# 2. 공공데이터 API 연동 시뮬레이션 (Interface Mock-up)
def fetch_pharmacy_data():
    # 실제 환경에서는 requests.get(url, params=...) 사용
    # 여기서는 데이터 구조 예시만 제시합니다.
    data = [
        {"이름": "강남약국", "주소": "서울 강남구...", "시간": "09:00-22:00", "상태": "영업 중"},
        {"이름": "행복약국", "주소": "서울 강남구...", "시간": "09:00-20:00", "상태": "영업 종료"}
    ]
    return pd.DataFrame(data)

st.title("💊 실시간 근처 약국 대시보드")

# 3. 사이드바 및 데이터 로드
if st.sidebar.button("실시간 약국 데이터 동기화"):
    df = fetch_pharmacy_data()
    st.session_state['pharmacy_df'] = df

# 4. 데이터 표시 및 컬러 칩 적용
if 'pharmacy_df' in st.session_state:
    df = st.session_state['pharmacy_df']
    
    # 컬러 칩 적용 로직
    df['상태_Chip'] = df['상태'].apply(get_status_chip)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(df, use_container_width=True)
    with col2:
        st.subheader("💡 약국 상태 요약")
        for idx, row in df.iterrows():
            st.markdown(f"**{row['이름']}** {row['상태_Chip']}")
else:
    st.info("데이터 동기화 버튼을 눌러주세요.")

# 5. 지도 임베딩 (공간 데이터 연동)
st.subheader("📍 지도로 확인하기")
st.components.v1.iframe("https://map.naver.com/", height=500)