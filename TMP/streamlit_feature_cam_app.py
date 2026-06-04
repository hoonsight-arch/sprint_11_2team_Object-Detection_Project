#streamlit_feature_cam_app.py
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import time
import datetime

# Streamlit의 Session State를 파일 시스템/DB 대용 버퍼로 활용하여 비동기 데이터 저장 시뮬레이션
if 'initialized' not in st.session_state:
    st.session_state['initialized'] = True
    
    # 1. 알약 마스터 DB (식약처 공공데이터 가상 매핑 데이터)
    st.session_state['master_db'] = [
        {"pill_id": "P001", "name": "타이레놀정 500mg", "shape": "타원형", "color": "하얀색", "imprint": "TYLENOL", "ingredient": "아세트아미노펜", "desc": "감기로 인한 발열 및 통증, 두통, 신경통 완화"},
        {"pill_id": "P002", "name": "아스피린정 100mg", "shape": "원형", "color": "하얀색", "imprint": "Bayer", "ingredient": "아스피린", "desc": "혈전 예방 및 일차 예방 심혈관 질환 위험 감소"},
        {"pill_id": "P003", "name": "이부프로펜정 200mg", "shape": "타원형", "color": "주황색", "imprint": "IBU", "ingredient": "이부프로펜", "desc": "관절염, 감기로 인한 발열 및 통증, 편두통 완화"},
        {"pill_id": "P004", "name": "아빌리파이정 5mg", "shape": "장방형", "color": "파란색", "imprint": "A5", "ingredient": "아리피프라졸", "desc": "조현병, 양극성 장애의 치료 및 우울증 보조치료"},
    ]
    
    # 2. 격리된 데이터베이스 버퍼 (Quarantine Storage) - 데이터 플라이휠의 핵심
    st.session_state['quarantine_db'] = []
    
    # 3. 누적 통계 지표
    st.session_state['stats'] = {
        "total_scans": 124,
        "normal_scans": 102,
        "quarantined_scans": 22,
        "corrected_scans": 8,
        "current_map": 0.892,
        "system_latency": "0.18s"
    }

def simulate_ai_inference(pil_image, simulate_score):
    """
    YOLO(Detection) + ResNet(Feature Extraction) 하이브리드 추론 과정을 시뮬레이션합니다.
    PIL ImageDraw를 이용해 업로드된 이미지 위에 바운딩 박스를 직접 렌더링합니다.
    """
    # 원본 이미지 복사 및 드로잉 객체 생성
    draw_image = pil_image.copy()
    width, height = draw_image.size
    draw = ImageDraw.Draw(draw_image)
    
    # 크롭된 이미지 중앙에 맞춤형 바운딩 박스 생성
    box = [width * 0.2, height * 0.2, width * 0.8, height * 0.8]
    
    # 신뢰도 점수에 따른 테두리 색상 분기 (정상: Teal, 저신뢰도 예외: Orange/Red)
    border_color = "#2dd4bf" if simulate_score >= 0.85 else "#f43f5e"
    draw.rectangle(box, outline=border_color, width=min(width, height) // 25)
    
    # 가상 피처 벡터 추출 연산 대기 시간 시뮬레이션
    time.sleep(0.3) 
    
    return draw_image, box

def apply_center_crop(pil_image, crop_ratio=0.6):
    """
    가이드 상자에 정렬해 찍은 것과 같은 효과를 주기 위해
    이미지의 중앙 영역(가로, 세로 crop_ratio 만큼)만 자동으로 크롭하는 전처리 함수입니다.
    """
    width, height = pil_image.size
    crop_w = int(width * crop_ratio)
    crop_h = int(height * crop_ratio)
    
    left = (width - crop_w) // 2
    top = (height - crop_h) // 2
    right = (width + crop_w) // 2
    bottom = (height + crop_h) // 2
    
    return pil_image.crop((left, top, right, bottom))

# Modern Dark Slate Theme 스타일 적용
st.set_page_config(page_title="AI Pill-ID MVP", layout="wide", page_icon="💊")

st.markdown("""
    <style>
    .main { background-color: #0f172a; color: #f1f5f9; }
    h1, h2, h3 { color: #f8fafc !important; font-family: 'Inter', sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 10px 20px;
        color: #94a3b8;
        border: 1px solid #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0ea5e9 !important;
        color: white !important;
    }
    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }
    .metric-value { font-size: 32px; font-weight: bold; color: #38bdf8; }
    .guide-box {
        border: 2px dashed #38bdf8;
        background-color: rgba(56, 189, 248, 0.05);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💊 AI 기반 알약 식별 및 데이터 플라이휠 시스템")
st.markdown("부트캠프 MVP 단계에서 검증해야 할 핵심 추론 파이프라인과 예외 수집 피드백 루프를 모니터링하는 프로토타입입니다.")

# 탭 구조 정의 (인퍼런스 인터페이스 / 전문가 백오피스 / 모니터링 대시보드 / 가상 마스터 DB)
tabs = st.tabs(["📸 알약 인식 (Inference)", "🛡️ 예외 처리 보정 (HITL Backoffice)", "📊 데이터 플라이휠 모니터링", "🗄️ 의약품 마스터 DB"])

with tabs[0]:
    st.header("📸 실시간 알약 식별 & 자동 예외 분기")
    st.write("알약을 촬영하거나 이미지를 업로드하세요. 가이드 영역에 맞춰 촬영하면 정확한 인식이 가능합니다.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. 입력 소스 선택")
        input_method = st.radio("촬영 방식 선택", ["📷 실시간 카메라 촬영", "📁 갤러리 파일 업로드"])
        
        uploaded_file = None
        
        if input_method == "📷 실시간 카메라 촬영":
            st.markdown("""
            <div class="guide-box">
                <p style="color: #38bdf8; margin: 0; font-weight: bold;">🎯 촬영 가이드라인</p>
                <p style="font-size: 13px; color: #cbd5e1; margin: 5px 0 0 0;">
                    알약을 화면 <b>중앙</b>에 위치시키고, 그림자가 지지 않도록 수평을 유지하여 촬영해 주세요.
                </p>
            </div>
            """, unsafe_allow_html=True)
            uploaded_file = st.camera_input("카메라 화면")
        else:
            uploaded_file = st.file_uploader("알약 촬영 이미지를 업로드하세요 (JPG, PNG)", type=["jpg", "png", "jpeg"])
        
        # 데모 시연 및 테스트를 위해 사용자가 직접 신뢰도 점수를 제어하는 시뮬레이터 인터페이스 탑재
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### ⚙️ AI 추론 파라미터 시뮬레이션")
        simulated_score = st.slider(
            "시뮬레이션할 추론 신뢰도(Confidence Score) 결정", 
            min_value=0.10, max_value=1.00, value=0.92, step=0.01,
            help="신뢰도를 0.85 미만으로 내리면 '예외 발생 및 비동기 격리' 로직이 트리거됩니다."
        )
        st.info("💡 **Tip:** 0.85 미만으로 슬라이더를 조정하면, 백엔드 격리 메커니즘을 테스트할 수 있습니다.")

    with col2:
        st.subheader("2. 전처리 및 하이브리드 추론 결과")
        if uploaded_file is not None:
            raw_image = Image.open(uploaded_file)
            
            with st.spinner("이미지 정렬 및 가이드라인 크롭 전처리 중..."):
                # 카메라 촬영 이미지일 경우에만 중앙 60% 영역만 자동 크롭하여 배경 노이즈 제거 시뮬레이션
                if input_method == "📷 실시간 카메라 촬영":
                    processed_input_img = apply_center_crop(raw_image, crop_ratio=0.6)
                    st.toast("📷 가이드라인 기준으로 자동 크롭이 완료되었습니다.")
                else:
                    processed_input_img = raw_image
                
                # 이미지 형식 정규화 (RGBA 등의 포맷 방지)
                if processed_input_img.mode != "RGB":
                    processed_input_img = processed_input_img.convert("RGB")
                    
            with st.spinner("YOLO 검출 및 ResNet 임베딩 추출 처리 중..."):
                processed_img, box_coords = simulate_ai_inference(processed_input_img, simulated_score)
                
            st.image(processed_img, caption="Spatial Filter (전처리 크롭 및 YOLO Bounding Box) 결과 시각화", use_container_width=True)
            
            # 매칭 대상 탐색
            matched_pill = st.session_state['master_db'][0] if simulated_score >= 0.85 else st.session_state['master_db'][2]
            
            # 신뢰도 임계값 판정 로직
            if simulated_score >= 0.85:
                st.success(f"✅ **정상 식별 완료** (신뢰도: {simulated_score:.2f})")
                st.markdown(f"""
                *   **매칭 식별명:** `{matched_pill['name']}` (일치율: {(simulated_score*100):.1f}%)
                *   **주요 성분:** {matched_pill['ingredient']}
                *   **효능/효과:** {matched_pill['desc']}
                """)
                st.session_state['stats']['total_scans'] += 1
                st.session_state['stats']['normal_scans'] += 1
            else:
                # 0.85 미만인 경우: 예외 트리거링 발생 및 S3 / Kafka 적재 시뮬레이션
                st.error(f"⚠️ **식별 예외 발생** (신뢰도: {simulated_score:.2f} < 임계값 0.85)")
                st.warning("예상 매칭 결과의 신뢰도가 낮습니다. 해당 트래픽은 메인 스레드 지연 없이 비동기 격리 DB(S3/Kafka)로 포크됩니다.")
                
                # 중복 등록 방지 로직을 포함한 예외 패킷 Quarantine DB 등록
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                packet_id = f"ERR-{len(st.session_state['quarantine_db']) + 1:03d}"
                
                new_quarantine = {
                    "packet_id": packet_id,
                    "timestamp": timestamp,
                    "raw_image": processed_input_img, # 전처리된 깔끔한 크롭 이미지를 저장
                    "confidence": simulated_score,
                    "meta_context": {"lighting": "Dim", "angle": "Vertical Fit", "device": "Webcam Stream"},
                    "status": "격리됨 (Pending Correction)",
                    "suggested_id": matched_pill['pill_id']
                }
                
                st.session_state['quarantine_db'].append(new_quarantine)
                st.session_state['stats']['total_scans'] += 1
                st.session_state['stats']['quarantined_scans'] += 1
                
                st.info(f"💾 **비동기 격리 패킷 생성 완료:** ID `{packet_id}`가 Quarantine Buffer에 백그라운드로 적재되었습니다.")
        else:
            st.info("카메라 촬영 버튼을 누르거나 이미지를 업로드하면 실시간 하이브리드 인퍼런스가 동작합니다.")

with tabs[1]:
    st.header("🛡️ Human-in-the-Loop 관리자 정답 보정 시스템")
    st.write("인공지능이 판단하지 못하고 보류하여 격리 보관 중인 예외 데이터를 도메인 전문가(약사)가 검토하고 매핑을 고쳐줍니다.")
    
    pending_list = [p for p in st.session_state['quarantine_db'] if p['status'] == "격리됨 (Pending Correction)"]
    
    if len(pending_list) == 0:
        st.success("🎉 현재 격리되어 있는 예외 건수가 전혀 없습니다! 시스템 무결성 상태 양호.")
    else:
        st.warning(f"총 {len(pending_list)}건의 검증 대기 예외 데이터가 검색되었습니다.")
        
        # 목록에서 보정할 예외 패킷 선택
        packet_options = [f"{p['packet_id']} (수집시각: {p['timestamp']})" for p in pending_list]
        selected_option = st.selectbox("보정 작업을 수행할 예외 패킷을 선택하세요:", packet_options)
        
        selected_id = selected_option.split(" ")[0]
        target_packet = next(p for p in pending_list if p['packet_id'] == selected_id)
        
        col_img, col_form = st.columns([1, 1])
        
        with col_img:
            st.image(target_packet['raw_image'], caption=f"격리 당시 전처리 적용 이미지 (ID: {selected_id})", use_container_width=True)
            st.json(target_packet['meta_context'])
            
        with col_form:
            st.subheader("전문가 정답 라벨 매핑 (Ground Truth Mapping)")
            st.write(f"**AI 제안 후보군:** {st.session_state['master_db'][2]['name']} (신뢰도: {target_packet['confidence']:.2f})")
            
            # 도메인 전문가가 수동 매핑할 알약 분류 선택 리스트
            options_db = {p['name']: p['pill_id'] for p in st.session_state['master_db']}
            selected_correct_name = st.selectbox("실제 정답 알약을 식별하여 선택하세요:", list(options_db.keys()))
            correct_id = options_db[selected_correct_name]
            
            feedback_comment = st.text_area("보정 특이사항 및 노이즈 사유 코멘트 입력", "촬영 반사광으로 인한 각인 인식 저하. 수동 클래스 교정 진행.")
            
            if st.button("정답 보정 완료 (Push to Training Database)"):
                # 상태 변경 및 지표 업데이트
                target_packet['status'] = "보정 완료 (Ready to Train)"
                target_packet['corrected_label'] = correct_id
                target_packet['comment'] = feedback_comment
                
                st.session_state['stats']['corrected_scans'] += 1
                st.session_state['stats']['quarantined_scans'] -= 1
                
                st.success(f"데이터셋 저장 완료: `{selected_id}`번 데이터가 '{selected_correct_name}'으로 올바르게 보정되어 차기 재학습 데이터셋으로 분류 이관되었습니다.")
                time.sleep(1)
                st.rerun()

with tabs[2]:
    st.header("📊 Continuous Training & 데이터 플라이휠 모니터링")
    st.write("사용자로부터 격리되어 도메인 전문가의 검증을 통과한 데이터셋의 파이프라인 누적 통계를 모니터링합니다.")
    
    # 4대 핵심 지표 시각화
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="glass-card">
            <p style='color: #94a3b8; margin-bottom: 5px;'>누적 추론 수</p>
            <p class="metric-value">{st.session_state['stats']['total_scans']}건</p>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="glass-card">
            <p style='color: #94a3b8; margin-bottom: 5px;'>현재 격리 중</p>
            <p class="metric-value" style="color: #fb923c;">{st.session_state['stats']['quarantined_scans']}건</p>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="glass-card">
            <p style='color: #94a3b8; margin-bottom: 5px;'>전문가 보정 완료</p>
            <p class="metric-value" style="color: #2dd4bf;">{st.session_state['stats']['corrected_scans']}건</p>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="glass-card">
            <p style='color: #94a3b8; margin-bottom: 5px;'>현재 추론 정확도 (mAP)</p>
            <p class="metric-value" style="color: #38bdf8;">{st.session_state['stats']['current_map']*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)

    # 데이터 플라이휠 동작 상태 및 Airflow 트리거 시뮬레이터
    st.subheader("🤖 CI/CD 파이프라인 컨트롤")
    
    col_ctrl, col_desc = st.columns([1, 2])
    with col_ctrl:
        st.write("**Airflow Fine-Tuning Trigger 조건:**")
        st.write("- 신규 보정 데이터 10건 이상 누적 시 자동 트리거")
        st.write(f"- 현재 보정 완료 데이터: **{st.session_state['stats']['corrected_scans']}** / 10 건")
        
        # 임의로 재학습 파이프라인 구동 시뮬레이션
        trigger_ready = st.session_state['stats']['corrected_scans'] >= 3
        if st.button("Airflow 재학습 파이프라인(DAG) 강제 실행", disabled=not trigger_ready):
            with st.spinner("미세조정(Fine-Tuning) 컨테이너 기동 및 가중치 업데이트 연산 중..."):
                time.sleep(3)
                # 모델 지표 및 누적 상태 조정
                st.session_state['stats']['current_map'] = min(0.995, st.session_state['stats']['current_map'] + 0.012)
                st.session_state['stats']['corrected_scans'] = 0
            st.success("🎉 Fine-Tuning이 성공적으로 수행되어 검증 데이터 성능 평가를 통과했습니다. 신규 모델이 무중단(Hot-Swap) 배포 완료되었습니다.")
            st.rerun()
    with col_desc:
        st.markdown("""
        **데이터 플라이휠 자동화 메커니즘 설명:**
        1. **Inference Exception:** 사용자가 현장에서 알약을 찍었을 때, 빛 번짐이나 그림자로 정확도가 낮게 판독되면 `Quarantine DB`로 비동기 포크 전송합니다.
        2. **HITL Correction:** 관리자 화면을 통해 도메인 전문가(약사)가 원클릭으로 정답 클래스를 매핑해 데이터 무결성을 보존합니다.
        3. **Auto Trigger:** 보정 데이터가 일정량 이상 누적되면, `Airflow`와 `MLflow` 파이프라인이 즉각 가동되어 기존 가중치 위에 점진적 추가 학습(Fine-Tuning)을 실시합니다.
        4. **Hot-Swap Deployment:** 배포 시점의 검증 셋 테스트 정확도가 개선되었을 때만 API 추론 서버가 신규 모델 가중치 파일을 재부팅 없이 Swap하여 성능의 계단식 우상향을 완성합니다.
        """)

with tabs[3]:
    st.header("🗄️ 약학 정보 마스터 데이터베이스 (Master DB)")
    st.write("본 시스템이 인식 가능한 식약처 마스터 의약품 정보입니다. 하이브리드 파이프라인 기술 설계의 핵심은 **[AI 모델]**과 **[의약품 정보 데이터베이스]**의 느슨한 결합(Decoupling)입니다.")
    
    df_master = pd.DataFrame(st.session_state['master_db'])
    st.dataframe(df_master, use_container_width=True)
    
    st.info("💡 **아키텍처 인사이트:** 신규 알약이 시장에 출시되어 추가되어도, AI 가중치(Weights) 자체를 재학습할 필요가 전혀 없습니다. 본 마스터 DB 테이블과 알약 벡터 DB에 특징 임베딩 한 행을 추가해주는 것만으로 즉시 인식 성능을 지원할 수 있어 관리 운영 비용을 획기적으로 낮춥니다.")
# ```

# ---

# ### ⚠️ 개발 및 실전에서 반드시 조심해야 할 부분 (Pitfalls)

# 카메라 입력을 MVP 시스템에 연동하여 발표할 때, 현장에서 가장 많이 일어나는 치명적 오류 3가지와 대처법입니다.

# 1.  **배포 서버의 HTTPS(보안 프로토콜) 요구 문제**
#     *   **이슈:** 웹브라우저는 보안상의 이유로 **HTTPS 환경이 아니면 웹캠 카메라 호출을 차단**합니다. 로컬 테스트(`localhost:8501`)에서는 잘 작동하지만, AWS EC2 등에 단순 HTTP로 배포하면 카메라 촬영 버튼이 차단되는 현상이 발생합니다.
#     *   **해결책:** Streamlit Community Cloud(`streamlit.app`)는 기본적으로 HTTPS가 기본 제공되어 정상 작동하지만, 자체 서버나 AWS에 배포할 예정이라면 반드시 **Let's Encrypt** 등으로 SSL 인증서를 적용해 도메인을 HTTPS화해야 합니다.
# 2.  **스마트폰 자체 그림자 노이즈**
#     *   **이슈:** 알약을 수평으로 두고 위에서 수직으로 찍다 보면, 스마트폰 본체의 그림자가 알약을 통째로 덮어버려 색상이 왜곡되는(Color Shift) 현상이 흔하게 발생합니다.
#     *   **해결책:** 사용자 가이드(Tab 1의 가이드 박스 영역)에 **"빛이 수평이 되는 측면에서 비스듬히 각도를 좁혀 촬영해 주세요"** 혹은 조명 아이콘 등의 경고 문구를 확실히 제시하여 그림자 발생을 억제해야 합니다.
# 3.  **다양한 카메라 해상도 및 화면 비율(Aspect Ratio) 대응**
#     *   **이슈:** 노트북 웹캠(주로 $640 \times 480$), 모바일 전면 카메라($16:9$), 아이폰 후면 카메라($4:3$) 등 디바이스마다 반환되는 이미지 가로세로 규격이 다릅니다. 이로 인해 강제 크롭 도중 이미지가 일그러지거나 인덱스 범위 초과 에러가 발생할 수 있습니다.
#     *   **해결책:** 위의 `apply_center_crop` 함수처럼 픽셀 크기를 상수로 지정($500 \times 500$ 등)하지 않고, 입력된 `pil_image` 가로세로 길이의 **상대적 비율(Ratio = 0.6)**을 구하여 연산하는 로직을 고수해야 어떤 브라우저로 접근해도 컴파일 에러가 나지 않습니다.