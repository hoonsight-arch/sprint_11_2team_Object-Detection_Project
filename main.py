# main.py
import io
import re
import json
import base64
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from app_wrapper import PillPredictorWrapper, DEFAULT_MODEL_PATH
from profiler import StageTimer, record_stats, timing_middleware_factory

# ── 복약 DB (56개 전체 클래스) ────────────────────────────────────────
PILL_DB = {
    "가바토파정 100mg": {
        "warning": "알코올·벤조디아제핀 등 중추신경억제제 병용 시 호흡억제 위험",
        "caution": "급격한 복용 중단 금지 — 발작 위험, 1~2주에 걸쳐 서서히 감량",
    },
    "글리아타민연질캡슐": {
        "warning": "항콜린제(아트로핀 등) 병용 시 효과 상쇄 가능",
        "caution": "식후 복용 권장, 임산부·수유부는 의사 지도 하에 복용",
    },
    "글리틴정": {
        "warning": "항콜린제 병용 시 효과 상쇄 가능",
        "caution": "식후 복용 권장, 임산부·수유부는 의사 지도 하에 복용",
    },
    "기넥신에프정": {
        "warning": "항혈소판제·항응고제(와파린, 아스피린) 병용 금지 — 출혈 위험 증가",
        "caution": "수술·시술 전 최소 36시간 이상 복용 중단 필요",
    },
    "노바스크정 5mg": {
        "warning": "자몽주스와 동시 복용 금지 — 혈중 농도 과다 상승",
        "caution": "기립성 저혈압 주의 — 복용 후 급격한 자세 변화 금지",
    },
    "놀텍정 10mg": {
        "warning": "알코올 병용 금지 — 졸음 증폭",
        "caution": "식전 복용 권장, 취침 전 복용 시 익일 졸음 주의",
    },
    "뉴로메드정": {
        "warning": "항경련제와 병용 시 상호작용 주의, 신기능 저하 환자 용량 조절 필요",
        "caution": "신기능 저하 환자는 반드시 의사 상담 후 복용",
    },
    "다보타민큐정 10mg/병": {
        "warning": "항응고제(와파린) 병용 시 약효 변화 가능 — 모니터링 필요",
        "caution": "식후 복용 권장, 직사광선·고온 다습 장소 보관 금지",
    },
    "동아가바펜틴정 800mg": {
        "warning": "알코올·중추신경억제제 병용 시 진정 효과 현저히 증폭, 호흡억제 위험",
        "caution": "급격한 복용 중단 금지 — 발작 위험, 신기능 저하 환자 용량 조절 필수",
    },
    "라비에트정 20mg": {
        "warning": "메토트렉사트 병용 시 혈중 농도 상승 가능 — 의사 상담 필수",
        "caution": "식전 30분 복용 권장, 장기 복용 시 마그네슘 결핍 주의",
    },
    "레일라정": {
        "warning": "35세 이상 흡연자 복용 금지, 혈전증·유방암 병력자 복용 금지",
        "caution": "두통·부종·유방통 지속 시 즉시 의사 상담, 매일 같은 시간 복용",
    },
    "로수바미브정 10/20mg": {
        "warning": "사이클로스포린·항진균제(이트라코나졸) 병용 금지 — 근병증 위험",
        "caution": "근육통·무력감 발생 시 즉시 복용 중단, 간기능 정기 검사",
    },
    "로수젯정10/5밀리그램": {
        "warning": "사이클로스포린·항진균제 병용 시 근병증 위험 현저 증가",
        "caution": "식사와 무관하게 복용 가능, 근육통 발생 시 즉시 복용 중단",
    },
    "리렉스펜정 300mg/PTP": {
        "warning": "알코올 병용 금지 — 간독성 위험 (아세트아미노펜 성분), 타 아세트아미노펜 함유 제품 중복 복용 금지",
        "caution": "졸음 유발 — 운전·기계 조작 주의 (클로르족사존 성분), 식후 복용 권장",
    },
    "리리카캡슐 150mg": {
        "warning": "중추신경억제제(벤조디아제핀) 병용 시 호흡억제 위험",
        "caution": "졸음 유발 — 운전·기계 조작 금지, 급격한 중단 금지",
    },
    "리바로정 4mg": {
        "warning": "사이클로스포린 병용 금지, 에리트로마이신 병용 시 근병증 위험",
        "caution": "임부·수유부 금기, 근육통·황달 발생 시 즉시 복용 중단",
    },
    "리피로우정 20mg": {
        "warning": "항진균제(이트라코나졸)·사이클로스포린 병용 금지",
        "caution": "근육통 발생 시 즉시 복용 중단, 간기능 검사 정기 실시",
    },
    "리피토정 20mg": {
        "warning": "사이클로스포린·피브레이트계 약물 병용 금지",
        "caution": "음식물과 무관하게 복용 가능, 저녁 복용 권장",
    },
    "마도파정": {
        "warning": "MAO-A 억제제 병용 절대 금지 — 고혈압 위기 위험, 항정신병약 병용 주의",
        "caution": "고단백 식사 후 흡수 저하 — 식전 30분 복용 권장, 갑작스러운 중단 금지",
    },
    "맥시부펜이알정 300mg": {
        "warning": "항응고제·아스피린 병용 시 출혈 위험 증가, 심혈관 질환자 장기 복용 금지",
        "caution": "식후 복용 필수 — 공복 복용 시 위장장애, 서방형 제제 분쇄 금지",
    },
    "무코스타정": {
        "warning": "특이 상호작용 드물지만 항궤양제 중복 복용 전 의사 상담",
        "caution": "식후 복용 권장, 임산부·수유부는 의사 지도 하에 복용",
    },
    "뮤테란캡슐 100mg": {
        "warning": "기관지천식 환자 초회 복용 시 기관지 경련 주의",
        "caution": "충분한 수분 섭취 권장 (하루 1.5L 이상), 식후 복용",
    },
    "보령부스파정 5mg": {
        "warning": "MAO 억제제 병용 금지, 자몽주스 복용 금지 (혈중 농도 상승)",
        "caution": "효과 발현까지 1~2주 소요 — 임의 중단 금지, 운전·기계 조작 주의",
    },
    "비모보정 500/20mg": {
        "warning": "항응고제·아스피린 병용 시 출혈 위험, 심혈관 질환자 장기 복용 금지",
        "caution": "식후 복용 필수, 통째로 삼킬 것 — 씹거나 분쇄 절대 금지",
    },
    "스토가정 10mg": {
        "warning": "클로피도그렐 병용 시 상호작용 주의 (CYP2C19)",
        "caution": "식사와 무관하게 복용 가능, 장기 복용 시 위산 반동 주의",
    },
    "신바로정": {
        "warning": "간기능 이상 환자 투여 금지, 간독성 약물 병용 주의",
        "caution": "임신·수유 중 투여 금지, 소화불량 발생 시 식후 복용으로 변경",
    },
    "써스펜8시간이알서방정 650mg": {
        "warning": "알코올 상시 복용자 간손상 위험 증가, 타 아세트아미노펜 함유 제품 중복 복용 절대 금지",
        "caution": "1일 최대 4,000mg 초과 금지, 서방형 제제 — 씹거나 분쇄 금지",
    },
    "아모잘탄정 5/100mg": {
        "warning": "자몽주스 복용 금지 (암로디핀 성분), 임신 중 절대 금기 — 태아 신기능 손상",
        "caution": "기립성 저혈압 주의, 칼륨 보충제·칼륨 보존 이뇨제 병용 주의",
    },
    "아빌리파이정 10mg": {
        "warning": "케토코나졸·파록세틴 병용 시 반드시 용량 조절",
        "caution": "갑작스러운 복용 중단 금지 — 의사 지도 하에 감량",
    },
    "아질렉트정": {
        "warning": "플루옥세틴·플루복사민 등 SSRI 병용 금지 — 세로토닌 증후군 위험, 티라민 풍부 식품(숙성 치즈·와인) 주의",
        "caution": "다른 MAO 억제제와 14일 이상 간격 필요, 갑작스러운 중단 금지",
    },
    "아토르바정 10mg": {
        "warning": "항진균제(이트라코나졸)·사이클로스포린 병용 금지",
        "caution": "근육통 발생 시 즉시 복용 중단, 자몽주스 다량 복용 금지",
    },
    "아토젯정 10/40mg": {
        "warning": "사이클로스포린·항진균제 병용 시 근병증 위험 현저 증가",
        "caution": "근육통·황달 발생 시 즉시 복용 중단, CK 수치 이상 시 투약 중지",
    },
    "알드린정": {
        "warning": "소염진통제(NSAIDs)·철분제·항생제 병용 시 약물 흡수 방해 — 최소 2시간 간격 필요",
        "caution": "제산제는 공복 또는 식후 1~3시간 복용 권장, 장기 복용 시 의사 상담",
    },
    "에빅사정": {
        "warning": "도파민 작용제·항경련제·바르비투레이트 병용 시 효과 변화 주의",
        "caution": "신기능 저하 환자 용량 조절 필수, 갑작스러운 중단 금지",
    },
    "에스원엠프정 20mg": {
        "warning": "자몽주스 동시 복용 금지 — 혈중 농도 과다 상승",
        "caution": "기립성 저혈압 주의 — 복용 후 급격한 자세 변화 금지",
    },
    "에어탈정": {
        "warning": "항응고제·아스피린 병용 시 출혈 위험, 심혈관·신장 기능 저하자 주의",
        "caution": "식후 복용 필수 — 공복 복용 시 위장장애, 장기 복용 주의",
    },
    "엑스포지정 5/160mg": {
        "warning": "임신 중 절대 금기 — 태아 신기능 손상, 자몽주스 복용 금지 (암로디핀)",
        "caution": "기립성 저혈압 주의, 고칼륨혈증 유발 가능 — 칼륨 보충제 병용 주의",
    },
    "오마코연질캡슐": {
        "warning": "항응고제(와파린) 병용 시 출혈 시간 연장 가능 — 주기적 모니터링 필요",
        "caution": "생선 알레르기 환자 주의, 수술 전 2주 이상 복용 중단 권장",
    },
    "울트라셋이알서방정": {
        "warning": "알코올·수면제·항불안제 병용 금지 — 호흡억제 위험",
        "caution": "서방형 제제 — 씹거나 쪼개서 복용 절대 금지",
    },
    "일양하이트린정 2mg": {
        "warning": "PDE5 억제제(비아그라·시알리스) 병용 시 심각한 저혈압 위험",
        "caution": "최초 복용 시 기립성 저혈압 심함 — 취침 전 복용 권장, 운전 주의",
    },
    "자누메트엑스알서방정 100/1000mg": {
        "warning": "조영제 사용 검사 전 복용 반드시 중단 — 유산산증 위험, 알코올 과다 복용 금지",
        "caution": "식사와 함께 복용, 서방형 제제 — 씹거나 분쇄 절대 금지",
    },
    "자누메트정 50/850mg": {
        "warning": "조영제 사용 검사 전 복용 중단, 알코올 과다 복용 시 유산산증 위험",
        "caution": "식사와 함께 복용, 구역·구토 발생 시 의사 상담",
    },
    "자누비아정 50mg": {
        "warning": "인슐린 병용 시 저혈당 위험 증가 — 혈당 모니터링 필수",
        "caution": "식사와 관계없이 복용 가능",
    },
    "제미메트서방정 50/1000mg": {
        "warning": "조영제 검사 전 복용 중단, 과량 음주 금지 — 유산산증 위험",
        "caution": "식사와 함께 복용, 서방형 제제 — 분쇄·씹기 금지",
    },
    "졸로푸트정 100mg": {
        "warning": "MAO 억제제 복용 후 14일 이내 절대 복용 금지",
        "caution": "초기 2~4주 불안·불면 일시적 증가 가능 — 임의 중단 금지",
    },
    "종근당글리아티린연질캡슐": {
        "warning": "항콜린제 병용 시 효과 상쇄 가능",
        "caution": "식후 복용 권장, 임산부·수유부는 의사 지도 하에 복용",
    },
    "카나브정 60mg": {
        "warning": "임신 중 절대 금기 — 태아 신기능 손상, ACE 억제제 병용 금지",
        "caution": "고칼륨혈증 주의 — 칼륨 보충제·칼륨 보존 이뇨제 병용 주의",
    },
    "카발린캡슐 25mg": {
        "warning": "알코올·중추신경억제제 병용 시 진정 효과 현저 증폭",
        "caution": "급격한 중단 금지 — 1주 이상에 걸쳐 서서히 감량, 졸음·어지러움 주의",
    },
    "콜리네이트연질캡슐 400mg": {
        "warning": "항콜린제 병용 시 효과 상쇄 가능",
        "caution": "식후 복용 권장, 임산부·수유부는 의사 지도 하에 복용",
    },
    "큐시드정 31.5mg/PTP": {
        "warning": "라니티딘 성분 불순물(NDMA) 검출 이슈 전력 — 현재 복용 중이라면 담당 의사·약사에게 안전성 재확인 필수",
        "caution": "다른 위장약과 중복 복용 전 약사 상담, 증상 지속 시 의사 진료 권장",
    },
    "크레스토정 20mg": {
        "warning": "항진균제(이트라코나졸) 병용 금지",
        "caution": "근육통 발생 시 즉시 복용 중단 후 의사 상담",
    },
    "트라젠타듀오정 2.5/850mg": {
        "warning": "조영제 사용 검사 전 복용 중단 필수, 알코올 과다 복용 금지 — 유산산증 위험",
        "caution": "식사와 함께 복용, 구역·복통 발생 시 의사 상담",
    },
    "트라젠타정": {
        "warning": "인슐린·설포닐우레아 병용 시 저혈당 위험 증가",
        "caution": "식사와 무관하게 복용 가능, 췌장염 증상(심한 복통) 발생 시 즉시 복용 중단",
    },
    "트루비타정 60mg/병": {
        "warning": "지용성 비타민(A·D·E·K) 함유 제품 중복 복용 주의",
        "caution": "식후 복용 권장, 직사광선·고온 다습 장소 보관 금지",
    },
    "트윈스타정 40/5mg": {
        "warning": "임신 중 절대 금기 — 태아 신기능 손상, 자몽주스 복용 금지 (암로디핀)",
        "caution": "기립성 저혈압 주의, 고칼륨혈증 유발 가능 — 칼륨 보충제 병용 주의",
    },
    "플라빅스정 75mg": {
        "warning": "아스피린 고용량(500mg↑) 병용 금지 — 출혈 위험",
        "caution": "수술 전 최소 5일 이상 복용 중단 필요",
    },
}

# ── App 초기화 ────────────────────────────────────────────────────────
app = FastAPI(title="💊 알약알쥐? API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(timing_middleware_factory())

# 모델 1회 로드 (서버 시작 시)
print("[LOG] FastAPI 서버 시작 — YOLO 모델 로딩 중...")
predictor = PillPredictorWrapper()
print("[LOG] 모델 로드 완료")

# Static 파일 디렉토리
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── 라우트 ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    print(f"[LOG] /predict 수신 — {file.filename}")
    image_bytes = await file.read()

    # app_wrapper 내부 스테이지 타이밍 포함 (3-tuple)
    refined_output, img_rgb, wrapper_timings = predictor.predict(image_bytes)

    post_timer = StageTimer()

    # ── 스테이지 5: JPEG 인코딩 ───────────────────────────────────────
    with post_timer.measure("5_jpeg_encode"):
        buf = io.BytesIO()
        Image.fromarray(img_rgb).save(buf, format="JPEG", quality=85)

    # ── 스테이지 6: base64 인코딩 ─────────────────────────────────────
    with post_timer.measure("6_base64_encode"):
        img_b64 = base64.b64encode(buf.getvalue()).decode()

    # ── 스테이지 7: DB 조회 + 레이블 정제 ────────────────────────────
    with post_timer.measure("7_db_lookup"):
        top4 = refined_output[:4]
        for pred in top4:
            label_key = re.sub(r'\([^)]*\)', '', pred["label"].strip()).strip()
            pred["pill_info"] = PILL_DB.get(label_key, {"warning": None, "caution": None})
            display = re.sub(r'\([^)]*\)', '', pred["label"]).strip()
            print(f"[LOG] 탐지: {display} ({pred['confidence']:.2%})")

    post_timer.print_report("main.predict (후처리)")

    # wrapper + postprocess 타이밍 병합 후 통계 누적
    merged_stages = {**wrapper_timings.get("stages_ms", {}), **post_timer.result()["stages_ms"]}
    merged_total  = wrapper_timings.get("total_ms", 0) + post_timer.result()["total_ms"]
    record_stats({"stages_ms": merged_stages})

    return {
        "predictions": top4,
        "image_b64": img_b64,
        "timings": {"stages_ms": merged_stages, "total_ms": round(merged_total, 2)},
    }


class HITLData(BaseModel):
    ai_label: str
    corrected_label: str
    confidence: float
    timestamp: str


@app.post("/hitl")
async def save_hitl(data: HITLData):
    log_path = Path(__file__).parent / "hitl_log.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data.model_dump(), ensure_ascii=False) + "\n")
    print(f"[LOG] HITL 저장: {data.ai_label} → {data.corrected_label}")
    return {"status": "saved"}


@app.get("/model-info")
async def model_info():
    return {
        "num_classes": len(predictor.model.names),
        "classes": predictor.model.names,
    }
