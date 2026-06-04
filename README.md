# 💊 알약 인식 Object Detection 프로젝트

> **스프린트 11 — 2팀**  
> AIHub 제공 한국 의약품 이미지 데이터셋을 활용한 알약 탐지 및 의약품 정보 제공 시스템

---

## 프로젝트 소개

카메라 또는 이미지 업로드를 통해 알약을 실시간으로 탐지하고, 해당 의약품의 **금기사항·주의사항**을 즉시 제공하는 시스템입니다.

- **학습 파이프라인**: Google Colab(GPU) 기반 멀티 모델 학습 및 앙상블
- **웹 애플리케이션**: FastAPI + 정적 UI(다크 글래스모피즘 디자인)로 모바일에서도 접근 가능
- **지원 의약품**: 56종 한국 의약품 DB 내장

---

## 폴더 구조

```
.
├── baseline_vomega/          # 학습 파이프라인 (전처리 → 학습 → 추론 → 앙상블)
│   ├── configs/
│   │   └── default.yaml      # 학습 하이퍼파라미터 설정
│   ├── data/
│   │   ├── dataset.py        # 데이터셋 클래스 및 어노테이션 파싱
│   │   └── transforms.py     # 데이터 증강
│   ├── models/
│   │   └── detector.py       # 모델 인터페이스 (Faster R-CNN용 슬롯)
│   ├── scripts/
│   │   ├── preprocess.py     # 중첩 어노테이션 → COCO JSON 병합
│   │   ├── convert_to_yolo.py# COCO → YOLO 포맷 변환 및 train/val/test 분리
│   │   ├── visualize.py      # 어노테이션 시각화
│   │   └── visualize_errors.py
│   ├── utils/
│   │   ├── coco_utils.py     # COCO 어노테이션 유틸
│   │   └── metrics.py        # mAP 계산
│   ├── train_yolo.py         # YOLO11 학습
│   ├── train_rtdetr.py       # RT-DETR 학습
│   ├── train.py              # Faster R-CNN 학습
│   ├── inference_yolo.py     # YOLO11 추론 → submission.csv
│   ├── inference_rtdetr.py   # RT-DETR 추론
│   ├── inference_ensemble_vomega.py  # YOLO11 + RT-DETR WBF 앙상블
│   ├── inference_yolo_ensemble.py    # YOLO 단독 앙상블
│   ├── colab_run.ipynb       # Colab 전체 실행 노트북
│   ├── confusion_matrix.ipynb
│   ├── requirements.txt
│   └── 하이퍼파라미터_설명서.md
│
└── ver4_0529_FastAPI/        # 웹 애플리케이션 (FastAPI)
    ├── main.py               # FastAPI 서버 (의약품 DB 56종 내장)
    ├── app_wrapper.py        # YOLO 모델 래퍼 (4단계 추론 파이프라인)
    ├── streamlit_app.py      # Streamlit 대체 UI
    ├── static/
    │   └── index.html        # 메인 웹 UI (다크 글래스모피즘)
    ├── profiler.py           # 단계별 지연 시간 측정
    ├── check_env.py          # 의존성 환경 확인
    └── NanumGothic.ttf       # 한글 폰트 (결과 시각화용)
```

---

## 기술 스택

| 분류 | 라이브러리 / 도구 |
|---|---|
| 딥러닝 프레임워크 | PyTorch ≥ 2.0, torchvision ≥ 0.15 |
| 객체 탐지 모델 | YOLO11 (ultralytics ≥ 8.3), RT-DETR, Faster R-CNN |
| 앙상블 | ensemble-boxes ≥ 1.0.9 (WBF) |
| 웹 서버 | FastAPI, Starlette |
| 대체 UI | Streamlit |
| 이미지 처리 | OpenCV, Pillow ≥ 9.0 |
| 설정 관리 | PyYAML ≥ 6.0 |
| 학습 환경 | Google Colab (T4 / A100 GPU) |
| 모바일 접근 | ngrok 터널링 |

---

## 학습 파이프라인

```
AIHub 데이터셋 (COCO 포맷)
        │
        ▼
  preprocess.py           ← 중첩 폴더 어노테이션 병합 → annotations.json
        │
        ▼
  convert_to_yolo.py      ← YOLO 포맷 변환 + train/val/test 분리 (기본 80/20)
        │
        ▼
  train_yolo.py           ← YOLO11 학습 (n/s/m/l/x 선택)
  train_rtdetr.py         ← RT-DETR 학습
        │
        ▼
  inference_ensemble_vomega.py  ← WBF 앙상블
        │
        ▼
  submission.csv          ← 캐글 제출 파일
```

### Colab에서 실행하기

1. Google Drive에 프로젝트 폴더 및 데이터셋 zip 업로드
2. `baseline_vomega/colab_run.ipynb` 열기
3. 셀 2에서 `DRIVE_PROJECT_DIR` 경로만 수정
4. 위에서부터 순서대로 실행

**본 학습 권장 파라미터:**
```bash
--epochs 60 --batch 8 --imgsz 1280
```

> 하이퍼파라미터 상세 설명 → [`baseline_vomega/하이퍼파라미터_설명서.md`](baseline_vomega/하이퍼파라미터_설명서.md)

---

## 모델 구성

| 모델 | 특징 | 용도 |
|---|---|---|
| **YOLO11m** | 속도-정확도 균형, 기본 생산 모델 | 단일 추론, 앙상블 |
| **RT-DETR** | Transformer 기반 고정밀 탐지 | 앙상블 보조 |
| **Faster R-CNN** | PyTorch 레퍼런스 구현 | 베이스라인 비교 |

**추론 기본값:**
- Score threshold: `0.5`
- NMS threshold: `0.5`
- 이미지당 최대 탐지: `4`개

---

## 웹 애플리케이션 실행

```bash
cd ver4_0529_FastAPI

# 환경 확인
python check_env.py

# 서버 실행
python main.py
# → http://localhost:8000
```

**기능:**
- 카메라 실시간 촬영 또는 이미지 업로드
- 탐지된 알약의 금기사항·주의사항 즉시 표시
- 56종 한국 의약품 정보 DB 내장
- HITL(Human-In-The-Loop) 피드백 로깅 (`hitl_log.jsonl`)
- 단계별 추론 지연 시간 프로파일링

**모바일 접근 (ngrok):**
```bash
# main.py 내 ngrok 설정 활성화 후 생성된 URL로 접속
```

---

## 학습 출력 파일

| 파일 | 경로 | 설명 |
|---|---|---|
| 학습된 가중치 | `outputs/yolo/{run_name}/weights/best.pt` | 최적 체크포인트 |
| 학습 결과 | `outputs/yolo/{run_name}/results.csv` | Epoch별 Loss/mAP |
| 제출 파일 | `outputs/predictions/submission_yolo.csv` | 캐글 제출용 CSV |
| COCO 어노테이션 | `data/processed/annotations.json` | 병합된 어노테이션 |
| 클래스 매핑 | `data/processed/class_mapping.json` | 카테고리 ID ↔ 이름 |

---

## 의존성 설치

```bash
pip install -r baseline_vomega/requirements.txt
```

```
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
PyYAML>=6.0
matplotlib>=3.5.0
ultralytics>=8.3.0
ensemble-boxes>=1.0.9
```

---

## 데이터셋

- **출처**: [AIHub — 한국 의약품 이미지](https://aihub.or.kr)
- **포맷**: COCO JSON (중첩 폴더 구조)
- **분할**: 기본 Train 80% / Val 20% (seed 고정)

---

## 검증 체크리스트 (Colab)

- [ ] 셀 3: 데이터셋 zip 해제가 수 분 안에 완료
- [ ] 셀 7: `data/processed/` 에 어노테이션 생성 확인
- [ ] 셀 8: `data/yolo/{images,labels}/{train,val}` 생성 확인
- [ ] 셀 9: GPU에서 학습 진행 및 `best.pt` 생성 확인
- [ ] 셀 10: `submission_yolo.csv` 생성 및 행 수 > 0 확인
- [ ] 셀 11: Google Drive 백업 완료 확인
