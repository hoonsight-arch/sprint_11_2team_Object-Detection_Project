# 💊 알약 인식 Object Detection 프로젝트

## 제출 
보고서 [2team 최종보고서.pdf](https://github.com/user-attachments/files/28588713/2team.pdf)
PPT https://drive.google.com/drive/folders/1HCIzn5HzvsI8LEeaaDdMsgf5cf72I48Q?usp=sharing
## 깃허브 제출 
- 전재완 https://traveling-hisser-ce1.notion.site/36422576234380e6b81cd2130ee8fd28?source=copy_link
- 이태훈 [(협업일지 AI부트캠프 초급 팀프로젝트 2팀 이태훈) (1).pdf](https://github.com/user-attachments/files/28632600/AI.2.1.pdf)
- 황인홍 https://concise-snowboard-3e4.notion.site/364082810eef80b49dabc5c1b76f25d2?v=364082810eef8031b2c3000caea27c76&source=copy_link 
- 김효진 https://insidious-flower-de8.notion.site/36912d20bf0a8063ac96d79c9c2c6e4a?source=copy_link


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
│   ├── train_yolo.py         # YOLO11 학습 (기본: yolo11m.pt)
│   ├── train_yolo26.py       # YOLO26 학습 (기본: yolo11x.pt, 앙상블 다양성용)
│   ├── train_rtdetr.py       # RT-DETR 학습
│   ├── train.py              # Faster R-CNN 학습
│   ├── inference_yolo.py     # YOLO11 단독 추론 → submission_yolo.csv
│   ├── inference_yolo26.py   # YOLO26 단독 추론 → submission_yolo26.csv
│   ├── inference_rtdetr.py   # RT-DETR 단독 추론
│   ├── inference_ensemble_vomega.py  # YOLO11 + YOLO26 + RT-DETR 3-모델 WBF 앙상블
│   ├── inference_yolo_ensemble.py    # YOLO 다중 체크포인트 앙상블
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
| 객체 탐지 모델 | YOLO11 · YOLO26 (ultralytics ≥ 8.3), RT-DETR, Faster R-CNN |
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
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  train_yolo.py    train_yolo26.py  train_rtdetr.py
  (YOLO11, m사이즈)  (YOLO26, x사이즈)  (RT-DETR)
        │               │               │
        ▼               ▼               ▼
  inference_yolo.py  inference_yolo26.py  inference_rtdetr.py
  (단독 추론)         (단독 추론)           (단독 추론)
        │               │               │
        └───────────────┼───────────────┘
                        ▼
        inference_ensemble_vomega.py  ← YOLO11 + YOLO26 + RT-DETR WBF 앙상블
                        │
                        ▼
              submission_final.csv   ← 최종 캐글 제출 파일
```

### Colab에서 실행하기

1. Google Drive에 프로젝트 폴더 및 데이터셋 zip 업로드
2. `baseline_vomega/colab_train_all.ipynb` 열기
3. 셀 2에서 `DRIVE_PROJECT_DIR` 경로만 수정
4. 위에서부터 순서대로 실행

**각 모델 학습 명령어:**
```bash
# YOLO11 (앙상블 모델 1)
python train_yolo.py --model yolo11m.pt --epochs 60 --batch 8 --imgsz 1280

# YOLO26 (앙상블 모델 2 — 더 큰 사이즈로 다양성 확보)
python train_yolo26.py --model yolo11x.pt --epochs 60 --batch 4 --imgsz 1280

# RT-DETR (앙상블 모델 3)
python train_rtdetr.py --model rtdetr-l.pt --epochs 150 --batch 16 --imgsz 1024
```

> 하이퍼파라미터 상세 설명 → [`baseline_vomega/하이퍼파라미터_설명서.md`](baseline_vomega/하이퍼파라미터_설명서.md)

---

## 모델 구성

### 앙상블 3-모델 구조

| 역할 | 모델 | 기본 사이즈 | imgsz | TTA | WBF 가중치 |
|---|---|---|---|---|---|
| 앙상블 모델 1 | **YOLO11** (`train_yolo.py`) | yolo11m.pt | 1280 | O | 2.0 |
| 앙상블 모델 2 | **YOLO26** (`train_yolo26.py`) | yolo11x.pt | 1280 | O | 2.0 |
| 앙상블 모델 3 | **RT-DETR** (`train_rtdetr.py`) | rtdetr-l.pt | 1024 | X | 1.0 |

> YOLO11과 YOLO26은 동일한 ultralytics YOLO 아키텍처를 다른 크기(m/x)로 학습해 앙상블 다양성을 확보합니다.  
> RT-DETR은 Transformer 기반으로 구조적 차이에 의한 예측 다양성을 제공합니다.

### 최종 앙상블 실행

```bash
python inference_ensemble_vomega.py \
    --yolo11 outputs/yolo/train/weights/best.pt \
    --yolo26 outputs/yolo/yolo26/weights/best.pt \
    --rtdetr outputs/yolo/rtdetr/weights/best.pt \
    --weights 2.0 2.0 1.0 \
    --wbf_iou 0.55
# → outputs/predictions/submission_final.csv
```

### 단독 추론 (모델별 성능 확인용)

```bash
# YOLO11 단독
python inference_yolo.py --checkpoint outputs/yolo/train/weights/best.pt
# → submission_yolo.csv

# YOLO26 단독
python inference_yolo26.py --checkpoint outputs/yolo/yolo26/weights/best.pt
# → submission_yolo26.csv
```

**추론 기본값:**
- Score threshold: `0.3` (앙상블 시 `0.15`)
- NMS threshold: `0.5`
- WBF IoU threshold: `0.55`
- 이미지당 최대 탐지: `4`개

### 베이스라인 비교 모델

| 모델 | 특징 |
|---|---|
| **Faster R-CNN** | PyTorch 레퍼런스 구현, 초기 베이스라인 용도 |

---

## 웹 애플리케이션 실행

```bash
실행 Sequence

준비 (최초 1회)
Ubuntu (WSL) 터미널에서(venv 인스톨되어있어야 함)
source .venv/bin/activate

로컬 PC에서만 쓸 때
터미널 1개만 필요

cd ver4_0529_FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

브라우저 → http://localhost:8000/

모바일 카메라로 쓸 때
터미널 2개 동시에 실행

터미널 1 — FastAPI 서버
source .venv/bin/activate
cd ver4_0529_FastAPI
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

터미널 2 — ngrok
ngrok http 8000

그러면 ngrok이 출력하는 주소 확인:
Forwarding   https://xxxx-xx-xx.ngrok.io/ → http://localhost:8000/
모바일 브라우저 → https://xxxx-xx-xx.ngrok.io/
```

**기능:**
- 카메라 실시간 촬영 또는 이미지 업로드
- 탐지된 알약의 금기사항·주의사항 즉시 표시
- 56종 한국 의약품 정보 DB 내장
- HITL(Human-In-The-Loop) 피드백 로깅 (`hitl_log.jsonl`)
- 단계별 추론 지연 시간 프로파일링


---

## 학습 출력 파일

| 파일 | 경로 | 설명 |
|---|---|---|
| YOLO11 가중치 | `outputs/yolo/train/weights/best.pt` | YOLO11 최적 체크포인트 |
| YOLO26 가중치 | `outputs/yolo/yolo26/weights/best.pt` | YOLO26 최적 체크포인트 |
| RT-DETR 가중치 | `outputs/yolo/rtdetr/weights/best.pt` | RT-DETR 최적 체크포인트 |
| 학습 결과 | `outputs/yolo/{run_name}/results.csv` | Epoch별 Loss/mAP |
| YOLO11 단독 제출 | `outputs/predictions/submission_yolo.csv` | YOLO11 단독 추론 결과 |
| YOLO26 단독 제출 | `outputs/predictions/submission_yolo26.csv` | YOLO26 단독 추론 결과 |
| **최종 앙상블 제출** | `outputs/predictions/submission_final.csv` | **3-모델 WBF 앙상블 (캐글 제출용)** |
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
