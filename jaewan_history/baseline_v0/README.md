# baseline_test — v0 구조 + YOLO11 단일 모델 동작 검증

## 목적

[baseline_v0/](../baseline_v0/) 템플릿(빈 모델 슬롯)에 [baseline_v2.0/](../baseline_v2.0/) 의 **YOLO11 단일 모델 경로**를 그대로 끼워 넣은 테스트 빌드.
**파이프라인 전체가 의도대로 돌아가는지** 확인하는 용도입니다 (Drive 복제 → 로컬 학습 → 추론 → Drive 백업).

## v0 와 무엇이 다른가

| 항목 | v0 | baseline_test |
|---|---|---|
| 코드 (data / utils / scripts / configs / *_yolo*.py) | 동일 | 동일 (변경 없음) |
| `models/detector.py` | `NotImplementedError` (빈 슬롯) | **그대로 둠** — YOLO 경로는 이 파일을 안 씀 |
| `colab_run.ipynb` | Faster R-CNN + YOLO + 앙상블 다 포함 | **YOLO 단일 모델 경로만** 남김 |

> Faster R-CNN 셀(`train.py` / `inference.py`)과 앙상블 셀은 노트북에서 제거했지만, 스크립트 파일 자체는 v0 그대로 남겨 두었습니다 (v0 구조 보존).

## 실행

콜랩에서 [colab_run.ipynb](colab_run.ipynb) 를 위에서부터 순서대로 실행하면 됩니다.

1. Drive 마운트
2. `DRIVE_PROJECT_DIR` 한 줄만 수정
3. Drive → 로컬 복제 (데이터셋 zip 해제 포함)
4. config 자동 갱신
5. 패키지 설치 (`requirements.txt` + `ultralytics`)
6. GPU 확인
7. `preprocess.py`
8. `convert_to_yolo.py`
9. `train_yolo.py` — 기본값: `yolo11m.pt`, **3 epoch, imgsz 640, batch 8** (테스트 우선, 빠르게 끝남)
10. `inference_yolo.py` → `outputs/predictions/submission_yolo.csv` 생성 + 미리보기
11. `outputs/predictions/`, `outputs/yolo/` 를 Drive 백업

## 본학습으로 바꿀 때

셀 9 의 학습 인자를 다음 정도로 올리세요:
```bash
--epochs 60 --batch 8 --imgsz 1280
```

## 검증 체크리스트

- [ ] 셀 3: dataset zip 해제가 수 분 안에 끝나는가
- [ ] 셀 7: `data/processed/` 에 어노테이션이 생성되는가
- [ ] 셀 8: `data/yolo/{images,labels}/{train,val}` 가 생성되는가
- [ ] 셀 9: 학습이 GPU 에서 돌고 `outputs/yolo/test_run/weights/best.pt` 가 생기는가
- [ ] 셀 10: `outputs/predictions/submission_yolo.csv` 가 생성되고 행 수가 0 보다 많은가
- [ ] 셀 11: Drive 의 `outputs/` 아래로 백업되는가
