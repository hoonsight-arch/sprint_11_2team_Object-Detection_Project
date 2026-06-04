"""
baseline_vomega 최종 앙상블 추론 — yolo11x + yolo26x + RT-DETR 3-모델 WBF
→ submission_final.csv 생성

WBF(Weighted Box Fusion)는 [0,1] 정규화 좌표로 박스를 합치므로 모델별 imgsz가 달라도 된다.
- YOLO 2개: 고해상도(1280) + TTA(augment=True)
- RT-DETR : 자기 해상도(1024), TTA 미지원이라 augment=False
- weights : 검증된 YOLO 2개에 더 큰 가중치, 신규 RT-DETR은 작게 (기본 2,2,1)

세 모델 모두 같은 convert_to_yolo.py 결과(56종, 동일 정렬)로 학습돼 클래스 인덱스가 일치해야 한다.

실행:
    python inference_ensemble_vomega.py \
        --yolo11 outputs/yolo/weights/yolo11.pt \
        --yolo26 outputs/yolo/weights/yolo26.pt \
        --rtdetr outputs/yolo/rtdetr/weights/best.pt
"""
import argparse
import csv
import json
from pathlib import Path

import yaml
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/default.yaml")
    parser.add_argument("--yolo11",      required=True, help="yolo11x best.pt 경로")
    parser.add_argument("--yolo26",      required=True, help="yolo26x best.pt 경로")
    parser.add_argument("--rtdetr",      required=True, help="RT-DETR best.pt 경로")
    parser.add_argument("--weights",     type=float, nargs=3, default=[2.0, 2.0, 1.0],
                        help="WBF 모델 가중치 [yolo11 yolo26 rtdetr]")
    parser.add_argument("--yolo_imgsz",  type=int,   default=1280, help="YOLO 추론 해상도")
    parser.add_argument("--rtdetr_imgsz",type=int,   default=1024, help="RT-DETR 추론 해상도")
    parser.add_argument("--conf",        type=float, default=0.15, help="모델별 confidence threshold")
    parser.add_argument("--iou",         type=float, default=0.6,  help="모델 내부 NMS IoU threshold")
    parser.add_argument("--wbf_iou",     type=float, default=0.55, help="WBF IoU threshold")
    parser.add_argument("--max_det",     type=int,   default=4,    help="이미지당 최대 검출 수")
    parser.add_argument("--out",         default="outputs/predictions/submission_final.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO, RTDETR
    from ensemble_boxes import weighted_boxes_fusion

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["data"]["data_root"])
    test_dir  = data_root / cfg["data"]["test_images"]
    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    orig_id_map_path = Path("data/yolo/orig_id_map.json")
    with open(orig_id_map_path) as f:
        orig_id_map = {int(k): v for k, v in json.load(f).items()}

    # 각 모델 스펙: (이름, 모델객체, imgsz, augment사용여부)
    specs = [
        ("yolo11", YOLO(args.yolo11),   args.yolo_imgsz,   True),
        ("yolo26", YOLO(args.yolo26),   args.yolo_imgsz,   True),
        ("rtdetr", RTDETR(args.rtdetr), args.rtdetr_imgsz, False),
    ]
    print("앙상블 모델 3개 로드 완료:")
    for name, _, imgsz, aug in specs:
        print(f"  - {name:7s} imgsz={imgsz} augment={aug}")
    print(f"  WBF weights = {args.weights}  wbf_iou = {args.wbf_iou}")

    test_images = sorted(test_dir.glob("*.png"))
    print(f"\n테스트 이미지 {len(test_images)}개 앙상블 추론 중...")

    rows = []
    ann_id = 1

    for img_path in test_images:
        img = Image.open(img_path)
        W, H = img.size

        stem = img_path.stem
        image_id = int(stem) if stem.isdigit() else stem

        all_boxes, all_scores, all_labels = [], [], []

        for _name, model, imgsz, use_aug in specs:
            result = model.predict(
                source=str(img_path),
                imgsz=imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                augment=use_aug,
                save=False,
                verbose=False,
            )[0]

            boxes  = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            labels = result.boxes.cls.cpu().tolist()

            # WBF는 [0,1] 정규화 좌표 필요
            norm_boxes = [
                [max(0.0, x1/W), max(0.0, y1/H), min(1.0, x2/W), min(1.0, y2/H)]
                for x1, y1, x2, y2 in boxes
            ]
            all_boxes.append(norm_boxes)
            all_scores.append(scores)
            all_labels.append([int(l) for l in labels])

        # 모든 모델 예측이 비어있으면 스킵
        if not any(b for b in all_boxes):
            continue

        merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            weights=args.weights,
            iou_thr=args.wbf_iou,
            skip_box_thr=args.conf,
        )

        # max_det 제한 (score 높은 순)
        if len(merged_scores) > args.max_det:
            top_idx = sorted(range(len(merged_scores)),
                             key=lambda i: merged_scores[i], reverse=True)[:args.max_det]
            merged_boxes  = [merged_boxes[i]  for i in top_idx]
            merged_scores = [merged_scores[i] for i in top_idx]
            merged_labels = [merged_labels[i] for i in top_idx]

        for box, score, label in zip(merged_boxes, merged_scores, merged_labels):
            x1, y1, x2, y2 = box[0]*W, box[1]*H, box[2]*W, box[3]*H
            orig_cat_id = orig_id_map.get(int(label), int(label))
            rows.append({
                "annotation_id": ann_id,
                "image_id":      image_id,
                "category_id":   orig_cat_id,
                "bbox_x":        round(x1, 2),
                "bbox_y":        round(y1, 2),
                "bbox_w":        round(x2 - x1, 2),
                "bbox_h":        round(y2 - y1, 2),
                "score":         round(float(score), 4),
            })
            ann_id += 1

    fieldnames = ["annotation_id", "image_id", "category_id",
                  "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n완료 — {len(rows)}개 예측 저장 → {out_path}")


if __name__ == "__main__":
    main()
