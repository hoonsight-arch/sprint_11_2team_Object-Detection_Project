"""
RT-DETR 단독 추론 스크립트 — submission_RTDETR.csv 생성

실행:
    python inference_rtdetr.py --checkpoint outputs/yolo/rtdetr/weights/best.pt
"""
import argparse
import csv
import json
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="outputs/yolo/rtdetr/weights/best.pt")
    parser.add_argument("--conf",       type=float, default=0.15, help="confidence threshold")
    parser.add_argument("--iou",        type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--max_det",    type=int,   default=4,   help="이미지당 최대 검출 수")
    parser.add_argument("--out",        default="outputs/predictions/submission_RTDETR.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import RTDETR

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_root = Path(cfg["data"]["data_root"])
    test_dir  = data_root / cfg["data"]["test_images"]
    out_path  = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 원본 category_id 매핑 로드 (YOLO 변환과 동일한 클래스 인덱스 → 원본 ID)
    orig_id_map_path = Path("data/yolo/orig_id_map.json")
    with open(orig_id_map_path) as f:
        orig_id_map = {int(k): v for k, v in json.load(f).items()}

    model = RTDETR(args.checkpoint)

    test_images = sorted(test_dir.glob("*.png"))
    print(f"테스트 이미지 {len(test_images)}개 RT-DETR 추론 중...")

    rows = []
    ann_id = 1

    # RT-DETR은 ultralytics TTA(augment)를 지원하지 않으므로 augment 미사용
    results = model.predict(
        source=str(test_dir),
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        save=False,
        verbose=False,
        stream=True,
    )

    for result in results:
        file_name = Path(result.path).name
        stem = Path(file_name).stem
        image_id = int(stem) if stem.isdigit() else stem

        boxes  = result.boxes.xyxy.cpu().tolist()
        labels = result.boxes.cls.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()

        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            orig_cat_id = orig_id_map.get(int(label), int(label))
            rows.append({
                "annotation_id": ann_id,
                "image_id":      image_id,
                "category_id":   orig_cat_id,
                "bbox_x":        round(x1, 2),
                "bbox_y":        round(y1, 2),
                "bbox_w":        round(x2 - x1, 2),
                "bbox_h":        round(y2 - y1, 2),
                "score":         round(score, 4),
            })
            ann_id += 1

    fieldnames = ["annotation_id", "image_id", "category_id",
                  "bbox_x", "bbox_y", "bbox_w", "bbox_h", "score"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"완료 — {len(rows)}개 예측 저장 → {out_path}")


if __name__ == "__main__":
    main()
