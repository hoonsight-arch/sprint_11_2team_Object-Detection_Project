"""
YOLO11 학습 스크립트

실행:
    python train_yolo.py
    python train_yolo.py --model yolo11m.pt --epochs 30 --batch 8
"""
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="yolo11m.pt", help="모델 크기: yolo11n/s/m/l/x.pt")
    parser.add_argument("--epochs",  type=int,   default=30)
    parser.add_argument("--batch",   type=int,   default=8)
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--data",    default="data/yolo/dataset.yaml")
    parser.add_argument("--output",  default="outputs/yolo")
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    if not Path(args.data).exists():
        print("dataset.yaml 없음 → 먼저 실행: python scripts/convert_to_yolo.py")
        return

    model = YOLO(args.model)

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.output,
        name="train",
        exist_ok=True,
        device=0,           # GPU 사용 (없으면 'cpu'로 자동 전환)
        patience=10,        # early stopping
        save=True,
        plots=True,
    )

    print(f"\n학습 완료 → {args.output}/train/weights/best.pt")


if __name__ == "__main__":
    main()
