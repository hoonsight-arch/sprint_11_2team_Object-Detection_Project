"""
RT-DETR 학습 스크립트 (baseline_vomega 앙상블용 3번째 모델)

YOLO 2개(yolo11x, yolo26x)와 같은 dataset.yaml로 학습해 클래스 인덱스를 맞춘다.
train_yolo.py와 동일한 백업/재개 구조를 사용한다.

실행:
    python train_rtdetr.py
    python train_rtdetr.py --model rtdetr-l.pt --epochs 150 --batch 16 --imgsz 1024

세션이 죽어도 이어서 학습하려면:
    python train_rtdetr.py ... --backup_dir /content/drive/.../outputs/yolo/<RUN_NAME> --backup_period 10
    python train_rtdetr.py ... --backup_dir <같은 경로> --resume
"""
import argparse
import shutil
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="rtdetr-l.pt", help="rtdetr-l.pt 또는 rtdetr-x.pt")
    parser.add_argument("--epochs",     type=int,   default=150)
    parser.add_argument("--batch",      type=int,   default=16, help="A100 80GB + rtdetr-l + imgsz1024 기준 16, 여유로 24까지")
    parser.add_argument("--imgsz",      type=int,   default=1024)
    parser.add_argument("--patience",   type=int,   default=50, help="early stop: DETR 계열은 수렴이 느려 YOLO보다 크게")
    parser.add_argument("--data",       default="data/yolo/dataset.yaml")
    parser.add_argument("--output",     default="outputs/yolo")
    parser.add_argument("--name",       default="rtdetr")
    # 재개 / 백업 옵션
    parser.add_argument("--resume",        action="store_true", help="last.pt에서 학습 이어가기")
    parser.add_argument("--save_period",   type=int, default=-1, help="N에폭마다 epochN.pt 영구 저장 (-1=끔)")
    parser.add_argument("--backup_dir",    default="", help="N에폭마다 가중치를 복사할 Drive 경로 (비우면 백업 끔)")
    parser.add_argument("--backup_period", type=int, default=10, help="backup_dir로 복사하는 주기(에폭)")
    return parser.parse_args()


def _backup_weights(save_dir, backup_dir):
    """학습 재개에 필요한 파일(last.pt, best.pt, results.csv)을 backup_dir로 복사."""
    save_dir = Path(save_dir)
    dst_w = Path(backup_dir) / "weights"
    dst_w.mkdir(parents=True, exist_ok=True)
    for fname in ("last.pt", "best.pt"):
        src = save_dir / "weights" / fname
        if src.exists():
            try:
                shutil.copy2(src, dst_w / fname)
            except Exception as e:
                print(f"[backup] {fname} 복사 실패: {e}")
    res = save_dir / "results.csv"
    if res.exists():
        try:
            shutil.copy2(res, Path(backup_dir) / "results.csv")
        except Exception as e:
            print(f"[backup] results.csv 복사 실패: {e}")


def _restore_from_backup(run_dir, backup_dir):
    """resume용: 로컬에 last.pt가 없으면 Drive 백업에서 내려받아 복원."""
    bak_last = Path(backup_dir) / "weights" / "last.pt"
    if not bak_last.exists():
        return False
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    shutil.copy2(bak_last, run_dir / "weights" / "last.pt")
    bak_best = Path(backup_dir) / "weights" / "best.pt"
    if bak_best.exists():
        shutil.copy2(bak_best, run_dir / "weights" / "best.pt")
    bak_res = Path(backup_dir) / "results.csv"
    if bak_res.exists():
        shutil.copy2(bak_res, run_dir / "results.csv")
    print(f"[resume] Drive 백업에서 복원 완료 → {run_dir / 'weights' / 'last.pt'}")
    return True


def main():
    args = parse_args()
    from ultralytics import RTDETR

    if not Path(args.data).exists():
        print("dataset.yaml 없음 → 먼저 실행: python scripts/convert_to_yolo.py")
        return

    project = str(Path(args.output).resolve())
    run_dir = Path(project) / args.name
    last_pt = run_dir / "weights" / "last.pt"

    device = 0 if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("⚠️  GPU 없음 → CPU로 학습 (매우 느림). Colab에서 GPU 런타임을 선택하세요.")

    # ── resume 준비: 로컬 last.pt 없으면 Drive 백업에서 복원 ──
    if args.resume and not last_pt.exists() and args.backup_dir:
        _restore_from_backup(run_dir, args.backup_dir)
    if args.resume and not last_pt.exists():
        print(f"⚠️  resume용 last.pt 없음: {last_pt} → 처음부터 학습으로 전환")
        args.resume = False

    model = RTDETR(str(last_pt)) if args.resume else RTDETR(args.model)

    # ── N에폭마다 Drive로 백업하는 콜백 등록 ──
    if args.backup_dir and args.backup_period > 0:
        def _on_fit_epoch_end(trainer):
            epoch = int(trainer.epoch) + 1  # 0-indexed → 사람 기준
            if epoch % args.backup_period == 0:
                _backup_weights(trainer.save_dir, args.backup_dir)
                print(f"[backup] {epoch}에폭 → Drive 복사 완료: {args.backup_dir}")
        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

    if args.resume:
        print(f"[resume] {last_pt} 에서 학습 재개")
        result = model.train(resume=True)
    else:
        result = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            project=project,
            name=args.name,
            exist_ok=True,
            device=device,
            patience=args.patience,
            cos_lr=True,
            save=True,
            save_period=args.save_period,
            plots=True,
        )

    # ── 학습 종료 후 최종 가중치도 Drive로 한 번 더 백업 ──
    if args.backup_dir:
        _backup_weights(result.save_dir, args.backup_dir)
        print(f"[backup] 최종 가중치 → Drive: {args.backup_dir}")

    best_pt = Path(result.save_dir) / "weights" / "best.pt"
    print(f"\n학습 완료 → {best_pt}")


if __name__ == "__main__":
    main()
